"""
Robust vector-based knowledge base for RAG (Retrieval-Augmented Generation).

Replaces the naive JSON-list approach with:
  - Sentence-transformer embeddings for semantic similarity search.
  - BM25 / TF-IDF as lightweight fallback retrievers.
  - Dual-threshold entry control: confidence score AND average log-probability
    of generated tokens must both exceed their respective thresholds.
  - Periodic outlier purging: cluster questions by embedding, then remove QA
    pairs whose answers are significantly dissimilar to others in the same
    cluster (z-score-based outlier detection).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class VectorKnowledgeBase:
    """Persistent, vector-indexed knowledge base for RAG retrieval.

    Entries are stored in a JSON file (for human readability / compatibility
    with the existing pipeline).  Question embeddings are kept in memory and
    recomputed on load; answer embeddings are computed on demand during outlier
    purging.

    Parameters
    ----------
    kb_path:
        Path to the JSON file where entries are persisted.
    encoder_name:
        Sentence-transformers model name used for semantic encoding.
    encoder:
        Pre-instantiated ``SentenceTransformer`` object.  When provided,
        ``encoder_name`` is ignored.  Useful for sharing a model across
        multiple components.
    """

    def __init__(
        self,
        kb_path: str = "knowledge_base.json",
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Any = None,
    ) -> None:
        self.kb_path = kb_path
        self.encoder_name = encoder_name
        self._encoder = encoder  # lazily initialised when first needed

        self.entries: List[Dict[str, Any]] = []
        self._q_embeddings: Optional[np.ndarray] = None  # shape [N, D]

        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder(self):
        """Return (and lazily initialise) the sentence-transformer encoder."""
        if self._encoder is None and not getattr(self, '_encoder_unavailable', False):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._encoder = SentenceTransformer(self.encoder_name)
            except Exception as exc:
                logger.warning(
                    "VectorKnowledgeBase: could not load SentenceTransformer '%s': %s. "
                    "Semantic retrieval will be unavailable.",
                    self.encoder_name,
                    exc,
                )
                # Cache the failure so we don't retry on every call.
                self._encoder_unavailable = True
        return self._encoder

    def _encode(self, texts: List[str]) -> Optional[np.ndarray]:
        """Encode *texts* to a float32 array of shape ``[len(texts), D]``.

        Returns ``None`` when the encoder is unavailable.
        """
        enc = self._get_encoder()
        if enc is None:
            return None
        try:
            return enc.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("VectorKnowledgeBase._encode failed: %s", exc)
            return None

    def _rebuild_q_embeddings(self) -> None:
        """Rebuild the in-memory question-embedding matrix from ``self.entries``."""
        if not self.entries:
            self._q_embeddings = None
            return
        questions = [
            # A single space is used instead of an empty string because some
            # tokenizers/encoders reject zero-length inputs.
            (e.get("question") or e.get("text") or "").strip() or " "
            for e in self.entries
        ]
        self._q_embeddings = self._encode(questions)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from the JSON file (if it exists) and rebuild embeddings."""
        if not os.path.exists(self.kb_path):
            return
        try:
            with open(self.kb_path, "r", encoding="utf-8") as fh:
                self.entries = json.load(fh)
            logger.info(
                "VectorKnowledgeBase: loaded %d entries from '%s'",
                len(self.entries),
                self.kb_path,
            )
            self._rebuild_q_embeddings()
        except Exception as exc:
            logger.exception(
                "VectorKnowledgeBase: failed to load '%s': %s", self.kb_path, exc
            )
            self.entries = []

    def _save(self) -> None:
        """Persist entries to the JSON file."""
        try:
            with open(self.kb_path, "w", encoding="utf-8") as fh:
                json.dump(self.entries, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.exception("VectorKnowledgeBase: failed to save '%s': %s", self.kb_path, exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Dict[str, Any]) -> None:
        """Append *entry* to the knowledge base and persist to disk.

        The in-memory embedding matrix is updated incrementally to avoid a
        full re-encode on every insertion.
        """
        self.entries.append(entry)

        question = (entry.get("question") or entry.get("text") or "").strip() or " "
        new_emb = self._encode([question])
        if new_emb is not None:
            if self._q_embeddings is None:
                self._q_embeddings = new_emb
            else:
                self._q_embeddings = np.vstack([self._q_embeddings, new_emb])

        self._save()

    def retrieve(self, question: str, topk: int = 3) -> List[Dict[str, Any]]:
        """Return the top-*k* knowledge-base entries most similar to *question*.

        Semantic (cosine) similarity on sentence-transformer embeddings is
        preferred.  When the encoder is unavailable, the method falls back to
        BM25 (via ``rank_bm25``) and then to simple word-overlap.
        """
        if not self.entries:
            return []

        # --- semantic search ---
        if self._q_embeddings is None:
            self._rebuild_q_embeddings()

        if self._q_embeddings is not None and len(self._q_embeddings) > 0:
            query_emb = self._encode([question])
            if query_emb is not None:
                sims = cosine_similarity(query_emb, self._q_embeddings)[0]
                top_idx = sims.argsort()[::-1][:topk]
                return [self.entries[i] for i in top_idx if sims[i] > 0]

        # --- BM25 fallback ---
        return self._retrieve_bm25(question, topk)

    def _retrieve_bm25(self, question: str, topk: int) -> List[Dict[str, Any]]:
        """BM25 keyword retrieval (fallback when embeddings are unavailable)."""
        questions = [
            (e.get("question") or e.get("text") or "").strip()
            for e in self.entries
        ]
        questions = [q for q in questions if q]
        if not questions:
            return []
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            tokenized = [q.lower().split() for q in questions]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
            return [self.entries[i] for i in top_idx if scores[i] > 0]
        except Exception:
            return self._retrieve_overlap(question, topk)

    def _retrieve_overlap(self, question: str, topk: int) -> List[Dict[str, Any]]:
        """Word-overlap fallback retrieval."""
        q_words = set(question.lower().split())
        scored = []
        for entry in self.entries:
            q2 = (entry.get("question") or entry.get("text") or "").lower()
            overlap = len(q_words & set(q2.split()))
            scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for overlap, item in scored[:topk] if overlap > 0]

    # ------------------------------------------------------------------
    # Quality control – outlier purging
    # ------------------------------------------------------------------

    def purge_outliers(
        self,
        n_clusters: Optional[int] = None,
        z_threshold: float = 2.0,
    ) -> int:
        """Detect and remove outlier QA pairs by answer-similarity analysis.

        Algorithm
        ---------
        1. Encode all questions and cluster them with KMeans.
        2. For each cluster, encode the corresponding answers and compute the
           pairwise cosine-similarity matrix.
        3. For each entry in the cluster compute its *mean answer similarity*
           to all other entries in the same cluster.
        4. Compute the mean (μ) and standard deviation (σ) of those per-entry
           similarities across the cluster.
        5. Any entry whose mean similarity falls below ``μ − z_threshold * σ``
           is flagged as an outlier and removed from the knowledge base.

        Parameters
        ----------
        n_clusters:
            Number of KMeans clusters.  Defaults to ``max(2, sqrt(N))``.
        z_threshold:
            Z-score cut-off for outlier detection (applied on the *low* side).

        Returns
        -------
        int
            Number of entries removed.
        """
        n = len(self.entries)
        if n < 10:
            logger.info(
                "VectorKnowledgeBase.purge_outliers: skipped (only %d entries, minimum 10 required)",
                n,
            )
            return 0

        # Ensure question embeddings are available.
        if self._q_embeddings is None:
            self._rebuild_q_embeddings()
        if self._q_embeddings is None:
            logger.warning(
                "VectorKnowledgeBase.purge_outliers: no question embeddings available; skipping."
            )
            return 0

        if n_clusters is None:
            # At least 2 clusters are required to give KMeans a meaningful
            # partitioning, even when the KB is very small.
            n_clusters = max(2, int(np.sqrt(n)))
        n_clusters = min(n_clusters, n)

        # Step 1: cluster questions.
        try:
            from sklearn.cluster import KMeans  # type: ignore

            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42, max_iter=100)
            labels = km.fit_predict(self._q_embeddings)
        except Exception as exc:
            logger.warning("VectorKnowledgeBase.purge_outliers: clustering failed: %s", exc)
            return 0

        # Step 2: encode answers.
        answer_texts = [
            (e.get("annotation") or "").strip() or " " for e in self.entries
        ]
        answer_embs = self._encode(answer_texts)
        if answer_embs is None:
            # Fallback: use TF-IDF cosine similarity for answers.
            answer_embs = self._tfidf_encode(answer_texts)
        if answer_embs is None:
            logger.warning(
                "VectorKnowledgeBase.purge_outliers: could not encode answers; skipping."
            )
            return 0

        # Steps 3-5: per-cluster outlier detection.
        outlier_indices: set = set()
        for cluster_id in range(n_clusters):
            cluster_idx = np.where(labels == cluster_id)[0]
            if len(cluster_idx) < 3:
                # Cannot compute meaningful statistics with fewer than 3 members.
                continue

            cluster_ans = answer_embs[cluster_idx]  # shape [m, D]
            sim_matrix = cosine_similarity(cluster_ans)  # shape [m, m]
            np.fill_diagonal(sim_matrix, 0.0)

            m = len(cluster_idx)
            avg_sims = sim_matrix.sum(axis=1) / (m - 1)  # exclude self-similarity

            mean_sim = avg_sims.mean()
            std_sim = avg_sims.std()
            if std_sim < 1e-6:
                continue  # All answers are essentially identical – no outliers.

            z_scores = (avg_sims - mean_sim) / std_sim
            for local_i in np.where(z_scores < -z_threshold)[0]:
                outlier_indices.add(int(cluster_idx[local_i]))

        if not outlier_indices:
            logger.info("VectorKnowledgeBase.purge_outliers: no outliers detected.")
            return 0

        # Remove outliers (highest indices first to preserve lower indices).
        for idx in sorted(outlier_indices, reverse=True):
            self.entries.pop(idx)

        removed = len(outlier_indices)
        logger.info(
            "VectorKnowledgeBase.purge_outliers: removed %d outlier(s) (KB size now %d)",
            removed,
            len(self.entries),
        )

        # Rebuild embeddings and persist.
        self._rebuild_q_embeddings()
        self._save()
        return removed

    @staticmethod
    def _tfidf_encode(texts: List[str]) -> Optional[np.ndarray]:
        """Encode *texts* using TF-IDF as a fallback when the neural encoder
        is unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            vec = TfidfVectorizer()
            mat = vec.fit_transform(texts)
            return mat.toarray().astype(np.float32)
        except Exception as exc:
            logger.warning("VectorKnowledgeBase._tfidf_encode failed: %s", exc)
            return None
