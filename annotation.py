import random
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from base_structure.base_task import Task
from tasks.qa import QATask
from rag import VectorKnowledgeBase


class HumanReviewQueue:
    """Human review queue for samples that fail quality thresholds."""
    def __init__(self):
        self.queue = []

    def add(self, sample: Dict[str, Any]):
        self.queue.append(sample)

    def export(self, filepath: str = "human_review.json"):
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, ensure_ascii=False, indent=2)
        logger = logging.getLogger(__name__)
        logger.info("Human review queue exported to %s", filepath)


class Annotator:
    """Use LLMs to annotate data samples with optional RAG and knowledge base support.

    Parameters
    ----------
    candidate_llms:
        List of LLM identifiers available for routing.
    llm_dict:
        Mapping from LLM identifier to an ``LLMBase`` instance.
    confidence_threshold:
        Minimum confidence score (0–1) required for a sample to be admitted
        to the knowledge base.
    avg_logprob_threshold:
        Minimum average log-probability of generated tokens required for KB
        admission.  Log-probs are always ≤ 0; values closer to 0 indicate
        higher confidence (e.g. ``-1.0``).  Set to ``None`` to disable.
    rag:
        Whether to enable RAG context injection at inference time.
    rag_method:
        Legacy option kept for backward compatibility.  The knowledge base
        now always uses semantic (sentence-transformer) search with BM25 as
        a fallback.  Ignored when ``rag=True``.
    kb_path:
        Path to the JSON file that backs the knowledge base.
    kb_encoder_name:
        Sentence-transformer model used for knowledge-base embeddings.
    kb_encoder:
        Pre-built encoder object (shared from another component).  When
        provided, ``kb_encoder_name`` is ignored.
    task:
        Task object defining prompt construction and output parsing.
    outlier_purge_interval:
        Number of new KB additions between consecutive outlier-purge runs.
        Set to ``0`` to disable periodic purging.
    outlier_z_threshold:
        Z-score cut-off for answer-similarity outlier removal.
    """

    def __init__(
        self,
        candidate_llms,
        llm_dict,
        confidence_threshold: float = 0.7,
        avg_logprob_threshold: float = None,
        rag: bool = False,
        rag_method: str = "bm25",
        kb_path: str = "knowledge_base.json",
        kb_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        kb_encoder=None,
        task: Task = None,
        outlier_purge_interval: int = 50,
        outlier_z_threshold: float = 2.0,
    ):
        self.candidate_llms = candidate_llms
        self.llm_dict = llm_dict
        self.confidence_threshold = confidence_threshold
        self.avg_logprob_threshold = avg_logprob_threshold
        self.human_review_queue = HumanReviewQueue()
        self.rag = rag
        # rag_method kept for backward-compat but VectorKnowledgeBase is always used.
        self.rag_method = rag_method.lower() if rag_method else "bm25"
        self.task = task or QATask()

        self.outlier_purge_interval = outlier_purge_interval
        self.outlier_z_threshold = outlier_z_threshold
        self._additions_since_purge: int = 0

        # Robust vector-based knowledge base.
        self.knowledge_base = VectorKnowledgeBase(
            kb_path=kb_path,
            encoder_name=kb_encoder_name,
            encoder=kb_encoder,
        )

    # ------------------------------------------------------------------
    # Backward-compatible property so code that reads
    # ``annotator.kb_path`` still works.
    # ------------------------------------------------------------------

    @property
    def kb_path(self) -> str:
        return self.knowledge_base.kb_path

    # ------------------------------------------------------------------
    # Deprecated helpers kept for backward compatibility
    # ------------------------------------------------------------------

    def _load_knowledge_base(self):
        """Deprecated – the VectorKnowledgeBase manages persistence internally."""
        return self.knowledge_base.entries

    def _save_knowledge_base(self):
        """Deprecated – the VectorKnowledgeBase manages persistence internally."""
        self.knowledge_base._save()

    def _rag_retrieve(self, question: str, topk: int = 3):
        """Delegate to VectorKnowledgeBase.retrieve (kept for back-compat)."""
        return self.knowledge_base.retrieve(question, topk=topk)

    # ------------------------------------------------------------------
    # Core annotation logic
    # ------------------------------------------------------------------

    def _passes_thresholds(self, conf, avg_logprob) -> bool:
        """Return True when *both* quality thresholds are satisfied.

        The confidence threshold is mandatory.  The log-probability threshold
        is only applied when ``avg_logprob_threshold`` is configured *and*
        the LLM actually returned a log-probability value (i.e. the value is
        not ``None``).
        """
        if conf < self.confidence_threshold:
            return False
        if (
            self.avg_logprob_threshold is not None
            and avg_logprob is not None
            and avg_logprob < self.avg_logprob_threshold
        ):
            return False
        return True

    def annotate(self, sample: Dict[str, Any], assigned_llm: str = None) -> Dict[str, Any]:
        if self.rag:
            # RAG 检索相似历史标注
            rag_examples = self.knowledge_base.retrieve(sample.get("question", ""), topk=3)
            # 通过Task对象生成prompt
            prompt = self.task.get_prompt(sample, rag_examples)
        else:
            prompt = self.task.get_prompt(sample)
        if assigned_llm is None:
            llm = sample.get('route')
        else:
            llm = assigned_llm
        if llm not in self.candidate_llms:
            best_llm = self.candidate_llms[0]
            for candidate in self.candidate_llms:
                if candidate in str(llm):
                    best_llm = candidate
            llm = best_llm

        # Use generate_with_logprobs when available for dual-threshold control.
        llm_obj = self.llm_dict[llm]
        avg_logprob = None
        if hasattr(llm_obj, 'generate_with_logprobs'):
            try:
                output, avg_logprob = llm_obj.generate_with_logprobs(prompt, max_new_tokens=50)
            except Exception:
                output = llm_obj.generate(prompt, max_new_tokens=50)
        else:
            output = llm_obj.generate(prompt, max_new_tokens=50)

        # 通过Task对象解析LLM输出
        parsed = self.task.parse_output(output)
        annotation = parsed.get("annotation", "unknown")
        conf = parsed.get("confidence", random.random())

        result = {**sample, "route": llm, "annotation": annotation, "confidence": conf}
        if avg_logprob is not None:
            result["avg_logprob"] = avg_logprob

        if not self._passes_thresholds(conf, avg_logprob):
            result["needs_human"] = True
            self.human_review_queue.add(result)
        else:
            result["needs_human"] = False
            # 标注通过的样本加入知识库
            self.knowledge_base.add(result)
            # Trigger periodic outlier purging.
            if self.outlier_purge_interval > 0:
                self._additions_since_purge += 1
                if self._additions_since_purge >= self.outlier_purge_interval:
                    self._additions_since_purge = 0
                    self.knowledge_base.purge_outliers(z_threshold=self.outlier_z_threshold)
        return result

    def annotate_batch(self, dataset: List[Dict[str, Any]], assigned_llm: str = None, progress_cb=None) -> List[Dict[str, Any]]:
        # normalize items: accept strings or non-dict items from routers/filters
        normalized = []
        for d in dataset:
            if isinstance(d, str):
                normalized.append({"text": d})
            elif isinstance(d, dict):
                normalized.append(d)
            else:
                try:
                    if hasattr(d, 'to_dict'):
                        normalized.append(d.to_dict())
                    elif hasattr(d, 'to_list'):
                        normalized.append({'items': d.to_list()})
                    else:
                        normalized.append(dict(d))
                except Exception:
                    normalized.append({"text": str(d)})

        results = []
        total = len(normalized)
        # If caller did not pass a progress_cb, allow instance-level `self.progress_cb` to be used
        if progress_cb is None:
            progress_cb = getattr(self, 'progress_cb', None)
        for i, d in enumerate(tqdm(normalized)):
            res = self.annotate(d, assigned_llm)
            results.append(res)
            # report progress if callback provided: (current, total, info)
            try:
                if progress_cb is not None:
                    progress_cb(i + 1, total, {'assigned_llm': assigned_llm})
            except Exception:
                pass
        return results
