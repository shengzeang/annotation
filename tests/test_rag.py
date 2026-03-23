"""
Unit tests for the robust RAG implementation.

Covers:
  - VectorKnowledgeBase: creation, add, retrieve (semantic + BM25 fallback)
  - VectorKnowledgeBase.purge_outliers: outlier removal logic
  - Annotator: dual-threshold entry control (confidence + avg_logprob)
  - LocalLLM / APILLM: generate_with_logprobs interface (mocked)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure the project root is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag import VectorKnowledgeBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entries(questions, annotations):
    """Build a list of KB-style dicts."""
    return [
        {"id": str(i), "question": q, "annotation": a, "confidence": 0.9}
        for i, (q, a) in enumerate(zip(questions, annotations))
    ]


def _mock_encoder(texts):
    """A deterministic pseudo-encoder that maps texts to fixed 4-D vectors."""
    rng = np.random.default_rng(abs(hash(frozenset(texts))))
    mat = rng.standard_normal((len(texts), 4)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


class FakeEncoder:
    """Fake SentenceTransformer-compatible encoder for testing."""

    def __init__(self, fixed_vecs: np.ndarray):
        self._vecs = fixed_vecs
        self._call_count = 0

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        n = len(texts)
        return np.vstack([self._vecs[i % len(self._vecs)] for i in range(n)])


# ---------------------------------------------------------------------------
# VectorKnowledgeBase – basic operations
# ---------------------------------------------------------------------------

class TestVectorKnowledgeBaseBasic(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)  # Let VectorKnowledgeBase create it fresh.

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def _make_kb(self, encoder=None):
        return VectorKnowledgeBase(
            kb_path=self.tmp.name,
            encoder=encoder,
        )

    # --- empty KB ---

    def test_empty_kb_has_zero_length(self):
        kb = self._make_kb()
        self.assertEqual(len(kb), 0)

    def test_empty_kb_retrieve_returns_empty(self):
        kb = self._make_kb()
        self.assertEqual(kb.retrieve("anything"), [])

    # --- add persists to disk ---

    def test_add_persists_entry(self):
        kb = self._make_kb()
        entry = {"id": "1", "question": "What is AI?", "annotation": "Artificial Intelligence"}
        kb.add(entry)
        self.assertEqual(len(kb), 1)
        self.assertTrue(os.path.exists(self.tmp.name))
        with open(self.tmp.name) as fh:
            saved = json.load(fh)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], "1")

    def test_add_multiple_entries(self):
        kb = self._make_kb()
        for i in range(5):
            kb.add({"id": str(i), "question": f"Q{i}", "annotation": f"A{i}"})
        self.assertEqual(len(kb), 5)

    # --- reload from disk ---

    def test_reload_restores_entries(self):
        kb = self._make_kb()
        kb.add({"id": "42", "question": "Reload test", "annotation": "ok"})
        # Create a fresh instance from same file.
        kb2 = self._make_kb()
        self.assertEqual(len(kb2), 1)
        self.assertEqual(kb2.entries[0]["id"], "42")

    # --- BM25 fallback retrieval (no encoder) ---

    def test_bm25_fallback_retrieve(self):
        kb = self._make_kb(encoder=None)
        # Patch the lazy encoder initialisation to always fail so BM25 is used.
        kb._encoder = None
        kb._q_embeddings = None
        with patch.object(kb, '_get_encoder', return_value=None):
            kb.entries = _make_entries(
                ["machine learning basics", "deep learning neural networks", "natural language processing"],
                ["ML basics answer", "DL answer", "NLP answer"],
            )
            results = kb.retrieve("machine learning", topk=2)
        # Should return at least one result with BM25.
        self.assertGreater(len(results), 0)
        questions_returned = [r["question"] for r in results]
        self.assertIn("machine learning basics", questions_returned)

    # --- semantic retrieval with fake encoder ---

    def test_semantic_retrieve_top_result(self):
        # Build fixed embeddings where entry 0 is very similar to the query.
        D = 8
        fixed = np.eye(D, dtype=np.float32)  # orthogonal vectors

        class OrderedEncoder:
            """Returns fixed_vecs[call_index % D] for each text."""
            def __init__(self):
                self._idx = 0

            def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
                out = []
                for _ in texts:
                    out.append(fixed[self._idx % D])
                    self._idx += 1
                return np.array(out, dtype=np.float32)

        enc = OrderedEncoder()
        kb = VectorKnowledgeBase(kb_path=self.tmp.name, encoder=enc)
        # Add D entries; each will receive a distinct orthogonal embedding.
        for i in range(D):
            kb.add({"id": str(i), "question": f"Q{i}", "annotation": f"A{i}"})

        # Query with the embedding of entry 0 (i.e. fixed[0]).
        # Reset encoder so next call returns fixed[0].
        enc._idx = 0
        results = kb.retrieve("Q0", topk=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "0")


# ---------------------------------------------------------------------------
# VectorKnowledgeBase – outlier purging
# ---------------------------------------------------------------------------

class TestPurgeOutliers(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_purge_skipped_on_small_kb(self):
        kb = VectorKnowledgeBase(kb_path=self.tmp.name)
        for i in range(5):  # fewer than minimum 10
            kb.entries.append({"id": str(i), "question": f"Q{i}", "annotation": f"A{i}"})
        removed = kb.purge_outliers()
        self.assertEqual(removed, 0)

    def test_purge_removes_outlier(self):
        """An answer that is completely unrelated to cluster peers is removed."""
        # Build a KB where most answers in a cluster are similar but one is different.
        np.random.seed(0)
        kb = VectorKnowledgeBase(kb_path=self.tmp.name)

        # 12 entries: 11 share similar answers; 1 is a clear outlier.
        similar_answers = [f"The capital of France is Paris (variant {i})" for i in range(11)]
        outlier_answer = "The sky is blue because of Rayleigh scattering in the atmosphere"

        all_answers = similar_answers + [outlier_answer]
        all_questions = [f"What is the capital of France? version {i}" for i in range(12)]

        for i, (q, a) in enumerate(zip(all_questions, all_answers)):
            kb.entries.append({"id": str(i), "question": q, "annotation": a, "confidence": 0.9})

        # Use TF-IDF encoding (no sentence-transformer needed).
        kb._encoder = None

        # Force rebuild of embeddings using TF-IDF.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cossim

            q_texts = [e["question"] for e in kb.entries]
            vec = TfidfVectorizer()
            kb._q_embeddings = vec.fit_transform(q_texts).toarray().astype(np.float32)
        except Exception:
            self.skipTest("sklearn not available")

        removed = kb.purge_outliers(n_clusters=1, z_threshold=1.5)
        # The outlier answer should have been removed.
        self.assertGreaterEqual(removed, 1)
        remaining_annotations = [e["annotation"] for e in kb.entries]
        self.assertNotIn(outlier_answer, remaining_annotations)

    def test_purge_no_false_positives_homogeneous(self):
        """When all answers are identical, no entries should be removed."""
        kb = VectorKnowledgeBase(kb_path=self.tmp.name)
        for i in range(12):
            kb.entries.append({
                "id": str(i),
                "question": f"Q{i}",
                "annotation": "Paris",
                "confidence": 0.9,
            })
        kb._encoder = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            q_texts = [e["question"] for e in kb.entries]
            vec = TfidfVectorizer()
            kb._q_embeddings = vec.fit_transform(q_texts).toarray().astype(np.float32)
        except Exception:
            self.skipTest("sklearn not available")

        removed = kb.purge_outliers(n_clusters=1, z_threshold=2.0)
        self.assertEqual(removed, 0)


# ---------------------------------------------------------------------------
# Annotator – dual threshold
# ---------------------------------------------------------------------------

class TestAnnotatorDualThreshold(unittest.TestCase):
    """Tests that Annotator correctly applies confidence + avg_logprob thresholds."""

    def _make_annotator(self, conf_threshold=0.7, logprob_threshold=None, kb_path=None):
        from annotation import Annotator
        from tasks.qa import QATask

        # Stub LLM that returns a controlled output.
        llm = MagicMock()
        llm.generate.return_value = "Answer: Paris Confidence: 0.90"
        llm.generate_with_logprobs.return_value = ("Answer: Paris Confidence: 0.90", -0.5)

        if kb_path is None:
            fd, kb_path = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            os.unlink(kb_path)  # Remove so VectorKnowledgeBase starts fresh.

        annotator = Annotator(
            candidate_llms=["stub_llm"],
            llm_dict={"stub_llm": llm},
            confidence_threshold=conf_threshold,
            avg_logprob_threshold=logprob_threshold,
            rag=False,
            kb_path=kb_path,
            task=QATask(),
            outlier_purge_interval=0,  # disable purging for unit tests
        )
        return annotator, llm

    def test_passes_both_thresholds(self):
        annotator, _ = self._make_annotator(conf_threshold=0.7, logprob_threshold=-1.0)
        sample = {"id": "1", "question": "What is the capital of France?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertFalse(result["needs_human"])
        self.assertEqual(len(annotator.knowledge_base), 1)

    def test_fails_confidence_threshold(self):
        annotator, llm = self._make_annotator(conf_threshold=0.95, logprob_threshold=None)
        # LLM returns confidence 0.90 which is below 0.95.
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertTrue(result["needs_human"])
        self.assertEqual(len(annotator.knowledge_base), 0)

    def test_fails_logprob_threshold(self):
        annotator, _ = self._make_annotator(conf_threshold=0.7, logprob_threshold=-0.1)
        # avg_logprob = -0.5 which is below -0.1 → should fail.
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertTrue(result["needs_human"])
        self.assertEqual(len(annotator.knowledge_base), 0)

    def test_no_logprob_threshold_ignores_logprob(self):
        annotator, _ = self._make_annotator(conf_threshold=0.7, logprob_threshold=None)
        # avg_logprob = -0.5 but threshold is None → should pass.
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertFalse(result["needs_human"])

    def test_none_logprob_skips_check(self):
        """When LLM does not return a logprob, the check is skipped."""
        annotator, llm = self._make_annotator(conf_threshold=0.7, logprob_threshold=-0.1)
        # Override to return None as avg_logprob.
        llm.generate_with_logprobs.return_value = ("Answer: Paris Confidence: 0.90", None)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        # Confidence 0.90 >= 0.7 and logprob is None → skip logprob check → admitted.
        self.assertFalse(result["needs_human"])

    def test_avg_logprob_stored_in_result(self):
        annotator, _ = self._make_annotator(conf_threshold=0.7, logprob_threshold=None)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertIn("avg_logprob", result)
        self.assertAlmostEqual(result["avg_logprob"], -0.5)


# ---------------------------------------------------------------------------
# LocalLLM – generate_with_logprobs (mocked model)
# ---------------------------------------------------------------------------

class TestLocalLLMLogprobs(unittest.TestCase):

    def test_generate_with_logprobs_returns_tuple(self):
        """generate_with_logprobs must return (str, float)."""
        from misc.llm_provider import LocalLLM

        # Build a minimal mock for the transformers components.
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "Paris"
        mock_inputs = MagicMock()
        mock_inputs.input_ids = MagicMock()
        mock_inputs.input_ids.shape = (1, 3)  # batch=1, prompt_len=3
        mock_inputs.__getitem__ = lambda s, k: mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        # Simulate 2 generated tokens.
        vocab_size = 10
        token1 = MagicMock()
        token1.shape = (1, vocab_size)
        token2 = MagicMock()
        token2.shape = (1, vocab_size)

        import torch

        # logits: each token gets uniform distribution so log_softmax = -log(vocab)
        logits = torch.zeros(1, vocab_size)
        token1.__getitem__ = lambda s, i: logits[0]
        token2.__getitem__ = lambda s, i: logits[0]

        mock_output = MagicMock()
        # sequences: prompt (3 tokens) + generated (2 tokens)
        seq = torch.tensor([[0, 1, 2, 3, 4]])
        mock_output.sequences = seq
        mock_output.scores = (logits, logits)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.generate.return_value = mock_output

        with patch("misc.llm_provider.AutoTokenizer") as mock_at, \
             patch("misc.llm_provider.AutoModelForCausalLM") as mock_am:
            mock_at.from_pretrained.return_value = mock_tokenizer
            mock_am.from_pretrained.return_value = mock_model

            llm = LocalLLM("fake-model")
            llm.tokenizer = mock_tokenizer
            llm.model = mock_model

            text, avg_lp = llm.generate_with_logprobs("What is the capital?", max_new_tokens=5)

        self.assertIsInstance(text, str)
        self.assertIsNotNone(avg_lp)
        self.assertIsInstance(avg_lp, float)
        # For uniform distribution over vocab_size tokens, log_softmax = -log(10).
        expected_lp = -np.log(vocab_size)
        self.assertAlmostEqual(avg_lp, expected_lp, places=4)


# ---------------------------------------------------------------------------
# APILLM – generate_with_logprobs (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAPILLMLogprobs(unittest.TestCase):

    def test_returns_avg_logprob_from_api(self):
        from misc.llm_provider import APILLM

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "text": "Paris",
                "logprobs": {
                    "token_logprobs": [-0.4, -0.6, None, -0.2]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("misc.llm_provider.requests.post", return_value=mock_response):
            llm = APILLM(api_url="http://fake-api/generate")
            text, avg_lp = llm.generate_with_logprobs("Q?", max_new_tokens=10)

        self.assertEqual(text, "Paris")
        # Average of [-0.4, -0.6, -0.2] (None excluded)
        self.assertAlmostEqual(avg_lp, (-0.4 + -0.6 + -0.2) / 3, places=5)

    def test_falls_back_on_api_error(self):
        from misc.llm_provider import APILLM

        with patch("misc.llm_provider.requests.post", side_effect=Exception("network error")):
            llm = APILLM(api_url="http://fake-api/generate")
            with patch.object(llm, "generate", return_value="fallback text"):
                text, avg_lp = llm.generate_with_logprobs("Q?")

        self.assertEqual(text, "fallback text")
        self.assertIsNone(avg_lp)


# ---------------------------------------------------------------------------
# VectorKnowledgeBase – datasets shadowing fix
# ---------------------------------------------------------------------------

_SENTINEL = object()  # used to detect "key was absent" in sys.modules


class TestGetEncoderDatasetsShadowFix(unittest.TestCase):
    """Verify that _get_encoder() restores the local ``datasets`` module after
    importing ``sentence_transformers``.

    The project contains a local ``datasets/`` package.  When ``sys.path``
    includes the project root, ``import datasets`` resolves to that local
    package, which does not export ``Dataset``.  ``sentence_transformers`` (and
    its sub-modules) try to do ``from datasets import Dataset`` at import time
    and would fail with ``ImportError`` unless we temporarily remove the local
    module from ``sys.modules`` while importing ``sentence_transformers``.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def _fresh_kb(self):
        kb = VectorKnowledgeBase(kb_path=self.tmp.name)
        kb._encoder = None
        kb._encoder_unavailable = False
        return kb

    def test_local_datasets_restored_after_successful_encoder_load(self):
        """sys.modules['datasets'] must point to the local module after a
        successful _get_encoder() call."""
        local_datasets = MagicMock(name="local_datasets_package")
        kb = self._fresh_kb()

        # Install a fake local datasets module.
        original = sys.modules.pop("datasets", _SENTINEL)
        sys.modules["datasets"] = local_datasets
        try:
            # Patch sentence_transformers so the import succeeds without a GPU
            # or a downloaded model.
            fake_encoder = MagicMock()
            fake_encoder.encode.return_value = np.zeros((1, 4), dtype=np.float32)
            fake_st = MagicMock()
            fake_st.SentenceTransformer.return_value = fake_encoder
            with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
                result = kb._get_encoder()

            # The local datasets module must be restored.
            self.assertIs(
                sys.modules.get("datasets"),
                local_datasets,
                "local 'datasets' module must be restored in sys.modules after _get_encoder()",
            )
            # The encoder must have been created.
            self.assertIsNotNone(result)
        finally:
            if original is not _SENTINEL:
                sys.modules["datasets"] = original
            else:
                sys.modules.pop("datasets", None)

    def test_local_datasets_restored_even_on_encoder_failure(self):
        """sys.modules['datasets'] must point to the local module even when
        SentenceTransformer raises an exception."""
        local_datasets = MagicMock(name="local_datasets_package")
        kb = self._fresh_kb()

        original = sys.modules.pop("datasets", _SENTINEL)
        sys.modules["datasets"] = local_datasets
        try:
            fake_st = MagicMock()
            fake_st.SentenceTransformer.side_effect = RuntimeError("model load failed")
            with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
                result = kb._get_encoder()

            # datasets is restored despite the failure.
            self.assertIs(sys.modules.get("datasets"), local_datasets)
            # Encoder is unavailable; result is None.
            self.assertIsNone(result)
            self.assertTrue(kb._encoder_unavailable)
        finally:
            if original is not _SENTINEL:
                sys.modules["datasets"] = original
            else:
                sys.modules.pop("datasets", None)

    def test_datasets_not_spuriously_added_when_absent(self):
        """When 'datasets' was not in sys.modules before the call and the
        encoder load fails, 'datasets' must not be left in sys.modules."""
        kb = self._fresh_kb()

        original = sys.modules.pop("datasets", _SENTINEL)
        try:
            fake_st = MagicMock()
            fake_st.SentenceTransformer.side_effect = RuntimeError("no model")
            with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
                kb._get_encoder()

            self.assertNotIn(
                "datasets",
                sys.modules,
                "'datasets' must not appear in sys.modules when it was absent before the call",
            )
        finally:
            if original is not _SENTINEL:
                sys.modules["datasets"] = original


if __name__ == "__main__":
    unittest.main()
