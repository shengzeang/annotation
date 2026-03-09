"""
Integration test: full annotation pipeline with mocked LLMs and no real models loaded.

Validates the end-to-end flow:
  raw_data -> LLMNaiveFilter -> CascadeRouter -> Annotator -> export
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_raw_dataset(n=10):
    return [
        {
            "id": str(i),
            "question": f"What is item number {i}?",
            "context": f"Item {i} is an example context for testing.",
            "answer": f"Item {i}",
            "text": f"Question: What is item number {i}?\nContext: Item {i} is an example.",
        }
        for i in range(n)
    ]


def _make_stub_llm(annotation="Paris", confidence=0.85, logprob=-0.4):
    """Build a fully stubbed LLM that returns controlled outputs."""
    llm = MagicMock()
    llm.generate.return_value = f"Answer: {annotation} Confidence: {confidence}"
    llm.generate_with_logprobs.return_value = (
        f"Answer: {annotation} Confidence: {confidence}",
        logprob,
    )
    return llm


# ---------------------------------------------------------------------------
# Pipeline smoke tests
# ---------------------------------------------------------------------------

class TestAnnotationPipeline(unittest.TestCase):
    """End-to-end pipeline tests using fully mocked LLMs (no GPU required)."""

    def setUp(self):
        self.raw_data = _make_raw_dataset(n=8)
        self.stub_llm = _make_stub_llm()
        fd, self.tmp_kb = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.tmp_kb)  # Let VectorKnowledgeBase/annotator create it fresh
        fd2, self.tmp_export = tempfile.mkstemp(suffix=".json")
        os.close(fd2)
        os.unlink(self.tmp_export)

    def tearDown(self):
        for p in (self.tmp_kb, self.tmp_export):
            if os.path.exists(p):
                os.unlink(p)

    def _make_annotator(self, conf_threshold=0.7, logprob_threshold=None):
        from annotation import Annotator
        from tasks.qa import QATask
        return Annotator(
            candidate_llms=["stub_llm"],
            llm_dict={"stub_llm": self.stub_llm},
            confidence_threshold=conf_threshold,
            avg_logprob_threshold=logprob_threshold,
            rag=False,
            kb_path=self.tmp_kb,
            task=QATask(),
            outlier_purge_interval=0,
        )

    def _make_cascade_router(self, judge_returns="1"):
        from routers.cascade_router import CascadeRouter
        judge = MagicMock()
        judge.generate.return_value = judge_returns
        return CascadeRouter(
            judge_llm=judge,
            candidate_llm=["stub_llm"],
            llm_dict={"stub_llm": self.stub_llm},
            threshold=0.7,
        )

    def _make_llm_naive_filter(self, budget=5):
        from filters.llm_filter import LLMNaiveFilter
        filter_llm = MagicMock()
        filter_llm.generate.return_value = "0.8"
        return LLMNaiveFilter(filter_llm, budget=budget)

    # --- Annotator unit tests ---

    def test_annotator_annotate_single_sample(self):
        annotator = self._make_annotator()
        sample = {
            "id": "x1",
            "question": "What is the capital of France?",
            "context": "",
            "route": "stub_llm",
        }
        result = annotator.annotate(sample)
        self.assertIn("annotation", result)
        self.assertIn("confidence", result)
        self.assertIn("needs_human", result)

    def test_annotator_batch_returns_all_results(self):
        annotator = self._make_annotator()
        dataset = [
            {"id": str(i), "question": f"Q{i}?", "context": "", "route": "stub_llm"}
            for i in range(5)
        ]
        results = annotator.annotate_batch(dataset)
        self.assertEqual(len(results), 5)

    def test_annotator_below_threshold_sets_needs_human(self):
        annotator = self._make_annotator(conf_threshold=0.99)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertTrue(result["needs_human"])

    def test_annotator_above_threshold_sets_not_needs_human(self):
        annotator = self._make_annotator(conf_threshold=0.5)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertFalse(result["needs_human"])

    def test_annotator_adds_to_kb_when_passes(self):
        annotator = self._make_annotator(conf_threshold=0.5)
        initial_len = len(annotator.knowledge_base)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        annotator.annotate(sample)
        self.assertEqual(len(annotator.knowledge_base), initial_len + 1)

    def test_annotator_does_not_add_to_kb_when_fails(self):
        annotator = self._make_annotator(conf_threshold=0.99)
        initial_len = len(annotator.knowledge_base)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        annotator.annotate(sample)
        self.assertEqual(len(annotator.knowledge_base), initial_len)

    def test_annotator_adds_to_human_review_queue_when_fails(self):
        annotator = self._make_annotator(conf_threshold=0.99)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        annotator.annotate(sample)
        self.assertEqual(len(annotator.human_review_queue.queue), 1)

    def test_annotator_handles_string_items_in_batch(self):
        """annotate_batch should gracefully handle non-dict items."""
        annotator = self._make_annotator()
        dataset = ["plain string item"]
        results = annotator.annotate_batch(dataset)
        self.assertEqual(len(results), 1)

    def test_annotator_progress_callback(self):
        annotator = self._make_annotator()
        calls = []
        dataset = [{"id": str(i), "question": f"Q{i}?", "context": "", "route": "stub_llm"} for i in range(3)]
        annotator.annotate_batch(dataset, progress_cb=lambda cur, tot, info: calls.append(cur))
        self.assertEqual(len(calls), 3)

    # --- Filter + Router + Annotator pipeline ---

    def test_filter_then_annotate(self):
        filt = self._make_llm_naive_filter(budget=5)
        annotator = self._make_annotator()

        filtered = filt.filter(self.raw_data)
        # Assign route manually (simulating router)
        for item in filtered:
            item["route"] = "stub_llm"
        results = annotator.annotate_batch(filtered)
        self.assertEqual(len(results), len(filtered))

    def test_router_then_annotate(self):
        router = self._make_cascade_router(judge_returns="1")
        annotator = self._make_annotator()

        routed = router.route(self.raw_data)
        results = annotator.annotate_batch(routed)
        self.assertEqual(len(results), len(self.raw_data))
        for r in results:
            self.assertIn("route", r)

    def test_full_pipeline_filter_route_annotate_export(self):
        """Full pipeline: filter -> route -> annotate -> export."""
        from utils import export_annotation_results

        filt = self._make_llm_naive_filter(budget=6)
        router = self._make_cascade_router(judge_returns="1")
        annotator = self._make_annotator(conf_threshold=0.5)

        filtered = filt.filter(self.raw_data)
        routed = router.route(filtered)
        results = annotator.annotate_batch(routed)

        auto_results = [r for r in results if not r.get("needs_human", False)]
        export_annotation_results(auto_results, self.raw_data, output_path=self.tmp_export)

        self.assertTrue(os.path.exists(self.tmp_export))
        with open(self.tmp_export) as f:
            exported = json.load(f)
        self.assertEqual(len(exported), len(auto_results))

    def test_full_pipeline_human_review_queue_populated(self):
        """Samples failing threshold should land in human review queue."""
        annotator = self._make_annotator(conf_threshold=0.99)
        routed = [{"id": str(i), "question": f"Q{i}?", "context": "", "route": "stub_llm"} for i in range(5)]
        annotator.annotate_batch(routed)
        self.assertEqual(len(annotator.human_review_queue.queue), 5)

    def test_human_review_queue_export(self):
        annotator = self._make_annotator(conf_threshold=0.99)
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        annotator.annotate(sample)

        fd, tmp_review = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(tmp_review)  # remove so export can write a fresh file
        try:
            annotator.human_review_queue.export(tmp_review)
            self.assertTrue(os.path.exists(tmp_review))
            with open(tmp_review) as f:
                queue = json.load(f)
            self.assertEqual(len(queue), 1)
        finally:
            if os.path.exists(tmp_review):
                os.unlink(tmp_review)

    # --- RAG-enabled annotator ---

    def test_annotator_rag_enabled(self):
        from annotation import Annotator
        from tasks.qa import QATask
        annotator = Annotator(
            candidate_llms=["stub_llm"],
            llm_dict={"stub_llm": self.stub_llm},
            confidence_threshold=0.5,
            rag=True,
            kb_path=self.tmp_kb,
            task=QATask(),
            outlier_purge_interval=0,
        )
        sample = {"id": "1", "question": "Q?", "context": "", "route": "stub_llm"}
        result = annotator.annotate(sample)
        self.assertIn("annotation", result)

    # --- Annotator backward-compat properties ---

    def test_kb_path_property(self):
        annotator = self._make_annotator()
        self.assertEqual(annotator.kb_path, self.tmp_kb)

    def test_deprecated_load_knowledge_base(self):
        annotator = self._make_annotator()
        result = annotator._load_knowledge_base()
        self.assertIsInstance(result, list)

    def test_deprecated_save_knowledge_base(self):
        annotator = self._make_annotator()
        # Should not raise
        annotator._save_knowledge_base()


if __name__ == "__main__":
    unittest.main()
