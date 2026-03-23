"""Tests for experiments/run_rag.py.

All tests use mock LLMs (no GPU, no network) following the
HumanLLMAnnotationSystem pipeline pattern from test.py.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_rag import (
    MockJudgeLLM,
    MockLLM,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_experiment,
    windowed_f1,
    write_sft_jsonl,
    _make_synthetic_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 20) -> list:
    return _make_synthetic_dataset(n=n, seed=0)


# ---------------------------------------------------------------------------
# compute_token_f1 / compute_exact_match
# ---------------------------------------------------------------------------

class TestComputeTokenF1(unittest.TestCase):
    def test_exact(self):
        self.assertAlmostEqual(compute_token_f1("relativity", "relativity"), 1.0)

    def test_zero(self):
        self.assertAlmostEqual(compute_token_f1("cat", "dog"), 0.0)

    def test_both_empty(self):
        self.assertAlmostEqual(compute_token_f1("", ""), 1.0)


class TestComputeExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertEqual(compute_exact_match("Paris", "Paris"), 1.0)

    def test_case(self):
        self.assertEqual(compute_exact_match("PARIS", "paris"), 1.0)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class TestMockLLMs(unittest.TestCase):
    def test_llm_returns_parseable_string(self):
        llm = MockLLM()
        out = llm.generate("prompt")
        self.assertIn("Answer:", out)
        self.assertIn("Confidence:", out)

    def test_judge_returns_one(self):
        judge = MockJudgeLLM()
        self.assertEqual(judge.generate("judge prompt").strip(), "1")

    def test_llm_logprobs(self):
        llm = MockLLM()
        text, lp = llm.generate_with_logprobs("prompt")
        self.assertIsInstance(text, str)
        self.assertIsInstance(lp, float)


# ---------------------------------------------------------------------------
# _make_synthetic_dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(_make_synthetic_dataset(n=20)), 20)

    def test_required_keys(self):
        for rec in _make_synthetic_dataset(n=3):
            for k in ("id", "question", "context", "answer", "text"):
                self.assertIn(k, rec)


# ---------------------------------------------------------------------------
# windowed_f1
# ---------------------------------------------------------------------------

class TestWindowedF1(unittest.TestCase):
    def test_exact_match_data(self):
        data = [{"annotation": "cat", "answer": "cat"} for _ in range(20)]
        windows = windowed_f1(data, window=10)
        self.assertEqual(len(windows), 2)
        for w in windows:
            self.assertAlmostEqual(w["mean_f1"], 1.0)

    def test_empty_returns_empty(self):
        self.assertEqual(windowed_f1([], window=10), [])

    def test_window_keys(self):
        data = [{"annotation": "cat", "answer": "cat"} for _ in range(10)]
        for w in windowed_f1(data, window=5):
            self.assertIn("window_start", w)
            self.assertIn("window_end", w)
            self.assertIn("mean_f1", w)


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_perfect(self):
        data = [{"answer": "yes", "annotation": "yes"} for _ in range(5)]
        self.assertAlmostEqual(evaluate_annotation_quality(data)["annotation_f1"], 1.0)

    def test_empty(self):
        self.assertEqual(evaluate_annotation_quality([])["annotation_f1"], 0.0)


# ---------------------------------------------------------------------------
# write_sft_jsonl
# ---------------------------------------------------------------------------

class TestWriteSftJsonl(unittest.TestCase):
    def test_writes_all(self):
        data = [{"text": "Q?", "annotation": "A"}] * 3
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            n = write_sft_jsonl(data, path)
            self.assertEqual(n, 3)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_experiment — uses actual Annotator(rag=True/False), ActiveLearningFilter
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, window: int = 8):
        return run_experiment(
            dataset=self.dataset,
            llm=MockLLM(),
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir,
            topk=3,
            window=window,
            force_fallback=True,
        )

    def test_returns_two_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 2)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("No RAG", names)
        self.assertIn("RAG", names)

    def test_required_keys(self):
        results = self._run()
        for r in results:
            for k in ("condition", "annotated", "annotation_f1",
                      "annotation_em", "final_kb_size", "windowed_f1", "sft_file"):
                self.assertIn(k, r)

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_no_rag_kb_size_is_zero(self):
        """No RAG condition should not build a KB."""
        results = self._run()
        no_rag = next(r for r in results if r["condition"] == "No RAG")
        self.assertEqual(no_rag["final_kb_size"], 0)

    def test_rag_kb_grows(self):
        """RAG condition accumulates KB entries (mock LLM confidence = 0.85 > threshold 0.7)."""
        results = self._run()
        rag = next(r for r in results if r["condition"] == "RAG")
        self.assertGreater(rag["final_kb_size"], 0)

    def test_windowed_f1_is_list(self):
        results = self._run(window=8)
        for r in results:
            self.assertIsInstance(r["windowed_f1"], list)

    def test_windowed_f1_entries_have_keys(self):
        results = self._run(window=8)
        for r in results:
            for w in r["windowed_f1"]:
                self.assertIn("mean_f1", w)
                self.assertIn("window_start", w)

    def test_sft_files_contain_valid_jsonl(self):
        results = self._run()
        for r in results:
            with open(r["sft_file"]) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("instruction", rec)
                    self.assertIn("output", rec)

    def test_uses_actual_annotator(self):
        """Verify that Annotator is used: annotations are non-None strings."""
        results = self._run()
        for r in results:
            with open(r["sft_file"]) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)
            for line in lines:
                rec = json.loads(line)
                self.assertIsInstance(rec["output"], str)


# ---------------------------------------------------------------------------
# print_results_table smoke-test
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [{
            "condition": "RAG",
            "annotation_f1": 0.6,
            "annotation_em": 0.5,
            "final_kb_size": 42,
            "annotated": 100,
            "windowed_f1": [{"window_start": 0, "window_end": 49, "mean_f1": 0.6}],
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results, window=50)
        self.assertIn("RAG", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
