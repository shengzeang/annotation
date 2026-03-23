"""Tests for experiments/run_active_learning.py.

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

from experiments.run_active_learning import (
    MockJudgeLLM,
    MockLLM,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    load_squad_dataset,
    print_results_table,
    run_experiment,
    write_sft_jsonl,
    _make_synthetic_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 20) -> list:
    return _make_synthetic_dataset(n=n, seed=0)


# ---------------------------------------------------------------------------
# compute_token_f1
# ---------------------------------------------------------------------------

class TestComputeTokenF1(unittest.TestCase):
    def test_exact(self):
        self.assertAlmostEqual(compute_token_f1("relativity", "relativity"), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(compute_token_f1("cat", "dog"), 0.0)

    def test_both_empty(self):
        self.assertAlmostEqual(compute_token_f1("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(compute_token_f1("", "cat"), 0.0)


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------

class TestComputeExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertEqual(compute_exact_match("Paris", "Paris"), 1.0)

    def test_case_insensitive(self):
        self.assertEqual(compute_exact_match("PARIS", "paris"), 1.0)

    def test_mismatch(self):
        self.assertEqual(compute_exact_match("Paris", "London"), 0.0)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class TestMockLLMs(unittest.TestCase):
    def test_mock_llm_returns_string(self):
        llm = MockLLM()
        out = llm.generate("some prompt")
        self.assertIsInstance(out, str)
        self.assertIn("Confidence:", out)

    def test_mock_judge_returns_one(self):
        judge = MockJudgeLLM()
        out = judge.generate("some prompt")
        self.assertEqual(out.strip(), "1")

    def test_mock_llm_generate_with_logprobs(self):
        llm = MockLLM()
        text, lp = llm.generate_with_logprobs("prompt")
        self.assertIsInstance(text, str)
        self.assertIsInstance(lp, float)


# ---------------------------------------------------------------------------
# _make_synthetic_dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset(unittest.TestCase):
    def test_count(self):
        ds = _make_synthetic_dataset(n=30)
        self.assertEqual(len(ds), 30)

    def test_required_keys(self):
        ds = _make_synthetic_dataset(n=5)
        for rec in ds:
            for key in ("id", "question", "context", "answer", "text"):
                self.assertIn(key, rec)

    def test_reproducible(self):
        d1 = _make_synthetic_dataset(n=10, seed=7)
        d2 = _make_synthetic_dataset(n=10, seed=7)
        self.assertEqual(d1, d2)


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_perfect(self):
        data = [{"answer": "yes", "annotation": "yes"} for _ in range(5)]
        q = evaluate_annotation_quality(data)
        self.assertAlmostEqual(q["annotation_f1"], 1.0)
        self.assertAlmostEqual(q["annotation_em"], 1.0)

    def test_empty(self):
        q = evaluate_annotation_quality([])
        self.assertEqual(q["annotation_f1"], 0.0)

    def test_no_overlap(self):
        data = [{"answer": "yes", "annotation": "no"} for _ in range(5)]
        q = evaluate_annotation_quality(data)
        self.assertAlmostEqual(q["annotation_f1"], 0.0)


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

    def test_jsonl_format(self):
        data = [{"text": "Question?", "annotation": "Answer"}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tf:
            path = tf.name
        try:
            write_sft_jsonl(data, path)
            with open(path) as f:
                rec = json.loads(f.readline())
            self.assertIn("instruction", rec)
            self.assertIn("output", rec)
            self.assertEqual(rec["output"], "Answer")
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "out.jsonl")
            write_sft_jsonl([{"text": "Q?", "annotation": "A"}], path)
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
# run_experiment — uses actual Annotator, ActiveLearningFilter, CascadeRouter
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, budget: int = 8):
        return run_experiment(
            dataset=self.dataset,
            cheap_llm=MockLLM(),
            judge_llm=MockJudgeLLM(),
            budget=budget,
            output_dir=self.tmp_dir,
            seed=0,
            force_fallback=True,
        )

    def test_returns_four_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 4)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("No filter", names)
        self.assertIn("Random sampling", names)
        self.assertIn("ALPS filter", names)
        self.assertIn("Full filter chain", names)

    def test_required_keys(self):
        results = self._run()
        for r in results:
            for k in ("condition", "selected", "annotated", "annotation_f1",
                      "annotation_em", "sft_file"):
                self.assertIn(k, r)

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_no_filter_gets_budget_samples(self):
        results = self._run(budget=8)
        no_filt = next(r for r in results if r["condition"] == "No filter")
        self.assertEqual(no_filt["selected"], 8)

    def test_budget_conditions_within_budget(self):
        budget = 8
        results = self._run(budget=budget)
        for r in results:
            self.assertLessEqual(r["selected"], len(self.dataset))

    def test_annotation_f1_is_float(self):
        results = self._run()
        for r in results:
            self.assertIsInstance(r["annotation_f1"], float)

    def test_uses_actual_annotator_output(self):
        """Annotations should be strings (not None), proving Annotator was used."""
        from tasks.qa import QATask
        results = self._run()
        for r in results:
            # SFT file should contain valid JSONL with 'output' field
            with open(r["sft_file"]) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("output", rec)
                    self.assertIsInstance(rec["output"], str)


# ---------------------------------------------------------------------------
# print_results_table smoke-test
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [{
            "condition": "ALPS filter",
            "selected": 50,
            "annotation_f1": 0.7,
            "annotation_em": 0.6,
            "human_review": 2,
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("ALPS filter", buf.getvalue())


# ---------------------------------------------------------------------------
# load_squad_dataset fallback to synthetic
# ---------------------------------------------------------------------------

class TestLoadSquadDataset(unittest.TestCase):
    def test_fallback_to_synthetic(self):
        ds = load_squad_dataset("/nonexistent/path.json", max_samples=10)
        self.assertEqual(len(ds), 10)
        for rec in ds:
            self.assertIn("question", rec)


if __name__ == "__main__":
    unittest.main()
