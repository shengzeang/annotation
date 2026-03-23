"""Tests for experiments/run_llm_routing.py.

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

from experiments.run_llm_routing import (
    MockAnnotationLLM,
    MockJudgeLLM,
    MockScorerLLM,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
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

    def test_case_insensitive(self):
        self.assertEqual(compute_exact_match("paris", "PARIS"), 1.0)

    def test_mismatch(self):
        self.assertEqual(compute_exact_match("a", "b"), 0.0)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class TestMockLLMs(unittest.TestCase):
    def test_annotation_llm_generates_answer(self):
        llm = MockAnnotationLLM()
        out = llm.generate("prompt")
        self.assertIn("Answer:", out)
        self.assertIn("Confidence:", out)

    def test_judge_returns_zero(self):
        """MockJudgeLLM always returns '0' to trigger cascade escalation."""
        judge = MockJudgeLLM()
        self.assertEqual(judge.generate("judge prompt").strip(), "0")

    def test_scorer_returns_json(self):
        scorer = MockScorerLLM()
        out = scorer.generate("score prompt")
        parsed = json.loads(out)
        self.assertIsInstance(parsed, list)
        self.assertTrue(all("model" in item and "score" in item for item in parsed))


# ---------------------------------------------------------------------------
# _make_synthetic_dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset(unittest.TestCase):
    def test_count(self):
        ds = _make_synthetic_dataset(n=15)
        self.assertEqual(len(ds), 15)

    def test_required_keys(self):
        for rec in _make_synthetic_dataset(n=3):
            for k in ("id", "question", "context", "answer", "text"):
                self.assertIn(k, rec)


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_all_cheap_zero_exp_rate(self):
        data = [{"answer": "a", "annotation": "a", "route": "cheap"} for _ in range(10)]
        q = evaluate_annotation_quality(data, expensive_llm_name="expensive")
        self.assertAlmostEqual(q["expensive_call_rate"], 0.0)

    def test_all_expensive_full_exp_rate(self):
        data = [{"answer": "a", "annotation": "a", "route": "expensive"} for _ in range(10)]
        q = evaluate_annotation_quality(data, expensive_llm_name="expensive")
        self.assertAlmostEqual(q["expensive_call_rate"], 1.0)

    def test_empty(self):
        q = evaluate_annotation_quality([])
        self.assertEqual(q["annotation_f1"], 0.0)


# ---------------------------------------------------------------------------
# write_sft_jsonl
# ---------------------------------------------------------------------------

class TestWriteSftJsonl(unittest.TestCase):
    def test_writes_count(self):
        data = [{"text": "Q?", "annotation": "A"}] * 4
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            n = write_sft_jsonl(data, path)
            self.assertEqual(n, 4)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_experiment — uses actual CascadeRouter, LLMRouter, Annotator
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=15)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        return run_experiment(
            dataset=self.dataset,
            cheap_llm=MockAnnotationLLM(),
            expensive_llm=MockAnnotationLLM(),
            judge_llm=MockJudgeLLM(),
            scorer_llm=MockScorerLLM(),
            output_dir=self.tmp_dir,
            force_fallback=True,
        )

    def test_returns_four_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 4)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("All-cheap", names)
        self.assertIn("All-expensive", names)
        self.assertIn("CascadeRouter", names)
        self.assertIn("LLMRouter", names)

    def test_required_keys(self):
        results = self._run()
        for r in results:
            for k in ("condition", "annotated", "annotation_f1",
                      "annotation_em", "expensive_call_rate", "sft_file"):
                self.assertIn(k, r)

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_all_cheap_zero_exp_rate(self):
        results = self._run()
        all_cheap = next(r for r in results if r["condition"] == "All-cheap")
        self.assertAlmostEqual(all_cheap["expensive_call_rate"], 0.0)

    def test_all_expensive_one_exp_rate(self):
        results = self._run()
        all_exp = next(r for r in results if r["condition"] == "All-expensive")
        self.assertAlmostEqual(all_exp["expensive_call_rate"], 1.0)

    def test_cascade_escalates_because_judge_returns_zero(self):
        """MockJudgeLLM returns 0 so CascadeRouter always escalates to expensive."""
        results = self._run()
        cascade = next(r for r in results if r["condition"] == "CascadeRouter")
        self.assertAlmostEqual(cascade["expensive_call_rate"], 1.0)

    def test_llm_router_uses_scorer_preference(self):
        """MockScorerLLM prefers expensive model, so LLMRouter routes to expensive."""
        results = self._run()
        llm_router = next(r for r in results if r["condition"] == "LLMRouter")
        self.assertAlmostEqual(llm_router["expensive_call_rate"], 1.0)

    def test_sft_files_contain_valid_jsonl(self):
        results = self._run()
        for r in results:
            with open(r["sft_file"]) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("instruction", rec)
                    self.assertIn("output", rec)


# ---------------------------------------------------------------------------
# print_results_table smoke-test
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [{
            "condition": "CascadeRouter",
            "annotation_f1": 0.7,
            "annotation_em": 0.6,
            "expensive_call_rate": 0.3,
            "annotated": 100,
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("CascadeRouter", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
