"""Unit and integration tests for experiments/run_llm_routing.py.

All tests are CPU-only and do not require real LLM models or GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_llm_routing import (
    CHEAP_LLM_ACCURACY,
    EXPENSIVE_LLM_ACCURACY,
    _InlineCascadeRouter,
    _InlineLLMRouter,
    _MockCheapLLM,
    _MockExpensiveLLM,
    _MockJudgeLLM,
    _SimpleQATask,
    _annotate_with_llm,
    _get_task,
    _make_synthetic_dataset,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_all_cheap,
    run_all_expensive,
    run_cascade,
    run_experiment,
    run_llm_router,
    write_sft_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 20) -> list:
    return _make_synthetic_dataset(n=n, seed=0)


def _make_llms(seed: int = 0):
    cheap = _MockCheapLLM(seed=seed)
    expensive = _MockExpensiveLLM(seed=seed)
    judge = _MockJudgeLLM()
    return cheap, expensive, judge


# ---------------------------------------------------------------------------
# compute_token_f1
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
# _SimpleQATask
# ---------------------------------------------------------------------------

class TestSimpleQATask(unittest.TestCase):
    def setUp(self):
        self.task = _SimpleQATask()

    def test_get_prompt_returns_string(self):
        sample = {"question": "Q?", "context": "Ctx"}
        p = self.task.get_prompt(sample)
        self.assertIsInstance(p, str)

    def test_parse_output_extracts_answer(self):
        out = self.task.parse_output("Answer: Paris Confidence: 0.85")
        self.assertEqual(out["annotation"], "Paris")
        self.assertAlmostEqual(float(out["confidence"]), 0.85, places=2)

    def test_parse_output_no_confidence(self):
        out = self.task.parse_output("Answer: something")
        self.assertIn("annotation", out)


class TestGetTask(unittest.TestCase):
    def test_returns_provided(self):
        stub = object()
        self.assertIs(_get_task(stub), stub)

    def test_none_returns_task(self):
        task = _get_task(None)
        self.assertTrue(hasattr(task, "get_prompt"))


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class TestMockLLMs(unittest.TestCase):
    def test_cheap_generate(self):
        llm = _MockCheapLLM()
        out = llm.generate("Answer: relativity Confidence: 0.9")
        self.assertIsInstance(out, str)
        self.assertIn("Confidence:", out)

    def test_expensive_generate(self):
        llm = _MockExpensiveLLM()
        out = llm.generate("Answer: relativity Confidence: 0.9")
        self.assertIsInstance(out, str)
        self.assertIn("Confidence:", out)

    def test_judge_generate_cascade(self):
        judge = _MockJudgeLLM()
        out = judge.generate("Determine if the answer is correct. output 0")
        self.assertIn("0", out)

    def test_judge_generate_llm_router(self):
        judge = _MockJudgeLLM()
        out = judge.generate("Rate which model is better (JSON)")
        # Should return JSON with expensive model score
        self.assertIn("[", out)


class TestExpensiveMoreAccurate(unittest.TestCase):
    """Expensive LLM should have higher base accuracy than cheap LLM."""

    def test_accuracy_constants(self):
        self.assertGreater(EXPENSIVE_LLM_ACCURACY, CHEAP_LLM_ACCURACY)


# ---------------------------------------------------------------------------
# _annotate_with_llm
# ---------------------------------------------------------------------------

class TestAnnotateWithLLM(unittest.TestCase):
    def test_keys_present(self):
        task = _SimpleQATask()
        sample = {"question": "Q?", "context": "", "answer": "A", "text": "Q?"}
        llm = _MockCheapLLM()
        result = _annotate_with_llm(sample, llm, "cheap", task)
        self.assertIn("annotation", result)
        self.assertIn("routed_to", result)
        self.assertEqual(result["routed_to"], "cheap")

    def test_original_fields_preserved(self):
        task = _SimpleQATask()
        sample = {"id": "1", "question": "Q?", "context": "", "answer": "A", "text": "Q?"}
        llm = _MockCheapLLM()
        result = _annotate_with_llm(sample, llm, "cheap", task)
        self.assertIn("id", result)


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------

class TestRunAllCheap(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        cheap, _, _ = _make_llms()
        task = _SimpleQATask()
        results = run_all_cheap(dataset, cheap, task)
        self.assertEqual(len(results), 10)

    def test_all_routed_to_cheap(self):
        dataset = _make_dataset(n=10)
        cheap, _, _ = _make_llms()
        task = _SimpleQATask()
        results = run_all_cheap(dataset, cheap, task)
        for r in results:
            self.assertEqual(r["routed_to"], "cheap")


class TestRunAllExpensive(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        _, expensive, _ = _make_llms()
        task = _SimpleQATask()
        results = run_all_expensive(dataset, expensive, task)
        self.assertEqual(len(results), 10)

    def test_all_routed_to_expensive(self):
        dataset = _make_dataset(n=10)
        _, expensive, _ = _make_llms()
        task = _SimpleQATask()
        results = run_all_expensive(dataset, expensive, task)
        for r in results:
            self.assertEqual(r["routed_to"], "expensive")


class TestRunCascade(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        cheap, expensive, judge = _make_llms()
        task = _SimpleQATask()
        results = run_cascade(dataset, cheap, expensive, judge, task)
        self.assertEqual(len(results), 10)

    def test_routing_key_present(self):
        dataset = _make_dataset(n=5)
        cheap, expensive, judge = _make_llms()
        task = _SimpleQATask()
        results = run_cascade(dataset, cheap, expensive, judge, task)
        for r in results:
            self.assertIn(r["routed_to"], ("cheap", "expensive"))


class TestRunLLMRouter(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        cheap, expensive, judge = _make_llms()
        task = _SimpleQATask()
        results = run_llm_router(dataset, cheap, expensive, judge, task)
        self.assertEqual(len(results), 10)

    def test_routing_key_present(self):
        dataset = _make_dataset(n=5)
        cheap, expensive, judge = _make_llms()
        task = _SimpleQATask()
        results = run_llm_router(dataset, cheap, expensive, judge, task)
        for r in results:
            self.assertIn(r["routed_to"], ("cheap", "expensive"))


# ---------------------------------------------------------------------------
# Inline routers
# ---------------------------------------------------------------------------

class TestInlineCascadeRouter(unittest.TestCase):
    def test_escalates_when_judge_says_wrong(self):
        class _AlwaysWrong:
            def generate(self, p, **kw):
                return "0"  # judge says cheap answer is wrong

        cheap = _MockCheapLLM()
        expensive = _MockExpensiveLLM()
        router = _InlineCascadeRouter(_AlwaysWrong(), {"cheap": cheap, "expensive": expensive})
        choice = router.route("Q: Who?")
        self.assertEqual(choice, "expensive")

    def test_keeps_cheap_when_judge_says_correct(self):
        class _AlwaysRight:
            def generate(self, p, **kw):
                return "1"  # judge says cheap answer is correct

        cheap = _MockCheapLLM()
        expensive = _MockExpensiveLLM()
        router = _InlineCascadeRouter(_AlwaysRight(), {"cheap": cheap, "expensive": expensive})
        choice = router.route("Q: Who?")
        self.assertEqual(choice, "cheap")


class TestInlineLLMRouter(unittest.TestCase):
    def test_picks_high_score_model(self):
        class _FakeLLM:
            def generate(self, p, **kw):
                return '[{"model": "cheap", "score": 0.2}, {"model": "expensive", "score": 0.9}]'

        router = _InlineLLMRouter(_FakeLLM(), ["cheap", "expensive"])
        choice = router.route("Q?")
        self.assertEqual(choice, "expensive")

    def test_fallback_on_invalid_json(self):
        class _BadLLM:
            def generate(self, p, **kw):
                return "not json at all"

        router = _InlineLLMRouter(_BadLLM(), ["cheap", "expensive"])
        choice = router.route("Q?")
        # Falls back to first candidate
        self.assertEqual(choice, "cheap")


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_expensive_call_rate_all_cheap(self):
        data = [{"answer": "a", "annotation": "a", "routed_to": "cheap"} for _ in range(10)]
        q = evaluate_annotation_quality(data)
        self.assertAlmostEqual(q["expensive_call_rate"], 0.0)

    def test_expensive_call_rate_all_expensive(self):
        data = [{"answer": "a", "annotation": "a", "routed_to": "expensive"} for _ in range(10)]
        q = evaluate_annotation_quality(data)
        self.assertAlmostEqual(q["expensive_call_rate"], 1.0)


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
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=15)
        self.tmp_dir = tempfile.mkdtemp()
        self.task = _SimpleQATask()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        cheap, expensive, judge = _make_llms()
        return run_experiment(
            dataset=self.dataset,
            cheap_llm=cheap,
            expensive_llm=expensive,
            judge_llm=judge,
            output_dir=self.tmp_dir,
            task=self.task,
        )

    def test_returns_four_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 4)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("All-cheap", names)
        self.assertIn("All-expensive", names)
        self.assertIn("Cascade", names)
        self.assertIn("LLM Router", names)

    def test_keys_present(self):
        results = self._run()
        for r in results:
            for k in ("condition", "annotated", "annotation_f1", "annotation_em",
                      "expensive_call_rate", "sft_file"):
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


# ---------------------------------------------------------------------------
# print_results_table
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [
            {"condition": "All-cheap", "annotation_f1": 0.6, "annotation_em": 0.5,
             "expensive_call_rate": 0.0, "annotated": 100}
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("All-cheap", buf.getvalue())
        self.assertIn("0.0000", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
