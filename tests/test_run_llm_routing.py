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
    MockKNNRouter,
    MockGraphRouter,
    MockMLPRouter,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_experiment,
    write_sft_jsonl,
    _make_synthetic_dataset,
    _safe_name,
    _condition_already_done,
    _condition_result_path,
    _save_condition_result,
    _load_condition_result,
    _route_direct,
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

    def test_returns_seven_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 7)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        for expected in (
            "All-cheap", "All-expensive", "CascadeRouter", "LLMRouter",
            "KNNRouter", "GraphRouter", "MLPRouter",
        ):
            self.assertIn(expected, names)

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


# ---------------------------------------------------------------------------
# Progress banner tests — verify condition banners are printed to stdout
# ---------------------------------------------------------------------------

class TestConditionProgressBanners(unittest.TestCase):
    """Verify that run_experiment() prints a [i/N] banner for each condition."""

    def setUp(self):
        self.dataset = _make_dataset(n=10)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _captured_output(self) -> str:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_experiment(
                dataset=self.dataset,
                cheap_llm=MockAnnotationLLM(),
                expensive_llm=MockAnnotationLLM(),
                judge_llm=MockJudgeLLM(),
                scorer_llm=MockScorerLLM(),
                output_dir=self.tmp_dir,
                force_fallback=True,
            )
        return buf.getvalue()

    def test_all_seven_banners_printed(self):
        out = self._captured_output()
        for name in ("All-cheap", "All-expensive", "CascadeRouter", "LLMRouter",
                     "KNNRouter", "GraphRouter", "MLPRouter"):
            self.assertIn(name, out, f"Banner for '{name}' not found in stdout")

    def test_banner_format_contains_fraction(self):
        out = self._captured_output()
        self.assertIn("[1/7]", out)
        self.assertIn("[7/7]", out)


# ---------------------------------------------------------------------------
# _safe_name helper
# ---------------------------------------------------------------------------

class TestSafeName(unittest.TestCase):
    def test_hyphen_replaced(self):
        self.assertEqual(_safe_name("All-cheap"), "all_cheap")

    def test_spaces_replaced(self):
        self.assertEqual(_safe_name("KNN Router"), "knn_router")

    def test_already_safe(self):
        self.assertEqual(_safe_name("knnrouter"), "knnrouter")

    def test_slash_replaced(self):
        self.assertEqual(_safe_name("LLM/Router"), "llm_router")


# ---------------------------------------------------------------------------
# Mock learning-based routers
# ---------------------------------------------------------------------------

class TestMockKNNRouter(unittest.TestCase):
    def setUp(self):
        self.router = MockKNNRouter(["cheap", "expensive"])

    def test_score_returns_all_candidates(self):
        scores = self.router.score("some text", ["cheap", "expensive"])
        self.assertEqual(len(scores), 2)
        models = {s["model"] for s in scores}
        self.assertIn("cheap", models)
        self.assertIn("expensive", models)

    def test_score_all_have_score_key(self):
        scores = self.router.score("text", ["cheap", "expensive"])
        for s in scores:
            self.assertIn("score", s)
            self.assertIsInstance(s["score"], float)

    def test_build_from_annotations_is_noop(self):
        """Should not raise even when passed None."""
        self.router.build_from_annotations(None)
        self.router.build_from_annotations([], out_dir="/tmp")

    def test_candidate_llms_stored(self):
        self.assertEqual(self.router.candidate_llms, ["cheap", "expensive"])


class TestMockGraphRouter(unittest.TestCase):
    def test_inherits_mock_knn(self):
        router = MockGraphRouter(["a", "b"])
        scores = router.score("text", ["a", "b"])
        self.assertEqual(len(scores), 2)


class TestMockMLPRouter(unittest.TestCase):
    def test_inherits_mock_knn(self):
        router = MockMLPRouter(["a", "b", "c"])
        scores = router.score("text", ["a", "b", "c"])
        self.assertEqual(len(scores), 3)


# ---------------------------------------------------------------------------
# _route_direct helper
# ---------------------------------------------------------------------------

class TestRouteDirect(unittest.TestCase):
    def _make_ds(self, n=5):
        return [{"id": i, "text": f"q{i}", "answer": "ans"} for i in range(n)]

    def test_adds_route_field(self):
        router = MockKNNRouter(["cheap", "expensive"])
        ds = self._make_ds(3)
        result = _route_direct(router, ds)
        for item in result:
            self.assertIn("route", item)
            self.assertIn("route_scores", item)

    def test_preserves_original_fields(self):
        router = MockKNNRouter(["cheap", "expensive"])
        ds = self._make_ds(2)
        result = _route_direct(router, ds)
        for orig, routed in zip(ds, result):
            self.assertEqual(orig["id"], routed["id"])
            self.assertEqual(orig["text"], routed["text"])

    def test_length_unchanged(self):
        router = MockKNNRouter(["cheap", "expensive"])
        ds = self._make_ds(7)
        result = _route_direct(router, ds)
        self.assertEqual(len(result), 7)


# ---------------------------------------------------------------------------
# Resume mechanism (_condition_already_done / _save/_load)
# ---------------------------------------------------------------------------

class TestResumeMechanism(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = _make_dataset(n=10)

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

    def test_not_done_initially(self):
        for cond in ("All-cheap", "All-expensive", "CascadeRouter",
                     "LLMRouter", "KNNRouter", "GraphRouter", "MLPRouter"):
            self.assertFalse(_condition_already_done(cond, self.tmp_dir))

    def test_all_conditions_done_after_run(self):
        self._run()
        for cond in ("All-cheap", "All-expensive", "CascadeRouter",
                     "LLMRouter", "KNNRouter", "GraphRouter", "MLPRouter"):
            self.assertTrue(
                _condition_already_done(cond, self.tmp_dir),
                f"Expected cached result for '{cond}' after run",
            )

    def test_second_run_returns_same_results(self):
        """Second call should load all conditions from cache — same results."""
        r1 = self._run()
        r2 = self._run()
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a["condition"], b["condition"])
            self.assertAlmostEqual(a["annotation_f1"], b["annotation_f1"])

    def test_second_run_prints_already_done_message(self):
        import io, contextlib
        self._run()  # first run
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run()  # second run — should skip everything
        self.assertIn("Already done", buf.getvalue())

    def test_save_and_load_result_roundtrip(self):
        result = {
            "condition": "All-cheap",
            "annotated": 10,
            "annotation_f1": 0.5,
            "annotation_em": 0.3,
            "expensive_call_rate": 0.0,
            "sft_file": "/tmp/foo.jsonl",
        }
        _save_condition_result(result, self.tmp_dir)
        self.assertTrue(_condition_already_done("All-cheap", self.tmp_dir))
        loaded = _load_condition_result("All-cheap", self.tmp_dir)
        self.assertEqual(loaded["condition"], result["condition"])
        self.assertAlmostEqual(loaded["annotation_f1"], result["annotation_f1"])


# ---------------------------------------------------------------------------
# New routing conditions in the full run
# ---------------------------------------------------------------------------

class TestNewRoutingConditions(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=12)
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

    def test_knn_router_condition_present(self):
        results = self._run()
        names = [r["condition"] for r in results]
        self.assertIn("KNNRouter", names)

    def test_graph_router_condition_present(self):
        results = self._run()
        names = [r["condition"] for r in results]
        self.assertIn("GraphRouter", names)

    def test_mlp_router_condition_present(self):
        results = self._run()
        names = [r["condition"] for r in results]
        self.assertIn("MLPRouter", names)

    def test_new_conditions_have_required_keys(self):
        results = self._run()
        for r in results:
            if r["condition"] in ("KNNRouter", "GraphRouter", "MLPRouter"):
                for k in ("condition", "annotated", "annotation_f1",
                          "annotation_em", "expensive_call_rate", "sft_file"):
                    self.assertIn(k, r, f"Key '{k}' missing from {r['condition']} result")

    def test_new_conditions_create_sft_files(self):
        results = self._run()
        for r in results:
            if r["condition"] in ("KNNRouter", "GraphRouter", "MLPRouter"):
                self.assertTrue(
                    os.path.exists(r["sft_file"]),
                    f"SFT file not created for {r['condition']}",
                )

    def test_new_conditions_create_result_json(self):
        self._run()
        for cond in ("KNNRouter", "GraphRouter", "MLPRouter"):
            path = _condition_result_path(cond, self.tmp_dir)
            self.assertTrue(os.path.exists(path), f"Result JSON missing for {cond}")


if __name__ == "__main__":
    unittest.main()
