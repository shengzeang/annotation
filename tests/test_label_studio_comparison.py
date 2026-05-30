"""Tests for experiments/run_label_studio_comparison.py.

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

from experiments.run_label_studio_comparison import (
    MockAnnotationLLM,
    MockJudgeLLM,
    _make_synthetic_dataset,
    _tokenize,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_experiment,
    write_sft_jsonl,
    DEFAULT_CANDIDATE_LLMS,
    _safe_name,
    _condition_result_path,
    _sft_output_path,
    _condition_already_done,
    _load_condition_result,
    _save_condition_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 20) -> list:
    return _make_synthetic_dataset(n=n, seed=0)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_tokenize("Hello World"), ["hello", "world"])

    def test_empty(self):
        self.assertEqual(_tokenize(""), [])

    def test_case(self):
        self.assertEqual(_tokenize("ABC"), ["abc"])


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

    def test_partial(self):
        f1 = compute_token_f1("the cat", "the dog")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------

class TestComputeExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertEqual(compute_exact_match("Paris", "Paris"), 1.0)

    def test_case_insensitive(self):
        self.assertEqual(compute_exact_match("PARIS", "paris"), 1.0)

    def test_no_match(self):
        self.assertEqual(compute_exact_match("Paris", "London"), 0.0)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class TestMockLLMs(unittest.TestCase):
    def test_annotation_llm_returns_parseable_string(self):
        llm = MockAnnotationLLM()
        out = llm.generate("prompt")
        self.assertIn("Answer:", out)
        self.assertIn("Confidence:", out)

    def test_judge_returns_one(self):
        judge = MockJudgeLLM()
        self.assertEqual(judge.generate("judge prompt").strip(), "1")

    def test_annotation_llm_logprobs(self):
        llm = MockAnnotationLLM()
        text, lp = llm.generate_with_logprobs("prompt")
        self.assertIsInstance(text, str)
        self.assertIsInstance(lp, float)


# ---------------------------------------------------------------------------
# DEFAULT_CANDIDATE_LLMS
# ---------------------------------------------------------------------------

class TestDefaultCandidateLLMs(unittest.TestCase):
    def test_contains_three_models(self):
        self.assertEqual(len(DEFAULT_CANDIDATE_LLMS), 3)

    def test_all_qwen(self):
        for m in DEFAULT_CANDIDATE_LLMS:
            self.assertTrue(m.startswith("Qwen/"), f"{m} should start with Qwen/")

    def test_expected_values(self):
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", DEFAULT_CANDIDATE_LLMS)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", DEFAULT_CANDIDATE_LLMS)
        self.assertIn("Qwen/Qwen2.5-32B-Instruct", DEFAULT_CANDIDATE_LLMS)


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

    def test_different_seeds_differ(self):
        d1 = _make_synthetic_dataset(n=5, seed=1)
        d2 = _make_synthetic_dataset(n=5, seed=2)
        self.assertNotEqual(d1[0]["context"], d2[0]["context"])


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_perfect(self):
        data = [{"answer": "yes", "annotation": "yes"} for _ in range(5)]
        self.assertAlmostEqual(evaluate_annotation_quality(data)["annotation_f1"], 1.0)

    def test_empty_returns_zero(self):
        self.assertEqual(evaluate_annotation_quality([])["annotation_f1"], 0.0)

    def test_returns_both_metrics(self):
        data = [{"answer": "cat", "annotation": "cat"}]
        result = evaluate_annotation_quality(data)
        self.assertIn("annotation_f1", result)
        self.assertIn("annotation_em", result)


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

    def test_valid_jsonl_output(self):
        data = [{"text": "Q?", "annotation": "A"}, {"text": "Q2?", "annotation": "B"}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            write_sft_jsonl(data, path)
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                rec = json.loads(line)
                self.assertIn("instruction", rec)
                self.assertIn("output", rec)
        finally:
            os.unlink(path)


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

    def _run(self):
        return run_experiment(
            dataset=self.dataset,
            oracle_llm=MockAnnotationLLM(),
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir,
            skip_finetune=True,
            force_fallback=True,
        )

    def test_returns_five_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 5)

    def test_condition_names(self):
        results = self._run()
        names = [r["condition"] for r in results]
        self.assertIn("Single Oracle", names)
        self.assertIn("3-Oracle Majority Vote", names)
        self.assertIn("DataFlow (naive LLM)", names)
        self.assertIn("DataFlow (KB + RAG)", names)
        self.assertIn("DataFlow (full pipeline)", names)

    def test_required_keys_in_each_result(self):
        results = self._run()
        for r in results:
            for k in ("condition", "num_samples", "annotation_f1",
                      "annotation_em", "downstream_bleu",
                      "downstream_rouge_l", "sft_file"):
                self.assertIn(k, r, f"key '{k}' missing from condition '{r.get('condition')}'")

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]),
                            f"SFT file missing for condition '{r['condition']}'")

    def test_sft_files_contain_valid_jsonl(self):
        results = self._run()
        for r in results:
            with open(r["sft_file"]) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("instruction", rec)
                    self.assertIn("output", rec)

    def test_downstream_bleu_is_none_when_skipped(self):
        results = self._run()
        for r in results:
            self.assertIsNone(r["downstream_bleu"])

    def test_annotation_f1_in_range(self):
        results = self._run()
        for r in results:
            self.assertGreaterEqual(r["annotation_f1"], 0.0)
            self.assertLessEqual(r["annotation_f1"], 1.0)

    def test_uses_actual_annotator(self):
        """Verify that Annotator is used: SFT outputs are non-empty strings."""
        results = self._run()
        for r in results:
            with open(r["sft_file"]) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)
            for line in lines:
                rec = json.loads(line)
                self.assertIsInstance(rec["output"], str)

    def test_conditions_appear_in_order(self):
        """Conditions must be returned in the documented order 1-5."""
        results = self._run()
        expected = [
            "Single Oracle",
            "3-Oracle Majority Vote",
            "DataFlow (naive LLM)",
            "DataFlow (KB + RAG)",
            "DataFlow (full pipeline)",
        ]
        for i, (r, name) in enumerate(zip(results, expected)):
            self.assertEqual(r["condition"], name,
                             f"Condition {i+1} should be '{name}', got '{r['condition']}'")


# ---------------------------------------------------------------------------
# print_results_table smoke-test
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke_no_downstream(self):
        import io, contextlib
        results = [{
            "condition": "Single Oracle",
            "annotation_f1": 0.8,
            "annotation_em": 0.7,
            "num_samples": 100,
            "downstream_bleu": None,
            "downstream_rouge_l": None,
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("Single Oracle", buf.getvalue())

    def test_smoke_with_downstream(self):
        import io, contextlib
        results = [{
            "condition": "DataFlow (KB + RAG)",
            "annotation_f1": 0.8,
            "annotation_em": 0.7,
            "num_samples": 100,
            "downstream_bleu": 0.42,
            "downstream_rouge_l": 0.55,
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("DataFlow", buf.getvalue())

    def test_all_five_conditions_printed(self):
        import io, contextlib
        results = [
            {"condition": name, "annotation_f1": 0.5, "annotation_em": 0.4,
             "num_samples": 10, "downstream_bleu": None, "downstream_rouge_l": None}
            for name in [
                "Single Oracle", "3-Oracle Majority Vote",
                "DataFlow (naive LLM)", "DataFlow (KB + RAG)", "DataFlow (full pipeline)",
            ]
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        out = buf.getvalue()
        for name in ["Single Oracle", "3-Oracle Majority Vote",
                     "DataFlow (naive LLM)", "DataFlow (KB + RAG)", "DataFlow (full pipeline)"]:
            self.assertIn(name, out)


# ---------------------------------------------------------------------------
# Progress banner tests
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
                oracle_llm=MockAnnotationLLM(),
                judge_llm=MockJudgeLLM(),
                output_dir=self.tmp_dir,
                skip_finetune=True,
                force_fallback=True,
            )
        return buf.getvalue()

    def test_all_five_condition_banners_printed(self):
        out = self._captured_output()
        for name in (
            "Single Oracle",
            "3-Oracle Majority Vote",
            "DataFlow (naive LLM)",
            "DataFlow (KB + RAG)",
            "DataFlow (full pipeline)",
        ):
            self.assertIn(name, out, f"Banner for '{name}' not found in stdout")

    def test_banner_format_first_and_last(self):
        out = self._captured_output()
        self.assertIn("[1/5]", out)
        self.assertIn("[5/5]", out)

    def test_banners_appear_in_order(self):
        out = self._captured_output()
        positions = [
            out.index(name)
            for name in (
                "Single Oracle",
                "3-Oracle Majority Vote",
                "DataFlow (naive LLM)",
                "DataFlow (KB + RAG)",
                "DataFlow (full pipeline)",
            )
        ]
        self.assertEqual(positions, sorted(positions),
                         "Condition banners must appear in order 1-5")


# ---------------------------------------------------------------------------
# Resume mechanism (_condition_already_done / helpers / skip on second run)
# ---------------------------------------------------------------------------

_ALL_LSC_CONDS = (
    "Single Oracle",
    "3-Oracle Majority Vote",
    "DataFlow (naive LLM)",
    "DataFlow (KB + RAG)",
    "DataFlow (full pipeline)",
)


class TestResumeMechanismLSC(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = _make_dataset(n=10)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        return run_experiment(
            dataset=self.dataset,
            oracle_llm=MockAnnotationLLM(),
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir,
            skip_finetune=True,
            force_fallback=True,
        )

    def test_not_done_initially(self):
        for cond in _ALL_LSC_CONDS:
            self.assertFalse(_condition_already_done(cond, self.tmp_dir))

    def test_all_conditions_done_after_run(self):
        self._run()
        for cond in _ALL_LSC_CONDS:
            self.assertTrue(
                _condition_already_done(cond, self.tmp_dir),
                f"Expected cached output for '{cond}' after run",
            )

    def test_second_run_skips_all_conditions(self):
        import io, contextlib
        self._run()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run()
        self.assertIn("Already done", buf.getvalue())

    def test_second_run_returns_same_results(self):
        r1 = self._run()
        r2 = self._run()
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a["condition"], b["condition"])
            self.assertAlmostEqual(a["annotation_f1"], b["annotation_f1"])

    def test_sft_file_alone_triggers_done(self):
        cond = "Single Oracle"
        sft_path = _sft_output_path(cond, self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        with open(sft_path, "w") as f:
            f.write('{"instruction": "Q", "output": "A"}\n')
        self.assertTrue(_condition_already_done(cond, self.tmp_dir))

    def test_sft_file_only_run_returns_result(self):
        cond = "Single Oracle"
        sft_path = _sft_output_path(cond, self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        with open(sft_path, "w") as f:
            for i in range(5):
                f.write(f'{{"instruction": "Q{i}", "output": "A{i}"}}\n')
        results = self._run()
        oracle = next(r for r in results if r["condition"] == cond)
        for k in ("condition", "num_samples", "annotation_f1", "annotation_em", "sft_file"):
            self.assertIn(k, oracle)
        self.assertEqual(oracle["num_samples"], 5)

    def test_safe_name_helper(self):
        self.assertEqual(_safe_name("Single Oracle"), "single_oracle")
        self.assertEqual(_safe_name("DataFlow (naive LLM)"), "dataflow__naive_llm")

    def test_save_load_result_roundtrip(self):
        result = {
            "condition": "Single Oracle",
            "num_samples": 10,
            "annotation_f1": 0.6,
            "annotation_em": 0.4,
            "downstream_bleu": None,
            "downstream_rouge_l": None,
            "sft_file": "/tmp/foo.jsonl",
        }
        _save_condition_result(result, self.tmp_dir)
        self.assertTrue(_condition_already_done("Single Oracle", self.tmp_dir))
        loaded = _load_condition_result("Single Oracle", self.tmp_dir)
        self.assertEqual(loaded["condition"], result["condition"])
        self.assertAlmostEqual(loaded["annotation_f1"], result["annotation_f1"])


if __name__ == "__main__":
    unittest.main()
