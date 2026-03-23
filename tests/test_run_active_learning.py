"""Unit and integration tests for experiments/run_active_learning.py.

All tests are CPU-only and do not require real LLM models, BERT, or GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_active_learning import (
    MockLLM,
    _SimpleQATask,
    _annotate_samples,
    _get_task,
    _make_synthetic_dataset,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_experiment,
    select_alps_fallback,
    select_diversity_tfidf,
    select_random,
    select_uncertainty_length,
    write_sft_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
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
# _SimpleQATask
# ---------------------------------------------------------------------------

class TestSimpleQATask(unittest.TestCase):
    def setUp(self):
        self.task = _SimpleQATask()
        self.sample = {"question": "What did Einstein develop?", "context": "He developed relativity.", "answer": "relativity"}

    def test_get_prompt_contains_question(self):
        prompt = self.task.get_prompt(self.sample)
        self.assertIn("Einstein", prompt)

    def test_get_prompt_with_rag(self):
        rag = [{"question": "Q?", "annotation": "A"}]
        prompt = self.task.get_prompt(self.sample, rag_examples=rag)
        self.assertIn("Q?", prompt)
        self.assertIn("A", prompt)

    def test_parse_output_standard(self):
        out = self.task.parse_output("Answer: relativity Confidence: 0.9")
        self.assertEqual(out["annotation"], "relativity")
        self.assertAlmostEqual(float(out["confidence"]), 0.9, places=2)

    def test_parse_output_no_confidence(self):
        out = self.task.parse_output("Answer: something")
        self.assertIn("annotation", out)

    def test_parse_output_empty(self):
        out = self.task.parse_output("")
        self.assertIn("annotation", out)


# ---------------------------------------------------------------------------
# _get_task
# ---------------------------------------------------------------------------

class TestGetTask(unittest.TestCase):
    def test_returns_provided_task(self):
        mock = object()
        self.assertIs(_get_task(mock), mock)

    def test_none_returns_task_instance(self):
        task = _get_task(None)
        self.assertTrue(hasattr(task, "get_prompt"))
        self.assertTrue(hasattr(task, "parse_output"))


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

class TestMockLLM(unittest.TestCase):
    def test_generate_returns_string(self):
        llm = MockLLM()
        out = llm.generate("some prompt")
        self.assertIsInstance(out, str)

    def test_generate_with_logprobs(self):
        llm = MockLLM()
        text, logprob = llm.generate_with_logprobs("some prompt")
        self.assertIsInstance(text, str)
        self.assertIsInstance(logprob, float)


# ---------------------------------------------------------------------------
# _make_synthetic_dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset(unittest.TestCase):
    def test_count(self):
        ds = _make_synthetic_dataset(n=30)
        self.assertEqual(len(ds), 30)

    def test_keys(self):
        ds = _make_synthetic_dataset(n=5)
        for rec in ds:
            for key in ("id", "question", "context", "answer", "text"):
                self.assertIn(key, rec)

    def test_reproducibility(self):
        d1 = _make_synthetic_dataset(n=10, seed=7)
        d2 = _make_synthetic_dataset(n=10, seed=7)
        self.assertEqual(d1, d2)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

class TestSelectRandom(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=30)

    def test_returns_budget_samples(self):
        result = select_random(self.dataset, budget=10)
        self.assertEqual(len(result), 10)

    def test_does_not_exceed_dataset(self):
        result = select_random(self.dataset, budget=100)
        self.assertLessEqual(len(result), len(self.dataset))

    def test_reproducible(self):
        r1 = select_random(self.dataset, budget=10, seed=0)
        r2 = select_random(self.dataset, budget=10, seed=0)
        self.assertEqual([r["id"] for r in r1], [r["id"] for r in r2])

    def test_different_seed_may_differ(self):
        r1 = select_random(self.dataset, budget=10, seed=0)
        r2 = select_random(self.dataset, budget=10, seed=99)
        # With 30 samples and budget 10 the probability of identical order is negligible
        ids1 = [r["id"] for r in r1]
        ids2 = [r["id"] for r in r2]
        self.assertNotEqual(ids1, ids2)


class TestSelectDiversityTFIDF(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=30)

    def test_returns_budget_samples(self):
        result = select_diversity_tfidf(self.dataset, budget=10)
        self.assertLessEqual(len(result), 10)

    def test_all_original_fields_preserved(self):
        result = select_diversity_tfidf(self.dataset, budget=5)
        for r in result:
            self.assertIn("question", r)


class TestSelectUncertaintyLength(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=30)

    def test_returns_exact_budget(self):
        result = select_uncertainty_length(self.dataset, budget=10)
        self.assertEqual(len(result), 10)

    def test_all_fields_preserved(self):
        result = select_uncertainty_length(self.dataset, budget=5)
        for r in result:
            self.assertIn("id", r)


class TestSelectAlpsFallback(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=30)

    def test_returns_budget_samples(self):
        result = select_alps_fallback(self.dataset, budget=10, batch_size=5)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 10)

    def test_all_fields_preserved(self):
        result = select_alps_fallback(self.dataset, budget=5, batch_size=3)
        for r in result:
            self.assertIn("id", r)


# ---------------------------------------------------------------------------
# _annotate_samples
# ---------------------------------------------------------------------------

class TestAnnotateSamples(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=10)
        self.task = _SimpleQATask()

    def test_returns_same_count(self):
        llm = MockLLM()
        result = _annotate_samples(self.dataset, llm, task=self.task)
        self.assertEqual(len(result), len(self.dataset))

    def test_annotation_key_present(self):
        llm = MockLLM()
        result = _annotate_samples(self.dataset, llm, task=self.task)
        for r in result:
            self.assertIn("annotation", r)

    def test_original_fields_preserved(self):
        llm = MockLLM()
        result = _annotate_samples(self.dataset, llm, task=self.task)
        for r in result:
            self.assertIn("id", r)
            self.assertIn("question", r)


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
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=25)
        self.tmp_dir = tempfile.mkdtemp()
        self.task = _SimpleQATask()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, budget=8):
        return run_experiment(
            dataset=self.dataset,
            llm=MockLLM(),
            budget=budget,
            output_dir=self.tmp_dir,
            seed=0,
            task=self.task,
        )

    def test_returns_five_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 5)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("Full dataset", names)
        self.assertIn("Random sampling", names)
        self.assertIn("Diversity (TF-IDF)", names)
        self.assertIn("Uncertainty (length)", names)
        self.assertIn("ALPS (force-fallback)", names)

    def test_keys_present(self):
        results = self._run()
        for r in results:
            for k in ("condition", "selected", "annotated", "annotation_f1", "annotation_em", "sft_file"):
                self.assertIn(k, r)

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_full_dataset_has_all_samples(self):
        results = self._run()
        full = next(r for r in results if r["condition"] == "Full dataset")
        self.assertEqual(full["selected"], len(self.dataset))

    def test_budget_conditions_respect_budget(self):
        budget = 8
        results = self._run(budget=budget)
        for r in results:
            if r["condition"] != "Full dataset":
                self.assertLessEqual(r["selected"], budget)


# ---------------------------------------------------------------------------
# print_results_table
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [{"condition": "Random sampling", "selected": 50, "annotation_f1": 0.7, "annotation_em": 0.6}]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("Random sampling", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
