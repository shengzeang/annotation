"""Unit and integration tests for experiments/run_label_studio_comparison.py.

All tests are CPU-only and do not require real LLM models, network access,
or GPU hardware.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_label_studio_comparison import (
    LabelStudioAnnotator,
    SimulatedLLM,
    _make_synthetic_dataset,
    _tokenize,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_dataflow_condition,
    run_experiment,
    simulate_sft_downstream,
    write_sft_jsonl,
    EM_CHANCE,
    EM_ORACLE,
    F1_CHANCE,
    F1_ORACLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 10) -> list:
    return _make_synthetic_dataset(n=n, seed=0)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_tokenize("Hello World"), ["hello", "world"])

    def test_empty(self):
        self.assertEqual(_tokenize(""), [])

    def test_lowercase(self):
        self.assertEqual(_tokenize("UPPER lower"), ["upper", "lower"])


# ---------------------------------------------------------------------------
# compute_token_f1
# ---------------------------------------------------------------------------

class TestComputeTokenF1(unittest.TestCase):
    def test_exact_match(self):
        self.assertAlmostEqual(compute_token_f1("relativity", "relativity"), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(compute_token_f1("cat", "dog"), 0.0)

    def test_partial_overlap(self):
        score = compute_token_f1("the cat sat", "cat sat on mat")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_prediction(self):
        self.assertAlmostEqual(compute_token_f1("", "cat"), 0.0)

    def test_both_empty(self):
        self.assertAlmostEqual(compute_token_f1("", ""), 1.0)


# ---------------------------------------------------------------------------
# compute_exact_match
# ---------------------------------------------------------------------------

class TestComputeExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertEqual(compute_exact_match("Paris", "Paris"), 1.0)

    def test_case_insensitive(self):
        self.assertEqual(compute_exact_match("paris", "Paris"), 1.0)

    def test_mismatch(self):
        self.assertEqual(compute_exact_match("Paris", "London"), 0.0)

    def test_whitespace_stripped(self):
        self.assertEqual(compute_exact_match("  Paris  ", "Paris"), 1.0)


# ---------------------------------------------------------------------------
# simulate_sft_downstream
# ---------------------------------------------------------------------------

class TestSimulateSftDownstream(unittest.TestCase):
    def test_zero_quality(self):
        result = simulate_sft_downstream(0.0)
        self.assertAlmostEqual(result["em"], EM_CHANCE, places=4)
        self.assertAlmostEqual(result["f1"], F1_CHANCE, places=4)

    def test_full_quality(self):
        result = simulate_sft_downstream(1.0)
        self.assertAlmostEqual(result["em"], EM_ORACLE, places=4)
        self.assertAlmostEqual(result["f1"], F1_ORACLE, places=4)

    def test_midpoint(self):
        result = simulate_sft_downstream(0.5)
        expected_em = EM_CHANCE + 0.5 * (EM_ORACLE - EM_CHANCE)
        self.assertAlmostEqual(result["em"], round(expected_em, 4), places=4)

    def test_clamps_above_one(self):
        result = simulate_sft_downstream(1.5)
        self.assertAlmostEqual(result["em"], EM_ORACLE, places=4)

    def test_clamps_below_zero(self):
        result = simulate_sft_downstream(-0.5)
        self.assertAlmostEqual(result["em"], EM_CHANCE, places=4)

    def test_returns_dict_with_keys(self):
        result = simulate_sft_downstream(0.7)
        self.assertIn("em", result)
        self.assertIn("f1", result)

    def test_monotone_increasing(self):
        low = simulate_sft_downstream(0.3)["em"]
        high = simulate_sft_downstream(0.8)["em"]
        self.assertGreater(high, low)


# ---------------------------------------------------------------------------
# SimulatedLLM
# ---------------------------------------------------------------------------

class TestSimulatedLLM(unittest.TestCase):
    def setUp(self):
        self.sample = {"answer": "relativity", "question": "What did Einstein develop?"}

    def test_high_accuracy_mostly_correct(self):
        llm = SimulatedLLM(base_accuracy=1.0, seed=0)
        for _ in range(20):
            self.assertEqual(llm.annotate(self.sample), "relativity")

    def test_zero_accuracy_never_correct(self):
        llm = SimulatedLLM(base_accuracy=0.0, seed=0)
        for _ in range(10):
            self.assertNotEqual(llm.annotate(self.sample), "relativity")

    def test_confidence_in_range(self):
        llm = SimulatedLLM(seed=0)
        for _ in range(20):
            c = llm.confidence()
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_seeded_reproducibility(self):
        llm1 = SimulatedLLM(base_accuracy=0.6, seed=99)
        llm2 = SimulatedLLM(base_accuracy=0.6, seed=99)
        results1 = [llm1.annotate(self.sample) for _ in range(10)]
        results2 = [llm2.annotate(self.sample) for _ in range(10)]
        self.assertEqual(results1, results2)

    def test_empty_answer(self):
        llm = SimulatedLLM(base_accuracy=0.0, seed=0)
        sample = {"answer": "", "question": "Q?"}
        result = llm.annotate(sample)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# LabelStudioAnnotator
# ---------------------------------------------------------------------------

class TestLabelStudioAnnotator(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)

    def test_single_annotator_output_count(self):
        ls = LabelStudioAnnotator(num_annotators=1, annotator_accuracy=0.9, seed=0)
        results = ls.annotate_dataset(self.dataset)
        self.assertEqual(len(results), len(self.dataset))

    def test_three_annotators_output_count(self):
        ls = LabelStudioAnnotator(num_annotators=3, annotator_accuracy=0.9, seed=0)
        results = ls.annotate_dataset(self.dataset)
        self.assertEqual(len(results), len(self.dataset))

    def test_annotation_key_present(self):
        ls = LabelStudioAnnotator(seed=0)
        results = ls.annotate_dataset(self.dataset)
        for r in results:
            self.assertIn("annotation", r)

    def test_majority_vote_improves_over_single(self):
        """3-annotator majority vote should have >= F1 of 1-annotator at 75% accuracy."""
        dataset = _make_dataset(n=200)

        ls1 = LabelStudioAnnotator(num_annotators=1, annotator_accuracy=0.75, seed=0)
        ls3 = LabelStudioAnnotator(num_annotators=3, annotator_accuracy=0.75, seed=0)

        res1 = ls1.annotate_dataset(dataset)
        res3 = ls3.annotate_dataset(dataset)

        q1 = evaluate_annotation_quality(res1)["annotation_f1"]
        q3 = evaluate_annotation_quality(res3)["annotation_f1"]
        self.assertGreaterEqual(q3, q1)

    def test_high_accuracy_annotates_mostly_correctly(self):
        dataset = [{"answer": "yes", "question": "Q?", "text": "Q? yes", "id": "0"}] * 50
        ls = LabelStudioAnnotator(num_annotators=3, annotator_accuracy=1.0, seed=0)
        results = ls.annotate_dataset(dataset)
        correct = sum(1 for r in results if r["annotation"] == "yes")
        self.assertEqual(correct, 50)


# ---------------------------------------------------------------------------
# _make_synthetic_dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset(unittest.TestCase):
    def test_count(self):
        ds = _make_synthetic_dataset(n=50)
        self.assertEqual(len(ds), 50)

    def test_keys(self):
        ds = _make_synthetic_dataset(n=5)
        for rec in ds:
            for key in ("id", "question", "context", "answer", "text"):
                self.assertIn(key, rec)

    def test_reproducibility(self):
        ds1 = _make_synthetic_dataset(n=10, seed=7)
        ds2 = _make_synthetic_dataset(n=10, seed=7)
        self.assertEqual(ds1, ds2)


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_perfect_annotations(self):
        dataset = [{"answer": "yes", "annotation": "yes"} for _ in range(10)]
        result = evaluate_annotation_quality(dataset)
        self.assertAlmostEqual(result["annotation_f1"], 1.0)
        self.assertAlmostEqual(result["annotation_em"], 1.0)

    def test_no_overlap_annotations(self):
        dataset = [{"answer": "yes", "annotation": "no"} for _ in range(10)]
        result = evaluate_annotation_quality(dataset)
        self.assertAlmostEqual(result["annotation_f1"], 0.0)
        self.assertAlmostEqual(result["annotation_em"], 0.0)

    def test_empty_list(self):
        result = evaluate_annotation_quality([])
        self.assertEqual(result["annotation_f1"], 0.0)


# ---------------------------------------------------------------------------
# run_dataflow_condition
# ---------------------------------------------------------------------------

class TestRunDataflowCondition(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)

    def test_output_count(self):
        results = run_dataflow_condition(self.dataset, llm_accuracy=0.7, seed=0)
        self.assertEqual(len(results), len(self.dataset))

    def test_annotation_key_present(self):
        results = run_dataflow_condition(self.dataset, llm_accuracy=0.7, seed=0)
        for r in results:
            self.assertIn("annotation", r)

    def test_confidence_key_present(self):
        results = run_dataflow_condition(self.dataset, llm_accuracy=0.7, seed=0)
        for r in results:
            self.assertIn("confidence", r)

    def test_higher_accuracy_better_quality(self):
        low = run_dataflow_condition(self.dataset, llm_accuracy=0.3, seed=0)
        high = run_dataflow_condition(self.dataset, llm_accuracy=0.9, seed=0)
        q_low = evaluate_annotation_quality(low)["annotation_f1"]
        q_high = evaluate_annotation_quality(high)["annotation_f1"]
        self.assertGreater(q_high, q_low)

    def test_seeded_reproducibility(self):
        r1 = run_dataflow_condition(self.dataset, llm_accuracy=0.7, seed=42)
        r2 = run_dataflow_condition(self.dataset, llm_accuracy=0.7, seed=42)
        anns1 = [r["annotation"] for r in r1]
        anns2 = [r["annotation"] for r in r2]
        self.assertEqual(anns1, anns2)


# ---------------------------------------------------------------------------
# write_sft_jsonl
# ---------------------------------------------------------------------------

class TestWriteSftJsonl(unittest.TestCase):
    def test_writes_correct_count(self):
        annotated = [
            {"text": "Q1?", "annotation": "A1", "needs_human": False},
            {"text": "Q2?", "annotation": "A2", "needs_human": False},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            n = write_sft_jsonl(annotated, path, skip_human_review=False)
            self.assertEqual(n, 2)
        finally:
            os.unlink(path)

    def test_skips_human_review(self):
        annotated = [
            {"text": "Q1?", "annotation": "A1", "needs_human": False},
            {"text": "Q2?", "annotation": "A2", "needs_human": True},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            n = write_sft_jsonl(annotated, path, skip_human_review=True)
            self.assertEqual(n, 1)
        finally:
            os.unlink(path)

    def test_jsonl_format(self):
        annotated = [{"text": "Question: Q?", "annotation": "Answer", "needs_human": False}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tf:
            path = tf.name
        try:
            write_sft_jsonl(annotated, path)
            with open(path, encoding="utf-8") as f:
                line = f.readline()
            record = json.loads(line)
            self.assertIn("instruction", record)
            self.assertIn("output", record)
            self.assertEqual(record["output"], "Answer")
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "out.jsonl")
            annotated = [{"text": "Q?", "annotation": "A", "needs_human": False}]
            write_sft_jsonl(annotated, path)
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=30)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_five_conditions(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        self.assertEqual(len(results), 5)

    def test_condition_keys(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        expected_keys = {
            "condition", "num_samples", "annotation_f1",
            "annotation_em", "downstream_em", "downstream_f1", "sft_file",
        }
        for r in results:
            self.assertEqual(set(r.keys()), expected_keys)

    def test_sft_files_created(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]), r["sft_file"])

    def test_condition_names_unique(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        names = [r["condition"] for r in results]
        self.assertEqual(len(names), len(set(names)))

    def test_downstream_metrics_ordering(self):
        """Full pipeline should have better or equal downstream EM than naive."""
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        by_name = {r["condition"]: r for r in results}
        naive_em = by_name["DataFlow (naive LLM)"]["downstream_em"]
        full_em = by_name["DataFlow (full pipeline)"]["downstream_em"]
        self.assertGreaterEqual(full_em, naive_em)

    def test_label_studio_3_annotators_better_than_1(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        by_name = {r["condition"]: r for r in results}
        em_1 = by_name["Label Studio (1 annotator)"]["downstream_em"]
        em_3 = by_name["Label Studio (3 annotators)"]["downstream_em"]
        self.assertGreaterEqual(em_3, em_1)

    def test_num_samples_positive(self):
        results = run_experiment(self.dataset, output_dir=self.tmp_dir, seed=0)
        for r in results:
            self.assertGreater(r["num_samples"], 0)


# ---------------------------------------------------------------------------
# print_results_table (smoke test)
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_runs_without_error(self):
        results = [
            {
                "condition": "Test",
                "annotation_f1": 0.8,
                "annotation_em": 0.75,
                "downstream_em": 0.65,
                "downstream_f1": 0.70,
                "num_samples": 100,
            }
        ]
        # Should not raise
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        output = buf.getvalue()
        self.assertIn("Test", output)
        self.assertIn("0.8000", output)


if __name__ == "__main__":
    unittest.main()
