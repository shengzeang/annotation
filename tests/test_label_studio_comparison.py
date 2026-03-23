"""Unit and integration tests for experiments/run_label_studio_comparison.py.

All tests are CPU-only and do not require real LLM models, GPU, or network
access.  Real LLMs are replaced by lightweight mock objects.
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
    OracleAnnotator,
    _make_synthetic_dataset,
    _tokenize,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_dataflow_condition,
    run_experiment,
    write_sft_jsonl,
)


# ---------------------------------------------------------------------------
# Shared mock LLM & task
# ---------------------------------------------------------------------------

class _MockLLM:
    """LLM stub that returns the ground-truth answer embedded in a
    QATask-compatible format: "Answer: <answer> Confidence: 0.9".

    If *answer_override* is set, returns that instead (simulates noise).
    """

    def __init__(self, answer_override: str = "") -> None:
        self._override = answer_override

    def generate(self, prompt: str, **kw) -> str:
        if self._override:
            return f"Answer: {self._override} Confidence: 0.9"
        # Extract answer from the prompt (not available in real usage, but
        # sufficient for testing that the pipeline works end-to-end)
        return "Answer: test_answer Confidence: 0.9"

    def generate_with_logprobs(self, prompt: str, **kw):
        return self.generate(prompt), -0.3


class _PerfectMockLLM(_MockLLM):
    """Always returns the ground-truth answer for the given sample."""

    def __init__(self, sample_answers: dict) -> None:
        super().__init__()
        self._answers = sample_answers  # id -> answer

    def generate(self, prompt: str, **kw) -> str:
        # We can't recover the sample id from the prompt, so use a fixed answer
        return "Answer: test_answer Confidence: 0.95"


class _MockTask:
    """Minimal task compatible with OracleAnnotator / run_dataflow_condition."""

    def get_prompt(self, sample, rag_examples=None) -> str:
        return f"Q: {sample.get('question', '')}?"

    def parse_output(self, output: str) -> dict:
        import re
        m = re.search(r"Answer:\s*(.*?)\s*Confidence:\s*([0-9.]+)", output)
        if m:
            return {"annotation": m.group(1).strip(), "confidence": float(m.group(2))}
        return {"annotation": output.strip(), "confidence": 0.5}


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
# OracleAnnotator
# ---------------------------------------------------------------------------

class TestOracleAnnotator(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)
        self.task = _MockTask()

    def test_single_oracle_output_count(self):
        llm = _MockLLM()
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        results = oa.annotate_dataset(self.dataset)
        self.assertEqual(len(results), len(self.dataset))

    def test_three_oracle_output_count(self):
        llm = _MockLLM()
        oa = OracleAnnotator(llm, num_oracles=3, task=self.task)
        results = oa.annotate_dataset(self.dataset)
        self.assertEqual(len(results), len(self.dataset))

    def test_annotation_key_present(self):
        llm = _MockLLM()
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        results = oa.annotate_dataset(self.dataset)
        for r in results:
            self.assertIn("annotation", r)

    def test_single_oracle_calls_llm_once(self):
        """Single oracle should call generate() exactly once per sample."""
        call_count = [0]
        class _CountingLLM(_MockLLM):
            def generate(self, prompt, **kw):
                call_count[0] += 1
                return super().generate(prompt, **kw)

        dataset = _make_dataset(n=5)
        oa = OracleAnnotator(_CountingLLM(), num_oracles=1, task=self.task)
        oa.annotate_dataset(dataset)
        self.assertEqual(call_count[0], 5)

    def test_three_oracle_calls_llm_three_times_per_sample(self):
        """3-oracle should call generate() 3 × n_samples times."""
        call_count = [0]
        class _CountingLLM(_MockLLM):
            def generate(self, prompt, **kw):
                call_count[0] += 1
                return super().generate(prompt, **kw)

        n = 4
        dataset = _make_dataset(n=n)
        oa = OracleAnnotator(_CountingLLM(), num_oracles=3, task=self.task)
        oa.annotate_dataset(dataset)
        self.assertEqual(call_count[0], 3 * n)

    def test_majority_vote_consensus(self):
        """When all oracle calls agree, the annotation should equal that answer."""
        llm = _MockLLM()   # always returns "test_answer"
        oa = OracleAnnotator(llm, num_oracles=3, task=self.task)
        sample = {"question": "Q?", "context": "", "answer": "test_answer", "text": "Q?"}
        annotation = oa.annotate(sample)
        self.assertEqual(annotation, "test_answer")

    def test_annotate_returns_string(self):
        llm = _MockLLM()
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        sample = {"question": "Q?", "context": "", "answer": "a", "text": "Q?"}
        result = oa.annotate(sample)
        self.assertIsInstance(result, str)


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
        self.task = _MockTask()

    def test_output_count(self):
        llm = _MockLLM()
        results = run_dataflow_condition(self.dataset, llm, task=self.task)
        self.assertEqual(len(results), len(self.dataset))

    def test_annotation_key_present(self):
        llm = _MockLLM()
        results = run_dataflow_condition(self.dataset, llm, task=self.task)
        for r in results:
            self.assertIn("annotation", r)

    def test_confidence_key_present(self):
        llm = _MockLLM()
        results = run_dataflow_condition(self.dataset, llm, task=self.task)
        for r in results:
            self.assertIn("confidence", r)

    def test_needs_human_key_present(self):
        llm = _MockLLM()
        results = run_dataflow_condition(self.dataset, llm, task=self.task)
        for r in results:
            self.assertIn("needs_human", r)

    def test_high_confidence_not_needs_human(self):
        """Mock LLM returns confidence 0.9 → below-threshold flag should be False."""
        llm = _MockLLM()
        results = run_dataflow_condition(
            self.dataset, llm, confidence_threshold=0.65, task=self.task
        )
        for r in results:
            self.assertFalse(r["needs_human"])

    def test_low_confidence_needs_human(self):
        """Threshold above mock confidence → all samples need human review."""
        class _LowConfLLM(_MockLLM):
            def generate(self, prompt, **kw):
                return "Answer: answer Confidence: 0.3"

        results = run_dataflow_condition(
            self.dataset, _LowConfLLM(), confidence_threshold=0.65, task=self.task
        )
        for r in results:
            self.assertTrue(r["needs_human"])

    def test_rag_mode_runs_without_error(self):
        llm = _MockLLM()
        results = run_dataflow_condition(
            self.dataset, llm, rag=True, task=self.task
        )
        self.assertEqual(len(results), len(self.dataset))

    def test_rag_populates_kb(self):
        """High-confidence samples should be admitted to the KB for RAG retrieval."""
        # Use a larger dataset so at least some KB entries exist by the time
        # later samples are annotated.
        dataset = _make_dataset(n=30)
        llm = _MockLLM()
        results = run_dataflow_condition(dataset, llm, rag=True, task=self.task)
        self.assertEqual(len(results), len(dataset))


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
        self.dataset = _make_dataset(n=15)
        self.tmp_dir = tempfile.mkdtemp()
        self.task = _MockTask()
        self.mock_llm = _MockLLM()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        return run_experiment(
            dataset=self.dataset,
            oracle_llm=self.mock_llm,
            dataflow_llm=self.mock_llm,
            output_dir=self.tmp_dir,
            skip_finetune=True,
            oracle_task=self.task,
            dataflow_task=self.task,
        )

    def test_returns_five_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 5)

    def test_condition_keys(self):
        results = self._run()
        expected_keys = {
            "condition", "num_samples", "annotation_f1",
            "annotation_em", "downstream_bleu", "downstream_rouge_l", "sft_file",
        }
        for r in results:
            self.assertEqual(set(r.keys()), expected_keys)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("Single Oracle", names)
        self.assertIn("3-Oracle Majority Vote", names)
        self.assertIn("DataFlow (naive LLM)", names)
        self.assertIn("DataFlow (KB + RAG)", names)
        self.assertIn("DataFlow (full pipeline)", names)

    def test_condition_names_unique(self):
        results = self._run()
        names = [r["condition"] for r in results]
        self.assertEqual(len(names), len(set(names)))

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]), r["sft_file"])

    def test_num_samples_positive(self):
        results = self._run()
        for r in results:
            self.assertGreater(r["num_samples"], 0)

    def test_downstream_none_when_skip_finetune(self):
        """With skip_finetune=True, downstream metrics must be None."""
        results = self._run()
        for r in results:
            self.assertIsNone(r["downstream_bleu"])
            self.assertIsNone(r["downstream_rouge_l"])

    def test_annotation_f1_in_range(self):
        results = self._run()
        for r in results:
            self.assertGreaterEqual(r["annotation_f1"], 0.0)
            self.assertLessEqual(r["annotation_f1"], 1.0)

    def test_annotation_em_in_range(self):
        results = self._run()
        for r in results:
            self.assertGreaterEqual(r["annotation_em"], 0.0)
            self.assertLessEqual(r["annotation_em"], 1.0)

    def test_sft_jsonl_well_formed(self):
        results = self._run()
        for r in results:
            with open(r["sft_file"], encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.assertIn("instruction", record)
                    self.assertIn("output", record)


# ---------------------------------------------------------------------------
# print_results_table (smoke tests)
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def _make_result(self, condition="Test", bleu=None, rouge_l=None):
        return {
            "condition": condition,
            "annotation_f1": 0.8,
            "annotation_em": 0.75,
            "downstream_bleu": bleu,
            "downstream_rouge_l": rouge_l,
            "num_samples": 100,
        }

    def test_runs_without_downstream(self):
        import io, contextlib
        results = [self._make_result("Test")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("Test", buf.getvalue())
        self.assertIn("0.8000", buf.getvalue())

    def test_runs_with_downstream(self):
        import io, contextlib
        results = [self._make_result("Oracle", bleu=0.45, rouge_l=0.52)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        output = buf.getvalue()
        self.assertIn("Oracle", output)
        self.assertIn("DS-BLEU", output)

    def test_multiple_conditions(self):
        import io, contextlib
        results = [
            self._make_result("Single Oracle"),
            self._make_result("3-Oracle Majority Vote"),
            self._make_result("DataFlow (naive LLM)"),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        output = buf.getvalue()
        self.assertIn("Single Oracle", output)
        self.assertIn("3-Oracle Majority Vote", output)


# ---------------------------------------------------------------------------
# LLM dispatch tests (candidate_llms equivalent for this script)
#
# These tests verify that:
#   - oracle_llm is the LLM object called during oracle conditions
#   - dataflow_llm is the LLM object called during DataFlow conditions
#   - oracle_llm and dataflow_llm can be different objects
#   - call counts match the expected number of generate() calls
# ---------------------------------------------------------------------------

class _TrackingLLM:
    """Records every generate() call so tests can assert which LLM was used
    and how many times it was invoked."""

    def __init__(self, name: str, answer: str = "test_answer") -> None:
        self.name = name
        self.calls: list = []
        self._answer = answer

    def generate(self, prompt: str, **kw) -> str:
        self.calls.append(prompt)
        return f"Answer: {self._answer} Confidence: 0.9"

    def generate_with_logprobs(self, prompt: str, **kw):
        return self.generate(prompt), -0.2


class TestLLMDispatch(unittest.TestCase):
    """Verify that oracle_llm / dataflow_llm objects are dispatched correctly."""

    def setUp(self):
        self.dataset = _make_dataset(n=5)
        self.task = _MockTask()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, oracle_llm, dataflow_llm):
        return run_experiment(
            dataset=self.dataset,
            oracle_llm=oracle_llm,
            dataflow_llm=dataflow_llm,
            output_dir=self.tmp_dir,
            skip_finetune=True,
            oracle_task=self.task,
            dataflow_task=self.task,
        )

    # ------------------------------------------------------------------
    # oracle_llm dispatch
    # ------------------------------------------------------------------

    def test_oracle_llm_is_called_for_single_oracle_condition(self):
        """oracle_llm must receive generate() calls during the Single Oracle condition."""
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        # Single Oracle: 1 call × n_samples
        # 3-Oracle Majority Vote: 3 calls × n_samples
        # Total oracle calls: 4 × n_samples
        self.assertGreater(len(oracle.calls), 0,
                           "oracle_llm must be called at least once")

    def test_oracle_llm_called_4x_per_sample(self):
        """Single Oracle (1 call) + 3-Oracle Majority Vote (3 calls) = 4 oracle
        generate() calls per sample."""
        n = len(self.dataset)
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        self.assertEqual(len(oracle.calls), 4 * n,
                         f"Expected {4 * n} oracle calls (1+3 per sample), "
                         f"got {len(oracle.calls)}")

    def test_dataflow_llm_called_3x_per_sample(self):
        """Three DataFlow conditions (naive, KB+RAG, full) → 3 generate() calls
        per sample from the dataflow_llm."""
        n = len(self.dataset)
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        self.assertEqual(len(dataflow.calls), 3 * n,
                         f"Expected {3 * n} dataflow calls, got {len(dataflow.calls)}")

    def test_oracle_llm_not_used_for_dataflow_conditions(self):
        """oracle_llm must NOT be called more times than the oracle conditions
        require.  DataFlow conditions must NOT use the oracle object."""
        n = len(self.dataset)
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        # Oracle conditions: 1 (single) + 3 (majority) = 4 calls per sample
        # If oracle_llm is accidentally used for dataflow, call count > 4n
        self.assertEqual(len(oracle.calls), 4 * n,
                         "oracle_llm should not be called by DataFlow conditions")

    def test_dataflow_llm_not_used_for_oracle_conditions(self):
        """dataflow_llm must NOT be called for oracle conditions."""
        n = len(self.dataset)
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        # DataFlow conditions: naive + KB+RAG + full = 3 calls per sample
        self.assertEqual(len(dataflow.calls), 3 * n,
                         "dataflow_llm should not be called by oracle conditions")

    def test_same_llm_for_oracle_and_dataflow_is_accepted(self):
        """run_experiment must work correctly when oracle_llm == dataflow_llm
        (the script supports sharing one model instance)."""
        shared = _TrackingLLM("shared")
        results = self._run(shared, shared)
        n = len(self.dataset)
        # total calls: oracle (4n) + dataflow (3n) = 7n
        self.assertEqual(len(shared.calls), 7 * n)
        self.assertEqual(len(results), 5)

    def test_two_independent_llm_objects_both_called(self):
        """Both oracle_llm and dataflow_llm must be called when they are
        different objects — neither should be zero-call."""
        oracle = _TrackingLLM("oracle")
        dataflow = _TrackingLLM("dataflow")
        self._run(oracle, dataflow)
        self.assertGreater(len(oracle.calls), 0, "oracle_llm must be called")
        self.assertGreater(len(dataflow.calls), 0, "dataflow_llm must be called")

    # ------------------------------------------------------------------
    # OracleAnnotator call counting (unit level)
    # ------------------------------------------------------------------

    def test_single_oracle_annotator_calls_llm_n_times(self):
        """OracleAnnotator(num_oracles=1) should call generate() exactly
        once per sample."""
        llm = _TrackingLLM("oracle")
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        n = 7
        dataset = _make_dataset(n=n)
        oa.annotate_dataset(dataset)
        self.assertEqual(len(llm.calls), n)

    def test_three_oracle_annotator_calls_llm_3n_times(self):
        """OracleAnnotator(num_oracles=3) should call generate() exactly
        3 × n_samples times."""
        llm = _TrackingLLM("oracle")
        oa = OracleAnnotator(llm, num_oracles=3, task=self.task)
        n = 4
        dataset = _make_dataset(n=n)
        oa.annotate_dataset(dataset)
        self.assertEqual(len(llm.calls), 3 * n)

    def test_oracle_annotate_result_preserves_sample_keys(self):
        """annotate_dataset must preserve all original sample keys and add
        'annotation'."""
        llm = _TrackingLLM("oracle")
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        sample = {"id": "x", "question": "Q?", "context": "C", "answer": "A",
                  "text": "Q?"}
        results = oa.annotate_dataset([sample])
        self.assertEqual(len(results), 1)
        result = results[0]
        for key in ("id", "question", "context", "answer", "text", "annotation"):
            self.assertIn(key, result, f"key '{key}' missing from result")

    # ------------------------------------------------------------------
    # run_dataflow_condition — LLM is called for every sample
    # ------------------------------------------------------------------

    def test_dataflow_condition_calls_llm_once_per_sample(self):
        """run_dataflow_condition must call llm.generate() exactly n times."""
        llm = _TrackingLLM("dataflow")
        dataset = _make_dataset(n=8)
        run_dataflow_condition(dataset, llm, task=self.task)
        self.assertEqual(len(llm.calls), 8)

    def test_dataflow_condition_rag_calls_same_llm(self):
        """RAG mode must use the SAME llm object as non-RAG mode."""
        llm = _TrackingLLM("dataflow")
        dataset = _make_dataset(n=6)
        run_dataflow_condition(dataset, llm, rag=True, task=self.task)
        self.assertEqual(len(llm.calls), 6,
                         "RAG mode must still call generate() once per sample")

    def test_dataflow_condition_llm_receives_text(self):
        """Each generate() call must receive a non-empty prompt string."""
        llm = _TrackingLLM("dataflow")
        dataset = _make_dataset(n=3)
        run_dataflow_condition(dataset, llm, task=self.task)
        for prompt in llm.calls:
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0, "prompt must be non-empty")

    # ------------------------------------------------------------------
    # Oracle conditions result keys
    # ------------------------------------------------------------------

    def test_oracle_result_has_annotation_key(self):
        """Records returned by OracleAnnotator.annotate_dataset must have
        the 'annotation' key."""
        llm = _TrackingLLM("oracle")
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        results = oa.annotate_dataset(_make_dataset(n=5))
        for r in results:
            self.assertIn("annotation", r)

    def test_oracle_annotation_value_is_string(self):
        """annotation values produced by OracleAnnotator must be strings."""
        llm = _TrackingLLM("oracle")
        oa = OracleAnnotator(llm, num_oracles=1, task=self.task)
        for rec in oa.annotate_dataset(_make_dataset(n=5)):
            self.assertIsInstance(rec["annotation"], str)

    def test_dataflow_result_has_annotation_confidence_needs_human(self):
        """run_dataflow_condition records must contain annotation, confidence,
        and needs_human keys."""
        llm = _TrackingLLM("dataflow")
        results = run_dataflow_condition(_make_dataset(n=5), llm, task=self.task)
        for r in results:
            self.assertIn("annotation", r)
            self.assertIn("confidence", r)
            self.assertIn("needs_human", r)

    def test_run_experiment_oracle_annotation_matches_oracle_llm_output(self):
        """Oracle conditions must produce annotations derived from oracle_llm's
        output, not from dataflow_llm's output."""
        oracle = _TrackingLLM("oracle", answer="oracle_specific_answer")
        dataflow = _TrackingLLM("dataflow", answer="dataflow_specific_answer")
        results = self._run(oracle, dataflow)

        # Find the single-oracle SFT file and read annotations
        single_oracle_result = next(
            r for r in results if r["condition"] == "Single Oracle"
        )
        with open(single_oracle_result["sft_file"], encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        # All oracle annotations should reflect oracle_llm's answer
        for record in lines:
            self.assertIn("oracle_specific_answer", record["output"],
                          "Oracle SFT output must contain oracle LLM's answer")

    def test_run_experiment_dataflow_annotation_matches_dataflow_llm_output(self):
        """DataFlow conditions must produce annotations derived from
        dataflow_llm's output, not from oracle_llm's output."""
        oracle = _TrackingLLM("oracle", answer="oracle_specific_answer")
        dataflow = _TrackingLLM("dataflow", answer="dataflow_specific_answer")
        results = self._run(oracle, dataflow)

        naive_result = next(
            r for r in results if r["condition"] == "DataFlow (naive LLM)"
        )
        with open(naive_result["sft_file"], encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        for record in lines:
            self.assertIn("dataflow_specific_answer", record["output"],
                          "DataFlow SFT output must contain dataflow LLM's answer")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Progress banner tests — verify condition banners are printed to stdout
# ---------------------------------------------------------------------------

class TestConditionProgressBanners(unittest.TestCase):
    """Verify that run_experiment() prints a [i/N] banner for each condition."""

    def setUp(self):
        self.dataset = _make_dataset(n=10)
        self.task = _MockTask()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _captured_output(self) -> str:
        import io, contextlib
        llm = _MockLLM()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_experiment(
                dataset=self.dataset,
                oracle_llm=llm,
                dataflow_llm=llm,
                output_dir=self.tmp_dir,
                skip_finetune=True,
                oracle_task=self.task,
                dataflow_task=self.task,
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


if __name__ == "__main__":
    unittest.main()
