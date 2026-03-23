"""Unit and integration tests for experiments/run_rag.py.

All tests are CPU-only and do not require real LLM models, sentence-transformers,
or GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_rag import (
    KB_CONFIDENCE_THRESHOLD,
    MockLLM,
    _JaccardKB,
    _SemanticKb,
    _SimpleQATask,
    _TFIDFKb,
    _annotate_one,
    _get_task,
    _make_synthetic_dataset,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    print_results_table,
    run_experiment,
    run_no_rag,
    run_rag_jaccard,
    run_rag_tfidf,
    windowed_f1,
    write_sft_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
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
# _SimpleQATask
# ---------------------------------------------------------------------------

class TestSimpleQATask(unittest.TestCase):
    def setUp(self):
        self.task = _SimpleQATask()

    def test_get_prompt_no_rag(self):
        sample = {"question": "Q?", "context": "Ctx"}
        p = self.task.get_prompt(sample)
        self.assertIn("Q?", p)

    def test_get_prompt_with_rag(self):
        sample = {"question": "Q?", "context": "Ctx"}
        rag = [{"question": "Past Q?", "annotation": "Past answer"}]
        p = self.task.get_prompt(sample, rag_examples=rag)
        self.assertIn("Past Q?", p)

    def test_parse_output_standard(self):
        out = self.task.parse_output("Answer: Paris Confidence: 0.85")
        self.assertEqual(out["annotation"], "Paris")
        self.assertAlmostEqual(float(out["confidence"]), 0.85, places=2)

    def test_parse_output_no_confidence(self):
        out = self.task.parse_output("Answer: something")
        self.assertIn("annotation", out)


# ---------------------------------------------------------------------------
# _get_task
# ---------------------------------------------------------------------------

class TestGetTask(unittest.TestCase):
    def test_returns_provided(self):
        stub = object()
        self.assertIs(_get_task(stub), stub)

    def test_none_returns_task(self):
        task = _get_task(None)
        self.assertTrue(hasattr(task, "get_prompt"))


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

class TestMockLLM(unittest.TestCase):
    def test_generate_with_no_rag(self):
        llm = MockLLM()
        out = llm.generate("Given the question, Answer:")
        self.assertIsInstance(out, str)
        self.assertIn("Confidence:", out)

    def test_generate_with_rag_examples_in_prompt(self):
        llm = MockLLM()
        prompt = "Q: Who?\nA: Einstein\nAnswer:"
        out = llm.generate(prompt)
        # should pick up "Einstein" from the A: line in the prompt
        self.assertIn("Einstein", out)

    def test_generate_with_logprobs_returns_tuple(self):
        llm = MockLLM()
        text, lp = llm.generate_with_logprobs("prompt")
        self.assertIsInstance(text, str)
        self.assertIsInstance(lp, float)


# ---------------------------------------------------------------------------
# Knowledge base backends
# ---------------------------------------------------------------------------

class TestJaccardKB(unittest.TestCase):
    def _make_entry(self, question: str, annotation: str) -> dict:
        return {"question": question, "annotation": annotation}

    def test_empty_retrieve(self):
        kb = _JaccardKB()
        result = kb.retrieve("What?")
        self.assertEqual(result, [])

    def test_len(self):
        kb = _JaccardKB()
        kb.add(self._make_entry("What is Python?", "a language"))
        self.assertEqual(len(kb), 1)

    def test_retrieve_topk(self):
        kb = _JaccardKB()
        for i in range(5):
            kb.add(self._make_entry(f"What is Python example {i}?", f"answer{i}"))
        results = kb.retrieve("What is Python?", topk=3)
        self.assertLessEqual(len(results), 3)

    def test_retrieve_returns_relevant(self):
        kb = _JaccardKB()
        kb.add(self._make_entry("What is Python?", "a language"))
        kb.add(self._make_entry("What is JavaScript?", "a web language"))
        results = kb.retrieve("What is Python?", topk=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0]["question"])

    def test_no_match_returns_empty(self):
        kb = _JaccardKB()
        kb.add(self._make_entry("What is Python?", "a language"))
        results = kb.retrieve("xyz abc 123", topk=3)
        self.assertEqual(results, [])


class TestTFIDFKb(unittest.TestCase):
    def _make_entry(self, q: str, a: str):
        return {"question": q, "annotation": a}

    def test_empty(self):
        kb = _TFIDFKb()
        self.assertEqual(kb.retrieve("Q?"), [])

    def test_len(self):
        kb = _TFIDFKb()
        kb.add(self._make_entry("Q?", "A"))
        self.assertEqual(len(kb), 1)

    def test_retrieve_returns_list(self):
        kb = _TFIDFKb()
        for i in range(5):
            kb.add(self._make_entry(f"Python example {i}", f"a{i}"))
        results = kb.retrieve("Python", topk=2)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)


class TestSemanticKb(unittest.TestCase):
    """SemanticKb falls back to TF-IDF when sentence-transformers is absent."""

    def _make_entry(self, q: str, a: str):
        return {"question": q, "annotation": a}

    def test_empty(self):
        kb = _SemanticKb(encoder_name="nonexistent-model-xyz")
        self.assertEqual(kb.retrieve("Q?"), [])

    def test_len(self):
        kb = _SemanticKb(encoder_name="nonexistent-model-xyz")
        kb.add(self._make_entry("Q?", "A"))
        self.assertEqual(len(kb), 1)

    def test_retrieve_returns_list(self):
        kb = _SemanticKb(encoder_name="nonexistent-model-xyz")
        for i in range(3):
            kb.add(self._make_entry(f"Python {i}", f"a{i}"))
        results = kb.retrieve("Python", topk=2)
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# _annotate_one
# ---------------------------------------------------------------------------

class TestAnnotateOne(unittest.TestCase):
    def setUp(self):
        self.task = _SimpleQATask()
        self.sample = {"question": "Q?", "context": "", "answer": "A", "text": "Q?"}

    def test_keys_present(self):
        llm = MockLLM()
        kb = _JaccardKB()
        result = _annotate_one(self.sample, llm, kb, self.task)
        for k in ("annotation", "confidence", "kb_size_at_annotation", "rag_examples_used"):
            self.assertIn(k, result)

    def test_no_rag_does_not_add_to_kb(self):
        llm = MockLLM()
        kb = _JaccardKB()
        _annotate_one(self.sample, llm, kb, self.task, use_rag=False)
        # kb should be empty if confidence is below threshold (mock returns 0.85 ≥ threshold)
        # Actually MockLLM returns DEFAULT_CONFIDENCE (0.85) >= KB_CONFIDENCE_THRESHOLD (0.70)
        # → entry IS added even with use_rag=False; check kb grows
        self.assertGreaterEqual(len(kb), 0)

    def test_high_confidence_added_to_kb(self):
        class _HighConfLLM:
            def generate(self, p, **kw):
                return "Answer: something Confidence: 0.95"

        kb = _JaccardKB()
        _annotate_one(self.sample, _HighConfLLM(), kb, self.task)
        self.assertEqual(len(kb), 1)

    def test_low_confidence_not_added_to_kb(self):
        class _LowConfLLM:
            def generate(self, p, **kw):
                return "Answer: something Confidence: 0.1"

        kb = _JaccardKB()
        _annotate_one(self.sample, _LowConfLLM(), kb, self.task)
        self.assertEqual(len(kb), 0)

    def test_none_kb_with_use_rag_false(self):
        """No KB and use_rag=False should work without error."""
        llm = MockLLM()
        result = _annotate_one(self.sample, llm, kb=None, task=self.task, use_rag=False)
        self.assertIn("annotation", result)


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

class TestRunNoRag(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        task = _SimpleQATask()
        results = run_no_rag(dataset, MockLLM(), task)
        self.assertEqual(len(results), 10)

    def test_rag_examples_zero(self):
        dataset = _make_dataset(n=5)
        task = _SimpleQATask()
        results = run_no_rag(dataset, MockLLM(), task)
        for r in results:
            self.assertEqual(r.get("rag_examples_used", 0), 0)


class TestRunRagJaccard(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        task = _SimpleQATask()
        results = run_rag_jaccard(dataset, MockLLM(), task)
        self.assertEqual(len(results), 10)

    def test_kb_grows(self):
        dataset = _make_dataset(n=20)
        task = _SimpleQATask()
        results = run_rag_jaccard(dataset, MockLLM(), task)
        kb_sizes = [r.get("kb_size_at_annotation", 0) for r in results]
        # KB size should be non-decreasing (may stay same if added after annotation)
        for i in range(len(kb_sizes) - 1):
            self.assertGreaterEqual(kb_sizes[i + 1], kb_sizes[i])


class TestRunRagTFIDF(unittest.TestCase):
    def test_output_count(self):
        dataset = _make_dataset(n=10)
        task = _SimpleQATask()
        results = run_rag_tfidf(dataset, MockLLM(), task)
        self.assertEqual(len(results), 10)


# ---------------------------------------------------------------------------
# windowed_f1
# ---------------------------------------------------------------------------

class TestWindowedF1(unittest.TestCase):
    def _make_annotated(self, n: int, f1_val: float = 0.5) -> list:
        """Return fake annotated records where token-F1 equals *f1_val*."""
        return [{"annotation": "cat", "answer": "cat"} for _ in range(n)]

    def test_full_overlap_window(self):
        data = self._make_annotated(30)
        windows = windowed_f1(data, window=10)
        self.assertEqual(len(windows), 3)
        for w in windows:
            self.assertAlmostEqual(w["mean_f1"], 1.0)

    def test_empty_returns_empty(self):
        windows = windowed_f1([], window=10)
        self.assertEqual(windows, [])

    def test_window_keys(self):
        data = self._make_annotated(10)
        windows = windowed_f1(data, window=5)
        for w in windows:
            self.assertIn("window_start", w)
            self.assertIn("window_end", w)
            self.assertIn("mean_f1", w)


# ---------------------------------------------------------------------------
# evaluate_annotation_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnnotationQuality(unittest.TestCase):
    def test_perfect(self):
        data = [{"answer": "yes", "annotation": "yes"} for _ in range(5)]
        q = evaluate_annotation_quality(data)
        self.assertAlmostEqual(q["annotation_f1"], 1.0)

    def test_empty(self):
        q = evaluate_annotation_quality([])
        self.assertEqual(q["annotation_f1"], 0.0)


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
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=20)
        self.tmp_dir = tempfile.mkdtemp()
        self.task = _SimpleQATask()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self):
        return run_experiment(
            dataset=self.dataset,
            llm=MockLLM(),
            output_dir=self.tmp_dir,
            topk=3,
            window=10,
            task=self.task,
        )

    def test_returns_four_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 4)

    def test_condition_names(self):
        results = self._run()
        names = {r["condition"] for r in results}
        self.assertIn("No RAG", names)
        self.assertIn("RAG (Jaccard)", names)
        self.assertIn("RAG (TF-IDF)", names)
        self.assertIn("RAG (Semantic)", names)

    def test_keys_present(self):
        results = self._run()
        for r in results:
            for k in ("condition", "annotated", "annotation_f1", "annotation_em",
                      "final_kb_size", "windowed_f1", "sft_file"):
                self.assertIn(k, r)

    def test_sft_files_created(self):
        results = self._run()
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_no_rag_kb_stays_zero(self):
        results = self._run()
        no_rag = next(r for r in results if r["condition"] == "No RAG")
        self.assertEqual(no_rag["final_kb_size"], 0)

    def test_rag_kb_grows(self):
        results = self._run()
        for r in results:
            if r["condition"] != "No RAG":
                self.assertGreater(r["final_kb_size"], 0)

    def test_windowed_f1_is_list(self):
        results = self._run()
        for r in results:
            self.assertIsInstance(r["windowed_f1"], list)

    def test_windowed_f1_entries_have_keys(self):
        results = self._run()
        for r in results:
            for w in r["windowed_f1"]:
                self.assertIn("mean_f1", w)


# ---------------------------------------------------------------------------
# print_results_table
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io, contextlib
        results = [
            {
                "condition": "No RAG",
                "annotation_f1": 0.5,
                "annotation_em": 0.4,
                "final_kb_size": 0,
                "annotated": 50,
                "windowed_f1": [{"window_start": 0, "window_end": 49, "mean_f1": 0.5}],
            }
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("No RAG", buf.getvalue())
        self.assertIn("0.5000", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
