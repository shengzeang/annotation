"""Tests for experiments/run_noisy_kb.py.

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

from experiments.run_noisy_kb import (
    DEFAULT_NOISE_RATES,
    MockContextAwareLLM,
    MockJudgeLLM,
    _WRONG_ANSWERS,
    _make_condition_name,
    _make_synthetic_dataset,
    _condition_already_done,
    _condition_result_path,
    _load_condition_result,
    _safe_name,
    _save_condition_result,
    _sft_output_path,
    build_seed_kb,
    compute_exact_match,
    compute_token_f1,
    evaluate_annotation_quality,
    inject_noise,
    load_squad_dataset,
    print_results_table,
    run_experiment,
    windowed_f1,
    write_sft_jsonl,
)

import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 40) -> list:
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

    def test_one_empty(self):
        self.assertAlmostEqual(compute_token_f1("", "cat"), 0.0)

    def test_partial_overlap(self):
        f1 = compute_token_f1("the cat sat", "the cat")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)


class TestComputeExactMatch(unittest.TestCase):
    def test_match(self):
        self.assertEqual(compute_exact_match("Paris", "Paris"), 1.0)

    def test_case_insensitive(self):
        self.assertEqual(compute_exact_match("PARIS", "paris"), 1.0)

    def test_no_match(self):
        self.assertEqual(compute_exact_match("London", "Paris"), 0.0)

    def test_whitespace_trimmed(self):
        self.assertEqual(compute_exact_match("  Paris  ", "Paris"), 1.0)


# ---------------------------------------------------------------------------
# MockContextAwareLLM
# ---------------------------------------------------------------------------

class TestMockContextAwareLLM(unittest.TestCase):
    def setUp(self):
        self.answer_map = {
            "What is the capital of France?": "Paris",
            "Who wrote Hamlet?": "Shakespeare",
        }
        self.llm = MockContextAwareLLM(answer_map=self.answer_map)

    def _rag_prompt(self, *kb_answers: str) -> str:
        """Build a minimal prompt that looks like QATask output with RAG context."""
        rag_block = "\nHere are some similar QA pairs from the knowledge base to help you answer:\n"
        for i, ans in enumerate(kb_answers):
            rag_block += f"Q: Sample question {i}\nA: {ans}\n"
        return (
            "Given the following question, please answer it as accurately as possible.\n"
            "Output format: Answer: <your answer> Confidence: <score>\n"
            "Question: What is the capital of France?\n"
            "Context: France is a country in Western Europe.\n"
            f"{rag_block}"
            "Answer:"
        )

    def _no_rag_prompt(self, question: str) -> str:
        return (
            "Given the following question, please answer it as accurately as possible.\n"
            "Output format: Answer: <your answer> Confidence: <score>\n"
            f"Question: {question}\n"
            "Context: Some context.\n"
            "Answer:"
        )

    def test_copies_single_kb_answer(self):
        prompt = self._rag_prompt("relativity")
        out = self.llm.generate(prompt)
        self.assertIn("relativity", out)
        self.assertIn("Confidence:", out)

    def test_majority_vote_picks_most_common(self):
        """With 2 noisy entries and 1 correct, majority vote returns noisy."""
        prompt = self._rag_prompt("wrong_answer", "wrong_answer", "Paris")
        out = self.llm.generate(prompt)
        self.assertIn("wrong_answer", out)
        self.assertNotIn("Paris", out)

    def test_majority_vote_correct_when_majority_correct(self):
        """With 2 correct entries and 1 noisy, majority vote returns correct."""
        prompt = self._rag_prompt("Paris", "Paris", "wrong_answer")
        out = self.llm.generate(prompt)
        self.assertIn("Paris", out)

    def test_falls_back_to_answer_map_when_no_rag(self):
        prompt = self._no_rag_prompt("What is the capital of France?")
        out = self.llm.generate(prompt)
        self.assertIn("Paris", out)

    def test_returns_unknown_when_no_map_entry(self):
        prompt = self._no_rag_prompt("An unknown question not in the map?")
        out = self.llm.generate(prompt)
        self.assertIn("unknown", out.lower())

    def test_output_format_contains_answer_and_confidence(self):
        prompt = self._no_rag_prompt("What is the capital of France?")
        out = self.llm.generate(prompt)
        self.assertIn("Answer:", out)
        self.assertIn("Confidence:", out)

    def test_generate_with_logprobs_returns_tuple(self):
        prompt = self._no_rag_prompt("Who wrote Hamlet?")
        text, lp = self.llm.generate_with_logprobs(prompt)
        self.assertIsInstance(text, str)
        self.assertIsInstance(lp, float)

    def test_empty_answer_map(self):
        llm = MockContextAwareLLM()
        prompt = self._no_rag_prompt("Unknown question?")
        out = llm.generate(prompt)
        self.assertIn("Answer:", out)


# ---------------------------------------------------------------------------
# MockJudgeLLM
# ---------------------------------------------------------------------------

class TestMockJudgeLLM(unittest.TestCase):
    def test_returns_one(self):
        judge = MockJudgeLLM()
        self.assertEqual(judge.generate("any prompt").strip(), "1")


# ---------------------------------------------------------------------------
# inject_noise
# ---------------------------------------------------------------------------

class TestInjectNoise(unittest.TestCase):
    def _make_entries(self, n: int = 10) -> list:
        return [{"annotation": "correct", "question": f"Q{i}"} for i in range(n)]

    def test_zero_noise_unchanged(self):
        entries = self._make_entries(10)
        rng = random.Random(0)
        noisy = inject_noise(entries, noise_rate=0.0, rng=rng)
        self.assertEqual(len(noisy), 10)
        for e in noisy:
            self.assertEqual(e["annotation"], "correct")

    def test_full_noise_all_wrong(self):
        entries = self._make_entries(10)
        rng = random.Random(0)
        noisy = inject_noise(entries, noise_rate=1.0, rng=rng)
        for e in noisy:
            self.assertIn(e["annotation"], _WRONG_ANSWERS)

    def test_partial_noise_in_range(self):
        entries = self._make_entries(100)
        rng = random.Random(42)
        noisy = inject_noise(entries, noise_rate=0.5, rng=rng)
        n_wrong = sum(1 for e in noisy if e["annotation"] != "correct")
        # With 100 samples at 50% noise, expect roughly 30–70 corrupted.
        self.assertGreater(n_wrong, 20)
        self.assertLess(n_wrong, 80)

    def test_does_not_modify_original(self):
        entries = self._make_entries(5)
        original = [dict(e) for e in entries]
        rng = random.Random(0)
        inject_noise(entries, noise_rate=1.0, rng=rng)
        for orig, curr in zip(original, entries):
            self.assertEqual(orig["annotation"], curr["annotation"])

    def test_other_fields_preserved(self):
        entries = [{"annotation": "correct", "question": "Q0", "extra": "keep"}]
        rng = random.Random(0)
        noisy = inject_noise(entries, noise_rate=1.0, rng=rng)
        self.assertEqual(noisy[0]["extra"], "keep")
        self.assertEqual(noisy[0]["question"], "Q0")


# ---------------------------------------------------------------------------
# build_seed_kb
# ---------------------------------------------------------------------------

class TestBuildSeedKb(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = _make_dataset(n=30)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build(self, seed_size: int = 10, noise_rate: float = 0.0) -> str:
        kb_path = os.path.join(self.tmp_dir, "test_kb.json")
        rng = random.Random(42)
        build_seed_kb(self.dataset, seed_size, noise_rate, rng, kb_path)
        return kb_path

    def test_creates_file(self):
        kb_path = self._build()
        self.assertTrue(os.path.exists(kb_path))

    def test_correct_entry_count(self):
        kb_path = self._build(seed_size=10)
        with open(kb_path) as f:
            entries = json.load(f)
        self.assertEqual(len(entries), 10)

    def test_clean_kb_all_correct(self):
        kb_path = self._build(seed_size=10, noise_rate=0.0)
        with open(kb_path) as f:
            entries = json.load(f)
        for entry, orig in zip(entries, self.dataset[:10]):
            self.assertEqual(entry["annotation"], orig["answer"])

    def test_fully_noisy_kb_all_wrong(self):
        kb_path = self._build(seed_size=20, noise_rate=1.0)
        with open(kb_path) as f:
            entries = json.load(f)
        correct_answers = {rec["answer"] for rec in self.dataset[:20]}
        for entry in entries:
            self.assertNotIn(entry["annotation"], correct_answers)

    def test_confidence_field_set(self):
        kb_path = self._build(seed_size=5)
        with open(kb_path) as f:
            entries = json.load(f)
        for e in entries:
            self.assertIn("confidence", e)

    def test_seed_larger_than_dataset_clips(self):
        """seed_size larger than dataset should not crash."""
        kb_path = os.path.join(self.tmp_dir, "big_kb.json")
        rng = random.Random(0)
        # dataset has 30 entries, seed_size=100 — should use all 30
        build_seed_kb(self.dataset, 100, 0.0, rng, kb_path)
        with open(kb_path) as f:
            entries = json.load(f)
        self.assertEqual(len(entries), 30)


# ---------------------------------------------------------------------------
# _make_condition_name
# ---------------------------------------------------------------------------

class TestMakeConditionName(unittest.TestCase):
    def test_zero_noise(self):
        self.assertEqual(_make_condition_name(0.0), "Noise 00pct")

    def test_twenty_five(self):
        self.assertEqual(_make_condition_name(0.25), "Noise 25pct")

    def test_fifty(self):
        self.assertEqual(_make_condition_name(0.5), "Noise 50pct")

    def test_seventy_five(self):
        self.assertEqual(_make_condition_name(0.75), "Noise 75pct")

    def test_hundred(self):
        self.assertEqual(_make_condition_name(1.0), "Noise 100pct")


# ---------------------------------------------------------------------------
# _safe_name / resume helpers
# ---------------------------------------------------------------------------

class TestSafeName(unittest.TestCase):
    def test_clean_condition(self):
        self.assertEqual(_safe_name("Noise 00pct"), "noise_00pct")

    def test_noise_condition(self):
        self.assertEqual(_safe_name("Noise 75pct"), "noise_75pct")

    def test_lowercase(self):
        name = _safe_name("SomeCondition")
        self.assertEqual(name, name.lower())


class TestResumeMechanismNoisyKb(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = _make_dataset(n=40)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, noise_rates=None):
        if noise_rates is None:
            noise_rates = [0.0, 0.75]
        answer_map = {r["question"]: r["answer"] for r in self.dataset}
        llm = MockContextAwareLLM(answer_map=answer_map)
        return run_experiment(
            dataset=self.dataset,
            llm=llm,
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir,
            noise_rates=noise_rates,
            seed_size=5,
            topk=3,
            window=8,
            seed=42,
            force_fallback=True,
        )

    def test_not_done_initially(self):
        for rate in [0.0, 0.75]:
            cond = _make_condition_name(rate)
            self.assertFalse(_condition_already_done(cond, self.tmp_dir))

    def test_all_conditions_done_after_run(self):
        self._run()
        for rate in [0.0, 0.75]:
            cond = _make_condition_name(rate)
            self.assertTrue(_condition_already_done(cond, self.tmp_dir))

    def test_second_run_skips_all(self):
        import io
        import contextlib
        self._run()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run()
        self.assertIn("Already done", buf.getvalue())

    def test_second_run_same_results(self):
        r1 = self._run()
        r2 = self._run()
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a["condition"], b["condition"])
            self.assertAlmostEqual(a["annotation_f1"], b["annotation_f1"])

    def test_sft_file_alone_triggers_done(self):
        cond = _make_condition_name(0.0)
        sft_path = _sft_output_path(cond, self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        with open(sft_path, "w") as f:
            f.write('{"instruction": "Q", "output": "A"}\n')
        self.assertTrue(_condition_already_done(cond, self.tmp_dir))

    def test_save_load_result_roundtrip(self):
        result = {
            "condition": "Noise 00pct",
            "noise_rate": 0.0,
            "annotated": 20,
            "annotation_f1": 0.8,
            "annotation_em": 0.6,
            "final_kb_size": 25,
            "windowed_f1": [],
            "sft_file": "/tmp/foo.jsonl",
        }
        _save_condition_result(result, self.tmp_dir)
        self.assertTrue(_condition_already_done("Noise 00pct", self.tmp_dir))
        loaded = _load_condition_result("Noise 00pct", self.tmp_dir)
        self.assertEqual(loaded["condition"], result["condition"])
        self.assertAlmostEqual(loaded["annotation_f1"], result["annotation_f1"])

    def test_routing_skipped_when_all_done(self):
        """When all conditions are already done, routing should not be recomputed."""
        import io
        import contextlib
        # Pre-create SFT files so all conditions are marked done
        for rate in [0.0, 0.75]:
            cond = _make_condition_name(rate)
            sft_path = _sft_output_path(cond, self.tmp_dir)
            with open(sft_path, "w") as f:
                f.write('{"instruction": "Q", "output": "A"}\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = self._run(noise_rates=[0.0, 0.75])
        self.assertEqual(buf.getvalue().count("Already done"), 2)
        self.assertEqual(len(results), 2)


# ---------------------------------------------------------------------------
# windowed_f1
# ---------------------------------------------------------------------------

class TestWindowedF1(unittest.TestCase):
    def test_perfect_quality(self):
        data = [{"annotation": "cat", "answer": "cat"} for _ in range(20)]
        windows = windowed_f1(data, window=10)
        self.assertEqual(len(windows), 2)
        for w in windows:
            self.assertAlmostEqual(w["mean_f1"], 1.0)

    def test_empty(self):
        self.assertEqual(windowed_f1([], window=10), [])

    def test_window_keys(self):
        data = [{"annotation": "cat", "answer": "cat"} for _ in range(10)]
        for w in windowed_f1(data, window=5):
            for k in ("window_start", "window_end", "mean_f1"):
                self.assertIn(k, w)


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
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("instruction", rec)
                    self.assertIn("output", rec)
        finally:
            os.unlink(path)


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

    def test_answers_non_empty(self):
        for rec in _make_synthetic_dataset(n=8):
            self.assertTrue(rec["answer"])


# ---------------------------------------------------------------------------
# load_squad_dataset — uses synthetic fallback when path does not exist
# ---------------------------------------------------------------------------

class TestLoadSquadDataset(unittest.TestCase):
    def test_fallback_synthetic(self):
        ds = load_squad_dataset("/nonexistent/path.json", max_samples=10)
        self.assertEqual(len(ds), 10)
        for rec in ds:
            self.assertIn("question", rec)
            self.assertIn("answer", rec)


# ---------------------------------------------------------------------------
# run_experiment — main integration tests
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset(n=50)
        self.tmp_dir = tempfile.mkdtemp()
        answer_map = {r["question"]: r["answer"] for r in self.dataset}
        self.llm = MockContextAwareLLM(answer_map=answer_map)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, noise_rates=None, seed_size=5, window=8):
        return run_experiment(
            dataset=self.dataset,
            llm=self.llm,
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir,
            noise_rates=noise_rates if noise_rates is not None else DEFAULT_NOISE_RATES,
            seed_size=seed_size,
            topk=3,
            window=window,
            seed=42,
            force_fallback=True,
        )

    def test_returns_correct_number_of_conditions(self):
        results = self._run(noise_rates=[0.0, 0.5])
        self.assertEqual(len(results), 2)

    def test_default_four_conditions(self):
        results = self._run()
        self.assertEqual(len(results), 4)

    def test_condition_names_match_noise_rates(self):
        noise_rates = [0.0, 0.75]
        results = self._run(noise_rates=noise_rates)
        names = {r["condition"] for r in results}
        self.assertIn(_make_condition_name(0.0), names)
        self.assertIn(_make_condition_name(0.75), names)

    def test_required_keys_in_results(self):
        results = self._run(noise_rates=[0.0])
        for r in results:
            for k in (
                "condition", "noise_rate", "annotated", "annotation_f1",
                "annotation_em", "final_kb_size", "windowed_f1", "sft_file",
            ):
                self.assertIn(k, r)

    def test_noise_rate_stored_in_result(self):
        results = self._run(noise_rates=[0.0, 0.75])
        rates = {r["noise_rate"] for r in results}
        self.assertIn(0.0, rates)
        self.assertIn(0.75, rates)

    def test_sft_files_created(self):
        results = self._run(noise_rates=[0.0])
        for r in results:
            self.assertTrue(os.path.exists(r["sft_file"]))

    def test_sft_files_contain_valid_jsonl(self):
        results = self._run(noise_rates=[0.0])
        for r in results:
            with open(r["sft_file"]) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertIn("instruction", rec)
                    self.assertIn("output", rec)

    def test_kb_grows_for_all_conditions(self):
        """All conditions use RAG — KB should accumulate entries."""
        results = self._run(noise_rates=[0.0, 0.75])
        for r in results:
            self.assertGreater(r["final_kb_size"], 0)

    def test_windowed_f1_is_list(self):
        results = self._run(noise_rates=[0.0])
        for r in results:
            self.assertIsInstance(r["windowed_f1"], list)

    def test_windowed_f1_entries_have_keys(self):
        results = self._run(noise_rates=[0.0], window=8)
        for r in results:
            for w in r["windowed_f1"]:
                for k in ("window_start", "window_end", "mean_f1"):
                    self.assertIn(k, w)

    def test_clean_kb_higher_f1_than_noisy_kb(self):
        """Clean KB (0% noise) should produce higher F1 than 75% noisy KB."""
        results = self._run(noise_rates=[0.0, 0.75])
        clean = next(r for r in results if r["noise_rate"] == 0.0)
        noisy = next(r for r in results if r["noise_rate"] == 0.75)
        self.assertGreater(clean["annotation_f1"], noisy["annotation_f1"])

    def test_f1_lower_at_high_noise_than_zero_noise(self):
        """Across the full noise range, F1 at 0% should exceed F1 at 75%.

        Strict per-step monotonicity can be fragile with small seed sizes;
        we instead verify the overall downward trend from the lowest to the
        highest noise rate, which is the key experimental claim.
        """
        dataset = _make_synthetic_dataset(n=80, seed=5)
        answer_map = {r["question"]: r["answer"] for r in dataset}
        llm = MockContextAwareLLM(answer_map=answer_map)
        tmp = tempfile.mkdtemp()
        try:
            results = run_experiment(
                dataset=dataset,
                llm=llm,
                judge_llm=MockJudgeLLM(),
                output_dir=tmp,
                noise_rates=[0.0, 0.25, 0.50, 0.75],
                seed_size=20,
                topk=3,
                window=10,
                seed=42,
                force_fallback=True,
            )
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        f1_by_noise = sorted(
            [(r["noise_rate"], r["annotation_f1"]) for r in results],
            key=lambda x: x[0],
        )
        f1_0pct = f1_by_noise[0][1]
        f1_75pct = f1_by_noise[-1][1]
        self.assertGreater(
            f1_0pct, f1_75pct,
            msg=f"F1 at 0% noise ({f1_0pct:.4f}) should exceed F1 at 75% noise ({f1_75pct:.4f})",
        )

    def test_high_noise_windowed_f1_declines(self):
        """For 75% noise, windowed F1 should not improve over time."""
        # Use more samples and smaller windows for visible trend
        dataset = _make_synthetic_dataset(n=80, seed=7)
        answer_map = {r["question"]: r["answer"] for r in dataset}
        llm = MockContextAwareLLM(answer_map=answer_map)
        results = run_experiment(
            dataset=dataset,
            llm=llm,
            judge_llm=MockJudgeLLM(),
            output_dir=self.tmp_dir + "_trend",
            noise_rates=[0.0, 0.75],
            seed_size=8,
            topk=3,
            window=10,
            seed=42,
            force_fallback=True,
        )
        import shutil
        shutil.rmtree(self.tmp_dir + "_trend", ignore_errors=True)

        noisy = next(r for r in results if r["noise_rate"] == 0.75)
        f1_windows = [w["mean_f1"] for w in noisy.get("windowed_f1", [])]
        # High-noise condition should have at least one window
        self.assertGreater(len(f1_windows), 0)
        # First window F1 should be >= last window F1 (quality does not improve)
        if len(f1_windows) >= 2:
            self.assertGreaterEqual(
                f1_windows[0], f1_windows[-1],
                msg="High-noise KB quality should not improve over time",
            )

    def test_banner_printed_for_each_condition(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run(noise_rates=[0.0, 0.75])
        for name in [_make_condition_name(0.0), _make_condition_name(0.75)]:
            self.assertIn(name, buf.getvalue())

    def test_banner_fraction_format(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run(noise_rates=[0.0, 0.75])
        self.assertIn("[1/2]", buf.getvalue())
        self.assertIn("[2/2]", buf.getvalue())


# ---------------------------------------------------------------------------
# print_results_table
# ---------------------------------------------------------------------------

class TestPrintResultsTable(unittest.TestCase):
    def test_smoke(self):
        import io
        import contextlib
        results = [
            {
                "condition": _make_condition_name(r),
                "noise_rate": r,
                "annotation_f1": 1.0 - r,
                "annotation_em": 1.0 - r,
                "final_kb_size": 30,
                "annotated": 100,
                "windowed_f1": [
                    {"window_start": 0, "window_end": 49, "mean_f1": 1.0 - r}
                ],
            }
            for r in [0.0, 0.25, 0.50, 0.75]
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results, window=50)
        output = buf.getvalue()
        for r in results:
            self.assertIn(r["condition"], output)
        self.assertIn("Per-window token-F1", output)

    def test_no_windowed_data_no_crash(self):
        import io
        import contextlib
        results = [{
            "condition": "Noise 00pct",
            "noise_rate": 0.0,
            "annotation_f1": 0.9,
            "annotation_em": 0.8,
            "final_kb_size": 20,
            "annotated": 50,
            "windowed_f1": [],
        }]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_results_table(results)
        self.assertIn("Noise 00pct", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
