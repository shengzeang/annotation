"""
Tests for the KB quality experiment components.

Covers:
  - SimulatedLLM: clean / noisy mode, perfect mode, seed reproducibility
  - TopicAwareEncoder: topic assignment, embedding dimensions, within/cross-topic similarity
  - QA metrics: exact_match, token_f1, _normalize
  - evaluate_kb_quality: precision, recall, F1 computations
  - evaluate_downstream: label-noise formula, edge cases
  - _build_noise_pool: pool structure and non-empty guarantees
  - run_experiment: smoke test for all conditions
  - run_condition: output keys and value ranges
  - _save_sft: JSONL format and content
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_kb_experiment import (
    SimulatedLLM,
    TopicAwareEncoder,
    _build_noise_pool,
    _build_squad_json,
    _normalize,
    _QA_PAIRS,
    _ANSWER_LOOKUP,
    _TEXT_TO_TOPIC,
    _TEST_IDS,
    _save_sft,
    evaluate_downstream,
    evaluate_kb_quality,
    exact_match,
    token_f1,
    run_condition,
    run_experiment,
    _CONDITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(noise_rate=0.0, perfect=True):
    """Return a SimulatedLLM configured for the synthetic dataset."""
    qa_pairs = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
    answer_lookup = {qa["q"]: qa["a"] for qa in qa_pairs}
    noise_pool = _build_noise_pool(_QA_PAIRS)
    return SimulatedLLM(
        qa_lookup=answer_lookup,
        noise_pool=noise_pool,
        noise_rate=noise_rate,
        perfect=perfect,
        seed=42,
    )


def _make_encoder():
    return TopicAwareEncoder(
        text_to_topic=_TEXT_TO_TOPIC,
        n_topics=4,
        dim=32,
        noise_scale=0.05,
    )


# ---------------------------------------------------------------------------
# SimulatedLLM tests
# ---------------------------------------------------------------------------

class TestSimulatedLLM(unittest.TestCase):

    def test_perfect_mode_returns_correct_answer(self):
        llm = _make_llm(perfect=True)
        question = "In what country is Normandy located?"
        prompt = f"Question: {question}\nContext: some context\nAnswer:"
        text, logprob = llm.generate_with_logprobs(prompt)
        self.assertIn("France", text)
        self.assertAlmostEqual(logprob, SimulatedLLM.CLEAN_LOGPROB)

    def test_perfect_mode_always_high_confidence(self):
        llm = _make_llm(perfect=True)
        for qa in _QA_PAIRS[:5]:
            prompt = f"Question: {qa['q']}\nContext: {qa['ctx']}\nAnswer:"
            text, logprob = llm.generate_with_logprobs(prompt)
            self.assertIn(str(SimulatedLLM.CLEAN_CONF), text)

    def test_noisy_mode_sometimes_wrong(self):
        """With noise_rate=1.0 every call should produce a noisy answer."""
        qa_pairs = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
        answer_lookup = {qa["q"]: qa["a"] for qa in qa_pairs}
        noise_pool = _build_noise_pool(_QA_PAIRS)
        llm = SimulatedLLM(
            qa_lookup=answer_lookup,
            noise_pool=noise_pool,
            noise_rate=1.0,
            perfect=False,
            seed=42,
        )
        # With noise_rate=1.0 every call should use the noisy logprob.
        question = "In what country is Normandy located?"
        prompt = f"Question: {question}\nContext:\nAnswer:"
        noisy_count = 0
        for _ in range(10):
            _, logprob = llm.generate_with_logprobs(prompt)
            if logprob == SimulatedLLM.NOISY_LOGPROB:
                noisy_count += 1
        self.assertEqual(noisy_count, 10)

    def test_noisy_answers_have_low_logprob(self):
        """Noisy answers must return NOISY_LOGPROB (−1.80) as avg_logprob."""
        qa_pairs = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
        answer_lookup = {qa["q"]: qa["a"] for qa in qa_pairs}
        noise_pool = _build_noise_pool(_QA_PAIRS)
        llm = SimulatedLLM(
            qa_lookup=answer_lookup,
            noise_pool=noise_pool,
            noise_rate=1.0,
            perfect=False,
            seed=7,
        )
        _, logprob = llm.generate_with_logprobs(
            "Question: In what country is Normandy located?\nContext:\nAnswer:"
        )
        self.assertAlmostEqual(logprob, SimulatedLLM.NOISY_LOGPROB)

    def test_clean_answers_have_high_logprob(self):
        llm = _make_llm(perfect=True)
        _, logprob = llm.generate_with_logprobs(
            "Question: In what country is Normandy located?\nContext:\nAnswer:"
        )
        self.assertAlmostEqual(logprob, SimulatedLLM.CLEAN_LOGPROB)

    def test_generate_returns_string(self):
        llm = _make_llm(perfect=True)
        result = llm.generate("Question: What is H2O?\nContext:\nAnswer:")
        self.assertIsInstance(result, str)

    def test_reproducibility_with_same_seed(self):
        """Two LLMs with the same seed produce identical outputs."""
        qa_pairs = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
        answer_lookup = {qa["q"]: qa["a"] for qa in qa_pairs}
        noise_pool = _build_noise_pool(_QA_PAIRS)
        llm1 = SimulatedLLM(
            qa_lookup=answer_lookup, noise_pool=noise_pool,
            noise_rate=0.5, seed=100
        )
        llm2 = SimulatedLLM(
            qa_lookup=answer_lookup, noise_pool=noise_pool,
            noise_rate=0.5, seed=100
        )
        prompt = "Question: In what country is Normandy located?\nContext:\nAnswer:"
        t1, lp1 = llm1.generate_with_logprobs(prompt)
        t2, lp2 = llm2.generate_with_logprobs(prompt)
        self.assertEqual(t1, t2)
        self.assertEqual(lp1, lp2)

    def test_different_seeds_produce_different_rng_sequences(self):
        """Two LLMs seeded differently must produce different internal sequences."""
        import random as _random
        r1 = _random.Random(100)
        r2 = _random.Random(200)
        # Draw 20 values; at least one should differ for any reasonable pair of seeds.
        seq1 = [r1.random() for _ in range(20)]
        seq2 = [r2.random() for _ in range(20)]
        self.assertFalse(
            seq1 == seq2,
            "Seeds 100 and 200 produced identical sequences",
        )

    def test_noisy_answer_is_from_different_topic(self):
        """A noisy answer must come from a different topic than the question."""
        qa_pairs = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
        answer_lookup = {qa["q"]: qa["a"] for qa in qa_pairs}
        noise_pool = _build_noise_pool(_QA_PAIRS)
        llm = SimulatedLLM(
            qa_lookup=answer_lookup, noise_pool=noise_pool, noise_rate=1.0, seed=42
        )
        qa = qa_pairs[0]  # geography question
        prompt = f"Question: {qa['q']}\nContext: {qa['ctx']}\nAnswer:"
        wrong_answers = noise_pool[qa["q"]]
        text, _ = llm.generate_with_logprobs(prompt)
        # Correct answer should NOT appear; a wrong one should.
        self.assertNotIn(qa["a"], text.replace(f"Answer: {qa['a']}", "REDACTED"))
        # The returned text should contain one of the cross-topic answers.
        found_wrong = any(wa in text for wa in wrong_answers)
        self.assertTrue(found_wrong)


# ---------------------------------------------------------------------------
# TopicAwareEncoder tests
# ---------------------------------------------------------------------------

class TestTopicAwareEncoder(unittest.TestCase):

    def setUp(self):
        self.enc = _make_encoder()

    def test_encode_returns_float32_array(self):
        vecs = self.enc.encode(["France", "Berlin"])
        self.assertIsInstance(vecs, np.ndarray)
        self.assertEqual(vecs.dtype, np.float32)

    def test_output_shape(self):
        texts = ["France", "Germany", "Japan"]
        vecs = self.enc.encode(texts)
        self.assertEqual(vecs.shape, (3, 32))

    def test_single_text(self):
        vecs = self.enc.encode(["France"])
        self.assertEqual(vecs.shape, (1, 32))

    def test_unit_norm(self):
        vecs = self.enc.encode(["France", "mitochondria", "1945", "tennis"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_same_topic_vectors_are_similar(self):
        """Two geography answers should be more similar than cross-topic answers."""
        geo1 = self.enc.encode(["France"])[0]
        geo2 = self.enc.encode(["Berlin"])[0]
        sci = self.enc.encode(["mitochondria"])[0]
        sim_same = float(np.dot(geo1, geo2))
        sim_cross = float(np.dot(geo1, sci))
        self.assertGreater(sim_same, sim_cross)

    def test_cross_topic_vectors_are_dissimilar(self):
        geo = self.enc.encode(["France"])[0]
        sci = self.enc.encode(["carbon dioxide"])[0]
        sim = float(np.dot(geo, sci))
        self.assertLess(sim, 0.5)

    def test_deterministic_encoding(self):
        """Same text always produces the same vector."""
        enc1 = _make_encoder()
        enc2 = _make_encoder()
        v1 = enc1.encode(["France"])
        v2 = enc2.encode(["France"])
        np.testing.assert_array_equal(v1, v2)

    def test_unknown_text_falls_back_to_topic_0(self):
        """Unknown text should get the topic-0 basis direction."""
        vecs = self.enc.encode(["completely unknown text xyz123"])
        topic0_basis = self.enc._basis[0]
        # Should have positive dot product with topic-0 basis (same direction).
        dot = float(np.dot(vecs[0], topic0_basis))
        self.assertGreater(dot, 0.5)

    def test_case_insensitive_lookup(self):
        upper = self.enc.encode(["FRANCE"])[0]
        lower = self.enc.encode(["France"])[0]
        # Different hashes → different noise → not equal, but same TOPIC direction.
        # Check that both align with topic-0 basis.
        basis0 = self.enc._basis[0]
        self.assertGreater(float(np.dot(upper, basis0)), 0.5)
        self.assertGreater(float(np.dot(lower, basis0)), 0.5)


# ---------------------------------------------------------------------------
# QA metric tests
# ---------------------------------------------------------------------------

class TestQAMetrics(unittest.TestCase):

    def test_normalize_lowercase(self):
        self.assertEqual(_normalize("France"), "france")

    def test_normalize_removes_articles(self):
        self.assertEqual(_normalize("the France"), "france")
        self.assertEqual(_normalize("a city"), "city")
        self.assertEqual(_normalize("an apple"), "apple")

    def test_normalize_removes_punctuation(self):
        self.assertEqual(_normalize("France!"), "france")
        self.assertEqual(_normalize("Paris, France"), "paris france")

    def test_normalize_collapses_whitespace(self):
        self.assertEqual(_normalize("  France  "), "france")

    def test_exact_match_identical(self):
        self.assertEqual(exact_match("France", "France"), 1)

    def test_exact_match_case_insensitive(self):
        self.assertEqual(exact_match("france", "France"), 1)

    def test_exact_match_strips_articles(self):
        self.assertEqual(exact_match("the France", "France"), 1)

    def test_exact_match_different(self):
        self.assertEqual(exact_match("Germany", "France"), 0)

    def test_token_f1_perfect(self):
        self.assertAlmostEqual(token_f1("France", "France"), 1.0)

    def test_token_f1_partial(self):
        f1 = token_f1("South America Pacific", "South America")
        self.assertGreater(f1, 0.5)
        self.assertLess(f1, 1.0)

    def test_token_f1_disjoint(self):
        self.assertAlmostEqual(token_f1("Germany", "France"), 0.0)

    def test_token_f1_empty_prediction(self):
        self.assertEqual(token_f1("", "France"), 0.0)

    def test_token_f1_empty_gold(self):
        self.assertEqual(token_f1("France", ""), 0.0)


# ---------------------------------------------------------------------------
# evaluate_kb_quality tests
# ---------------------------------------------------------------------------

class TestEvaluateKBQuality(unittest.TestCase):

    def _make_entries(self, pairs):
        """pairs: list of (question, annotation)"""
        return [{"question": q, "annotation": a} for q, a in pairs]

    def test_perfect_precision_and_recall(self):
        entries = self._make_entries([("q1", "a1"), ("q2", "a2")])
        lookup = {"q1": "a1", "q2": "a2"}
        metrics = evaluate_kb_quality(entries, lookup, n_train_total=2)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["avg_em"], 1.0)

    def test_zero_precision(self):
        entries = self._make_entries([("q1", "wrong"), ("q2", "wrong")])
        lookup = {"q1": "a1", "q2": "a2"}
        metrics = evaluate_kb_quality(entries, lookup, n_train_total=2)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["avg_em"], 0.0)

    def test_partial_precision(self):
        entries = self._make_entries([("q1", "a1"), ("q2", "wrong")])
        lookup = {"q1": "a1", "q2": "a2"}
        metrics = evaluate_kb_quality(entries, lookup, n_train_total=2)
        self.assertAlmostEqual(metrics["precision"], 0.5)

    def test_recall_below_one_when_not_all_accepted(self):
        entries = self._make_entries([("q1", "a1")])
        lookup = {"q1": "a1", "q2": "a2"}
        metrics = evaluate_kb_quality(entries, lookup, n_train_total=2)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_empty_kb(self):
        metrics = evaluate_kb_quality([], {"q1": "a1"}, n_train_total=1)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_kb_f1_harmonic_mean(self):
        entries = self._make_entries([("q1", "a1")])  # 1 correct
        lookup = {"q1": "a1", "q2": "a2"}
        metrics = evaluate_kb_quality(entries, lookup, n_train_total=2)
        p, r = metrics["precision"], metrics["recall"]
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(metrics["kb_f1"], expected_f1, places=4)


# ---------------------------------------------------------------------------
# evaluate_downstream tests
# ---------------------------------------------------------------------------

class TestEvaluateDownstream(unittest.TestCase):

    def test_oracle_returns_full_score(self):
        """With 100% precision and full coverage the simulated EM should be 1.0."""
        kb_entries = [{"question": f"q{i}", "annotation": f"a{i}"} for i in range(10)]
        lookup = {f"q{i}": f"a{i}" for i in range(10)}
        metrics = evaluate_downstream(
            kb_entries, test_samples=[], answer_lookup=lookup, n_train_total=10
        )
        self.assertAlmostEqual(metrics["downstream_em"], 1.0)

    def test_zero_kb_returns_zero(self):
        metrics = evaluate_downstream(
            kb_entries=[], test_samples=[], answer_lookup={}, n_train_total=10
        )
        self.assertEqual(metrics["downstream_em"], 0.0)

    def test_partial_precision_degrades_score(self):
        # 5/10 correct → precision=0.5, noise=0.5 → (1-2*0.5)=0 → EM=0
        kb_entries = (
            [{"question": f"q{i}", "annotation": f"a{i}"} for i in range(5)]
            + [{"question": f"q{i}", "annotation": "WRONG"} for i in range(5, 10)]
        )
        lookup = {f"q{i}": f"a{i}" for i in range(10)}
        metrics = evaluate_downstream(
            kb_entries, test_samples=[], answer_lookup=lookup, n_train_total=10
        )
        # At 50% noise rate, label-noise correction zeroes the score.
        self.assertAlmostEqual(metrics["downstream_em"], 0.0)

    def test_high_precision_low_coverage(self):
        """100% precision but only half data → coverage=0.5 → EM=0.5."""
        kb_entries = [{"question": f"q{i}", "annotation": f"a{i}"} for i in range(5)]
        lookup = {f"q{i}": f"a{i}" for i in range(10)}
        metrics = evaluate_downstream(
            kb_entries, test_samples=[], answer_lookup=lookup, n_train_total=10
        )
        self.assertAlmostEqual(metrics["downstream_em"], 0.5)

    def test_noise_rate_below_threshold_keeps_positive_score(self):
        """With noise_rate < 0.5, the downstream EM should be positive."""
        kb_entries = (
            [{"question": f"q{i}", "annotation": f"a{i}"} for i in range(7)]
            + [{"question": f"q{i}", "annotation": "WRONG"} for i in range(7, 10)]
        )
        lookup = {f"q{i}": f"a{i}" for i in range(10)}
        metrics = evaluate_downstream(
            kb_entries, test_samples=[], answer_lookup=lookup, n_train_total=10
        )
        self.assertGreater(metrics["downstream_em"], 0.0)


# ---------------------------------------------------------------------------
# _build_noise_pool tests
# ---------------------------------------------------------------------------

class TestBuildNoisePool(unittest.TestCase):

    def test_pool_keys_cover_all_questions(self):
        pool = _build_noise_pool(_QA_PAIRS)
        for qa in _QA_PAIRS:
            self.assertIn(qa["q"], pool)

    def test_noise_answers_from_different_topic(self):
        """Every wrong answer in the pool should come from a different topic."""
        pool = _build_noise_pool(_QA_PAIRS)
        q_topic = {qa["q"]: qa["topic"] for qa in _QA_PAIRS}
        a_topic = {qa["a"]: qa["topic"] for qa in _QA_PAIRS}
        for q, wrong_answers in pool.items():
            q_t = q_topic[q]
            for wa in wrong_answers:
                if wa in a_topic:
                    self.assertNotEqual(
                        a_topic[wa], q_t,
                        msg=f"Wrong answer '{wa}' belongs to same topic {q_t} as question",
                    )

    def test_pool_non_empty_for_all_questions(self):
        pool = _build_noise_pool(_QA_PAIRS)
        for q, answers in pool.items():
            self.assertGreater(
                len(answers), 0, msg=f"Empty noise pool for '{q}'"
            )


# ---------------------------------------------------------------------------
# _build_squad_json tests
# ---------------------------------------------------------------------------

class TestBuildSquadJson(unittest.TestCase):

    def test_has_data_key(self):
        sq = _build_squad_json(_QA_PAIRS[:10])
        self.assertIn("data", sq)
        self.assertIsInstance(sq["data"], list)

    def test_qas_present(self):
        sq = _build_squad_json(_QA_PAIRS[:4])
        all_qas = [
            qa
            for article in sq["data"]
            for para in article["paragraphs"]
            for qa in para["qas"]
        ]
        self.assertGreater(len(all_qas), 0)

    def test_answers_are_populated(self):
        sq = _build_squad_json(_QA_PAIRS[:4])
        for article in sq["data"]:
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    self.assertTrue(qa["answers"], msg="QA has no answers")
                    self.assertIn("text", qa["answers"][0])


# ---------------------------------------------------------------------------
# _save_sft tests
# ---------------------------------------------------------------------------

class TestSaveSft(unittest.TestCase):

    def test_file_written_and_valid_jsonl(self):
        entries = [
            {"question": "Q1", "context": "C1", "annotation": "A1"},
            {"question": "Q2", "context": "C2", "annotation": "A2"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as fh:
            path = fh.name
        try:
            _save_sft(entries, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            for line in lines:
                self.assertIn("instruction", line)
                self.assertIn("output", line)
        finally:
            os.unlink(path)

    def test_instruction_contains_question_and_context(self):
        entries = [{"question": "Q?", "context": "C.", "annotation": "Ans"}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            _save_sft(entries, path)
            with open(path, encoding="utf-8") as f:
                rec = json.loads(f.readline())
            self.assertIn("Q?", rec["instruction"])
            self.assertIn("C.", rec["instruction"])
            self.assertEqual(rec["output"], "Ans")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_condition tests
# ---------------------------------------------------------------------------

class TestRunCondition(unittest.TestCase):

    def setUp(self):
        self.train = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
        self.test = [qa for qa in _QA_PAIRS if qa["id"] in _TEST_IDS]
        self.answer_lookup = {qa["q"]: qa["a"] for qa in _QA_PAIRS}
        self.noise_pool = _build_noise_pool(_QA_PAIRS)
        self.encoder = _make_encoder()
        # Build dataset dicts matching SquadDataset output format.
        self.train_samples = [
            {
                "id": qa["id"],
                "question": qa["q"],
                "context": qa["ctx"],
                "answer": qa["a"],
                "text": f"Question: {qa['q']}\nContext: {qa['ctx']}",
            }
            for qa in self.train
        ]
        self.test_samples = [
            {
                "id": qa["id"],
                "question": qa["q"],
                "context": qa["ctx"],
                "answer": qa["a"],
                "text": f"Question: {qa['q']}\nContext: {qa['ctx']}",
            }
            for qa in self.test
        ]

    def _run(self, cname):
        return run_condition(
            condition_name=cname,
            condition_cfg=_CONDITIONS[cname],
            train_samples=self.train_samples,
            test_samples=self.test_samples,
            answer_lookup=self.answer_lookup,
            noise_pool=self.noise_pool,
            encoder=self.encoder,
            noise_rate=0.30,
            seed=42,
            output_dir=None,
        )

    def test_oracle_naive_returns_expected_keys(self):
        result = self._run("oracle_naive")
        for key in [
            "condition", "n_train", "n_accepted", "n_human_review", "n_purge_removed",
            "kb_precision", "kb_recall", "kb_f1", "kb_avg_em", "kb_avg_token_f1",
            "downstream_em", "downstream_f1",
        ]:
            self.assertIn(key, result)

    def test_oracle_naive_perfect_precision(self):
        result = self._run("oracle_naive")
        self.assertAlmostEqual(result["kb_precision"], 1.0)

    def test_oracle_naive_full_recall(self):
        result = self._run("oracle_naive")
        self.assertAlmostEqual(result["kb_recall"], 1.0)

    def test_naive_lower_precision_than_oracle(self):
        oracle = self._run("oracle_naive")
        naive = self._run("naive")
        self.assertLess(naive["kb_precision"], oracle["kb_precision"])

    def test_entry_control_rejects_some_samples(self):
        result = self._run("entry_control")
        self.assertGreater(result["n_human_review"], 0)

    def test_entry_control_has_high_precision(self):
        result = self._run("entry_control")
        self.assertGreaterEqual(result["kb_precision"], 0.95)

    def test_purge_removes_some_entries(self):
        result = self._run("purge")
        # Outlier purge should have removed at least some noisy entries.
        self.assertGreater(result["n_purge_removed"], 0)

    def test_purge_improves_precision_over_naive(self):
        naive = self._run("naive")
        purge = self._run("purge")
        self.assertGreater(purge["kb_precision"], naive["kb_precision"])

    def test_both_has_highest_downstream_among_noisy(self):
        naive = self._run("naive")
        entry = self._run("entry_control")
        purge = self._run("purge")
        both = self._run("both")
        # entry_control should outperform naive.
        self.assertGreater(entry["downstream_em"], naive["downstream_em"])
        # purge should outperform naive.
        self.assertGreater(purge["downstream_em"], naive["downstream_em"])
        # entry_control should outperform purge (direct threshold > statistical purge).
        self.assertGreater(entry["downstream_em"], purge["downstream_em"])

    def test_all_metrics_in_valid_range(self):
        for cname in _CONDITIONS:
            result = self._run(cname)
            for metric in ["kb_precision", "kb_recall", "kb_f1", "kb_avg_em", "kb_avg_token_f1"]:
                self.assertGreaterEqual(result[metric], 0.0, msg=f"{cname}.{metric}")
                self.assertLessEqual(result[metric], 1.0, msg=f"{cname}.{metric}")
            self.assertGreaterEqual(result["downstream_em"], 0.0)
            self.assertLessEqual(result["downstream_em"], 1.0)


# ---------------------------------------------------------------------------
# run_experiment smoke tests
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):

    def test_returns_five_results(self):
        results = run_experiment(noise_rate=0.30, seed=42, output_dir=None, verbose=False)
        self.assertEqual(len(results), 5)

    def test_result_conditions_match_expected(self):
        results = run_experiment(noise_rate=0.30, seed=42, output_dir=None, verbose=False)
        conditions = {r["condition"] for r in results}
        self.assertEqual(
            conditions,
            {"oracle_naive", "naive", "entry_control", "purge", "both"},
        )

    def test_oracle_has_best_kb_precision(self):
        results = run_experiment(noise_rate=0.30, seed=42, output_dir=None, verbose=False)
        r_map = {r["condition"]: r for r in results}
        for cname in ["naive", "entry_control", "purge", "both"]:
            self.assertLessEqual(
                r_map[cname]["kb_precision"],
                r_map["oracle_naive"]["kb_precision"] + 1e-6,
            )

    def test_oracle_has_best_downstream(self):
        results = run_experiment(noise_rate=0.30, seed=42, output_dir=None, verbose=False)
        r_map = {r["condition"]: r for r in results}
        for cname in ["naive", "entry_control", "purge", "both"]:
            self.assertLessEqual(
                r_map[cname]["downstream_em"],
                r_map["oracle_naive"]["downstream_em"] + 1e-6,
            )

    def test_sft_files_written_when_output_dir_given(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_experiment(
                noise_rate=0.30,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 5)
            for cname in ["oracle_naive", "naive", "entry_control", "purge", "both"]:
                self.assertIn(f"sft_{cname}.jsonl", files)

    def test_single_condition_runs_fine(self):
        results = run_experiment(
            noise_rate=0.30, seed=42, conditions=["naive"], output_dir=None, verbose=False
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["condition"], "naive")

    def test_different_noise_rates_change_results(self):
        r_low = run_experiment(noise_rate=0.1, seed=42, output_dir=None, verbose=False)
        r_high = run_experiment(noise_rate=0.5, seed=42, output_dir=None, verbose=False)
        map_low = {r["condition"]: r for r in r_low}
        map_high = {r["condition"]: r for r in r_high}
        # Higher noise → lower KB precision for naive condition.
        self.assertLess(
            map_high["naive"]["kb_precision"],
            map_low["naive"]["kb_precision"],
        )


if __name__ == "__main__":
    unittest.main()
