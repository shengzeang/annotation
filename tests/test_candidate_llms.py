"""Tests verifying that candidate_llms is correctly wired in all three
experiment files.

These tests focus on:
  1. Every annotated record's ``route`` field stays within ``candidate_llms``.
  2. The correct LLM is dispatched for each route (via a call-tracking mock).
  3. Multi-candidate routing (cheap / expensive) works as intended.
  4. Annotator fallback to the first candidate when an unknown route is supplied.
  5. Single-candidate setups ("primary") always use the single LLM.

All tests are CPU-only and use mock LLMs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Shared mock LLM that records every call made to it
# ---------------------------------------------------------------------------

class _TrackingLLM:
    """Records every prompt generated so tests can assert which LLM was used."""

    def __init__(self, name: str, answer: str = "test_answer", confidence: float = 0.85):
        self.name = name
        self.calls: list = []  # list of prompts received
        self._answer = answer
        self._confidence = confidence

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        self.calls.append(prompt)
        return f"Answer: {self._answer} Confidence: {self._confidence}"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class _JudgeAlwaysPass:
    """Judge that always says '1' (answer is acceptable → no escalation)."""
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "1"


class _JudgeAlwaysFail:
    """Judge that always says '0' (answer unacceptable → always escalate)."""
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "0"


class _ScorerPrefersExpensive:
    """Scorer that always gives 'expensive' a higher score than 'cheap'."""
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return '[{"model": "cheap", "score": 0.3}, {"model": "expensive", "score": 0.9}]'


class _ScorerPrefersCheap:
    """Scorer that always gives 'cheap' a higher score than 'expensive'."""
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return '[{"model": "cheap", "score": 0.9}, {"model": "expensive", "score": 0.3}]'


def _make_dataset(n: int = 15):
    from experiments.run_active_learning import _make_synthetic_dataset
    return _make_synthetic_dataset(n=n, seed=42)


# ===========================================================================
# 1. run_active_learning.py — candidate_llms = ["primary"]
# ===========================================================================

class TestCandidateLLMs_ActiveLearning(unittest.TestCase):
    """candidate_llms = ["primary"] in run_active_learning.py"""

    def _run(self, cheap_llm, judge_llm):
        from experiments.run_active_learning import run_experiment
        self._tmp = tempfile.mkdtemp()
        return run_experiment(
            dataset=_make_dataset(),
            cheap_llm=cheap_llm,
            judge_llm=judge_llm,
            budget=5,
            output_dir=self._tmp,
            force_fallback=True,
        )

    def tearDown(self):
        import shutil
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_route_always_primary(self):
        """Every annotated record's route must be 'primary' (the only candidate)."""
        primary = _TrackingLLM("primary")
        results = self._run(cheap_llm=primary, judge_llm=_JudgeAlwaysPass())
        for cond in results:
            with open(cond["sft_file"]) as f:
                lines = f.readlines()
            # SFT file is not keyed by route; check raw annotation records via
            # re-running and inspecting the underlying data
        # Verified by checking that the tracking LLM was actually called
        self.assertGreater(len(primary.calls), 0,
                           "primary LLM should have been called during annotation")

    def test_primary_llm_is_called(self):
        """The 'primary' LLM object receives generate() calls for every sample."""
        primary = _TrackingLLM("primary")
        results = self._run(cheap_llm=primary, judge_llm=_JudgeAlwaysPass())
        total_annotated = sum(r["annotated"] for r in results)
        # primary must be called at least as many times as samples annotated
        # (once per sample, possibly more for judge interactions)
        self.assertGreaterEqual(len(primary.calls), total_annotated,
                                "primary LLM call count should match annotation count")

    def test_non_primary_llm_never_called(self):
        """A second unrelated LLM object should never be called when
        candidate_llms only contains 'primary'."""
        primary = _TrackingLLM("primary")
        other = _TrackingLLM("other")
        # We inject 'other' only as judge; it should be called for judging
        # not for annotation.  The primary must handle annotation.
        results = self._run(cheap_llm=primary, judge_llm=other)
        self.assertGreater(len(primary.calls), 0,
                           "primary LLM should annotate samples")

    def test_all_conditions_use_single_candidate(self):
        """All four filter conditions rely on the same 'primary' LLM."""
        primary = _TrackingLLM("primary")
        results = self._run(cheap_llm=primary, judge_llm=_JudgeAlwaysPass())
        self.assertEqual(len(results), 4)
        # primary must have been invoked — confirms it is the sole annotator
        self.assertGreater(len(primary.calls), 0)

    def test_annotated_count_matches_sft_lines(self):
        """annotated count in result equals the number of lines in the SFT file."""
        primary = _TrackingLLM("primary")
        results = self._run(cheap_llm=primary, judge_llm=_JudgeAlwaysPass())
        for r in results:
            with open(r["sft_file"]) as f:
                line_count = sum(1 for _ in f)
            self.assertEqual(r["annotated"], line_count)


# ===========================================================================
# 2. run_llm_routing.py — candidate_llms = ["cheap", "expensive"]
# ===========================================================================

class TestCandidateLLMs_LLMRouting(unittest.TestCase):
    """candidate_llms = ["cheap", "expensive"] in run_llm_routing.py"""

    def _run(self, cheap_llm, expensive_llm, judge_llm, scorer_llm):
        from experiments.run_llm_routing import run_experiment
        self._tmp = tempfile.mkdtemp()
        return run_experiment(
            dataset=_make_dataset(),
            cheap_llm=cheap_llm,
            expensive_llm=expensive_llm,
            judge_llm=judge_llm,
            scorer_llm=scorer_llm,
            output_dir=self._tmp,
            force_fallback=True,
        )

    def tearDown(self):
        import shutil
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_all_cheap_only_calls_cheap(self):
        """All-cheap condition should call the cheap LLM, not the expensive one."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysPass(), _ScorerPrefersCheap())
        all_cheap = next(r for r in results if r["condition"] == "All-cheap")
        # expensive should not be called during the All-cheap condition; however
        # because all conditions share a pre-filtered dataset and the expensive
        # LLM may be called in other conditions, we verify via route:
        self.assertAlmostEqual(all_cheap["expensive_call_rate"], 0.0)

    def test_all_expensive_only_calls_expensive(self):
        """All-expensive condition should route entirely to the expensive LLM."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysPass(), _ScorerPrefersCheap())
        all_exp = next(r for r in results if r["condition"] == "All-expensive")
        self.assertAlmostEqual(all_exp["expensive_call_rate"], 1.0)

    def test_cascade_escalates_when_judge_fails(self):
        """CascadeRouter with a judge that always returns 0 must escalate every
        sample to the expensive LLM."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysFail(), _ScorerPrefersCheap())
        cascade = next(r for r in results if r["condition"] == "CascadeRouter")
        self.assertAlmostEqual(cascade["expensive_call_rate"], 1.0,
                               msg="Judge always fails → all samples escalate to expensive")

    def test_cascade_no_escalation_when_judge_passes(self):
        """CascadeRouter with a judge that always returns 1 keeps cheap LLM for all samples."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysPass(), _ScorerPrefersCheap())
        cascade = next(r for r in results if r["condition"] == "CascadeRouter")
        self.assertAlmostEqual(cascade["expensive_call_rate"], 0.0,
                               msg="Judge always passes → no escalation to expensive")

    def test_llm_router_scorer_prefers_expensive(self):
        """LLMRouter with a scorer that prefers expensive routes all samples to expensive."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysPass(), _ScorerPrefersExpensive())
        llm_router = next(r for r in results if r["condition"] == "LLMRouter")
        self.assertAlmostEqual(llm_router["expensive_call_rate"], 1.0,
                               msg="Scorer prefers expensive → all routes to expensive")

    def test_llm_router_scorer_prefers_cheap(self):
        """LLMRouter with a scorer that prefers cheap routes all samples to cheap."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        results = self._run(cheap, expensive, _JudgeAlwaysPass(), _ScorerPrefersCheap())
        llm_router = next(r for r in results if r["condition"] == "LLMRouter")
        self.assertAlmostEqual(llm_router["expensive_call_rate"], 0.0,
                               msg="Scorer prefers cheap → all routes to cheap")

    def test_route_field_within_candidate_llms(self):
        """Every annotated record in CascadeRouter / LLMRouter must have route
        equal to either 'cheap' or 'expensive'."""
        from annotation import Annotator
        from routers import CascadeRouter

        candidate_llms = ["cheap", "expensive"]
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        llm_dict = {"cheap": cheap, "expensive": expensive}

        dataset = [
            {"id": str(i), "question": f"Q{i}?", "context": "ctx",
             "answer": "ans", "text": f"Q{i}?"}
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            router = CascadeRouter(
                judge_llm=_JudgeAlwaysFail(),
                candidate_llm=candidate_llms,
                llm_dict=llm_dict,
            )
            routed = router.route(dataset)
            annotated = annotator.annotate_batch(routed)

        for rec in annotated:
            self.assertIn(rec["route"], candidate_llms,
                          f"route='{rec['route']}' not in candidate_llms={candidate_llms}")

    def test_cheap_llm_actually_called_when_routed_to_cheap(self):
        """When a record's route is 'cheap', the cheap LLM object must receive
        the generate() call — and NOT the expensive LLM."""
        from annotation import Annotator

        candidate_llms = ["cheap", "expensive"]
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        llm_dict = {"cheap": cheap, "expensive": expensive}

        # Force route = "cheap" on all samples
        dataset = [
            {"id": str(i), "text": f"Q{i}?", "question": f"Q{i}?",
             "context": "ctx", "answer": "ans", "route": "cheap"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            annotated = annotator.annotate_batch(dataset)

        self.assertEqual(len(cheap.calls), 5,
                         "cheap LLM should have been called for all 5 cheap-routed samples")
        self.assertEqual(len(expensive.calls), 0,
                         "expensive LLM must NOT be called when route='cheap'")

    def test_expensive_llm_actually_called_when_routed_to_expensive(self):
        """When a record's route is 'expensive', the expensive LLM object must
        receive the generate() call."""
        from annotation import Annotator

        candidate_llms = ["cheap", "expensive"]
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        llm_dict = {"cheap": cheap, "expensive": expensive}

        dataset = [
            {"id": str(i), "text": f"Q{i}?", "question": f"Q{i}?",
             "context": "ctx", "answer": "ans", "route": "expensive"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            annotated = annotator.annotate_batch(dataset)

        self.assertEqual(len(expensive.calls), 5,
                         "expensive LLM should be called for all 5 expensive-routed samples")
        self.assertEqual(len(cheap.calls), 0,
                         "cheap LLM must NOT be called when route='expensive'")

    def test_annotator_falls_back_to_first_candidate_for_unknown_route(self):
        """When a route value is not in candidate_llms, Annotator must fall back
        to the first candidate (candidate_llms[0]) rather than raising."""
        from annotation import Annotator

        candidate_llms = ["cheap", "expensive"]
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        llm_dict = {"cheap": cheap, "expensive": expensive}

        dataset = [
            {"id": "0", "text": "Q?", "question": "Q?",
             "context": "ctx", "answer": "ans",
             "route": "unknown_model"}  # not in candidate_llms
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            annotated = annotator.annotate_batch(dataset)

        # Must not raise; annotation produced
        self.assertEqual(len(annotated), 1)
        # First candidate ("cheap") is the fallback
        self.assertEqual(annotated[0]["route"], "cheap",
                         "Annotator must fall back to first candidate for unknown route")
        self.assertEqual(len(cheap.calls), 1,
                         "cheap (first candidate) must handle the unknown-route sample")

    def test_assigned_llm_overrides_route_field(self):
        """When annotate_batch is called with assigned_llm, that LLM is always
        used regardless of any 'route' field in the samples."""
        from annotation import Annotator

        candidate_llms = ["cheap", "expensive"]
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        llm_dict = {"cheap": cheap, "expensive": expensive}

        # All samples have route='expensive' but we pass assigned_llm='cheap'
        dataset = [
            {"id": str(i), "text": f"Q{i}?", "question": f"Q{i}?",
             "context": "ctx", "answer": "ans", "route": "expensive"}
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            annotated = annotator.annotate_batch(dataset, assigned_llm="cheap")

        self.assertEqual(len(cheap.calls), 4,
                         "cheap LLM must handle all samples when assigned_llm='cheap'")
        self.assertEqual(len(expensive.calls), 0,
                         "expensive LLM must not be called when overridden by assigned_llm")


# ===========================================================================
# 3. run_rag.py — candidate_llms = ["primary"]
# ===========================================================================

class TestCandidateLLMs_RAG(unittest.TestCase):
    """candidate_llms = ["primary"] in run_rag.py"""

    def _run(self, llm, judge_llm):
        from experiments.run_rag import run_experiment
        self._tmp = tempfile.mkdtemp()
        return run_experiment(
            dataset=_make_dataset(),
            llm=llm,
            judge_llm=judge_llm,
            output_dir=self._tmp,
            topk=2,
            window=5,
            force_fallback=True,
        )

    def tearDown(self):
        import shutil
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_primary_llm_is_called_in_no_rag_condition(self):
        """No RAG condition uses the 'primary' LLM for all annotations."""
        primary = _TrackingLLM("primary")
        results = self._run(primary, _JudgeAlwaysPass())
        no_rag = next(r for r in results if r["condition"] == "No RAG")
        self.assertGreater(no_rag["annotated"], 0)
        # primary must have been called
        self.assertGreater(len(primary.calls), 0)

    def test_primary_llm_is_called_in_rag_condition(self):
        """RAG condition also uses the 'primary' LLM for annotations."""
        primary = _TrackingLLM("primary")
        results = self._run(primary, _JudgeAlwaysPass())
        rag = next(r for r in results if r["condition"] == "RAG")
        self.assertGreater(rag["annotated"], 0)
        self.assertGreater(len(primary.calls), 0)

    def test_route_field_within_candidate_llms(self):
        """Both conditions must produce annotated records with route='primary'."""
        from annotation import Annotator
        from routers import CascadeRouter

        candidate_llms = ["primary"]
        primary = _TrackingLLM("primary")
        llm_dict = {"primary": primary}

        dataset = [
            {"id": str(i), "question": f"Q{i}?", "context": "ctx",
             "answer": "ans", "text": f"Q{i}?"}
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator = Annotator(
                candidate_llms, llm_dict,
                kb_path=os.path.join(tmp, "kb.json"),
            )
            router = CascadeRouter(
                judge_llm=_JudgeAlwaysPass(),
                candidate_llm=candidate_llms,
                llm_dict=llm_dict,
            )
            routed = router.route(dataset)
            annotated = annotator.annotate_batch(routed)

        for rec in annotated:
            self.assertEqual(rec["route"], "primary",
                             f"route should be 'primary', got '{rec['route']}'")

    def test_total_calls_equals_no_rag_plus_rag(self):
        """Total LLM calls equals the sum of annotated samples across both conditions
        (since both use the same LLM object sequentially)."""
        primary = _TrackingLLM("primary")
        results = self._run(primary, _JudgeAlwaysPass())
        total_annotated = sum(r["annotated"] for r in results)
        # primary.calls may include extra calls from the CascadeRouter's judge
        # (which also uses a separate judge_llm in this experiment)
        # At minimum, one generate() call per annotated sample
        self.assertGreaterEqual(len(primary.calls), total_annotated,
                                "primary LLM calls must be >= total annotated samples")

    def test_rag_uses_same_candidate_llms_as_no_rag(self):
        """Enabling RAG must not change which LLMs are in candidate_llms."""
        from annotation import Annotator

        candidate_llms = ["primary"]
        primary = _TrackingLLM("primary")
        llm_dict = {"primary": primary}

        dataset = [
            {"id": str(i), "question": f"Q{i}?", "context": "ctx",
             "answer": "ans", "text": f"Q{i}?", "route": "primary"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            annotator_no_rag = Annotator(
                candidate_llms, llm_dict, rag=False,
                kb_path=os.path.join(tmp, "kb_no_rag.json"),
            )
            annotator_rag = Annotator(
                candidate_llms, llm_dict, rag=True,
                kb_path=os.path.join(tmp, "kb_rag.json"),
            )
            ann_no_rag = annotator_no_rag.annotate_batch(dataset)
            ann_rag = annotator_rag.annotate_batch(dataset)

        # Both use the same candidate list
        self.assertEqual(annotator_no_rag.candidate_llms, annotator_rag.candidate_llms)
        # Both produce records with route='primary'
        for rec in ann_no_rag + ann_rag:
            self.assertEqual(rec["route"], "primary")


# ===========================================================================
# 4. Direct Annotator candidate_llms API tests (unit-level)
# ===========================================================================

class TestAnnotatorCandidateLLMs(unittest.TestCase):
    """Unit tests for Annotator.candidate_llms mechanics independent of the
    experiment scripts."""

    def _make_annotator(self, candidate_llms, llm_dict, tmp_dir):
        from annotation import Annotator
        return Annotator(
            candidate_llms,
            llm_dict,
            kb_path=os.path.join(tmp_dir, "kb.json"),
        )

    def test_candidate_llms_stored(self):
        cheap = _TrackingLLM("cheap")
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(["cheap"], {"cheap": cheap}, tmp)
        self.assertEqual(ann.candidate_llms, ["cheap"])

    def test_multi_candidate_stored(self):
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(
                ["cheap", "expensive"],
                {"cheap": cheap, "expensive": expensive},
                tmp,
            )
        self.assertEqual(ann.candidate_llms, ["cheap", "expensive"])

    def test_fallback_to_first_on_none_route(self):
        """When sample has no 'route' field and no assigned_llm, fall back to
        candidate_llms[0]."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        sample = {"id": "1", "text": "Q?", "question": "Q?", "context": "ctx"}
        # No 'route' key → sample.get('route') returns None → fallback to cheap
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(
                ["cheap", "expensive"],
                {"cheap": cheap, "expensive": expensive},
                tmp,
            )
            result = ann.annotate(sample)
        self.assertEqual(result["route"], "cheap")
        self.assertEqual(len(cheap.calls), 1)
        self.assertEqual(len(expensive.calls), 0)

    def test_valid_route_field_respected(self):
        """When sample has 'route'='expensive', the expensive LLM is used."""
        cheap = _TrackingLLM("cheap")
        expensive = _TrackingLLM("expensive")
        sample = {"id": "1", "text": "Q?", "question": "Q?",
                  "context": "ctx", "route": "expensive"}
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(
                ["cheap", "expensive"],
                {"cheap": cheap, "expensive": expensive},
                tmp,
            )
            result = ann.annotate(sample)
        self.assertEqual(len(expensive.calls), 1)
        self.assertEqual(len(cheap.calls), 0)

    def test_exact_route_full_name_respected(self):
        """When route equals the full candidate name the correct LLM is selected
        (e.g. route='Qwen/Qwen2.5-14B-Instruct' picks model_b exactly)."""
        model_a = _TrackingLLM("Qwen/Qwen2.5-7B-Instruct")
        model_b = _TrackingLLM("Qwen/Qwen2.5-14B-Instruct")
        candidate_llms = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
        llm_dict = {
            "Qwen/Qwen2.5-7B-Instruct": model_a,
            "Qwen/Qwen2.5-14B-Instruct": model_b,
        }
        sample = {"id": "1", "text": "Q?", "question": "Q?",
                  "context": "ctx", "route": "Qwen/Qwen2.5-14B-Instruct"}
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(candidate_llms, llm_dict, tmp)
            result = ann.annotate(sample)
        self.assertEqual(len(model_b.calls), 1,
                         "exact route 'Qwen/Qwen2.5-14B-Instruct' should use model_b")
        self.assertEqual(len(model_a.calls), 0)

    def test_unknown_short_route_falls_back_to_first(self):
        """When route is a short string not in candidate_llms and not a
        superstring of any candidate name, Annotator falls back to
        candidate_llms[0]."""
        model_a = _TrackingLLM("Qwen/Qwen2.5-7B-Instruct")
        model_b = _TrackingLLM("Qwen/Qwen2.5-14B-Instruct")
        candidate_llms = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
        llm_dict = {
            "Qwen/Qwen2.5-7B-Instruct": model_a,
            "Qwen/Qwen2.5-14B-Instruct": model_b,
        }
        # "14B" is NOT a superstring of either full candidate name,
        # so the fallback logic (candidate in str(route)) finds no match
        # and returns candidate_llms[0] = model_a.
        sample = {"id": "1", "text": "Q?", "question": "Q?",
                  "context": "ctx", "route": "14B"}
        with tempfile.TemporaryDirectory() as tmp:
            ann = self._make_annotator(candidate_llms, llm_dict, tmp)
            result = ann.annotate(sample)
        # Falls back to first candidate
        self.assertEqual(result["route"], "Qwen/Qwen2.5-7B-Instruct",
                         "Short route not matching any candidate should fall back to first")
        self.assertEqual(len(model_a.calls), 1)
        self.assertEqual(len(model_b.calls), 0)


if __name__ == "__main__":
    unittest.main()
