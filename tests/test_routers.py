"""
Unit tests for router modules:
  - CascadeRouter
  - LLMRouter
  - BaseRouter helpers (choose_best, route)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(generate_return="Some answer"):
    llm = MagicMock()
    llm.generate.return_value = generate_return
    return llm


def _make_dataset(n=3):
    return [{"id": str(i), "text": f"sample text {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------

class TestCascadeRouter(unittest.TestCase):

    def _make_router(self, judge_returns="1", llm_returns="Paris", threshold=0.7):
        from routers.cascade_router import CascadeRouter
        judge = _make_llm(judge_returns)
        llm_a = _make_llm(llm_returns)
        llm_b = _make_llm(llm_returns)
        llm_dict = {"model-a": llm_a, "model-b": llm_b}
        router = CascadeRouter(
            judge_llm=judge,
            candidate_llm=["model-a", "model-b"],
            llm_dict=llm_dict,
            threshold=threshold,
        )
        return router, judge, llm_a, llm_b

    # --- if_train ---

    def test_if_train_is_false(self):
        router, _, _, _ = self._make_router()
        self.assertFalse(router.if_train)

    # --- build_from_annotations (no-op) ---

    def test_build_from_annotations_is_noop(self):
        router, _, _, _ = self._make_router()
        # Should not raise
        router.build_from_annotations([], out_dir="./")

    # --- evaluate ---

    def test_evaluate_returns_1_when_judge_says_1(self):
        router, judge, _, _ = self._make_router(judge_returns="1")
        reward = router.evaluate("What is the capital?", "Paris")
        self.assertEqual(reward, 1.0)

    def test_evaluate_returns_0_when_judge_says_0(self):
        router, judge, _, _ = self._make_router(judge_returns="0")
        reward = router.evaluate("What is the capital?", "Wrong answer")
        self.assertEqual(reward, 0.0)

    # --- score ---

    def test_score_chooses_first_model_that_passes_threshold(self):
        router, _, _, _ = self._make_router(judge_returns="1", threshold=0.7)
        scores = router.score("Some question", ["model-a", "model-b"])
        chosen = [s for s in scores if s["score"] == 1.0]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["model"], "model-a")

    def test_score_falls_back_to_last_model_when_none_pass(self):
        router, _, _, _ = self._make_router(judge_returns="0", threshold=0.7)
        scores = router.score("Some question", ["model-a", "model-b"])
        chosen = [s for s in scores if s["score"] == 1.0]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["model"], "model-b")

    def test_score_returns_entry_for_each_candidate(self):
        router, _, _, _ = self._make_router()
        candidates = ["model-a", "model-b"]
        scores = router.score("test", candidates)
        self.assertEqual(len(scores), len(candidates))
        models_returned = {s["model"] for s in scores}
        self.assertEqual(models_returned, set(candidates))

    def test_score_scores_sum_to_1(self):
        router, _, _, _ = self._make_router()
        scores = router.score("test", ["model-a", "model-b"])
        total = sum(s["score"] for s in scores)
        self.assertAlmostEqual(total, 1.0, places=5)

    # --- choose_best (inherited from BaseRouter) ---

    def test_choose_best_returns_model_name(self):
        router, _, _, _ = self._make_router(judge_returns="1", threshold=0.7)
        best, _ = router.choose_best("test", ["model-a", "model-b"])
        self.assertIn(best, ["model-a", "model-b"])


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------

class TestLLMRouter(unittest.TestCase):

    def _make_router(self, scorer_output=None):
        from routers.llm_router import LLMRouter
        scorer = MagicMock()
        if scorer_output is not None:
            scorer.generate.return_value = scorer_output
        else:
            scorer.generate.return_value = '[{"model": "model-a", "score": 0.9}, {"model": "model-b", "score": 0.4}]'
        return LLMRouter(scorer), scorer

    # --- if_train ---

    def test_if_train_is_false(self):
        router, _ = self._make_router()
        self.assertFalse(router.if_train)

    # --- build_from_annotations (no-op) ---

    def test_build_from_annotations_is_noop(self):
        router, _ = self._make_router()
        router.build_from_annotations([], out_dir="./")

    # --- score - valid JSON response ---

    def test_score_parses_json_response(self):
        router, _ = self._make_router()
        scores = router.score("Some question", ["model-a", "model-b"])
        self.assertEqual(len(scores), 2)

    def test_score_returns_float_scores(self):
        router, _ = self._make_router()
        scores = router.score("test", ["model-a", "model-b"])
        for s in scores:
            self.assertIsInstance(s["score"], float)

    def test_score_highest_first_in_json_mode(self):
        router, _ = self._make_router('[{"model": "model-a", "score": 0.9}, {"model": "model-b", "score": 0.3}]')
        scores = router.score("test", ["model-a", "model-b"])
        # JSON is returned as-is (order may vary), check that model-a has higher score
        by_model = {s["model"]: s["score"] for s in scores}
        self.assertGreater(by_model["model-a"], by_model["model-b"])

    # --- score - fallback when JSON fails ---

    def test_score_fallback_on_invalid_json(self):
        router, _ = self._make_router("This is not JSON at all")
        scores = router.score("test", ["model-a", "model-b"])
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIn("model", s)
            self.assertIn("score", s)

    def test_score_fallback_when_scorer_raises(self):
        """If scorer.generate() raises an exception, score() must fall back to
        heuristic scoring instead of propagating the exception."""
        from routers.llm_router import LLMRouter
        scorer = MagicMock()
        scorer.generate.side_effect = RuntimeError("CUDA out of memory")
        router = LLMRouter(scorer, candidate_llms=["model-a", "model-b"])
        # Must not raise; must return a valid list of scored candidates
        scores = router.score("some sample text", ["model-a", "model-b"])
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIn("model", s)
            self.assertIn("score", s)
            self.assertIsInstance(s["score"], float)

    def test_score_fallback_when_scorer_returns_empty_json_array(self):
        """scorer.generate() returning '[]' (valid but empty JSON) must trigger
        the heuristic fallback — not return an empty list that would crash
        max() in BaseRouter.route()."""
        from routers.llm_router import LLMRouter
        scorer = MagicMock()
        scorer.generate.return_value = "[]"
        router = LLMRouter(scorer, candidate_llms=["model-a", "model-b"])
        scores = router.score("some sample text", ["model-a", "model-b"])
        # Must return a non-empty list so max() in route() doesn't crash
        self.assertGreater(len(scores), 0)
        for s in scores:
            self.assertIn("model", s)
            self.assertIn("score", s)
            self.assertIsInstance(s["score"], float)

        router, _ = self._make_router("invalid json")
        scores = router.score("model-a model-b", ["model-a", "model-b"])
        result_scores = [s["score"] for s in scores]
        self.assertEqual(result_scores, sorted(result_scores, reverse=True))

    # --- _build_prompt ---

    def test_build_prompt_includes_candidates(self):
        from routers.llm_router import LLMRouter
        scorer = MagicMock()
        scorer.generate.return_value = "[]"
        router = LLMRouter(scorer)
        prompt = router._build_prompt("test text", ["model-a", "model-b"])
        self.assertIn("model-a", prompt)
        self.assertIn("model-b", prompt)

    def test_build_prompt_includes_sample(self):
        from routers.llm_router import LLMRouter
        scorer = MagicMock()
        scorer.generate.return_value = "[]"
        router = LLMRouter(scorer)
        prompt = router._build_prompt("This is the test sample", ["model-a"])
        self.assertIn("This is the test sample", prompt)

    # --- choose_best ---

    def test_choose_best_returns_highest_score_model(self):
        router, _ = self._make_router()
        best, _ = router.choose_best("test", ["model-a", "model-b"])
        self.assertEqual(best, "model-a")


# ---------------------------------------------------------------------------
# BaseRouter.route integration (using CascadeRouter as concrete impl)
# ---------------------------------------------------------------------------

class TestBaseRouterRoute(unittest.TestCase):

    def test_route_adds_route_field(self):
        from routers.cascade_router import CascadeRouter
        judge = _make_llm("1")
        llm_a = _make_llm("Paris")
        router = CascadeRouter(
            judge_llm=judge,
            candidate_llm=["model-a"],
            llm_dict={"model-a": llm_a},
        )
        dataset = _make_dataset(n=2)
        result = router.route(dataset)
        for item in result:
            self.assertIn("route", item)
            self.assertEqual(item["route"], "model-a")

    def test_route_adds_route_scores_field(self):
        from routers.cascade_router import CascadeRouter
        judge = _make_llm("1")
        llm_a = _make_llm("Paris")
        router = CascadeRouter(
            judge_llm=judge,
            candidate_llm=["model-a"],
            llm_dict={"model-a": llm_a},
        )
        dataset = _make_dataset(n=2)
        result = router.route(dataset)
        for item in result:
            self.assertIn("route_scores", item)

    def test_route_preserves_original_fields(self):
        from routers.cascade_router import CascadeRouter
        judge = _make_llm("1")
        llm_a = _make_llm("Paris")
        router = CascadeRouter(
            judge_llm=judge,
            candidate_llm=["model-a"],
            llm_dict={"model-a": llm_a},
        )
        dataset = _make_dataset(n=2)
        result = router.route(dataset)
        for orig, routed in zip(dataset, result):
            self.assertEqual(orig["id"], routed["id"])
            self.assertEqual(orig["text"], routed["text"])

    def test_route_handles_empty_scores_gracefully(self):
        """BaseRouter.route() must not crash when score() returns [].
        It should fall back to the first candidate instead of raising
        ValueError from max()."""
        from routers.llm_router import LLMRouter

        # Craft a scorer that always returns "[]" — an edge case that
        # previously caused max() to raise ValueError inside route().
        class _EmptyScorerLLM:
            def generate(self, prompt, **kwargs):
                return "[]"

        router = LLMRouter(
            scorer=_EmptyScorerLLM(),
            candidate_llms=["model-a", "model-b"],
        )
        dataset = _make_dataset(n=3)
        # Must not raise; every item should have a 'route' field
        result = router.route(dataset)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn("route", item)
            # Falls back to first candidate when scores are empty
            self.assertEqual(item["route"], "model-a")

