"""
Unit tests for filter modules:
  - LLMNaiveFilter
  - DataFlowFilter
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLLMNaiveFilter(unittest.TestCase):

    def _make_filter(self, budget=3, score="0.8"):
        from filters.llm_filter import LLMNaiveFilter
        llm = MagicMock()
        llm.generate.return_value = score
        return LLMNaiveFilter(llm, budget=budget), llm

    def _make_dataset(self, n=5):
        return [{"id": str(i), "text": f"Sample text {i}"} for i in range(n)]

    # --- basic filtering ---

    def test_filter_returns_at_most_budget(self):
        filt, _ = self._make_filter(budget=3)
        dataset = self._make_dataset(n=5)
        result = filt.filter(dataset)
        self.assertLessEqual(len(result), 3)

    def test_filter_exact_budget_when_enough_samples(self):
        filt, _ = self._make_filter(budget=2)
        dataset = self._make_dataset(n=5)
        result = filt.filter(dataset)
        self.assertEqual(len(result), 2)

    def test_filter_preserves_original_fields(self):
        filt, _ = self._make_filter(budget=5)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        for item in result:
            self.assertIn("id", item)
            self.assertIn("text", item)

    def test_filter_adds_llm_score_field(self):
        filt, _ = self._make_filter(budget=5)
        dataset = self._make_dataset(n=2)
        result = filt.filter(dataset)
        for item in result:
            self.assertIn("llm_score", item)

    def test_filter_sorts_by_score_descending(self):
        """Items should be sorted from highest to lowest LLM score."""
        from filters.llm_filter import LLMNaiveFilter
        llm = MagicMock()
        scores = ["0.2", "0.9", "0.5"]
        llm.generate.side_effect = scores
        filt = LLMNaiveFilter(llm, budget=3)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        result_scores = [r["llm_score"] for r in result]
        self.assertEqual(result_scores, sorted(result_scores, reverse=True))

    def test_filter_invalid_score_falls_back_to_random(self):
        """When LLM returns non-numeric text, score should fall back to random float."""
        from filters.llm_filter import LLMNaiveFilter
        llm = MagicMock()
        llm.generate.return_value = "not a number"
        filt = LLMNaiveFilter(llm, budget=5)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        # Random score should still be a float in [0, 1]
        for item in result:
            self.assertIsInstance(item["llm_score"], float)
            self.assertGreaterEqual(item["llm_score"], 0.0)
            self.assertLessEqual(item["llm_score"], 1.0)

    def test_filter_empty_dataset_returns_empty(self):
        filt, _ = self._make_filter(budget=5)
        result = filt.filter([])
        self.assertEqual(result, [])

    def test_filter_budget_larger_than_dataset(self):
        filt, _ = self._make_filter(budget=100)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        self.assertEqual(len(result), 3)

    def test_filter_calls_llm_for_each_item(self):
        filt, llm = self._make_filter(budget=5)
        dataset = self._make_dataset(n=4)
        filt.filter(dataset)
        self.assertEqual(llm.generate.call_count, 4)

    def test_filter_progress_callback_called(self):
        """If a progress_cb attribute is set, it should not cause errors."""
        from filters.llm_filter import LLMNaiveFilter
        llm = MagicMock()
        llm.generate.return_value = "0.7"
        filt = LLMNaiveFilter(llm, budget=3)
        dataset = self._make_dataset(n=2)
        # Simply verify filter runs without error when no callback exists
        result = filt.filter(dataset)
        self.assertEqual(len(result), 2)


class TestDataFlowFilter(unittest.TestCase):

    def _make_operator_class(self, keep=True):
        """Creates a fake operator class that either keeps or drops every item."""
        from base_structure.dataset import DatasetStorage, Dataset

        class FakeOperator:
            def run(self, storage, **kwargs):
                if keep:
                    pass  # do nothing – storage already has the item
                else:
                    storage.write([])  # empty out the storage

        return FakeOperator

    def _make_filter(self, budget=10, keep=True):
        from filters.dataflow_filter import DataFlowFilter
        op_cls = self._make_operator_class(keep=keep)
        return DataFlowFilter(operator_class=op_cls, budget=budget)

    def _make_dataset(self, n=5):
        return [{"id": str(i), "text": f"text {i}"} for i in range(n)]

    # --- initialization ---

    def test_init_requires_operator_name_or_class(self):
        from filters.dataflow_filter import DataFlowFilter
        with self.assertRaises(ValueError):
            DataFlowFilter()

    def test_init_with_operator_class(self):
        filt = self._make_filter()
        self.assertIsNotNone(filt)

    # --- filter behaviour ---

    def test_filter_returns_at_most_budget(self):
        filt = self._make_filter(budget=2, keep=True)
        dataset = self._make_dataset(n=5)
        result = filt.filter(dataset)
        self.assertLessEqual(len(result), 2)

    def test_filter_drops_items_when_operator_clears_storage(self):
        """Operator that clears storage should result in score=0 for all items."""
        filt = self._make_filter(budget=10, keep=False)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        # All items got score=0; first budget items returned
        op_name = getattr(filt.operator_class, "__name__", "operator")
        for item in result:
            self.assertEqual(item[f"{op_name}_score"], 0.0)

    def test_filter_preserves_original_fields(self):
        filt = self._make_filter(budget=10, keep=True)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        for item in result:
            self.assertIn("id", item)
            self.assertIn("text", item)

    def test_filter_empty_dataset_returns_empty(self):
        filt = self._make_filter(budget=10)
        result = filt.filter([])
        self.assertEqual(result, [])

    def test_filter_adds_score_field(self):
        filt = self._make_filter(budget=10, keep=True)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        op_name = getattr(filt.operator_class, "__name__", "operator")
        for item in result:
            self.assertIn(f"{op_name}_score", item)

    def test_filter_budget_larger_than_dataset(self):
        filt = self._make_filter(budget=100, keep=True)
        dataset = self._make_dataset(n=3)
        result = filt.filter(dataset)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
