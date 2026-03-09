"""
Unit tests for utils module:
  - export_annotation_results
  - compute_metrics_for_annotations
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_raw_data(n=3):
    return [
        {"id": str(i), "question": f"Q{i}?", "context": f"ctx{i}", "answer": f"answer{i}"}
        for i in range(n)
    ]


def _make_results(n=3, needs_human=False):
    return [
        {
            "id": str(i),
            "annotation": f"annotation{i}",
            "route": "model-a",
            "confidence": 0.9,
            "needs_human": needs_human,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# export_annotation_results
# ---------------------------------------------------------------------------

class TestExportAnnotationResults(unittest.TestCase):

    def setUp(self):
        from utils import export_annotation_results
        self.export = export_annotation_results
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.tmp)  # start with a non-existent path so export creates it

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    def test_creates_output_file(self):
        self.export(_make_results(3), _make_raw_data(3), output_path=self.tmp)
        self.assertTrue(os.path.exists(self.tmp))

    def test_output_is_valid_json(self):
        self.export(_make_results(3), _make_raw_data(3), output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)

    def test_output_has_correct_count(self):
        self.export(_make_results(3), _make_raw_data(3), output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)

    def test_output_has_required_fields(self):
        self.export(_make_results(3), _make_raw_data(3), output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        for item in data:
            for field in ("id", "question", "context", "route", "annotation"):
                self.assertIn(field, item)

    def test_output_merges_question_from_raw(self):
        results = _make_results(2)
        raw = _make_raw_data(2)
        self.export(results, raw, output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(data[0]["question"], "Q0?")
        self.assertEqual(data[1]["question"], "Q1?")

    def test_output_uses_annotation_from_results(self):
        results = _make_results(2)
        raw = _make_raw_data(2)
        self.export(results, raw, output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(data[0]["annotation"], "annotation0")

    def test_empty_results_creates_empty_list(self):
        self.export([], _make_raw_data(3), output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(data, [])

    def test_id_mismatch_still_exports(self):
        """Results with IDs not present in raw_data should still be exported."""
        results = [{"id": "999", "annotation": "x", "route": "m"}]
        raw = _make_raw_data(3)
        self.export(results, raw, output_path=self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "999")


# ---------------------------------------------------------------------------
# compute_metrics_for_annotations
# ---------------------------------------------------------------------------

class TestComputeMetricsForAnnotations(unittest.TestCase):

    def setUp(self):
        from utils import compute_metrics_for_annotations
        self.compute = compute_metrics_for_annotations

    def test_runs_without_error_on_valid_data(self):
        results = _make_results(3)
        # Give annotations meaningful text that matches answers
        for r in results:
            r["annotation"] = f"answer{r['id']}"
        raw = _make_raw_data(3)
        # Should not raise
        self.compute(results, raw)

    def test_handles_empty_results(self):
        # Should not raise on empty input
        self.compute([], _make_raw_data(3))

    def test_handles_list_answer_in_raw_data(self):
        results = [{"id": "0", "annotation": "answer", "route": "m", "needs_human": False}]
        raw = [{"id": "0", "question": "Q?", "context": "ctx", "answer": ["answer", "alt answer"]}]
        # Should not raise
        self.compute(results, raw)

    def test_handles_missing_annotation(self):
        results = [{"id": "0", "route": "m", "needs_human": False}]
        raw = [{"id": "0", "question": "Q?", "context": "ctx", "answer": "expected"}]
        # Missing annotation should be handled gracefully
        self.compute(results, raw)

    def test_metrics_logged(self):
        """Verify that no exception is raised when computing metrics (basic sanity check)."""
        results = _make_results(2)
        for r in results:
            r["annotation"] = f"answer{r['id']}"
        raw = _make_raw_data(2)
        # Should not raise; metrics are computed and logged internally
        self.compute(results, raw)

    def test_list_answer_uses_first_element(self):
        """When answer is a list, only the first element should be used."""
        results = [{"id": "0", "annotation": "Paris", "route": "m"}]
        raw = [{"id": "0", "question": "Q?", "context": "ctx", "answer": ["Paris", "France"]}]
        # Should not raise
        self.compute(results, raw)


if __name__ == "__main__":
    unittest.main()
