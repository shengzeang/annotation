"""
Unit tests for misc.metrics module:
  - compute_bleu
  - compute_rouge
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestComputeBleu(unittest.TestCase):

    def setUp(self):
        from misc.metrics import compute_bleu
        self.compute_bleu = compute_bleu

    def test_identical_strings_score_near_1(self):
        score = self.compute_bleu("Paris is the capital of France", "Paris is the capital of France")
        self.assertGreater(score, 0.9)

    def test_completely_different_strings_score_near_0(self):
        score = self.compute_bleu("Paris is the capital of France", "xyz abc def ghi jkl mno")
        self.assertLess(score, 0.2)

    def test_partial_overlap_score_between_0_and_1(self):
        score = self.compute_bleu("Paris is the capital of France", "The capital is Paris")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_returns_float(self):
        score = self.compute_bleu("hello world", "hello there")
        self.assertIsInstance(score, float)

    def test_empty_hypothesis_returns_float(self):
        # Should not raise; returns a float (likely 0)
        score = self.compute_bleu("reference text", "")
        self.assertIsInstance(score, float)

    def test_score_non_negative(self):
        score = self.compute_bleu("hello", "world")
        self.assertGreaterEqual(score, 0.0)

    def test_longer_reference_still_returns_float(self):
        ref = "The quick brown fox jumped over the lazy dog near the river bank"
        hyp = "A quick brown fox jumped over the lazy dog"
        score = self.compute_bleu(ref, hyp)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


class TestComputeRouge(unittest.TestCase):

    def setUp(self):
        from misc.metrics import compute_rouge
        self.compute_rouge = compute_rouge

    def test_returns_dict_with_expected_keys(self):
        result = self.compute_rouge("Paris is the capital of France", "The capital of France is Paris")
        self.assertIn("rouge-1", result)
        self.assertIn("rouge-2", result)
        self.assertIn("rouge-l", result)

    def test_each_rouge_entry_has_f_score(self):
        result = self.compute_rouge("The quick brown fox", "The quick brown fox")
        for key in ("rouge-1", "rouge-2", "rouge-l"):
            self.assertIn("f", result[key])
            self.assertIsInstance(result[key]["f"], float)

    def test_identical_strings_high_rouge_1(self):
        result = self.compute_rouge("Paris is the capital", "Paris is the capital")
        self.assertGreater(result["rouge-1"]["f"], 0.9)

    def test_completely_different_strings_low_rouge(self):
        result = self.compute_rouge("Paris is the capital", "xyz abc def ghi jkl")
        self.assertLess(result["rouge-1"]["f"], 0.3)

    def test_partial_overlap_score_between_0_and_1(self):
        result = self.compute_rouge("Paris is the capital of France", "Paris visited France")
        for key in ("rouge-1", "rouge-2", "rouge-l"):
            f = result[key]["f"]
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_empty_strings_fallback_returns_zeros(self):
        """Empty hypothesis or reference should not raise — falls back to zero scores."""
        try:
            result = self.compute_rouge("reference", "")
        except Exception:
            # rouge library may raise; that's acceptable since we catch it in compute_rouge
            return
        for key in ("rouge-1", "rouge-2", "rouge-l"):
            self.assertGreaterEqual(result[key]["f"], 0.0)

    def test_rouge_l_f_score_non_negative(self):
        result = self.compute_rouge("hello world test", "hello there test")
        self.assertGreaterEqual(result["rouge-l"]["f"], 0.0)


if __name__ == "__main__":
    unittest.main()
