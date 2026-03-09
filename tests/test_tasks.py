"""
Unit tests for all annotation task classes:
  - QATask
  - ClassificationTask
  - TextSummarization
  - Translation
  - NERTask
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestQATask(unittest.TestCase):

    def setUp(self):
        from tasks.qa import QATask
        self.task = QATask()

    # --- get_prompt ---

    def test_get_prompt_contains_question(self):
        sample = {"question": "What is the capital of France?", "context": "France is in Europe."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("What is the capital of France?", prompt)

    def test_get_prompt_contains_context(self):
        sample = {"question": "Q?", "context": "Some context here."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("Some context here.", prompt)

    def test_get_prompt_rag_examples_injected(self):
        sample = {"question": "Q?", "context": ""}
        rag = [{"question": "Old Q?", "annotation": "Old Answer"}]
        prompt = self.task.get_prompt(sample, rag_examples=rag)
        self.assertIn("Old Q?", prompt)
        self.assertIn("Old Answer", prompt)

    def test_get_prompt_no_rag_no_rag_str(self):
        sample = {"question": "Q?", "context": ""}
        prompt = self.task.get_prompt(sample)
        self.assertNotIn("knowledge base", prompt.lower())

    def test_get_prompt_uses_text_fallback(self):
        sample = {"text": "Fallback text question"}
        prompt = self.task.get_prompt(sample)
        self.assertIn("Fallback text question", prompt)

    # --- parse_output ---

    def test_parse_output_standard_format(self):
        out = "Answer: Paris Confidence: 0.92"
        result = self.task.parse_output(out)
        self.assertEqual(result["annotation"], "Paris")
        self.assertAlmostEqual(result["confidence"], 0.92, places=3)

    def test_parse_output_percentage_confidence(self):
        out = "Answer: Berlin Confidence: 90%"
        result = self.task.parse_output(out)
        self.assertAlmostEqual(result["confidence"], 0.90, places=2)

    def test_parse_output_no_confidence(self):
        out = "The answer is London"
        result = self.task.parse_output(out)
        self.assertIn("annotation", result)
        self.assertNotIn("confidence", result)

    def test_parse_output_multiline_answer(self):
        out = "Answer: Paris is the capital\nof France Confidence: 0.88"
        result = self.task.parse_output(out)
        self.assertIn("Paris", result["annotation"])
        self.assertAlmostEqual(result["confidence"], 0.88, places=3)

    def test_parse_output_empty_string(self):
        result = self.task.parse_output("")
        self.assertIn("annotation", result)

    def test_parse_output_confidence_close_to_zero(self):
        out = "Answer: unknown Confidence: 0.0"
        result = self.task.parse_output(out)
        self.assertAlmostEqual(result["confidence"], 0.0, places=3)


class TestClassificationTask(unittest.TestCase):

    def setUp(self):
        from tasks.classification import ClassificationTask
        self.task = ClassificationTask()

    def test_get_prompt_contains_text(self):
        sample = {"Text": "I love this product!", "Categories": "positive, negative, neutral"}
        prompt = self.task.get_prompt(sample)
        self.assertIn("I love this product!", prompt)
        self.assertIn("positive", prompt)

    def test_get_prompt_rag_examples_injected(self):
        sample = {"Text": "Great item", "Categories": "positive, negative"}
        rag = [{"text": "Good product", "annotation": "positive"}]
        prompt = self.task.get_prompt(sample, rag_examples=rag)
        self.assertIn("Good product", prompt)
        self.assertIn("positive", prompt)

    def test_parse_output_standard_format(self):
        out = "Category: positive Confidence: 0.95"
        result = self.task.parse_output(out)
        self.assertEqual(result["annotation"], "positive")
        self.assertAlmostEqual(result["confidence"], 0.95, places=3)

    def test_parse_output_no_confidence(self):
        out = "The text is positive."
        result = self.task.parse_output(out)
        self.assertIn("annotation", result)

    def test_parse_output_strips_trailing_comma(self):
        out = "Category: negative, Confidence: 0.8"
        result = self.task.parse_output(out)
        self.assertNotIn(",", result["annotation"])


class TestTextSummarization(unittest.TestCase):

    def setUp(self):
        from tasks.summary import TextSummarization
        self.task = TextSummarization(max_len=50)

    def test_get_prompt_contains_text(self):
        sample = {"text": "A long article about AI research..."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("A long article about AI research", prompt)

    def test_get_prompt_contains_max_len(self):
        sample = {"text": "Some text."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("50", prompt)

    def test_get_prompt_rag_examples_injected(self):
        sample = {"text": "Article text."}
        rag = [{"text": "Old article", "summary": "Short summary"}]
        prompt = self.task.get_prompt(sample, rag_examples=rag)
        self.assertIn("Old article", prompt)
        self.assertIn("Short summary", prompt)

    def test_parse_output_standard_format(self):
        out = "Summary: AI is transforming industries. Confidence: 0.88"
        result = self.task.parse_output(out)
        self.assertIn("AI is transforming industries", result["annotation"])
        self.assertAlmostEqual(result["confidence"], 0.88, places=3)

    def test_parse_output_no_confidence(self):
        out = "This is a summary without confidence."
        result = self.task.parse_output(out)
        self.assertIn("annotation", result)
        self.assertNotIn("confidence", result)


class TestTranslation(unittest.TestCase):

    def setUp(self):
        from tasks.translation import Translation
        self.task = Translation(target_language="Chinese")

    def test_get_prompt_contains_text(self):
        sample = {"text": "Hello world"}
        prompt = self.task.get_prompt(sample)
        self.assertIn("Hello world", prompt)

    def test_get_prompt_target_language(self):
        sample = {"text": "Good morning"}
        prompt = self.task.get_prompt(sample)
        self.assertIn("Chinese", prompt)

    def test_get_prompt_dictionary_hints(self):
        sample = {"text": "Hello world"}
        dictionary = {"Hello": "你好", "world": "世界"}
        prompt = self.task.get_prompt(sample, dictionary=dictionary)
        self.assertIn("你好", prompt)
        self.assertIn("世界", prompt)

    def test_get_prompt_no_dictionary_hints(self):
        sample = {"text": "Hello world"}
        prompt = self.task.get_prompt(sample)
        self.assertIn("None", prompt)

    def test_parse_output_standard_format(self):
        out = "Translation: 你好世界 Confidence: 0.9"
        result = self.task.parse_output(out)
        self.assertIn("你好世界", result["annotation"])
        self.assertAlmostEqual(result["confidence"], 0.9, places=3)

    def test_parse_output_no_confidence(self):
        out = "Some untranslated output"
        result = self.task.parse_output(out)
        self.assertIn("annotation", result)
        self.assertNotIn("confidence", result)

    def test_get_dictionary_hints_missing_words(self):
        hints = self.task.get_dictionary_hints("foo bar baz", {"hello": "你好"})
        self.assertEqual(hints, "")

    def test_get_dictionary_hints_partial_match(self):
        hints = self.task.get_dictionary_hints("Hello world", {"Hello": "你好"})
        self.assertIn("你好", hints)


class TestNERTask(unittest.TestCase):

    def setUp(self):
        from tasks.ner import NERTask
        self.task = NERTask(entity_types=["PERSON", "ORG", "LOC"], language="en")

    def test_get_prompt_contains_text(self):
        sample = {"text": "Barack Obama visited New York."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("Barack Obama visited New York", prompt)

    def test_get_prompt_contains_entity_types(self):
        sample = {"text": "John works at Google."}
        prompt = self.task.get_prompt(sample)
        self.assertIn("PERSON", prompt)
        self.assertIn("ORG", prompt)

    def test_get_prompt_rag_examples_injected(self):
        sample = {"text": "Alice met Bob in Paris."}
        rag = [{"text": "John visited London", "annotation": "John|PERSON, London|LOC"}]
        prompt = self.task.get_prompt(sample, rag_examples=rag)
        self.assertIn("John visited London", prompt)

    def test_parse_output_standard_format(self):
        out = "Entities: Barack Obama|PERSON, New York|LOC Confidence: 0.91"
        result = self.task.parse_output(out)
        self.assertIn("Barack Obama|PERSON", result["annotation"])
        self.assertAlmostEqual(result["confidence"], 0.91, places=3)

    def test_parse_output_no_confidence(self):
        out = "Some NER output without confidence"
        result = self.task.parse_output(out)
        self.assertIn("annotation", result)
        self.assertNotIn("confidence", result)

    def test_pre_process_normalizes_whitespace(self):
        sample = {"text": "Hello   world\t!"}
        processed = self.task.pre_process(sample)
        self.assertNotIn("   ", processed["processed_text"])

    def test_default_entity_types(self):
        from tasks.ner import NERTask
        task = NERTask()
        self.assertIn("PERSON", task.entity_types)
        self.assertIn("ORG", task.entity_types)


if __name__ == "__main__":
    unittest.main()
