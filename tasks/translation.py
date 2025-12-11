from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from ..base_structure.base_task import Task


class Translation(Task):
    """翻译任务实现: English to Chinese"""
    def __init__(self, target_language="Chinese"):
        self.target_language = target_language

    def get_dictionary_hints(self, en_text, dictionary: Dict[str, str]) -> str:
        hints = []
        for token in en_text.split():
            if token in dictionary:
                hints.append(f"{token}: {dictionary[token]}")
        if hints:
            return "\n".join(hints)
        else:
            return ""
        
    def get_prompt(self, sample: Dict[str, Any], dictionary = None, rag_examples=None) -> str:
        text = sample.get("text", "")
        prompt = (
            f"You are a English to Chinese Translator. Translate the following text into {self.target_language}.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Text: {text}\n"
            f"Dictionary hints: {self.get_dictionary_hints(text, dictionary) if dictionary else 'None'}\n"
            f"Output format: Translation: <translated text> Confidence: <score>\n"
            f"Translation:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        translation, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                translation = parts[0].split("Translation:")[-1].strip().replace(",", "")
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
                return {"annotation": translation, "confidence": conf}
            except:
                pass
        if conf == None:
            return {"annotation": translation}
        else:
            return {"annotation": translation, "confidence": conf}