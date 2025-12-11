from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from ..base_structure.base_task import Task


class ClassificationTask(Task):
    """分类任务实现"""
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar classification examples from the knowledge base to help you classify:\n"
            for ex in rag_examples:
                rag_str += f"Text: {ex.get('text','')}\nCategory: {ex.get('annotation','')}\n"
        prompt = (
            f"Classify the following text into one of the predefined categories.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Your answer should be one of the categories listed below.\n"
            f"Output format: Category: <category> Confidence: <score>\n"
            f"Categories: {sample.get('Categories', '')}\n"
            f"Text: {sample.get('Text','')}\n"
            f"{rag_str}"
            f"Answer:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        category, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                category = parts[0].split(":")[-1].strip().replace(",","") #extract category
                conf = float(parts[1].split("\n")[0].replace(":", "").strip()) #extract confidence
            except:
                pass
        if conf == None:
            return {"annotation": category}
        else:
            return {"annotation": category, "confidence": conf}