from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from ..base_structure.base_task import Task


class QATask(Task):
    """QA任务实现"""
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar QA pairs from the knowledge base to help you answer:\n"
            for ex in rag_examples:
                rag_str += f"Q: {ex.get('question','')}\nA: {ex.get('annotation','')}\n"
        prompt = (
            f"Given the following question, please answer it as accurately as possible.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Output format: Answer: <your answer> Confidence: <score>\n"
            f"Question: {sample.get('question', sample.get('text',''))}\n"
            f"Context: {sample.get('context', '')}\n"
            f"{rag_str}"
            f"Answer:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        annotation, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                annotation = parts[0].split(":")[-1].strip().replace(",", "")
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
            except:
                pass
        if conf == None:
            return {"annotation": annotation}
        else:
            return {"annotation": annotation, "confidence": conf}