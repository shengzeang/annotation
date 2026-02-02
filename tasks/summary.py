from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from base_structure.base_task import Task


class TextSummarization(Task):
    """Text summary within predefined length."""
    def __init__(self, max_len=150):
        self.summary_type = "extractive"  # or "abstractive"
        self.max_len = max_len
    
    def get_prompt(self, sample: Dict[str, Any], rag_examples=None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar summarization examples from the knowledge base to help you summarize:\n"
            for ex in rag_examples:
                rag_str += f"Text: {ex.get('text','')}\nSummary: {ex.get('summary','')}\n"
        prompt = (
            f"Summarize the following text into a concise summary.\n"
            f"The summary should not exceed {self.max_len} words.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for your answer, representing how confident you are in your answer.\n"
            f"Output format: Summary: <summary text> Confidence: <score>\n"
            f"Text: {sample.get('text', '')}\n"
            f"{rag_str}"
            f"Summary:"
        )
        return prompt

    def parse_output(self, output: str) -> Dict[str, Any]:
        summary, conf = "unknown", None
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                summary = parts[0].split("Summary:")[-1].strip()
                conf = float(parts[1].split("\n")[0].replace(":", "").strip())
            except:
                pass
        if conf == None:
            return {"annotation": summary}
        else:
            return {"annotation": summary, "confidence": conf}