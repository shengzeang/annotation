from abc import ABC, abstractmethod
import unicodedata, re
from typing import Dict, Any
from base_structure.base_task import Task


class QATask(Task):
    """Question answering with confidence score output"""
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
        try:
            """extract a confidence score. handles different formats"""
            m = re.search(r'confidence\s*[:\-]?\s*([0-9]*\.?[0-9]+)\s*%?', output, re.I)
            if m:
                conf_raw = float(m.group(1))
                # if confidence appears as percentage (>1), normalize
                if conf_raw > 1.0:
                    conf = min(1.0, conf_raw / 100.0)
                else:
                    conf = conf_raw
        except Exception:
            conf = None

        try:
            # text labeled after 'Answer:' and before 'Confidence'
            parts = re.split(r'confidence\s*[:\-]?', output, flags=re.I)
            first = parts[0]
            m_ans = re.search(r'answer\s*[:\-]?\s*(.*)', first, re.I | re.S)
            if m_ans:
                annotation = m_ans.group(1).strip()
            else:
                # fallback: take the first non-empty line (strip any leading 'Answer:')
                lines = first.strip().splitlines()
                if lines:
                    line0 = lines[0]
                    annotation = re.sub(r'^\s*Answer\s*[:\-]?\s*', '', line0, flags=re.I).strip()
                    if len(lines) > 1:
                        rest = '\n'.join([l.strip() for l in lines[1:]]).strip()
                        if rest:
                            annotation = annotation + '\n' + rest
        except Exception:
            pass

        if conf is None:
            return {"annotation": annotation}
        return {"annotation": annotation, "confidence": conf}