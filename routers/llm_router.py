from typing import List, Dict, Any
from misc.llm_provider import LLMBase
import math

from .base_router import BaseRouter


class LLMRouter(BaseRouter):
    """
    Scores candidate LLMs for each sample using a scoring LLM (which can be local or API-backed).
    """

    def __init__(self, scorer: LLMBase, temperature: float = 0.0):
        self.scorer = scorer
        self.temperature = temperature

    @property
    def if_train(self):
        self.ready = True
        return False

    def _build_prompt(self, sample_text: str, candidate_llms: List[str]) -> str:
        prompt = (
            "You are an evaluator that rates how well different LLMs will perform on a given text sample.\n"
            "Given a text sample and a list of candidate LLM names, output a JSON list of objects {\n"
            "  \"model\": \"<model name>\",\n"
            "  \"score\": <float score between 0.0 and 1.0>\n"
            "}\n"
            "Only output valid JSON, no extra explanation.\n\n"
            f"Sample: {sample_text}\n\n"
            f"Candidates: {candidate_llms}\n\n"
            "Output:"
        )
        return prompt

    def score(self, sample_text: str, candidate_llms: List[str], max_new_tokens: int = 80) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(sample_text, candidate_llms)
        raw = self.scorer.generate(prompt, max_new_tokens=max_new_tokens)
        # Try to parse JSON out of the raw response robustly
        import json
        out = raw.strip()
        # Heuristic: find the first '[' and last ']' to extract JSON array
        try:
            start = out.index('[')
            end = out.rindex(']')
            json_text = out[start:end+1]
            parsed = json.loads(json_text)
            # Normalize scores
            for item in parsed:
                if 'score' in item:
                    try:
                        item['score'] = float(item['score'])
                    except:
                        item['score'] = 0.0
                else:
                    item['score'] = 0.0
            return parsed
        except Exception:
            # Fallback: simple heuristic scoring based on name overlap
            scores = []
            sample_words = set(sample_text.lower().split())
            for cand in candidate_llms:
                cand_words = set(cand.lower().split('/'))
                overlap = len(sample_words & cand_words)
                score = 1.0 - math.exp(-overlap)
                scores.append({'model': cand, 'score': score})
            # sort by score desc
            scores.sort(key=lambda x: x['score'], reverse=True)
            return scores
