from typing import List, Dict, Any
from tqdm import tqdm
import random
from base_structure.base_filter import BaseFilter

class LLMNaiveFilter(BaseFilter):
    """
    LLM-based naive filter implementation.
    """
    def __init__(self, llm, budget: int = 200):
        self.llm = llm
        self.budget = budget

    def _generate(self, prompt, max_new_tokens=10):
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)

    def filter(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the LLM naive filter to the dataset.

        Args:
            dataset (List[Dict[str, Any]]): The input dataset.

        Returns:
            List[Dict[str, Any]]: The filtered dataset.
        """
        filtered = []
        for d in tqdm(dataset):
            prompt = (
                f"You are an expert evaluator assisting in selecting the most useful samples for a downstream task.\n"
                f"Given a candidate sample, your job is to analyze how useful the sample is for improving performance on the target task.\n"
                f"You must assign a usefulness score between 0.0 (completely useless) and 1.0 (highly useful) based on the sample's relevance, \n"
                f"    informativeness, and contribution to solving or learning the target task.\n"
                f"When judging usefulness, consider the following aspects: \n"
                f"    1. Relevance: How closely does the sample relate to the target task's domain or objectives? \n"
                f"    2. Informative Value: Does the sample contain unique or high-quality information that can improve understanding or model performance? \n"
                f"    3. Clarity and Correctness: Is the sample accurate, well-structured, and unambiguous? \n"
                f"    4. Diversity Contribution (optional): Does the sample add valuable variety to the dataset without redundancy?\n"
                f"Output only the usefulness score as a float number between 0.0 and 1.0.\n"
                f"Candiate sample: {d['text']}"
            )
            text_out = self._generate(prompt, max_new_tokens=10)
            try:
                score = float(text_out)
            except:
                score = random.random()
            filtered.append({**d, "llm_score": score})

        filtered.sort(key=lambda x: x["llm_score"], reverse=True)
        return filtered[:self.budget]
