from typing import List, Dict, Any, Optional, Tuple

from base_structure.base_router import BaseRouter


class CascadeRouter(BaseRouter):
    """
    FrugalGPT-style cascade router.
    Calls models until output > confidence threshold.
    """
    def __init__(self, judge_llm, candidate_llm, llm_dict, threshold=0.7):
        self.model_order = candidate_llm
        self.judge = judge_llm
        self.llm_dict = llm_dict
        self.threshold = threshold

    @property
    def if_train(self):
        return False

    def evaluate(self, query, answer):
        """Use judge LLM to score answer quality."""
        prompt = f"""
        Determine if the answer is correct.
        Question: {query}
        Answer: {answer}
        If correct output 1, else output 0.
        """
        score = self.judge.generate(prompt).strip()
        return 1.0 if "1" in score else 0.0

    def score(self, sample: str, candidate_llms: List[str]):
        """Return only one model (1.0) based on cascade selection."""
        chosen = None
        for model_name in self.model_order:
            # call the model and get output
            llm = self.llm_dict[model_name] 
            answer = llm.generate(sample, max_new_tokens=50)
            # evaluate quality
            reward = self.evaluate(sample, answer)
            if reward >= self.threshold:
                chosen = model_name
                break

        if chosen is None:
            chosen = self.model_order[-1]
        return [
            {"model": m, "score": 1.0 if m == chosen else 0.0}
            for m in candidate_llms
        ]
    