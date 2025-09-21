import random
import numpy as np
from typing import List, Dict, Any
from transformers import pipeline


# ==============================
# Recommendation 模块
# ==============================

class ActiveLearningRanker:
    """使用分类模型不确定性进行粗排"""
    def __init__(self, budget: int = 1000, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.budget = budget
        self.clf = pipeline("text-classification", model=model_name, return_all_scores=True)

    def rank(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored_data = []
        for d in dataset:
            probs = self.clf(d["text"])[0]
            prob_values = np.array([p["score"] for p in probs])
            entropy = -np.sum(prob_values * np.log(prob_values + 1e-12))  # 信息熵
            scored_data.append({**d, "score": entropy})
        scored_data.sort(key=lambda x: x["score"], reverse=True)
        return scored_data[:self.budget]


class QwenRefiner:
    """使用 Qwen 模型进行精排和路由"""
    def __init__(self, candidate_llms, budget: int = 200, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.budget = budget
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")
        self.candidate_llms = candidate_llms

    def llm_rerank(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = []
        for d in dataset:
            prompt = (
                f"Please rate the usefulness of the following text for improving "
                f"the downstream task on a scale of 0.0 to 1.0. Output only the number.\n"
                f"Text: {d['text']}"
            )
            response = self.pipe(prompt, max_new_tokens=10, do_sample=False)
            text_out = response[0]["generated_text"].split("\n")[-1].strip()
            try:
                score = float(text_out)
            except:
                score = random.random()
            reranked.append({**d, "llm_score": score})

        reranked.sort(key=lambda x: x["llm_score"], reverse=True)
        return reranked[:self.budget]
    
    def llm_route(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        routed = []
        for d in dataset:
            prompt = (
                f"You have the following candidate LLMs: {self.candidate_llms} "
                f"Based on the content of the input text, choose the most suitable LLM for annotation. "
                f"Text: {d['text']}"
                f"Output format: only output the LLM name."
            )

            response = self.pipe(prompt, max_new_tokens=10, do_sample=False)
            text_out = response[0]["generated_text"].split("\n")[-1].strip()

            routed.append({
                "text": d['text'],
                "route": text_out
            })
        return routed