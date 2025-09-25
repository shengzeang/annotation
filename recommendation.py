import random
import numpy as np
from typing import List, Dict, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch


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
        self.candidate_llms = candidate_llms
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def _generate(self, prompt, max_new_tokens=10):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            return output_text[len(prompt):].strip()
        return output_text.strip()

    def llm_rerank(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = []
        for d in dataset:
            prompt = (
                f"Please rate the usefulness of the following text for improving "
                f"the downstream task on a scale of 0.0 to 1.0. Output only the number.\n"
                f"Text: {d['text']}"
            )
            text_out = self._generate(prompt, max_new_tokens=10)
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
            text_out = self._generate(prompt, max_new_tokens=10)
            routed.append({
                "text": d['text'],
                "route": text_out
            })
        return routed