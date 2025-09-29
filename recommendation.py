import random
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from active_learning import DataPool, BertEmbeddings, BertKM, SurprisalEmbeddings, ALPS
from llm_provider import LocalLLM, APILLM


# ==============================
# Recommendation 模块
# ==============================
class ActiveLearningFilter:
    """
    Active Learning整体流程API, 简化主流程调用。
    用法：
        api = ActiveLearningAPI(method="bertkm", budget=100, batch_size=20)
        picked_data = api.select(raw_dataset)
    """
    def __init__(self, method="bertkm", budget=100, batch_size=20, model_name="bert-base-uncased"):
        self.method = method.lower()
        self.budget = budget
        self.batch_size = batch_size
        self.model_name = model_name

        if self.method == "bertkm":
            self.emb = BertEmbeddings(model_name=self.model_name)
            self.selector = BertKM(self.emb, budget=self.budget, batch_size=self.batch_size)
        elif self.method == "alps":
            self.emb = SurprisalEmbeddings(model_name=self.model_name, batch_size=self.batch_size)
            self.selector = ALPS(self.emb, budget=self.budget, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unknown active learning method: {self.method}")

    def select(self, raw_dataset: list) -> list:
        """
        输入: 原始样本list[dict], 每个dict至少有id/question/context。
        输出: 采样后的样本list[dict]。
        """
        texts = [d["text"] if "text" in d else f"Q: {d['question']}\nContext: {d['context']}" for d in raw_dataset]
        ids = [str(d.get("id", i)) for i, d in enumerate(raw_dataset)]
        pool = DataPool(texts, ids)
        picked_ids = set(self.selector.run(pool))
        # 支持id为str/int/索引
        picked_data = [d for i, d in enumerate(raw_dataset) if str(d.get("id", i)) in picked_ids]
        return picked_data


class Refiner:
    """使用 LLM 进行精排和路由"""
    def __init__(self, candidate_llms, self_llm: str = "Qwen/Qwen2.5-7B-Instruct", budget: int = 200, llm_mode: str = "local", api_config: dict = None):
        self.budget = budget
        self.candidate_llms = candidate_llms
        self.llm_mode = llm_mode
        if llm_mode == "local":
            self.llm = LocalLLM(self_llm)
        elif llm_mode == "api":
            # api_config: {llm_name: {"api_url":..., "api_key":..., ...}}
            conf = api_config.get(self.llm, {}) if api_config else {}
            self.llm = APILLM(conf.get("api_url", ""), conf.get("api_key"), conf.get("extra_headers"))
        else:
            raise ValueError(f"Unknown llm_mode: {llm_mode}")

    def _generate(self, prompt, max_new_tokens=10):
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)

    def refine_and_route(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = self.llm_rerank(dataset)
        routed = self.llm_route(reranked)
        return routed

    def llm_rerank(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = []
        for d in tqdm(dataset):
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
        for d in tqdm(dataset):
            prompt = (
                f"You have the following candidate LLMs: {self.candidate_llms}\n"
                f"Based on the content of the input text, choose the most suitable LLM for annotation.\n"
                f"Jointly consider performance, cost, and the question difficulty.\n"
                f"Text: {d['text']}\n"
                f"Output format: <LLM name>.\n"
                f"Output: "
            )
            text_out = self._generate(prompt, max_new_tokens=50)
            routed.append({
                **d,
                "text": d['text'],
                "route": text_out
            })
        return routed
