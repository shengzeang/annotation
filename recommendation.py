import random
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from active_learning import DataPool, BertEmbeddings, BertKM, SurprisalEmbeddings, ALPS


from task import Task, QATask
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
    def __init__(self, candidate_llms, self_llm, budget: int = 200, task: Task = None, router=None):
        self.budget = budget
        self.candidate_llms = candidate_llms
        self.llm = self_llm
        self.task = task or QATask()
        self.router = router  # optional router implementing score(sample, candidate_llms)

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
                f"You are an expert evaluator assisting in selecting the most useful samples for a downstream task.\n"
                f"Given a candidate sample, your job is to analyze how useful the sample is for improving performance on the target task.\n"
                f"You must assign a usefulness score between 0.0 (completely useless) and 1.0 (highly useful) based on the sample's relevance, \
                    informativeness, and contribution to solving or learning the target task.\n"
                f"When judging usefulness, consider the following aspects: \
                    1. Relevance: How closely does the sample relate to the target task's domain or objectives? \
                    2. Informative Value: Does the sample contain unique or high-quality information that can improve understanding or model performance? \
                    3. Clarity and Correctness: Is the sample accurate, well-structured, and unambiguous? \
                    4. Diversity Contribution (optional): Does the sample add valuable variety to the dataset without redundancy?\n"
                f"Output only the usefulness score as a float number between 0.0 and 1.0.\n"
                f"Candiate sample: {d['text']}"
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
            # If a router is provided (e.g., MLPRouter), use it to score candidates
            if self.router is not None:
                try:
                    scores = self.router.score(d.get('text', ''), self.candidate_llms)
                    # scores -> list of {model, score}
                    best = max(scores, key=lambda x: x.get('score', 0.0))
                    chosen = best.get('model')
                except Exception:
                    chosen = self.candidate_llms[0]
                    scores = [{'model': m, 'score': 0.0} for m in self.candidate_llms]
                routed.append({**d, 'route': chosen, 'route_scores': scores})
            else:
                prompt = (
                    f"You are an expert system responsible for recommending the most appropriate candidate LLM to annotate a given data sample.\n"
                    f"You have the following candidate LLMs: {self.candidate_llms}\n"
                    f"Your goal is to analyze the sample's characteristics, difficulty, and domain, then choose the LLM that is best suited for producing accurate and high-quality annotations on this sample.\n"
                    f"When choosing the best LLM, consider: 1. Domain Expertise: Which model is most familiar with the domain (e.g., legal, medical, conversational)? \n"
                    f"2. Complexity Handling: Which model performs best on tasks of similar difficulty? \n"
                    f"3. Instruction Following & Alignment: Which model is most reliable for annotation-style outputs? \n"
                    f"4. Cost-Performance Tradeoff: Prefer smaller models for simple samples, larger models for complex reasoning.\n"
                    f"Data sample: {d['text']}\n"
                    f"Output format: <LLM name>.\n"
                    f"Output: "
                )
                text_out = self._generate(prompt, max_new_tokens=50)
                routed.append({**d, 'route': text_out})
        return routed
