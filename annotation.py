import random
from typing import List, Dict, Any
from tqdm import tqdm

from llm_provider import LocalLLM, APILLM


# ==============================
# Annotation 模块
# ==============================
class HumanReviewQueue:
    """人工复审池"""
    def __init__(self):
        self.queue = []

    def add(self, sample: Dict[str, Any]):
        self.queue.append(sample)

    def export(self, filepath: str = "human_review.json"):
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, ensure_ascii=False, indent=2)
        print(f"Human review queue exported to {filepath}")


class Annotator:
    """使用 Qwen 进行标注, LLM调用抽象化"""
    def __init__(self, candidate_llms, confidence_threshold: float = 0.7, llm_mode: str = "local", api_config: dict = None):
        self.candidate_llms = candidate_llms
        self.llm_mode = llm_mode
        self.llm_dict = {}
        if llm_mode == "local":
            for llm in candidate_llms:
                self.llm_dict[llm] = LocalLLM(llm)
        elif llm_mode == "api":
            # api_config: {llm_name: {"api_url":..., "api_key":..., ...}}
            for llm in candidate_llms:
                conf = api_config.get(llm, {}) if api_config else {}
                self.llm_dict[llm] = APILLM(conf.get("api_url", ""), conf.get("api_key"), conf.get("extra_headers"))
        else:
            raise ValueError(f"Unknown llm_mode: {llm_mode}")
        self.confidence_threshold = confidence_threshold
        self.human_review_queue = HumanReviewQueue()

    def annotate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 兼容QA样本
        text = sample.get("text") or f"Q: {sample['question']}\nContext: {sample['context']}"
        # QA风格prompt，输入Q，期望LLM输出A
        prompt = (
            f"Given the following question, please answer it as accurately as possible.\n"
            f"Also output a confidence score (between 0.0 and 1.0) for you answer, representing how confident you are in your answer.\n"
            f"Output format: Answer: <your answer> Confidence: <score>\n"
            f"Question: {sample.get('question', text)}\n"
            f"Context: {sample.get('context', '')}\n"
            f"Answer:"
        )
        llm = sample.get('route')
        if llm not in self.candidate_llms:
            best_llm = self.candidate_llms[0]
            for candidate in self.candidate_llms:
                if candidate in str(llm):
                    best_llm = candidate
            llm = best_llm
        output = self.llm_dict[llm].generate(prompt, max_new_tokens=50)

        label, conf = "unknown", random.random()
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                label = parts[0].split(":")[-1].strip().replace(",", "")
                conf = float(parts[1].replace(":", "").strip())
            except:
                pass

        result = {**sample, "route": llm, "label": label, "confidence": conf}
        if conf < self.confidence_threshold:
            result["needs_human"] = True
            self.human_review_queue.add(result)
        else:
            result["needs_human"] = False
        return result

    def annotate_batch(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.annotate(d) for d in tqdm(dataset)]
