import random
from typing import List, Dict, Any
from transformers import pipeline


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


class QwenAnnotator:
    """使用 Qwen 进行标注"""
    def __init__(self, candidate_llms, confidence_threshold: float = 0.7):
        self.pipe_dict = {}
        for llm in candidate_llms:
            self.pipe_dict[llm] = pipeline("text-generation", model=llm, device_map="auto")
        self.confidence_threshold = confidence_threshold
        self.human_review_queue = HumanReviewQueue()

    def annotate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"Please assign a label to the following text and provide a confidence score "
            f"between 0.0 and 1.0. Use the format: Label: <label>, Confidence: <score>\n"
            f"Text: {sample['text']}"
        )
        response = self.pipe_dict[sample['route']](prompt, max_new_tokens=50, do_sample=False)
        output = response[0]["generated_text"].split("\n")[-1]

        label, conf = "unknown", random.random()
        if "Confidence" in output:
            try:
                parts = output.split("Confidence")
                label = parts[0].split(":")[-1].strip().replace(",", "")
                conf = float(parts[1].replace(":", "").strip())
            except:
                pass

        result = {**sample, "label": label, "confidence": conf}
        if conf < self.confidence_threshold:
            result["needs_human"] = True
            self.human_review_queue.add(result)
        else:
            result["needs_human"] = False
        return result

    def annotate_batch(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.annotate(d) for d in dataset]