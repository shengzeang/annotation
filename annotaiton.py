import random
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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
        self.model_dict = {}
        self.tokenizer_dict = {}
        for llm in candidate_llms:
            self.tokenizer_dict[llm] = AutoTokenizer.from_pretrained(llm)
            self.model_dict[llm] = AutoModelForCausalLM.from_pretrained(llm, device_map="auto")
        self.confidence_threshold = confidence_threshold
        self.human_review_queue = HumanReviewQueue()

    def _generate(self, model, tokenizer, prompt, max_new_tokens=50):
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Only return the new generated part after the prompt
        if output_text.startswith(prompt):
            return output_text[len(prompt):].strip()
        return output_text.strip()

    def annotate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"Please assign a label to the following text and provide a confidence score "
            f"between 0.0 and 1.0. Use the format: Label: <label>, Confidence: <score>\n"
            f"Text: {sample['text']}"
        )
        llm = sample['route']
        model = self.model_dict[llm]
        tokenizer = self.tokenizer_dict[llm]
        output = self._generate(model, tokenizer, prompt, max_new_tokens=50)

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