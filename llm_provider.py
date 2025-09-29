import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Optional
import requests


class LLMBase:
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        raise NotImplementedError


class LocalLLM(LLMBase):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            return output_text[len(prompt):].strip()
        return output_text.strip()


class APILLM(LLMBase):
    def __init__(self, api_url: str, api_key: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens
        }
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Assume the API returns {'generated_text': ...} or similar
        if "generated_text" in data:
            return data["generated_text"].strip()
        elif "choices" in data and isinstance(data["choices"], list):
            return data["choices"][0]["text"].strip()
        else:
            return str(data)
