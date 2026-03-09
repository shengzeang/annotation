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
        try:
            # prefer automatic device mapping when available
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        except ValueError as e:
            # transformers raises a ValueError if `accelerate` is required but not installed
            msg = str(e)
            if 'requires `accelerate`' in msg or 'device_map' in msg:
                # fallback to loading without device_map (loads to CPU or default device)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                raise
        # Ensure pad token is set to avoid transformers warning during generation
        try:
            if getattr(self.tokenizer, 'pad_token', None) is None:
                # prefer eos token as pad if available
                if getattr(self.tokenizer, 'eos_token', None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            # sync model config pad token id
            if getattr(self.model, 'config', None) is not None and getattr(self.tokenizer, 'pad_token_id', None) is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # explicitly pass pad_token_id to silence tokenizer/model warnings
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True)
            try:
                pad_id = getattr(self.tokenizer, 'pad_token_id', None)
                if pad_id is not None:
                    gen_kwargs['pad_token_id'] = int(pad_id)
            except Exception:
                pass
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            return output_text[len(prompt):].strip()
        return output_text.strip()

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        """Generate text and return ``(text, avg_log_prob)``.

        ``avg_log_prob`` is the mean of the per-token log-probabilities for
        the *generated* (non-prompt) tokens.  It is always ``≤ 0``; values
        closer to ``0`` indicate higher model confidence.

        Returns
        -------
        tuple[str, float]
            ``(generated_text, avg_log_prob)``
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
            try:
                pad_id = getattr(self.tokenizer, 'pad_token_id', None)
                if pad_id is not None:
                    gen_kwargs['pad_token_id'] = int(pad_id)
            except Exception:
                pass
            outputs = self.model.generate(**inputs, **gen_kwargs)

        output_ids = outputs.sequences[0]  # [prompt_length + num_generated]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()
        else:
            output_text = output_text.strip()

        # Compute average log-probability of generated tokens.
        scores = outputs.scores  # tuple of [1, vocab_size] tensors
        avg_log_prob = None
        if scores:
            log_probs = []
            for i, score in enumerate(scores):
                token_id = output_ids[prompt_length + i].item()
                lp = torch.log_softmax(score[0], dim=-1)[token_id].item()
                log_probs.append(lp)
            if log_probs:
                avg_log_prob = sum(log_probs) / len(log_probs)

        return output_text, avg_log_prob


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

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        """Generate text and return ``(text, avg_log_prob)``.

        Requests token log-probabilities from the remote API if the endpoint
        supports it (OpenAI-compatible ``logprobs`` parameter).  When
        log-probabilities cannot be obtained, ``avg_log_prob`` is ``None``
        and the caller should skip the log-probability threshold check.

        Returns
        -------
        tuple[str, float | None]
            ``(generated_text, avg_log_prob)``
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "logprobs": 1,  # request per-token logprobs (OpenAI-compatible)
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "generated_text" in data:
                text = data["generated_text"].strip()
            elif "choices" in data and isinstance(data["choices"], list):
                text = data["choices"][0].get("text", "").strip()
            else:
                text = str(data)

            # Extract avg log-probability when provided by the API.
            avg_log_prob = None
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                logprobs_data = choice.get("logprobs") or {}
                token_logprobs = logprobs_data.get("token_logprobs", [])
                valid = [lp for lp in token_logprobs if lp is not None]
                if valid:
                    avg_log_prob = sum(valid) / len(valid)

            return text, avg_log_prob
        except Exception:
            # Fall back to regular generate without log-probability.
            text = self.generate(prompt, max_new_tokens)
            return text, None
