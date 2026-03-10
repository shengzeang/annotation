import argparse
import importlib.util
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    # Keep third-party packages ahead of local modules with generic names
    # (e.g., `datasets`) to avoid import collisions.
    sys.path.append(_PROJECT_ROOT)


def _load_local_module(module_name: str, rel_path: str):
    module_path = os.path.join(_PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Annotator = _load_local_module("annotation_local", "annotation.py").Annotator
SquadDataset = _load_local_module("qa_datasets_local", os.path.join("datasets", "qa_datasets.py")).SquadDataset
from rag import VectorKnowledgeBase
from tasks.qa import QATask


QWEN_3_06B = "Qwen/Qwen3-0.6B"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"


@dataclass
class ConditionConfig:
    name: str
    confidence_threshold: float
    avg_logprob_threshold: Optional[float]
    outlier_purge_interval: int


class SimulatedLLM:
    """LLM simulator with controllable noisy high-confidence failure mode."""

    def __init__(
        self,
        answer_key: Dict[str, str],
        rng_seed: int = 42,
        error_rate: float = 0.3,
        high_confidence_error_rate: float = 0.6,
    ) -> None:
        self.answer_key = answer_key
        self.rng = random.Random(rng_seed)
        self.error_rate = error_rate
        self.high_confidence_error_rate = high_confidence_error_rate
        self._wrong_pool = [v for v in answer_key.values() if isinstance(v, str) and v.strip()]
        if not self._wrong_pool:
            self._wrong_pool = ["unknown"]

    def _extract_question(self, prompt: str) -> str:
        m = re.search(r"Question:\s*(.*)\nContext:", prompt, flags=re.S)
        if m:
            return m.group(1).strip()
        return ""

    def _make_bad_answer(self, gold: str) -> str:
        for _ in range(10):
            cand = self.rng.choice(self._wrong_pool)
            if cand != gold:
                return cand
        return "incorrect"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        """Return (output, avg_logprob) matching LocalLLM/APILLM interface."""
        # Keep max_new_tokens for compatibility with real LLM interfaces.
        _ = max_new_tokens
        question = self._extract_question(prompt)
        gold = self.answer_key.get(question, "unknown")

        if self.rng.random() < self.error_rate:
            bad = self._make_bad_answer(gold)
            if self.rng.random() < self.high_confidence_error_rate:
                # High confidence but poor generation quality.
                return f"Answer: {bad} Confidence: 0.95", -3.0
            return f"Answer: {bad} Confidence: 0.25", -3.0

        return f"Answer: {gold} Confidence: 0.85", -0.2

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        output, _ = self.generate_with_logprobs(prompt, max_new_tokens=max_new_tokens)
        return output


class OracleLLM(SimulatedLLM):
    """Perfect oracle model used to quantify noise-induced performance loss."""

    def __init__(self, answer_key: Dict[str, str]) -> None:
        super().__init__(answer_key=answer_key, error_rate=0.0, high_confidence_error_rate=0.0)

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        """Return (output, avg_logprob) matching LocalLLM/APILLM interface."""
        # Keep max_new_tokens for compatibility with real LLM interfaces.
        _ = max_new_tokens
        question = self._extract_question(prompt)
        gold = self.answer_key.get(question, "unknown")
        return f"Answer: {gold} Confidence: 0.99", -0.01


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def evaluate_kb_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    admitted = [r for r in results if not r.get("needs_human", False)]
    if not admitted:
        return {"kb_size": 0, "kb_exact_match": 0.0, "kb_contamination_rate": 0.0}
    em = [exact_match(r.get("annotation", ""), r.get("answer", "")) for r in admitted]
    kb_exact_match = sum(em) / len(em)
    return {
        "kb_size": len(admitted),
        "kb_exact_match": kb_exact_match,
        "kb_contamination_rate": 1.0 - kb_exact_match,
    }


def evaluate_downstream_rag(kb_path: str, eval_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    kb = VectorKnowledgeBase(kb_path=kb_path, encoder=None)
    if not eval_set:
        return {"rag_top1_exact_match": 0.0}
    scores = []
    for sample in eval_set:
        retrieved = kb.retrieve(sample.get("question", ""), topk=1)
        pred = retrieved[0].get("annotation", "") if retrieved else ""
        scores.append(exact_match(pred, sample.get("answer", "")))
    return {"rag_top1_exact_match": sum(scores) / len(scores)}


def run_condition(
    train_set: List[Dict[str, Any]],
    eval_set: List[Dict[str, Any]],
    llm_obj: Any,
    cfg: ConditionConfig,
    output_dir: str,
    outlier_z_threshold: float = 2.0,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    kb_path = os.path.join(output_dir, f"{cfg.name}_kb.json")
    results_path = os.path.join(output_dir, f"{cfg.name}_annotations.json")

    annotator = Annotator(
        candidate_llms=["sim"],
        llm_dict={"sim": llm_obj},
        confidence_threshold=cfg.confidence_threshold,
        avg_logprob_threshold=cfg.avg_logprob_threshold,
        rag=True,
        kb_path=kb_path,
        task=QATask(),
        outlier_purge_interval=cfg.outlier_purge_interval,
        outlier_z_threshold=outlier_z_threshold,
    )

    routed = [{**s, "route": "sim"} for s in train_set]
    results = annotator.annotate_batch(routed)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics = {
        "condition": cfg.name,
        **evaluate_kb_quality(results),
        **evaluate_downstream_rag(kb_path, eval_set),
        "annotation_path": results_path,
        "kb_path": kb_path,
    }
    return metrics


def load_squad_500(cache_path: str = "squad_train.json", sample_count: int = 500) -> List[Dict[str, Any]]:
    try:
        ds = SquadDataset.from_url(save_path=cache_path, max_samples=sample_count, overwrite=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SQuAD samples. Provide a local cached SQuAD file via --squad-cache-path "
            "or enable network access for download."
        ) from exc
    rows = ds.to_list()
    if len(rows) < sample_count:
        raise ValueError(f"Expected {sample_count} SQuAD samples, but only loaded {len(rows)}")
    return rows[:sample_count]


def split_train_eval(samples: List[Dict[str, Any]], eval_size: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if eval_size <= 0 or eval_size >= len(samples):
        raise ValueError(
            f"eval_size must be between 1 and {len(samples)-1} "
            f"(got {eval_size} with {len(samples)} total samples)"
        )
    return samples[:-eval_size], samples[-eval_size:]


def save_sft_jsonl(annotations: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in annotations:
            if row.get("needs_human", False):
                continue
            prompt = f"Question: {row.get('question', '')}\nContext: {row.get('context', '')}"
            f.write(
                json.dumps(
                    {"instruction": prompt, "output": row.get("annotation", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )


def maybe_finetune_models(
    sft_path: str,
    output_dir: str,
    model_names: List[str],
    execute_finetune: bool,
) -> List[Dict[str, str]]:
    jobs = []
    for model_name in model_names:
        model_slug = model_name.replace("/", "_")
        model_out = os.path.join(output_dir, model_slug)
        status = "prepared"
        if execute_finetune:
            # Lazy import keeps experiment execution lightweight when users only
            # want baseline comparison and SFT-file generation.
            from misc.evaluate import finetune_sft

            finetune_sft(sft_path, model_name, model_out, epochs=1, batch_size=1)
            status = "completed"
        jobs.append({"model_name": model_name, "output_dir": model_out, "status": status})
    return jobs


def run_experiment(
    output_dir: str = "experiment_outputs",
    sample_count: int = 500,
    run_finetune: bool = False,
    execute_finetune: bool = False,
    squad_cache_path: str = "squad_train.json",
) -> Dict[str, Any]:
    if execute_finetune and not run_finetune:
        raise ValueError(
            "Cannot execute fine-tuning without generating SFT files. "
            "Set --run-finetune when using --execute-finetune."
        )

    samples = load_squad_500(cache_path=squad_cache_path, sample_count=sample_count)
    train_set, eval_set = split_train_eval(samples, eval_size=100)
    answer_key = {s.get("question", ""): s.get("answer", "") for s in train_set}

    noisy = SimulatedLLM(answer_key=answer_key)
    oracle = OracleLLM(answer_key=answer_key)

    conditions = [
        (ConditionConfig("naive", 0.0, None, 0), noisy),
        (ConditionConfig("entry_control_only", 0.7, -1.0, 0), noisy),
        (ConditionConfig("outlier_purge_only", 0.0, None, 50), noisy),
        (ConditionConfig("entry_control_and_outlier_purge", 0.7, -1.0, 50), noisy),
        (ConditionConfig("naive_oracle", 0.0, None, 0), oracle),
    ]

    model_targets = [QWEN_3_06B, LLAMA_3_2_1B]
    summary = {"sample_count": sample_count, "models": model_targets, "conditions": []}
    for cfg, llm in conditions:
        metrics = run_condition(train_set, eval_set, llm, cfg, output_dir=output_dir)
        summary["conditions"].append(metrics)

        if run_finetune:
            sft_path = os.path.join(output_dir, f"{cfg.name}_sft.jsonl")
            with open(metrics["annotation_path"], "r", encoding="utf-8") as f:
                annotations = json.load(f)
            save_sft_jsonl(annotations, sft_path)
            metrics["sft_path"] = sft_path
            metrics["finetune_jobs"] = maybe_finetune_models(
                sft_path=sft_path,
                output_dir=os.path.join(output_dir, f"{cfg.name}_finetuned"),
                model_names=model_targets,
                execute_finetune=execute_finetune,
            )

    by_name = {item["condition"]: item for item in summary["conditions"]}
    if "naive" in by_name and "naive_oracle" in by_name:
        summary["naive_vs_oracle_loss"] = {
            "kb_exact_match_loss": by_name["naive_oracle"]["kb_exact_match"] - by_name["naive"]["kb_exact_match"],
            "rag_top1_exact_match_loss": by_name["naive_oracle"]["rag_top1_exact_match"] - by_name["naive"]["rag_top1_exact_match"],
        }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_path"] = summary_path
    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG+annotation experiment for entry-control and outlier-purge effects")
    parser.add_argument("--output-dir", default="experiment_outputs")
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--squad-cache-path", default="squad_train.json")
    parser.add_argument("--run-finetune", action="store_true", help="Also materialize SFT files for Qwen3-0.6B and Llama3.2-1B")
    parser.add_argument("--execute-finetune", action="store_true", help="Execute actual finetuning jobs (resource-intensive)")
    args = parser.parse_args()

    summary = run_experiment(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        run_finetune=args.run_finetune,
        execute_finetune=args.execute_finetune,
        squad_cache_path=args.squad_cache_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
