"""Experiment: LLM Routing for Cost-Quality Tradeoff in QA Annotation.

This experiment demonstrates how the repository's routing strategies balance
annotation quality against cost, following the same pipeline pattern from
``test.py`` (``HumanLLMAnnotationSystem``):

    filter → router.route() → annotator.annotate_batch()

Four routing conditions are compared using the same ``ActiveLearningFilter``
+ ``Annotator``:

1. **All-cheap**   – every sample is annotated by the cheap LLM (no routing).
2. **All-expensive** – every sample is annotated by the expensive LLM (quality
                       upper-bound).
3. **CascadeRouter** – the cheap LLM is tried first; the judge LLM evaluates
                       the answer and escalates to the expensive LLM when the
                       answer does not meet the quality threshold.
4. **LLMRouter**     – a scorer LLM rates each candidate and the highest-rated
                       model is chosen per sample.

Metrics show annotation quality (token-F1 / exact-match vs. ground truth)
and the fraction of samples routed to the expensive LLM (cost proxy).

Usage
-----
    # Offline smoke-test (no GPU / network required)
    python experiments/run_llm_routing.py --samples 100 --skip-llm

    # Real Qwen routing (requires GPU)
    python experiments/run_llm_routing.py \\
        --samples 500 \\
        --cheap-model   Qwen/Qwen2.5-7B-Instruct \\
        --expensive-model Qwen/Qwen2.5-72B-Instruct \\
        --judge-model   Qwen/Qwen2.5-7B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/routing_out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Sys-path fix
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Repository imports — actual system components
# ---------------------------------------------------------------------------
from annotation import Annotator
from filters import ActiveLearningFilter
from routers import CascadeRouter, LLMRouter
from tasks.qa import QATask


# ---------------------------------------------------------------------------
# Mock LLMs for offline testing
# ---------------------------------------------------------------------------

class MockAnnotationLLM:
    """Primary annotation LLM stub.  Returns a QATask-parseable string."""

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "Answer: test_answer Confidence: 0.85"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class MockJudgeLLM:
    """Judge LLM stub for ``CascadeRouter``.

    Returns ``"0"`` so the router ALWAYS escalates to the expensive model,
    ensuring the cascade path is exercised in tests.
    """

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "0"


class MockScorerLLM:
    """Scorer LLM stub for ``LLMRouter``.

    Returns JSON scores preferring the expensive model.
    """

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        return '[{"model": "cheap", "score": 0.4}, {"model": "expensive", "score": 0.8}]'


# ---------------------------------------------------------------------------
# QA metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_t = _tokenize(prediction)
    gt_t = _tokenize(ground_truth)
    if not pred_t and not gt_t:
        return 1.0
    if not pred_t or not gt_t:
        return 0.0
    common = set(pred_t) & set(gt_t)
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(gt_t)
    return 2 * p * r / (p + r)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if *prediction* matches *ground_truth* (case-insensitive, whitespace-trimmed), else 0.0."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def evaluate_annotation_quality(
    annotated: List[Dict[str, Any]],
    expensive_llm_name: str = "expensive",
) -> Dict[str, float]:
    f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    ems = [compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    exp_rate = sum(1 for r in annotated if r.get("route") == expensive_llm_name) / max(1, len(annotated))
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
        "expensive_call_rate": round(exp_rate, 4),
    }


# ---------------------------------------------------------------------------
# SFT JSONL
# ---------------------------------------------------------------------------

def write_sft_jsonl(annotated: List[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in annotated:
            f.write(json.dumps(
                {"instruction": rec.get("text", ""), "output": rec.get("annotation", "")},
                ensure_ascii=False,
            ) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    topics = [
        ("Albert Einstein", "Einstein developed the theory of relativity.", "relativity"),
        ("Python language", "Python is a high-level programming language.", "high-level"),
        ("Mount Everest", "Mount Everest is the highest mountain.", "highest"),
        ("Marie Curie", "Marie Curie discovered polonium and radium.", "polonium"),
        ("The Sun", "The Sun is the star at the center of the Solar System.", "star"),
    ]
    dataset = []
    for i in range(n):
        subj, ctx, ans = topics[i % len(topics)]
        q = f"What is associated with {subj}? (sample {i})"
        dataset.append({
            "id": f"synthetic-{i}",
            "question": q,
            "context": ctx,
            "answer": ans,
            "text": f"Question: {q}\nContext: {ctx}",
        })
    return dataset


def load_squad_dataset(squad_path: str, max_samples: int = 200) -> List[Dict[str, Any]]:
    if squad_path and os.path.exists(squad_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qa_datasets",
            os.path.join(_ROOT, "datasets", "qa_datasets.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ds = mod.SquadDataset.from_file(squad_path, max_samples=max_samples)
        return list(ds._data)
    return _make_synthetic_dataset(n=max_samples)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: List[Dict[str, Any]],
    cheap_llm: Any,
    expensive_llm: Any,
    judge_llm: Any,
    scorer_llm: Any,
    output_dir: str = "/tmp/routing_out",
    cascade_threshold: float = 0.7,
    force_fallback: bool = True,
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Run four routing conditions following the HumanLLMAnnotationSystem pattern.

    All conditions use the same ``ActiveLearningFilter`` pre-filter; only the
    router changes.

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    cheap_llm:
        Fast / inexpensive LLM.
    expensive_llm:
        Slow / capable LLM.
    judge_llm:
        LLM used as judge by ``CascadeRouter``.
    scorer_llm:
        LLM used as scorer by ``LLMRouter``.
    output_dir:
        Directory for SFT JSONL outputs.
    cascade_threshold:
        Minimum quality score below which ``CascadeRouter`` escalates.
    force_fallback:
        Passed to ``ActiveLearningFilter`` for offline mode.
    task:
        Task object (default: ``QATask``).

    Returns
    -------
    List of result dicts with keys ``condition``, ``annotated``,
    ``annotation_f1``, ``annotation_em``, ``expensive_call_rate``,
    ``sft_file``.
    """
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    candidate_llms = ["cheap", "expensive"]
    llm_dict = {"cheap": cheap_llm, "expensive": expensive_llm}

    # Shared filter (same for all conditions, as in HumanLLMAnnotationSystem)
    al_filter = ActiveLearningFilter(
        method="alps",
        budget=len(dataset),
        batch_size=max(2, len(dataset) // 10),
        force_fallback=force_fallback,
    )
    filtered = al_filter.filter(dataset)

    # Four routing conditions
    def _annotate_no_router(llm_name: str) -> List[Dict[str, Any]]:
        """Bypass routing: annotate every sample with a fixed LLM."""
        kb_path = os.path.join(output_dir, f"kb_no_router_{llm_name}.json")
        annotator = Annotator(candidate_llms, llm_dict, task=task, kb_path=kb_path)
        return annotator.annotate_batch(filtered, assigned_llm=llm_name)

    def _annotate_cascade() -> List[Dict[str, Any]]:
        """CascadeRouter: cheap first, escalate when judge says wrong."""
        kb_path = os.path.join(output_dir, "kb_cascade.json")
        annotator = Annotator(candidate_llms, llm_dict, task=task, kb_path=kb_path)
        router = CascadeRouter(
            judge_llm=judge_llm,
            candidate_llm=candidate_llms,
            llm_dict=llm_dict,
            threshold=cascade_threshold,
        )
        routed = router.route(filtered)
        return annotator.annotate_batch(routed)

    def _annotate_llm_router() -> List[Dict[str, Any]]:
        """LLMRouter: scorer LLM picks the best candidate per sample."""
        kb_path = os.path.join(output_dir, "kb_llm_router.json")
        annotator = Annotator(candidate_llms, llm_dict, task=task, kb_path=kb_path)
        router = LLMRouter(scorer=scorer_llm, candidate_llms=candidate_llms)
        routed = router.route(filtered)
        return annotator.annotate_batch(routed)

    _N = 4
    print(f"\n[1/{_N}] Running condition: All-cheap …")
    _data_cheap = _annotate_no_router("cheap")
    print(f"\n[2/{_N}] Running condition: All-expensive …")
    _data_expensive = _annotate_no_router("expensive")
    print(f"\n[3/{_N}] Running condition: CascadeRouter …")
    _data_cascade = _annotate_cascade()
    print(f"\n[4/{_N}] Running condition: LLMRouter …")
    _data_llm_router = _annotate_llm_router()

    conditions_data = [
        ("All-cheap",     _data_cheap),
        ("All-expensive", _data_expensive),
        ("CascadeRouter", _data_cascade),
        ("LLMRouter",     _data_llm_router),
    ]

    results = []
    for cond_name, annotated in conditions_data:
        quality = evaluate_annotation_quality(annotated, expensive_llm_name="expensive")
        safe_name = re.sub(r"[() /\-]", "_", cond_name.lower()).strip("_")
        sft_path = os.path.join(output_dir, f"sft_routing_{safe_name}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path)
        results.append({
            "condition": cond_name,
            "annotated": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "expensive_call_rate": quality["expensive_call_rate"],
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Condition':<18} {'Ann-F1':>7} {'Ann-EM':>7} {'Exp-Rate':>9} {'#Samples':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  LLM Routing — Quality vs Cost Tradeoff")
    print("  (Exp-Rate = fraction of samples routed to expensive LLM)")
    print("  (all conditions use the same ActiveLearningFilter + Annotator)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<18} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r['expensive_call_rate']:>9.4f} "
            f"{r['annotated']:>9}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="LLM Routing experiment: cost-quality tradeoff in QA annotation"
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--squad-path", default="squad_train.json")
    parser.add_argument("--cheap-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--expensive-model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge-model", default=None,
                        help="Judge LLM for CascadeRouter (defaults to --cheap-model)")
    parser.add_argument("--scorer-model", default=None,
                        help="Scorer LLM for LLMRouter (defaults to --cheap-model)")
    parser.add_argument("--cascade-threshold", type=float, default=0.7)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/routing_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    if args.skip_llm:
        print("Using mock LLMs (--skip-llm)")
        cheap_llm: Any = MockAnnotationLLM()
        expensive_llm: Any = MockAnnotationLLM()
        judge_llm: Any = MockJudgeLLM()
        scorer_llm: Any = MockScorerLLM()
        force_fallback = True
    else:
        from misc.llm_provider import LocalLLM
        cheap_llm = LocalLLM(args.cheap_model)
        expensive_llm = LocalLLM(args.expensive_model)
        judge_name = args.judge_model or args.cheap_model
        judge_llm = cheap_llm if judge_name == args.cheap_model else LocalLLM(judge_name)
        scorer_name = args.scorer_model or args.cheap_model
        scorer_llm = cheap_llm if scorer_name == args.cheap_model else LocalLLM(scorer_name)
        force_fallback = False

    results = run_experiment(
        dataset=dataset,
        cheap_llm=cheap_llm,
        expensive_llm=expensive_llm,
        judge_llm=judge_llm,
        scorer_llm=scorer_llm,
        output_dir=args.output_dir,
        cascade_threshold=args.cascade_threshold,
        force_fallback=force_fallback,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "routing_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
