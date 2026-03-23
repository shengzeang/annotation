"""Experiment: Active Learning for Efficient QA Annotation.

This experiment demonstrates the value of the repository's active-learning
sampling strategies by comparing four annotation pipelines that differ ONLY
in their first-stage filter — following the same
``HumanLLMAnnotationSystem`` pipeline pattern from ``test.py``:

    filter_1 → filter_2 (optional) → router.route() → annotator.annotate_batch()

Four filter conditions are compared:

1. **No filter**         – all samples are passed directly to the router and
                           annotator (upper-bound throughput, no selection).
2. **Random sampling**   – a random budget-sized subset is selected (cheap
                           baseline with no learned selection).
3. **ALPS filter**       – ``ActiveLearningFilter(method="alps")`` selects the
                           most informative samples (surprisal-based).
4. **Full filter chain** – AL filter followed by ``LLMNaiveFilter``, matching
                           the two-stage filter used in
                           ``HumanLLMAnnotationSystem``.

All four conditions share the same ``CascadeRouter`` and ``Annotator``.

Metrics show annotation quality (token-F1 / exact-match vs. ground truth)
and pipeline throughput.

Usage
-----
    # Offline smoke-test (no GPU / network required)
    python experiments/run_active_learning.py --samples 200 --budget 50 --skip-llm

    # Real Qwen annotation (requires GPU + HuggingFace model access)
    python experiments/run_active_learning.py \\
        --samples 500 --budget 100 \\
        --cheap-model  Qwen/Qwen2.5-7B-Instruct \\
        --judge-model  Qwen/Qwen2.5-7B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/al_out
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
# Sys-path fix so the script can be run from the repo root or any CWD
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Repository imports — the actual system components from test.py
# ---------------------------------------------------------------------------
from annotation import Annotator
from filters import ActiveLearningFilter, LLMNaiveFilter
from routers import CascadeRouter
from tasks.qa import QATask
from utils import export_annotation_results

# Default candidate LLMs — shared with test.py / HumanLLMAnnotationSystem.
DEFAULT_CANDIDATE_LLMS: List[str] = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


# ---------------------------------------------------------------------------
# Mock LLM – duck-types LLMBase interface, no torch / network needed
# ---------------------------------------------------------------------------

class MockLLM:
    """Minimal LLM stub for offline testing.

    Returns an output that ``QATask.parse_output`` can parse so that
    ``Annotator`` produces valid annotation records.
    """

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "Answer: test_answer Confidence: 0.85"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class MockJudgeLLM:
    """Judge LLM stub for ``CascadeRouter``.

    Always returns ``"1"`` so the router keeps the first (cheap) model
    and does not escalate — making offline annotation fast.
    """

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "1"


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


def evaluate_annotation_quality(annotated: List[Dict[str, Any]]) -> Dict[str, float]:
    f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    ems = [compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# SFT JSONL output
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
        ("Isaac Newton", "Newton formulated the laws of motion and universal gravitation.", "motion"),
        ("William Shakespeare", "Shakespeare wrote plays including Hamlet.", "Hamlet"),
        ("Leonardo da Vinci", "Da Vinci painted the Mona Lisa.", "Mona Lisa"),
    ]
    dataset: List[Dict[str, Any]] = []
    for i in range(n):
        subj, ctx, ans = topics[i % len(topics)]
        extra = " ".join([f"word{j}" for j in range(rng.randint(0, 20))])
        question = f"What is associated with {subj}? (sample {i})"
        context = ctx + (" " + extra if extra else "")
        dataset.append({
            "id": f"synthetic-{i}",
            "question": question,
            "context": context,
            "answer": ans,
            "text": f"Question: {question}\nContext: {context}",
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
    judge_llm: Any,
    budget: int = 50,
    output_dir: str = "/tmp/al_out",
    seed: int = 42,
    force_fallback: bool = True,
    task: Any = None,
    candidate_llms: Optional[List[str]] = None,
    llm_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run four filter conditions following the HumanLLMAnnotationSystem pattern.

    Each condition uses the SAME ``CascadeRouter`` + ``Annotator`` pipeline;
    only the first-stage filter changes.

    Parameters
    ----------
    dataset:
        Full pool of QA samples.
    cheap_llm:
        Primary annotation LLM (and LLMNaiveFilter scorer).
    judge_llm:
        LLM used by ``CascadeRouter`` to judge cheap-LLM answers.
    budget:
        Annotation budget per AL condition (samples selected).
    output_dir:
        Directory for SFT JSONL outputs and KB files.
    seed:
        Random seed.
    force_fallback:
        Passed to ``ActiveLearningFilter`` — set ``True`` to skip BERT
        loading (offline / test mode).
    task:
        Task object (default: ``QATask``).
    candidate_llms:
        Ordered list of LLM identifiers available for routing.  When
        provided, ``llm_dict`` must also be supplied.  Defaults to the
        single-entry ``["primary"]`` fallback using ``cheap_llm``.
    llm_dict:
        Mapping from LLM identifier to LLM instance.  Must be consistent
        with ``candidate_llms``.

    Returns
    -------
    List of result dicts with keys ``condition``, ``selected``,
    ``annotated``, ``annotation_f1``, ``annotation_em``, ``sft_file``.
    """
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    # --- shared pipeline components (same for all conditions) ---
    # Use caller-supplied candidate list (full pipeline) or fall back to a
    # single "primary" entry for backward-compatible / mock-mode calls.
    if candidate_llms is None or llm_dict is None:
        candidate_llms = ["primary"]
        llm_dict = {"primary": cheap_llm}

    # Build conditions — each returns a filtered list
    rng = random.Random(seed)
    shuffled = dataset[:]
    rng.shuffle(shuffled)
    random_subset = shuffled[:budget]

    al_filter = ActiveLearningFilter(
        method="alps",
        budget=budget,
        batch_size=max(2, budget // 5),
        force_fallback=force_fallback,
    )
    al_filtered = al_filter.filter(dataset)

    al_filter_full = ActiveLearningFilter(
        method="alps",
        budget=max(budget * 2, len(dataset)),
        batch_size=max(2, budget // 5),
        force_fallback=force_fallback,
    )
    llm_naive_filter = LLMNaiveFilter(cheap_llm, budget=budget)
    al_then_llm = llm_naive_filter.filter(al_filter_full.filter(dataset))

    conditions = [
        ("No filter",         dataset[:budget]),
        ("Random sampling",   random_subset),
        ("ALPS filter",       al_filtered),
        ("Full filter chain", al_then_llm),
    ]

    results: List[Dict[str, Any]] = []
    n_cond = len(conditions)
    for i, (cond_name, selected) in enumerate(conditions, 1):
        print(f"\n[{i}/{n_cond}] Running condition: {cond_name}  ({len(selected)} samples) …")
        # Build fresh annotator per condition (fresh KB to avoid cross-contamination)
        kb_path = os.path.join(output_dir, f"kb_{re.sub(r'[^a-z0-9]', '_', cond_name.lower())}.json")
        annotator = Annotator(
            candidate_llms,
            llm_dict,
            task=task,
            kb_path=kb_path,
        )
        router = CascadeRouter(
            judge_llm=judge_llm,
            candidate_llm=candidate_llms,
            llm_dict=llm_dict,
        )

        # filter → route → annotate  (HumanLLMAnnotationSystem pattern)
        routed = router.route(selected)
        annotated = annotator.annotate_batch(routed)

        quality = evaluate_annotation_quality(annotated)
        safe_name = re.sub(r"[() /\-]", "_", cond_name.lower()).strip("_")
        sft_path = os.path.join(output_dir, f"sft_al_{safe_name}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path)

        results.append({
            "condition": cond_name,
            "selected": len(selected),
            "annotated": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "human_review": sum(1 for r in annotated if r.get("needs_human", False)),
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Condition':<22} {'Selected':>9} {'Ann-F1':>7} {'Ann-EM':>7} {'HumanRev':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Active Learning — Filter Strategy Comparison")
    print("  (all conditions use the same CascadeRouter + Annotator)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<22} "
            f"{r['selected']:>9} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r.get('human_review', 0):>9}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Active Learning filter comparison for QA annotation"
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--squad-path", default="squad_train.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANDIDATE_LLMS,
        help="Candidate LLMs to annotate with (default: the 3 standard Qwen models)",
    )
    parser.add_argument("--judge-model", default=None,
                        help="Judge LLM for CascadeRouter (defaults to last --models entry)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use mock LLMs and force_fallback for offline testing")
    parser.add_argument("--output-dir", default="/tmp/al_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.  Budget = {args.budget}.")

    if args.skip_llm:
        print("Using mock LLMs and force_fallback (--skip-llm)")
        cheap_llm: Any = MockLLM()
        judge_llm: Any = MockJudgeLLM()
        force_fallback = True
        # In mock mode keep the single-entry fallback for lightweight testing
        _candidate_llms = None
        _llm_dict = None
    else:
        from misc.llm_provider import LocalLLM
        models = args.models
        print(f"Loading LLMs: {models}")
        _llm_dict = {m: LocalLLM(m) for m in models}
        _candidate_llms = models
        cheap_llm = _llm_dict[models[0]]
        judge_name = args.judge_model or models[-1]
        judge_llm = _llm_dict.get(judge_name) or LocalLLM(judge_name)
        force_fallback = False

    results = run_experiment(
        dataset=dataset,
        cheap_llm=cheap_llm,
        judge_llm=judge_llm,
        budget=args.budget,
        output_dir=args.output_dir,
        seed=args.seed,
        force_fallback=force_fallback,
        candidate_llms=_candidate_llms,
        llm_dict=_llm_dict,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "al_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
