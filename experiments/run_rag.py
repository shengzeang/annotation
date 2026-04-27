"""Experiment: RAG-Augmented Annotation for QA.

This experiment demonstrates how enabling RAG (Retrieval-Augmented
Generation) in the ``Annotator`` improves annotation quality as the
knowledge base accumulates high-confidence examples.  The pipeline follows
the same ``HumanLLMAnnotationSystem`` pattern from ``test.py``:

    filter → router.route() → annotator.annotate_batch()

Two conditions are compared, varying only the ``Annotator`` RAG configuration:

1. **No RAG**  – ``Annotator(rag=False)``:  plain LLM annotation without
                  any KB retrieval context.
2. **RAG**     – ``Annotator(rag=True)``:   each sample is annotated with
                  retrieved similar QA pairs prepended to the prompt.  The KB
                  grows progressively: high-confidence answers are added after
                  each batch so later samples benefit from richer context.

Both conditions use the same ``ActiveLearningFilter`` + ``CascadeRouter``.
Per-window token-F1 shows the improvement over time as the KB fills.

Usage
-----
    # Offline smoke-test (no GPU / network required)
    python experiments/run_rag.py --samples 200 --skip-llm

    # Real Qwen annotation (requires GPU)
    python experiments/run_rag.py \\
        --samples 10000 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --hotpot-path path/to/hotpot_train_v1.1.json \\
        --output-dir /tmp/rag_out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
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
from routers import CascadeRouter
from tasks.qa import QATask

# Default candidate LLMs — shared with test.py / HumanLLMAnnotationSystem.
DEFAULT_CANDIDATE_LLMS: List[str] = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


# ---------------------------------------------------------------------------
# Mock LLMs for offline testing
# ---------------------------------------------------------------------------

class MockLLM:
    """Primary annotation LLM stub.  Returns QATask-parseable output."""

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "Answer: test_answer Confidence: 0.85"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class MockJudgeLLM:
    """Judge LLM stub for CascadeRouter — always keeps the cheap model."""

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


def windowed_f1(annotated: List[Dict[str, Any]], window: int = 50) -> List[Dict[str, Any]]:
    """Return mean token-F1 for successive non-overlapping windows.

    Shows how annotation quality trends as the KB grows over time.
    """
    windows = []
    for start in range(0, len(annotated), window):
        chunk = annotated[start: start + window]
        if not chunk:
            break
        f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in chunk]
        windows.append({
            "window_start": start,
            "window_end": start + len(chunk) - 1,
            "mean_f1": round(sum(f1s) / len(f1s), 4),
        })
    return windows


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

def _make_synthetic_dataset(n: int = 10000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    topics = [
        ("Albert Einstein", "Einstein developed the theory of relativity.", "relativity"),
        ("Python language", "Python is a high-level programming language.", "high-level"),
        ("Mount Everest", "Mount Everest is the highest mountain.", "highest"),
        ("Marie Curie", "Marie Curie discovered polonium and radium.", "polonium"),
        ("The Sun", "The Sun is the star at the center of the Solar System.", "star"),
        ("Isaac Newton", "Newton formulated the laws of motion.", "motion"),
        ("William Shakespeare", "Shakespeare wrote plays including Hamlet.", "Hamlet"),
        ("Leonardo da Vinci", "Da Vinci painted the Mona Lisa.", "Mona Lisa"),
    ]
    dataset = []
    for i in range(n):
        subj, ctx, ans = topics[i % len(topics)]
        extra = " ".join([f"word{j}" for j in range(rng.randint(0, 10))])
        q = f"What is associated with {subj}? (sample {i})"
        context = ctx + (" " + extra if extra else "")
        dataset.append({
            "id": f"synthetic-{i}",
            "question": q,
            "context": context,
            "answer": ans,
            "text": f"Question: {q}\nContext: {context}",
        })
    return dataset


def load_squad_dataset(squad_path: str, max_samples: int = 10000) -> List[Dict[str, Any]]:
    if squad_path and os.path.exists(squad_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qa_datasets",
            os.path.join(_ROOT, "qa_data", "qa_datasets.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ds = mod.SquadDataset.from_file(squad_path, max_samples=max_samples)
        return list(ds._data)
    return _make_synthetic_dataset(n=max_samples)


def load_hotpot_dataset(hotpot_path: str, max_samples: int = 10000) -> List[Dict[str, Any]]:
    """Load HotpotQA dataset from *hotpot_path*; fall back to synthetic data."""
    if hotpot_path and os.path.exists(hotpot_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qa_datasets",
            os.path.join(_ROOT, "qa_data", "qa_datasets.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ds = mod.HotpotDataset.from_file(hotpot_path, max_samples=max_samples)
        return list(ds._data)
    return _make_synthetic_dataset(n=max_samples)


# ---------------------------------------------------------------------------
# Resume-mechanism helpers
# ---------------------------------------------------------------------------

def _safe_name(cond_name: str) -> str:
    """Convert a condition name to a filesystem-safe lowercase string."""
    return re.sub(r"[^a-z0-9]", "_", cond_name.lower())


def _condition_result_path(cond_name: str, output_dir: str) -> str:
    """Return the path to the per-condition result JSON file."""
    return os.path.join(output_dir, f"result_{_safe_name(cond_name)}.json")


def _sft_output_path(cond_name: str, output_dir: str) -> str:
    """Return the path to the per-condition SFT JSONL output file."""
    return os.path.join(output_dir, f"sft_rag_{_safe_name(cond_name)}.jsonl")


def _condition_already_done(cond_name: str, output_dir: str) -> bool:
    """Return ``True`` if *cond_name* has already produced output in *output_dir*.

    A condition is considered done if **either** the per-condition result JSON
    file **or** the SFT JSONL output file already exists on disk.
    """
    return (
        os.path.exists(_condition_result_path(cond_name, output_dir))
        or os.path.exists(_sft_output_path(cond_name, output_dir))
    )


def _load_condition_result(cond_name: str, output_dir: str) -> Dict[str, Any]:
    """Load and return the cached per-condition result dict from disk."""
    with open(_condition_result_path(cond_name, output_dir), encoding="utf-8") as f:
        return json.load(f)


def _save_condition_result(result: Dict[str, Any], output_dir: str) -> None:
    """Persist a per-condition result dict to *output_dir*."""
    path = _condition_result_path(result["condition"], output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: List[Dict[str, Any]],
    llm: Any,
    judge_llm: Any,
    output_dir: str = "/tmp/rag_out",
    topk: int = 3,
    window: int = 50,
    force_fallback: bool = True,
    task: Any = None,
    candidate_llms: Optional[List[str]] = None,
    llm_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run RAG vs No-RAG annotation comparison.

    Both conditions use the same ``ActiveLearningFilter`` + ``CascadeRouter``
    pipeline — only the ``Annotator`` RAG flag differs.  This directly shows
    the impact of knowledge-base retrieval on annotation quality.

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    llm:
        LLM instance used for annotation.
    judge_llm:
        LLM used as judge by ``CascadeRouter``.
    output_dir:
        Directory for SFT JSONL outputs and KB files.
    topk:
        Number of KB entries retrieved per sample when RAG is enabled.
    window:
        Sliding-window size for per-window F1 trend computation.
    force_fallback:
        Passed to ``ActiveLearningFilter`` for offline mode.
    task:
        Task object (default: ``QATask``).
    candidate_llms:
        Ordered list of LLM identifiers available for routing.  When
        provided, ``llm_dict`` must also be supplied.  Defaults to the
        single-entry ``["primary"]`` fallback using ``llm``.
    llm_dict:
        Mapping from LLM identifier to LLM instance.  Must be consistent
        with ``candidate_llms``.

    Returns
    -------
    List of result dicts with keys ``condition``, ``annotated``,
    ``annotation_f1``, ``annotation_em``, ``final_kb_size``,
    ``windowed_f1``, ``sft_file``.
    """
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    # Use caller-supplied candidate list (full pipeline) or fall back to a
    # single "primary" entry for backward-compatible / mock-mode calls.
    if candidate_llms is None or llm_dict is None:
        candidate_llms = ["primary"]
        llm_dict = {"primary": llm}

    _N = 2
    conditions = [
        ("No RAG", False),
        ("RAG",    True),
    ]

    # Only compute shared routing when at least one condition still needs to run.
    _needs_run = [not _condition_already_done(cn, output_dir) for cn, _ in conditions]
    if any(_needs_run):
        # Shared filter — same for both conditions
        al_filter = ActiveLearningFilter(
            method="alps",
            budget=1000,
            batch_size=max(2, 1000 // 10),
            force_fallback=force_fallback,
        )
        filtered = al_filter.filter(dataset)

        router = CascadeRouter(
            judge_llm=judge_llm,
            candidate_llm=candidate_llms,
            llm_dict=llm_dict,
        )
        routed = router.route(filtered)
    else:
        routed = []  # unused — all conditions will be loaded from cache

    results: List[Dict[str, Any]] = []
    for i, (cond_name, use_rag) in enumerate(conditions, 1):
        print(f"\n[{i}/{_N}] Running condition: {cond_name} …")
        if _condition_already_done(cond_name, output_dir):
            print(f"  ↳ Already done — skipping (output file exists).")
            result_path = _condition_result_path(cond_name, output_dir)
            if os.path.exists(result_path):
                results.append(_load_condition_result(cond_name, output_dir))
            else:
                # SFT file exists but result JSON was not written — reconstruct minimal result.
                sft_path = _sft_output_path(cond_name, output_dir)
                n_lines = sum(1 for _ in open(sft_path, encoding="utf-8"))
                results.append({
                    "condition": cond_name,
                    "annotated": n_lines,
                    "annotation_f1": 0.0,
                    "annotation_em": 0.0,
                    "final_kb_size": 0,
                    "windowed_f1": [],
                    "sft_file": sft_path,
                })
            continue

        kb_path = os.path.join(output_dir, f"kb_{re.sub(r'[^a-z0-9]', '_', cond_name.lower())}.json")
        annotator = Annotator(
            candidate_llms,
            llm_dict,
            task=task,
            rag=use_rag,
            kb_path=kb_path,
        )
        # For RAG condition, set the retrieve topk
        if use_rag:
            annotator.knowledge_base._topk = topk

        annotated = annotator.annotate_batch(routed)

        quality = evaluate_annotation_quality(annotated)
        w_f1 = windowed_f1(annotated, window=window)
        final_kb_size = len(annotator.knowledge_base.entries) if use_rag else 0

        sft_path = _sft_output_path(cond_name, output_dir)
        n_written = write_sft_jsonl(annotated, sft_path)

        result: Dict[str, Any] = {
            "condition": cond_name,
            "annotated": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "final_kb_size": final_kb_size,
            "windowed_f1": w_f1,
            "sft_file": sft_path,
        }
        _save_condition_result(result, output_dir)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]], window: int = 50) -> None:
    header = (
        f"{'Condition':<12} {'Ann-F1':>7} {'Ann-EM':>7} {'KB-Final':>9} {'#Samples':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  RAG — Retrieval-Augmented QA Annotation Comparison")
    print("  (both conditions use the same filter + CascadeRouter + Annotator)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<12} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r['final_kb_size']:>9} "
            f"{r['annotated']:>9}"
        )
    print(sep)

    # Per-window F1 trend
    if any(r.get("windowed_f1") for r in results):
        cond_windows = {}
        for r in results:
            cond_windows[r["condition"]] = {
                w["window_start"]: w["mean_f1"]
                for w in r.get("windowed_f1", [])
            }

        all_starts = sorted({s for wmap in cond_windows.values() for s in wmap})
        if all_starts:
            win_size = window
            col_labels = [f"[{s}-{s + win_size - 1}]" for s in all_starts]
            col_w = max(8, *(len(c) for c in col_labels))
            hdr = f"\n  Per-window token-F1 (higher later = KB growing helps):\n"
            hdr += f"  {'Condition':<12} " + " ".join(f"{c:>{col_w}}" for c in col_labels)
            print(hdr)
            for r in results:
                wmap = cond_windows[r["condition"]]
                row = f"  {r['condition']:<12} "
                row += " ".join(f"{wmap.get(s, 0.0):>{col_w}.4f}" for s in all_starts)
                print(row)
    print("")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG-augmented annotation experiment for QA"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--hotpot-path", default="hotpot_train_v1.json",
                        help="HotpotQA training JSON path (default: hotpot_train_v1.json)")
    parser.add_argument("--squad-path", default=None,
                        help="SQuAD training JSON path; used only when --hotpot-path is absent")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANDIDATE_LLMS,
        help="Candidate LLMs to annotate with (default: the 3 standard Qwen models)",
    )
    parser.add_argument("--judge-model", default=None,
                        help="Judge LLM for CascadeRouter (defaults to last --models entry)")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/rag_out")
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    if args.hotpot_path and os.path.exists(args.hotpot_path):
        dataset = load_hotpot_dataset(args.hotpot_path, max_samples=args.samples)
    else:
        dataset = load_squad_dataset(args.squad_path or "", max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    if args.skip_llm:
        print("Using mock LLMs (--skip-llm)")
        llm: Any = MockLLM()
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
        llm = _llm_dict[models[0]]
        judge_name = args.judge_model or models[-1]
        judge_llm = _llm_dict.get(judge_name) or LocalLLM(judge_name)
        force_fallback = False

    results = run_experiment(
        dataset=dataset,
        llm=llm,
        judge_llm=judge_llm,
        output_dir=args.output_dir,
        topk=args.topk,
        window=args.window,
        force_fallback=force_fallback,
        candidate_llms=_candidate_llms,
        llm_dict=_llm_dict,
    )

    print_results_table(results, window=args.window)

    summary_path = os.path.join(args.output_dir, "rag_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
