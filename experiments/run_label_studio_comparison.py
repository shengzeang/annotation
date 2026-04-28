"""Experiment: Comparing DataFlow-Annotator vs Label-Studio-style oracle annotation.

This experiment follows the same ``HumanLLMAnnotationSystem`` pipeline pattern
from ``test.py``:

    filter → router.route() → annotator.annotate_batch()

Five conditions are compared:

1. **Single Oracle**          – all samples annotated by the most capable LLM
                                (no filter/router; models[-1] is used).
2. **3-Oracle Majority Vote** – three independent oracle annotations resolved
                                by majority vote.
3. **DataFlow (naive LLM)**   – ``ActiveLearningFilter`` + ``KNNRouter`` +
                                ``Annotator(rag=False)``.
4. **DataFlow (KB + RAG)**    – same pipeline with ``Annotator(rag=True)``.
5. **DataFlow (full)**        – same as (4) but with a stricter confidence
                                threshold (0.75 instead of 0.7).

Usage
-----
    # Offline smoke-test (no GPU required)
    python experiments/run_label_studio_comparison.py \\
        --samples 200 --skip-llm

    # Real Qwen annotation (requires GPU)
    python experiments/run_label_studio_comparison.py \\
        --samples 10000 \\
        --models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct \\
        --hotpot-path path/to/hotpot_train_v1.1.json \\
        --output-dir /tmp/lsc_out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
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
from routers import KNNRouter
from tasks.qa import QATask

# Default candidate LLMs — shared with test.py / HumanLLMAnnotationSystem.
DEFAULT_CANDIDATE_LLMS: List[str] = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]


# ---------------------------------------------------------------------------
# Mock LLMs for offline testing
# ---------------------------------------------------------------------------

class MockAnnotationLLM:
    """Primary annotation LLM stub.  Returns QATask-parseable output."""

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "Answer: test_answer Confidence: 0.85"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class MockJudgeLLM:
    """Judge LLM stub for CascadeRouter — always keeps the cheap model."""

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "1"


class MockKNNRouter:
    """Offline-safe KNNRouter stub for ``force_fallback`` / ``--skip-llm`` mode.

    Routes all samples to the first (cheapest) candidate with uniform scores,
    without loading any sentence-transformer model.
    """

    def __init__(self, candidate_llms: List[str]):
        self.candidate_llms = list(candidate_llms)

    def route(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(self.candidate_llms)
        uniform = 1.0 / n if n else 0.0
        scores = [{"model": c, "score": uniform} for c in self.candidate_llms]
        chosen = self.candidate_llms[0] if self.candidate_llms else None
        return [{**d, "route": chosen, "route_scores": scores} for d in dataset]


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
    """Compute mean token-F1 and exact-match of annotations vs ground truth."""
    f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    ems = [compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
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

def _make_synthetic_dataset(n: int = 10000, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a synthetic SQuAD-style dataset for offline testing."""
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
        extra = " ".join([f"word{j}" for j in range(rng.randint(0, 5))])
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
    """Load SQuAD dataset from *squad_path*; fall back to synthetic data."""
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
# Downstream fine-tuning helper (optional — skipped by default)
# ---------------------------------------------------------------------------

def run_downstream_finetune(
    sft_path: str,
    model_name: str,
    model_output_dir: str,
    val_data_path: Optional[str] = None,
    epochs: int = 2,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Fine-tune *model_name* on *sft_path* and evaluate (requires GPU).

    Delegates to ``misc.evaluate.finetune_sft`` and
    ``misc.evaluate.evaluate`` so that all Qwen fine-tuning logic lives in
    one place.

    Returns
    -------
    Dict with ``model_dir`` and optionally ``bleu`` / ``rouge_l`` keys.
    """
    from misc.evaluate import finetune_sft
    from misc.evaluate import evaluate as evaluate_model

    finetune_sft(sft_path, model_name, model_output_dir, epochs=epochs, batch_size=batch_size)

    result: Dict[str, Any] = {"model_dir": model_output_dir}
    if val_data_path and os.path.exists(val_data_path):
        metrics = evaluate_model(model_output_dir, val_data_path)
        if isinstance(metrics, dict):
            result.update(metrics)

    return result


# ---------------------------------------------------------------------------
# Resume-mechanism helpers
# ---------------------------------------------------------------------------

def _safe_name(cond_name: str) -> str:
    """Convert a condition name to a filesystem-safe lowercase string."""
    return re.sub(r"[() /\-]", "_", cond_name.lower()).strip("_")


def _condition_result_path(cond_name: str, output_dir: str) -> str:
    """Return the path to the per-condition result JSON file."""
    return os.path.join(output_dir, f"result_{_safe_name(cond_name)}.json")


def _sft_output_path(cond_name: str, output_dir: str) -> str:
    """Return the path to the per-condition SFT JSONL output file."""
    return os.path.join(output_dir, f"sft_lsc_{_safe_name(cond_name)}.jsonl")


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
    oracle_llm: Any,
    judge_llm: Any,
    output_dir: str = "/tmp/lsc_out",
    skip_finetune: bool = True,
    finetune_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    val_data_path: Optional[str] = None,
    force_fallback: bool = True,
    task: Any = None,
    candidate_llms: Optional[List[str]] = None,
    llm_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run all five annotation conditions and return a list of result dicts.

    Follows the ``HumanLLMAnnotationSystem`` pipeline pattern:
    ``filter → router.route() → annotator.annotate_batch()``.

    Five conditions are compared:

    1. **Single Oracle**          – ``Annotator`` with ``assigned_llm`` fixed to
                                    the most capable model (last in
                                    ``candidate_llms``); bypasses filter/router.
    2. **3-Oracle Majority Vote** – same as (1) but run 3 times; per-sample
                                    majority vote decides the final annotation.
    3. **DataFlow (naive LLM)**   – ``ActiveLearningFilter`` + ``KNNRouter``
                                    + ``Annotator(rag=False)``.
    4. **DataFlow (KB + RAG)**    – same pipeline with ``Annotator(rag=True)``.
    5. **DataFlow (full)**        – same as (4) with ``confidence_threshold=0.75``.

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    oracle_llm:
        LLM instance used by the oracle conditions when no ``llm_dict`` is
        supplied (mock / single-model mode).
    judge_llm:
        Unused; retained for backward compatibility.
    output_dir:
        Directory for SFT JSONL output and (optionally) fine-tuned models.
    skip_finetune:
        When ``True`` (default), downstream fine-tuning is skipped.
    finetune_model_name:
        Qwen model to fine-tune when ``skip_finetune=False``.
    val_data_path:
        Validation JSON path for downstream BLEU/ROUGE-L evaluation.
    force_fallback:
        Passed to ``ActiveLearningFilter`` for offline/CPU mode.
    task:
        Task object (default: ``QATask``).
    candidate_llms:
        Ordered list of LLM identifiers (first = cheapest, last = most
        capable).  When provided, ``llm_dict`` must also be supplied.
        Defaults to the single-entry ``["primary"]`` fallback using
        ``oracle_llm``.
    llm_dict:
        Mapping from LLM identifier to LLM instance.

    Returns
    -------
    List of result dicts; each contains:
    ``condition``, ``num_samples``, ``annotation_f1``, ``annotation_em``,
    ``downstream_bleu``, ``downstream_rouge_l``, ``sft_file``.
    """
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    # Use caller-supplied candidate list or fall back to a single "primary"
    # entry for backward-compatible / mock-mode calls.
    if candidate_llms is None or llm_dict is None:
        candidate_llms = ["primary"]
        llm_dict = {"primary": oracle_llm}

    oracle_name = candidate_llms[-1]  # most capable model

    # ------------------------------------------------------------------ #
    # Lazy shared routing — computed only when needed by a DataFlow cond  #
    # ------------------------------------------------------------------ #
    _dataflow_cond_names = (
        "DataFlow (naive LLM)",
        "DataFlow (KB + RAG)",
        "DataFlow (full pipeline)",
    )
    _routed_cache: List[Any] = []  # populated on first DataFlow call

    def _get_routed() -> List[Dict[str, Any]]:
        if not _routed_cache:
            al_filter = ActiveLearningFilter(
                method="alps",
                budget=1000,
                batch_size=max(2, 1000 // 10),
                force_fallback=force_fallback,
            )
            filtered = al_filter.filter(dataset)
            if force_fallback:
                router: Any = MockKNNRouter(candidate_llms)
            else:
                _knn_kb_path = os.path.join(output_dir, "kb_knn_bootstrap.json")
                _bootstrap_annotator = Annotator(candidate_llms, llm_dict, task=task, kb_path=_knn_kb_path)
                router = KNNRouter(annotator=_bootstrap_annotator, candidate_llms=candidate_llms)
            _routed_cache.append(router.route(filtered))
        return _routed_cache[0]

    # ------------------------------------------------------------------ #
    # Helper: annotate with majority vote across N independent runs       #
    # ------------------------------------------------------------------ #
    def _oracle_annotate(num_oracles: int, suffix: str) -> List[Dict[str, Any]]:
        """Annotate *dataset* ``num_oracles`` times, resolve by majority vote."""
        all_runs: List[List[Dict[str, Any]]] = []
        for i in range(num_oracles):
            kb_path = os.path.join(output_dir, f"kb_{suffix}_{i}.json")
            annotator = Annotator(
                candidate_llms, llm_dict, task=task, kb_path=kb_path,
            )
            all_runs.append(annotator.annotate_batch(dataset, assigned_llm=oracle_name))
        if num_oracles == 1:
            return all_runs[0]
        # Majority vote per sample
        final: List[Dict[str, Any]] = []
        for j in range(len(dataset)):
            votes = [all_runs[i][j].get("annotation", "") for i in range(num_oracles)]
            winner = Counter(votes).most_common(1)[0][0]
            merged = dict(all_runs[0][j])
            merged["annotation"] = winner
            final.append(merged)
        return final

    # ------------------------------------------------------------------ #
    # Define conditions as (name, fn) pairs — fn is only called if needed #
    # ------------------------------------------------------------------ #
    _N = 5

    def _annotate_dataflow_naive() -> List[Dict[str, Any]]:
        annotator = Annotator(
            candidate_llms, llm_dict, task=task, rag=False,
            kb_path=os.path.join(output_dir, "kb_dataflow_naive.json"),
        )
        return annotator.annotate_batch(_get_routed())

    def _annotate_dataflow_rag() -> List[Dict[str, Any]]:
        annotator = Annotator(
            candidate_llms, llm_dict, task=task, rag=True,
            kb_path=os.path.join(output_dir, "kb_dataflow_rag.json"),
        )
        return annotator.annotate_batch(_get_routed())

    def _annotate_dataflow_full() -> List[Dict[str, Any]]:
        annotator = Annotator(
            candidate_llms, llm_dict, task=task, rag=True,
            confidence_threshold=0.75,
            kb_path=os.path.join(output_dir, "kb_dataflow_full.json"),
        )
        return annotator.annotate_batch(_get_routed())

    conditions: List[tuple] = [
        ("Single Oracle",            lambda: _oracle_annotate(1, "single_oracle")),
        ("3-Oracle Majority Vote",   lambda: _oracle_annotate(3, "three_oracle")),
        ("DataFlow (naive LLM)",     _annotate_dataflow_naive),
        ("DataFlow (KB + RAG)",      _annotate_dataflow_rag),
        ("DataFlow (full pipeline)", _annotate_dataflow_full),
    ]

    # ------------------------------------------------------------------ #
    # Evaluate and collect results                                        #
    # ------------------------------------------------------------------ #
    results: List[Dict[str, Any]] = []
    for i, (cond_name, fn) in enumerate(conditions, 1):
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
                    "num_samples": n_lines,
                    "annotation_f1": 0.0,
                    "annotation_em": 0.0,
                    "downstream_bleu": None,
                    "downstream_rouge_l": None,
                    "sft_file": sft_path,
                })
            continue

        annotated = fn()
        quality = evaluate_annotation_quality(annotated)

        sft_path = _sft_output_path(cond_name, output_dir)
        n_written = write_sft_jsonl(annotated, sft_path)

        downstream_bleu: Optional[float] = None
        downstream_rouge_l: Optional[float] = None
        if not skip_finetune:
            model_dir = os.path.join(output_dir, f"qwen_sft_{_safe_name(cond_name)}")
            ft_result = run_downstream_finetune(
                sft_path=sft_path,
                model_name=finetune_model_name,
                model_output_dir=model_dir,
                val_data_path=val_data_path,
            )
            downstream_bleu = ft_result.get("bleu")
            downstream_rouge_l = ft_result.get("rouge_l")

        result: Dict[str, Any] = {
            "condition": cond_name,
            "num_samples": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "downstream_bleu": downstream_bleu,
            "downstream_rouge_l": downstream_rouge_l,
            "sft_file": sft_path,
        }
        _save_condition_result(result, output_dir)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    has_downstream = any(r.get("downstream_bleu") is not None for r in results)
    if has_downstream:
        header = (
            f"{'Condition':<35} {'Ann-F1':>7} {'Ann-EM':>7} "
            f"{'DS-BLEU':>8} {'DS-ROUGE-L':>10} {'#Samples':>9}"
        )
    else:
        header = (
            f"{'Condition':<35} {'Ann-F1':>7} {'Ann-EM':>7} {'#Samples':>9}"
        )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  DataFlow-Annotator vs Oracle — QA Annotation Comparison (Qwen)")
    print("  (conditions 3-5 use ActiveLearningFilter + KNNRouter + Annotator)")
    if not has_downstream:
        print("  (downstream metrics skipped; re-run without --skip-finetune for Qwen SFT eval)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        if has_downstream:
            bleu = r["downstream_bleu"]
            rouge = r["downstream_rouge_l"]
            print(
                f"{r['condition']:<35} "
                f"{r['annotation_f1']:>7.4f} "
                f"{r['annotation_em']:>7.4f} "
                f"{bleu if bleu is not None else 'N/A':>8} "
                f"{rouge if rouge is not None else 'N/A':>10} "
                f"{r['num_samples']:>9}"
            )
        else:
            print(
                f"{r['condition']:<35} "
                f"{r['annotation_f1']:>7.4f} "
                f"{r['annotation_em']:>7.4f} "
                f"{r['num_samples']:>9}"
            )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DataFlow-Annotator vs oracle annotation on QA quality "
            "with optional Qwen LLM fine-tuning as the downstream task."
        )
    )
    parser.add_argument(
        "--samples", type=int, default=10000,
        help="Number of QA samples (default: 200)",
    )
    parser.add_argument(
        "--hotpot-path", default="hotpot_train_v1.json",
        help="HotpotQA training JSON path (default: hotpot_train_v1.json)",
    )
    parser.add_argument(
        "--squad-path", default=None,
        help="SQuAD training JSON path; used only when --hotpot-path is absent",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANDIDATE_LLMS,
        help="Candidate LLMs (first = cheapest, last = oracle). Default: the 3 standard Qwen models.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge LLM (unused with KNNRouter; retained for compatibility)",
    )
    parser.add_argument(
        "--finetune-model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen model to fine-tune on SFT data (default: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--val-path", default=None,
        help="Validation JSON path for downstream BLEU/ROUGE-L evaluation",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/lsc_out",
        help="Directory for SFT JSONL files and model checkpoints (default: /tmp/lsc_out)",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip Qwen fine-tuning; report annotation quality only",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Use mock LLMs for offline testing (no GPU / network required)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic dataset (default: 42)",
    )
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    if args.hotpot_path and os.path.exists(args.hotpot_path):
        dataset = load_hotpot_dataset(args.hotpot_path, max_samples=args.samples)
    else:
        dataset = load_squad_dataset(args.squad_path or "", max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    if args.skip_llm:
        print("Using mock LLMs (--skip-llm)")
        oracle_llm: Any = MockAnnotationLLM()
        judge_llm: Any = MockJudgeLLM()
        force_fallback = True
        _candidate_llms = None
        _llm_dict = None
    else:
        from misc.llm_provider import LocalLLM
        models = args.models
        print(f"Loading LLMs: {models}")
        _llm_dict = {m: LocalLLM(m) for m in models}
        _candidate_llms = models
        oracle_llm = _llm_dict[models[-1]]
        judge_name = args.judge_model or models[-1]
        judge_llm = _llm_dict.get(judge_name) or LocalLLM(judge_name)
        force_fallback = False

    print("Running annotation conditions…")
    results = run_experiment(
        dataset=dataset,
        oracle_llm=oracle_llm,
        judge_llm=judge_llm,
        output_dir=args.output_dir,
        skip_finetune=args.skip_finetune,
        finetune_model_name=args.finetune_model,
        val_data_path=args.val_path,
        force_fallback=force_fallback,
        candidate_llms=_candidate_llms,
        llm_dict=_llm_dict,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "lsc_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
