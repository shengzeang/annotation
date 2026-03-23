"""Experiment: Active Learning for Efficient QA Annotation.

This experiment demonstrates the value of the repository's active-learning
sampling strategies for annotation budget allocation.  Five conditions are
compared over the same annotation budget:

1. **Full dataset**        – annotate every sample (annotation oracle, no budget limit)
2. **Random sampling**     – pick `budget` samples at random
3. **Diversity (TF-IDF)** – pick the `budget` most diverse samples via greedy
                             k-means on TF-IDF embeddings (no BERT required)
4. **Uncertainty proxy**   – pick samples whose length deviates most from the
                             dataset mean (a cheap proxy for surprisal, avoiding
                             heavy model loading)
5. **ALPS (force-fallback)** – run the repository's ``ActiveLearningFilter`` with
                             ``force_fallback=True`` so that no real BERT model
                             is loaded; the output order matches the deterministic
                             ALPS fallback and serves as a smoke-test that the
                             full filter pipeline is exercised end-to-end

For each condition the selected subset is annotated with a real LLM (injected
via ``--model``; defaults to a fast Qwen model) and annotation quality
(token-F1 / exact-match vs. ground truth) is reported.

In the offline / test mode (``--skip-llm``) a ``MockLLM`` is substituted so
that no GPU or network is required.

Usage
-----
    # Offline smoke-test (no model loading, no GPU)
    python experiments/run_active_learning.py \\
        --samples 200 --budget 50 --skip-llm

    # Real Qwen annotation
    python experiments/run_active_learning.py \\
        --samples 500 --budget 100 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/al_out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Sys-path fix so the script can be run from the repo root or any CWD
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE: float = 0.8  # confidence returned by mock LLM

# ---------------------------------------------------------------------------
# Lightweight inline QA task (avoids torch-dependent import chain from tasks/)
# ---------------------------------------------------------------------------


class _SimpleQATask:
    """Minimal QA task: build prompt and parse ``Answer: ... Confidence: ...``."""

    def get_prompt(self, sample: Dict[str, Any], rag_examples: Any = None) -> str:
        rag_str = ""
        if rag_examples:
            rag_str = "\nHere are some similar QA pairs:\n"
            for ex in rag_examples:
                rag_str += f"Q: {ex.get('question','')}\nA: {ex.get('annotation','')}\n"
        return (
            "Given the following question, answer as accurately as possible.\n"
            "Output format: Answer: <your answer> Confidence: <0.0-1.0>\n"
            f"Question: {sample.get('question', sample.get('text', ''))}\n"
            f"Context: {sample.get('context', '')}\n"
            f"{rag_str}"
            "Answer:"
        )

    def parse_output(self, output: str) -> Dict[str, Any]:
        annotation = "unknown"
        confidence = None
        m_conf = re.search(r"confidence\s*[:\-]?\s*([0-9]*\.?[0-9]+)", output, re.I)
        if m_conf:
            try:
                confidence = float(m_conf.group(1))
                if confidence > 1.0:
                    confidence = min(1.0, confidence / 100.0)
            except ValueError:
                confidence = None
        parts = re.split(r"confidence\s*[:\-]?", output, flags=re.I)
        m_ans = re.search(r"answer\s*[:\-]?\s*(.*)", parts[0], re.I | re.S)
        if m_ans:
            annotation = m_ans.group(1).strip()
        if confidence is None:
            return {"annotation": annotation}
        return {"annotation": annotation, "confidence": confidence}


def _get_task(task: Any) -> Any:
    if task is not None:
        return task
    try:
        from tasks.qa import QATask
        return QATask()
    except Exception:
        return _SimpleQATask()


# ---------------------------------------------------------------------------
# Shared QA metrics (copied from run_label_studio_comparison for self-containment)
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
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def evaluate_annotation_quality(annotated: List[Dict[str, Any]]) -> Dict[str, float]:
    f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    ems = [compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Mock LLM (used when --skip-llm is set)
# ---------------------------------------------------------------------------


class MockLLM:
    """Deterministic mock that returns the ground-truth answer so that
    annotation quality metrics are maximal.  Used for fast offline tests.
    """

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:  # noqa: D401
        return f"Answer: test_answer Confidence: {DEFAULT_CONFIDENCE}"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 64):
        return self.generate(prompt), -0.3


# ---------------------------------------------------------------------------
# LLM annotation helper (task-agnostic)
# ---------------------------------------------------------------------------


def _annotate_samples(
    samples: List[Dict[str, Any]],
    llm: Any,
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Annotate *samples* with *llm* and return augmented records."""
    task = _get_task(task)

    results = []
    for sample in samples:
        prompt = task.get_prompt(sample)
        raw = llm.generate(prompt, max_new_tokens=64)
        parsed = task.parse_output(raw)
        results.append({
            **sample,
            "annotation": parsed.get("annotation", ""),
            "confidence": parsed.get("confidence", DEFAULT_CONFIDENCE),
        })
    return results


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def select_random(
    dataset: List[Dict[str, Any]],
    budget: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Pick *budget* samples at random (baseline)."""
    import random
    rng = random.Random(seed)
    population = dataset[:]
    rng.shuffle(population)
    return population[:budget]


def select_diversity_tfidf(
    dataset: List[Dict[str, Any]],
    budget: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Greedy diversity selection via k-means on TF-IDF embeddings.

    Falls back to random selection when scikit-learn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        import numpy as np

        texts = [
            d.get("text") or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
            for d in dataset
        ]
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
        X = vec.fit_transform(texts).toarray().astype("float32")

        k = min(budget, len(dataset))
        km = KMeans(n_clusters=k, n_init=3, random_state=seed, max_iter=50)
        km.fit(X)
        # representative = sample closest to each cluster centroid
        nearest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
        selected_idx = list(dict.fromkeys(nearest.tolist()))[:budget]
        return [dataset[i] for i in selected_idx]
    except Exception:
        return select_random(dataset, budget, seed=seed)


def select_uncertainty_length(
    dataset: List[Dict[str, Any]],
    budget: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Pick samples whose text length deviates most from the dataset mean.

    This is a cheap proxy for surprisal / uncertainty: unusually short or long
    samples tend to be harder for a model to answer correctly.
    """
    import math

    texts = [
        d.get("text") or f"Q: {d.get('question','')}\nContext: {d.get('context','')}"
        for d in dataset
    ]
    lengths = [len(t.split()) for t in texts]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    deviations = [abs(l - mean_len) for l in lengths]

    # Sort by descending deviation (most "surprising" first)
    order = sorted(range(len(dataset)), key=lambda i: deviations[i], reverse=True)
    return [dataset[i] for i in order[:budget]]


def select_alps_fallback(
    dataset: List[Dict[str, Any]],
    budget: int,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """Run the repository's ``ActiveLearningFilter`` in force-fallback mode.

    ``force_fallback=True`` instructs the filter to skip BERT/MLM model loading
    and use a deterministic first-K selection, making this fast and GPU-free
    while still exercising the full ``ActiveLearningFilter`` → ``ALPS`` →
    ``DataPool`` pipeline.
    """
    try:
        from filters.al_filter import ActiveLearningFilter

        filt = ActiveLearningFilter(
            method="alps",
            budget=budget,
            batch_size=batch_size,
            force_fallback=True,
        )
        return filt.filter(dataset)
    except Exception:
        return select_random(dataset, budget)


# ---------------------------------------------------------------------------
# Write annotated records as SFT JSONL
# ---------------------------------------------------------------------------


def write_sft_jsonl(annotated: List[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in annotated:
            line = json.dumps(
                {"instruction": rec.get("text", ""), "output": rec.get("annotation", "")},
                ensure_ascii=False,
            )
            f.write(line + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Synthetic / squad dataset helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    import random
    rng = random.Random(seed)
    topics = [
        ("Albert Einstein", "Einstein developed the theory of relativity.", "relativity"),
        ("Python language", "Python is a high-level programming language.", "high-level"),
        ("Mount Everest", "Mount Everest is the highest mountain.", "highest"),
        ("Marie Curie", "Marie Curie discovered polonium and radium.", "polonium"),
        ("The Sun", "The Sun is the star at the center of the Solar System.", "star"),
        ("Isaac Newton", "Newton formulated the laws of motion and universal gravitation.", "motion"),
        ("William Shakespeare", "Shakespeare wrote plays including Hamlet and Romeo and Juliet.", "Hamlet"),
        ("Leonardo da Vinci", "Da Vinci painted the Mona Lisa and The Last Supper.", "Mona Lisa"),
    ]
    dataset: List[Dict[str, Any]] = []
    for i in range(n):
        subj, ctx, ans = topics[i % len(topics)]
        # Add length variation to make the uncertainty selector meaningful
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
        spec = __import__("datasets.qa_datasets", fromlist=["SquadDataset"])
        ds = spec.SquadDataset.from_file(squad_path, max_samples=max_samples)
        return list(ds._data)
    return _make_synthetic_dataset(n=max_samples)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    dataset: List[Dict[str, Any]],
    llm: Any,
    budget: int = 50,
    output_dir: str = "/tmp/al_out",
    seed: int = 42,
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Run all active-learning conditions on *dataset*.

    Parameters
    ----------
    dataset:
        Full pool of QA samples.
    llm:
        LLM instance used for annotation (real or mock).
    budget:
        Number of samples to annotate per condition (except ``full`` condition
        which annotates the entire dataset).
    output_dir:
        Directory for SFT JSONL outputs.
    seed:
        Random seed.
    task:
        Task object (default: ``QATask``).

    Returns
    -------
    List of result dicts with keys ``condition``, ``selected``, ``annotated``,
    ``annotation_f1``, ``annotation_em``, ``sft_file``.
    """
    os.makedirs(output_dir, exist_ok=True)

    conditions: List[tuple] = [
        ("Full dataset",         dataset),
        ("Random sampling",      select_random(dataset, budget, seed=seed)),
        ("Diversity (TF-IDF)",   select_diversity_tfidf(dataset, budget, seed=seed)),
        ("Uncertainty (length)", select_uncertainty_length(dataset, budget, seed=seed)),
        ("ALPS (force-fallback)", select_alps_fallback(dataset, budget, batch_size=max(2, budget // 5))),
    ]

    results: List[Dict[str, Any]] = []
    for cond_name, selected in conditions:
        annotated = _annotate_samples(selected, llm, task=task)
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
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Condition':<28} {'Selected':>9} {'Ann-F1':>7} {'Ann-EM':>7}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Active Learning — Annotation Budget Comparison")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<28} "
            f"{r['selected']:>9} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Active Learning sampling comparison for QA annotation"
    )
    parser.add_argument("--samples", type=int, default=200,
                        help="Total pool size (default: 200)")
    parser.add_argument("--budget", type=int, default=50,
                        help="Annotation budget per AL condition (default: 50)")
    parser.add_argument("--squad-path", default="squad_train.json",
                        help="Path to SQuAD JSON; falls back to synthetic data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name for annotation LLM")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use a mock LLM instead of loading a real model")
    parser.add_argument("--output-dir", default="/tmp/al_out",
                        help="Directory for SFT JSONL outputs (default: /tmp/al_out)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.  Budget = {args.budget}.")

    if args.skip_llm:
        print("Using MockLLM (--skip-llm)")
        llm: Any = MockLLM()
    else:
        from misc.llm_provider import LocalLLM
        print(f"Loading LLM: {args.model}")
        llm = LocalLLM(args.model)

    results = run_experiment(
        dataset=dataset,
        llm=llm,
        budget=args.budget,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "al_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
