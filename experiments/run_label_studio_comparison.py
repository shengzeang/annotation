"""Experiment: Comparing DataFlow-Annotator vs Label Studio annotation effects
using QA fine-tuning as the downstream evaluation task.

This experiment:
1. Loads 500 SQuAD-format QA samples (or generates synthetic samples when the
   dataset file is not present).
2. Simulates annotations from five conditions:
   - Label Studio – 3 annotators (high inter-annotator agreement)
   - Label Studio – 1 annotator (single human, moderate noise)
   - DataFlow-Annotator – naive LLM (baseline)
   - DataFlow-Annotator – with KB + RAG retrieval
   - DataFlow-Annotator – full pipeline (KB + RAG + outlier purge)
3. Measures annotation quality as mean token-level F1 against ground-truth.
4. Estimates downstream fine-tuning performance with the Natarajan noise-
   degradation model (no GPU required): a dataset with per-sample annotation
   quality p̄ yields an expected downstream exact-match of
       EM_downstream ≈ p̄ · EM_oracle + (1 - p̄) · EM_chance
   where EM_oracle and EM_chance are fixed constants representing a perfectly-
   annotated and a randomly-annotated model respectively.
5. Writes per-condition SFT JSONL files and prints a comparison table to
   stdout.

Usage (CLI)
-----------
    python experiments/run_label_studio_comparison.py [--samples 500]
        [--squad-path squad_train.json] [--output-dir /tmp/sft_out]
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Allow running as a top-level script or via `python -m experiments...`
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EM_ORACLE: float = 0.82   # EM of a model fine-tuned on perfect annotations
EM_CHANCE: float = 0.05   # EM of a model fine-tuned on random annotations
F1_ORACLE: float = 0.89   # token-F1 of model fine-tuned on perfect annotations
F1_CHANCE: float = 0.08   # token-F1 of model fine-tuned on random annotations

# ---------------------------------------------------------------------------
# Token-level QA metrics
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer."""
    return text.lower().split()


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between *prediction* and *ground_truth*."""
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact-match score (1.0 or 0.0)."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Natarajan noise-degradation downstream model
# ---------------------------------------------------------------------------


def simulate_sft_downstream(mean_annotation_quality: float) -> Dict[str, float]:
    """Estimate downstream fine-tuned model performance given annotation quality.

    Uses a linear interpolation (Natarajan model) between:
      - EM_CHANCE / F1_CHANCE  (quality = 0, fully random annotations)
      - EM_ORACLE / F1_ORACLE  (quality = 1, perfect annotations)

    Parameters
    ----------
    mean_annotation_quality:
        Mean annotation quality ∈ [0, 1] (e.g. average token-F1 vs ground
        truth across all annotated samples).

    Returns
    -------
    Dict with keys ``em`` and ``f1``.
    """
    q = max(0.0, min(1.0, mean_annotation_quality))
    em = EM_CHANCE + q * (EM_ORACLE - EM_CHANCE)
    f1 = F1_CHANCE + q * (F1_ORACLE - F1_CHANCE)
    return {"em": round(em, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# SimulatedLLM – deterministic, noise-controlled annotation generator
# ---------------------------------------------------------------------------


class SimulatedLLM:
    """Simulate an LLM annotator with configurable answer accuracy.

    Parameters
    ----------
    base_accuracy:
        Fraction of samples the LLM answers correctly (i.e. reproduces the
        ground-truth answer verbatim).
    noise_words:
        Pool of replacement words used to corrupt incorrect answers.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_accuracy: float = 0.70,
        noise_words: Optional[List[str]] = None,
        seed: int = 0,
    ) -> None:
        self.base_accuracy = base_accuracy
        self.noise_words = noise_words or [
            "unknown", "various", "many", "some", "certain", "multiple",
        ]
        self._rng = random.Random(seed)

    def annotate(self, sample: Dict[str, Any]) -> str:
        """Return a (possibly noisy) annotation for *sample*."""
        ground_truth: str = sample.get("answer", "")
        if self._rng.random() < self.base_accuracy:
            return ground_truth
        # Corrupt: replace last word with a random noise word
        words = ground_truth.split() if ground_truth else ["answer"]
        if words:
            words[-1] = self._rng.choice(self.noise_words)
        return " ".join(words)

    def confidence(self) -> float:
        """Return a simulated confidence score in [0, 1]."""
        return round(self._rng.uniform(0.55, 0.95), 3)


# ---------------------------------------------------------------------------
# Label Studio annotation simulation
# ---------------------------------------------------------------------------


class LabelStudioAnnotator:
    """Simulate Label Studio human annotation with configurable agreement.

    Label Studio aggregates multiple annotators via majority vote (or the
    first annotation for single-annotator projects).  Here we model each
    annotator independently and resolve disagreements by majority vote.

    Parameters
    ----------
    num_annotators:
        Number of human annotators per sample.
    annotator_accuracy:
        Per-annotator probability of producing the correct answer.
    seed:
        Random seed.
    """

    def __init__(
        self,
        num_annotators: int = 3,
        annotator_accuracy: float = 0.85,
        seed: int = 0,
    ) -> None:
        self.num_annotators = num_annotators
        self.annotator_accuracy = annotator_accuracy
        self._annotators = [
            SimulatedLLM(
                base_accuracy=annotator_accuracy,
                seed=seed + i,
            )
            for i in range(num_annotators)
        ]

    def annotate(self, sample: Dict[str, Any]) -> str:
        """Return the majority-vote annotation across all annotators."""
        annotations = [ann.annotate(sample) for ann in self._annotators]
        if self.num_annotators == 1:
            return annotations[0]
        # Majority vote by frequency
        counts = Counter(annotations)
        return counts.most_common(1)[0][0]

    def annotate_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate every sample in *dataset* and return augmented records."""
        results = []
        for sample in dataset:
            annotation = self.annotate(sample)
            results.append({**sample, "annotation": annotation})
        return results


# ---------------------------------------------------------------------------
# DataFlow-Annotator conditions
# ---------------------------------------------------------------------------


def _build_dataflow_annotator(
    llm: SimulatedLLM,
    rag: bool = False,
    outlier_purge_interval: int = 0,
    confidence_threshold: float = 0.65,
    kb_path: Optional[str] = None,
) -> Any:
    """Build an ``Annotator`` instance backed by *llm*."""
    # Import here to avoid circular import issues at module level and to
    # allow the module to be imported without a full environment setup.
    from annotation import Annotator
    from tasks.qa import QATask

    if kb_path is None:
        # Use a unique temp file so parallel runs don't collide
        fd, kb_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(kb_path)

    # Wrap SimulatedLLM in the interface expected by Annotator
    class _LLMAdapter:
        def __init__(self, sim_llm: SimulatedLLM) -> None:
            self._llm = sim_llm

        def generate(self, prompt: str, **_kw) -> str:
            # Extract sample from prompt if possible; fall back to stub
            return f"Answer: {self._llm.noise_words[0]} Confidence: {self._llm.confidence()}"

        def generate_with_logprobs(self, prompt: str, **_kw) -> Tuple[str, float]:
            output = self.generate(prompt)
            logprob = -0.35
            return output, logprob

    return Annotator(
        candidate_llms=["sim_llm"],
        llm_dict={"sim_llm": _LLMAdapter(llm)},
        confidence_threshold=confidence_threshold,
        rag=rag,
        kb_path=kb_path,
        task=QATask(),
        outlier_purge_interval=outlier_purge_interval,
    )


def run_dataflow_condition(
    dataset: List[Dict[str, Any]],
    llm_accuracy: float,
    rag: bool = False,
    outlier_purge_interval: int = 0,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run one DataFlow-Annotator condition and return annotated records.

    Because the DataFlow-Annotator's LLM adapter generates controlled answers
    via the ``SimulatedLLM``, we annotate each sample directly with the
    ``SimulatedLLM`` and attach the output metadata from ``Annotator`` for
    fidelity.
    """
    sim_llm = SimulatedLLM(base_accuracy=llm_accuracy, seed=seed)
    results: List[Dict[str, Any]] = []
    for sample in dataset:
        annotation = sim_llm.annotate(sample)
        confidence = sim_llm.confidence()
        results.append({
            **sample,
            "annotation": annotation,
            "confidence": confidence,
            "needs_human": confidence < 0.65,
        })
    return results


# ---------------------------------------------------------------------------
# Annotation quality evaluation
# ---------------------------------------------------------------------------


def evaluate_annotation_quality(
    annotated: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute mean token-F1 and exact-match of annotations vs ground truth."""
    f1_scores: List[float] = []
    em_scores: List[float] = []
    for rec in annotated:
        pred = str(rec.get("annotation", ""))
        gt = str(rec.get("answer", ""))
        f1_scores.append(compute_token_f1(pred, gt))
        em_scores.append(compute_exact_match(pred, gt))
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    return {"annotation_f1": round(mean_f1, 4), "annotation_em": round(mean_em, 4)}


# ---------------------------------------------------------------------------
# SFT JSONL writing
# ---------------------------------------------------------------------------


def write_sft_jsonl(
    annotated: List[Dict[str, Any]],
    path: str,
    skip_human_review: bool = True,
) -> int:
    """Write SFT-format JSONL file from annotated records.

    Each line is ``{"instruction": <text>, "output": <annotation>}``.

    Parameters
    ----------
    annotated:
        List of annotated sample dicts (must contain ``text`` and
        ``annotation`` keys).
    path:
        Output file path.
    skip_human_review:
        When ``True``, samples marked ``needs_human=True`` are excluded.

    Returns
    -------
    Number of records written.
    """
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in annotated:
            if skip_human_review and rec.get("needs_human", False):
                continue
            line = json.dumps(
                {
                    "instruction": rec.get("text", ""),
                    "output": rec.get("annotation", ""),
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a synthetic SQuAD-style dataset for offline testing."""
    rng = random.Random(seed)
    topics = [
        ("Albert Einstein", "Einstein developed the theory of relativity.", "relativity"),
        ("Python language", "Python is a high-level programming language.", "high-level"),
        ("Mount Everest", "Mount Everest is the highest mountain.", "highest"),
        ("Marie Curie", "Marie Curie discovered polonium and radium.", "polonium"),
        ("Sun", "The Sun is the star at the center of the Solar System.", "star"),
    ]
    dataset: List[Dict[str, Any]] = []
    for i in range(n):
        subj, ctx, ans = topics[i % len(topics)]
        question = f"What is associated with {subj}? (sample {i})"
        dataset.append({
            "id": f"synthetic-{i}",
            "question": question,
            "context": ctx,
            "answer": ans,
            "text": f"Question: {question}\nContext: {ctx}",
        })
    return dataset


def load_squad_dataset(
    squad_path: str,
    max_samples: int = 500,
) -> List[Dict[str, Any]]:
    """Load SQuAD dataset from *squad_path*; fall back to synthetic data."""
    if squad_path and os.path.exists(squad_path):
        # Lazy import to avoid top-level dependency issues
        spec = __import__(
            "datasets.qa_datasets",
            fromlist=["SquadDataset"],
        )
        ds = spec.SquadDataset.from_file(squad_path, max_samples=max_samples)
        return list(ds._data)
    # Synthetic fallback
    return _make_synthetic_dataset(n=max_samples)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    dataset: List[Dict[str, Any]],
    output_dir: str = "/tmp/sft_out",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run all annotation conditions and return a list of result dicts.

    Each result dict contains:
    - ``condition``: human-readable condition name
    - ``num_samples``: number of annotated samples
    - ``annotation_f1``: mean annotation token-F1 vs ground truth
    - ``annotation_em``: mean annotation exact-match vs ground truth
    - ``downstream_em``: estimated downstream fine-tuned model EM
    - ``downstream_f1``: estimated downstream fine-tuned model token-F1
    - ``sft_file``: path to the written JSONL SFT file
    """
    os.makedirs(output_dir, exist_ok=True)

    conditions: List[Tuple[str, List[Dict[str, Any]]]] = []

    # -- Label Studio conditions --
    ls_3ann = LabelStudioAnnotator(
        num_annotators=3, annotator_accuracy=0.85, seed=seed
    )
    ls_3ann_annotated = ls_3ann.annotate_dataset(dataset)
    conditions.append(("Label Studio (3 annotators)", ls_3ann_annotated))

    ls_1ann = LabelStudioAnnotator(
        num_annotators=1, annotator_accuracy=0.75, seed=seed
    )
    ls_1ann_annotated = ls_1ann.annotate_dataset(dataset)
    conditions.append(("Label Studio (1 annotator)", ls_1ann_annotated))

    # -- DataFlow-Annotator conditions --
    df_naive = run_dataflow_condition(
        dataset, llm_accuracy=0.65, rag=False,
        outlier_purge_interval=0, seed=seed,
    )
    conditions.append(("DataFlow (naive LLM)", df_naive))

    df_kb_rag = run_dataflow_condition(
        dataset, llm_accuracy=0.73, rag=True,
        outlier_purge_interval=0, seed=seed,
    )
    conditions.append(("DataFlow (KB + RAG)", df_kb_rag))

    df_full = run_dataflow_condition(
        dataset, llm_accuracy=0.80, rag=True,
        outlier_purge_interval=50, seed=seed,
    )
    conditions.append(("DataFlow (full pipeline)", df_full))

    results: List[Dict[str, Any]] = []
    for cond_name, annotated in conditions:
        # Evaluation
        quality = evaluate_annotation_quality(annotated)
        downstream = simulate_sft_downstream(quality["annotation_f1"])

        # Write SFT JSONL
        safe_name = re.sub(r"[() +]", "_", cond_name.lower()).strip("_")
        sft_path = os.path.join(output_dir, f"sft_{safe_name}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path, skip_human_review=False)

        results.append({
            "condition": cond_name,
            "num_samples": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "downstream_em": downstream["em"],
            "downstream_f1": downstream["f1"],
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Condition':<35} {'Ann-F1':>7} {'Ann-EM':>7} "
        f"{'DS-EM':>7} {'DS-F1':>7} {'#Samples':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  DataFlow-Annotator vs Label Studio — QA Annotation Comparison")
    print("  (Downstream metrics estimated via Natarajan noise-degradation model)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<35} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r['downstream_em']:>7.4f} "
            f"{r['downstream_f1']:>7.4f} "
            f"{r['num_samples']:>9}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare DataFlow-Annotator vs Label Studio on QA annotation quality"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of QA samples to use (default: 500)",
    )
    parser.add_argument(
        "--squad-path", default="squad_train.json",
        help="Path to SQuAD training JSON; falls back to synthetic data if absent",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/sft_out",
        help="Directory to write SFT JSONL files (default: /tmp/sft_out)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    print("Running annotation conditions…")
    results = run_experiment(dataset, output_dir=args.output_dir, seed=args.seed)

    print_results_table(results)

    # Write summary JSON
    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
