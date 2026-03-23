"""Experiment: Comparing DataFlow-Annotator vs oracle-annotator annotation effects
using Qwen LLM fine-tuning as the downstream evaluation task.

This experiment:
1. Loads SQuAD-format QA samples (or generates synthetic samples when the
   dataset file is not present).
2. Annotates data with five conditions using real LLM calls:
   - Single Oracle            – one call to a large Qwen model per sample
   - 3-Oracle Majority Vote   – three independent calls, majority vote decides
   - DataFlow (naive LLM)     – Qwen-7B annotation without KB/RAG
   - DataFlow (KB + RAG)      – Qwen-7B with in-context KB retrieval
   - DataFlow (full pipeline) – Qwen-7B with KB retrieval + confidence filter
3. Measures annotation quality (token-level F1, exact-match vs ground truth).
4. Optionally fine-tunes a Qwen model on each condition's SFT JSONL using
   ``misc/evaluate.py`` and reports downstream BLEU / ROUGE-L (requires GPU).
5. Writes per-condition SFT JSONL files and prints a comparison table.

Usage (CLI)
-----------
    # Annotation only (no GPU required)
    python experiments/run_label_studio_comparison.py \\
        --samples 500 --skip-finetune

    # Full pipeline with Qwen fine-tuning (requires GPU)
    python experiments/run_label_studio_comparison.py \\
        --samples 500 \\
        --squad-path squad_train.json \\
        --oracle-model  Qwen/Qwen2.5-72B-Instruct \\
        --dataflow-model Qwen/Qwen2.5-7B-Instruct \\
        --finetune-model Qwen/Qwen2.5-7B-Instruct \\
        --val-path validation.json \\
        --output-dir /tmp/sft_out \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running as a top-level script or via `python -m experiments...`
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE: float = 0.5  # fallback when LLM output has no parseable confidence

# ---------------------------------------------------------------------------
# Token-level QA metrics (no heavy dependencies)
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
    """Return 1.0 if *prediction* matches *ground_truth* (case-insensitive, whitespace-trimmed), else 0.0."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Oracle annotator – wraps a real LLM (e.g. Qwen/Qwen2.5-72B-Instruct)
# ---------------------------------------------------------------------------


class OracleAnnotator:
    """Oracle annotator that calls a real LLM for every sample.

    Two modes:
    - ``num_oracles=1``  → "Single Oracle": one LLM call per sample.
    - ``num_oracles=3``  → "3-Oracle Majority Vote": three independent calls
      whose answers are resolved by majority vote.

    The oracle LLM should be a high-capability model
    (e.g. ``Qwen/Qwen2.5-72B-Instruct``) loaded via
    ``misc.llm_provider.LocalLLM`` or an API-backed equivalent.

    Parameters
    ----------
    llm:
        Any object with a ``generate(prompt, max_new_tokens=…) -> str`` method
        (satisfies ``misc.llm_provider.LLMBase``).
    num_oracles:
        Number of independent LLM calls to make per sample (1 or 3).
    task:
        Task object providing ``get_prompt`` / ``parse_output``.  Defaults to
        a lazily-loaded ``tasks.qa.QATask`` when ``None``.
    """

    def __init__(
        self,
        llm: Any,
        num_oracles: int = 1,
        task: Any = None,
    ) -> None:
        self.llm = llm
        self.num_oracles = num_oracles
        self._task = task  # resolved lazily to avoid heavy imports at module level

    @property
    def task(self) -> Any:
        if self._task is None:
            from tasks.qa import QATask  # lazy import
            self._task = QATask()
        return self._task

    def annotate(self, sample: Dict[str, Any]) -> str:
        """Call the LLM ``num_oracles`` times and return the resolved answer."""
        prompt = self.task.get_prompt(sample)
        annotations: List[str] = []
        for _ in range(self.num_oracles):
            raw = self.llm.generate(prompt, max_new_tokens=64)
            parsed = self.task.parse_output(raw)
            annotations.append(parsed.get("annotation", ""))
        if self.num_oracles == 1:
            return annotations[0]
        # Majority vote across oracle calls
        counts = Counter(annotations)
        return counts.most_common(1)[0][0]

    def annotate_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate every sample in *dataset* and return augmented records."""
        results = []
        for sample in tqdm(dataset, desc=f"Oracle ×{self.num_oracles}", unit="sample", leave=False):
            annotation = self.annotate(sample)
            results.append({**sample, "annotation": annotation})
        return results


# ---------------------------------------------------------------------------
# DataFlow-Annotator condition runner (real LLM, no heavy framework deps)
# ---------------------------------------------------------------------------


def run_dataflow_condition(
    dataset: List[Dict[str, Any]],
    llm: Any,
    rag: bool = False,
    confidence_threshold: float = 0.65,
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Annotate *dataset* with a DataFlow-style pipeline using a real LLM.

    Parameters
    ----------
    dataset:
        List of sample dicts with at least ``question``, ``context``, and
        ``answer`` keys.
    llm:
        Real LLM instance (``misc.llm_provider.LLMBase`` or compatible).
        Recommended: ``LocalLLM("Qwen/Qwen2.5-7B-Instruct")``.
    rag:
        When ``True``, previously high-confidence annotations are stored in a
        simple in-memory knowledge base and retrieved as few-shot examples for
        subsequent samples (keyword-Jaccard retrieval, no heavy dependencies).
    confidence_threshold:
        Minimum LLM confidence score for a sample to be auto-accepted;
        samples below this score are flagged ``needs_human=True``.
    task:
        Task object (defaults to lazily-loaded ``tasks.qa.QATask``).

    Returns
    -------
    List of dicts – each input sample augmented with ``annotation``,
    ``confidence``, and ``needs_human`` keys.
    """
    if task is None:
        from tasks.qa import QATask  # lazy import
        task = QATask()

    kb: List[Dict[str, Any]] = []  # simple in-memory KB for RAG retrieval

    results: List[Dict[str, Any]] = []
    rag_label = "on" if rag else "off"
    for sample in tqdm(dataset, desc=f"DataFlow [rag={rag_label}, thr={confidence_threshold:.2f}]", unit="sample", leave=False):
        # RAG: retrieve similar examples from the in-memory KB
        rag_examples: List[Dict[str, Any]] = []
        if rag and kb:
            q_toks = set(sample.get("question", "").lower().split())
            scored = []
            for entry in kb:
                e_toks = set(entry.get("question", "").lower().split())
                union = q_toks | e_toks
                score = len(q_toks & e_toks) / len(union) if union else 0.0
                scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            rag_examples = [e for _, e in scored[:3]]

        prompt = task.get_prompt(sample, rag_examples if rag else None)
        raw_output = llm.generate(prompt, max_new_tokens=64)
        parsed = task.parse_output(raw_output)

        annotation = parsed.get("annotation", "")
        confidence = parsed.get("confidence", DEFAULT_CONFIDENCE)
        if not isinstance(confidence, (int, float)):
            confidence = DEFAULT_CONFIDENCE

        needs_human = float(confidence) < confidence_threshold
        record = {
            **sample,
            "annotation": annotation,
            "confidence": float(confidence),
            "needs_human": needs_human,
        }
        results.append(record)

        # Admit high-confidence answers to the RAG KB
        if rag and not needs_human:
            kb.append(record)

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
# SFT JSONL writing (shared by all conditions)
# ---------------------------------------------------------------------------


def write_sft_jsonl(
    annotated: List[Dict[str, Any]],
    path: str,
    skip_human_review: bool = True,
) -> int:
    """Write SFT-format JSONL for fine-tuning Qwen (or any causal LM).

    Each line: ``{"instruction": <prompt text>, "output": <annotation>}``.

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
# Downstream fine-tuning with Qwen via misc/evaluate.py
# ---------------------------------------------------------------------------


def run_downstream_finetune(
    sft_path: str,
    model_name: str,
    model_output_dir: str,
    val_data_path: Optional[str] = None,
    epochs: int = 2,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Fine-tune *model_name* (a Qwen model) on *sft_path* and evaluate.

    Delegates to ``misc.evaluate.finetune_sft`` and ``misc.evaluate.evaluate``
    so that all Qwen-specific fine-tuning logic lives in one place.

    Parameters
    ----------
    sft_path:
        Path to the SFT JSONL file produced by :func:`write_sft_jsonl`.
    model_name:
        HuggingFace model identifier for the Qwen model to fine-tune
        (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    model_output_dir:
        Directory where the fine-tuned model checkpoint is saved.
    val_data_path:
        Path to a JSON validation file (list of dicts with ``question``,
        ``context``, ``annotation`` keys) used for BLEU / ROUGE-L evaluation.
        When ``None``, evaluation is skipped.
    epochs:
        Number of training epochs.
    batch_size:
        Per-device training batch size.

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
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a synthetic SQuAD-style dataset for offline testing."""
    import random
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
        spec = __import__(
            "datasets.qa_datasets",
            fromlist=["SquadDataset"],
        )
        ds = spec.SquadDataset.from_file(squad_path, max_samples=max_samples)
        return list(ds._data)
    return _make_synthetic_dataset(n=max_samples)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    dataset: List[Dict[str, Any]],
    oracle_llm: Any,
    dataflow_llm: Any,
    output_dir: str = "/tmp/sft_out",
    skip_finetune: bool = True,
    finetune_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    val_data_path: Optional[str] = None,
    oracle_task: Any = None,
    dataflow_task: Any = None,
) -> List[Dict[str, Any]]:
    """Run all annotation conditions and return a list of result dicts.

    Five conditions are compared:

    1. **Single Oracle** – ``OracleAnnotator(oracle_llm, num_oracles=1)``
    2. **3-Oracle Majority Vote** – ``OracleAnnotator(oracle_llm, num_oracles=3)``
    3. **DataFlow (naive LLM)** – ``run_dataflow_condition(rag=False)``
    4. **DataFlow (KB + RAG)** – ``run_dataflow_condition(rag=True)``
    5. **DataFlow (full pipeline)** – ``run_dataflow_condition(rag=True)`` with a
       stricter confidence threshold (tighter quality gate)

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    oracle_llm:
        Real LLM instance used for oracle conditions (high-capability model
        such as ``Qwen/Qwen2.5-72B-Instruct``).
    dataflow_llm:
        Real LLM instance used for DataFlow conditions (e.g.
        ``Qwen/Qwen2.5-7B-Instruct``).
    output_dir:
        Directory for SFT JSONL output and (optionally) fine-tuned models.
    skip_finetune:
        When ``True`` (default), downstream fine-tuning is skipped and
        ``downstream_bleu`` / ``downstream_rouge_l`` are ``None``.
    finetune_model_name:
        Qwen model to fine-tune when ``skip_finetune=False``.
    val_data_path:
        Validation JSON path for downstream evaluation.
    oracle_task:
        Custom task for oracle conditions (default: ``QATask``).
    dataflow_task:
        Custom task for DataFlow conditions (default: ``QATask``).

    Returns
    -------
    List of result dicts; each contains:
    ``condition``, ``num_samples``, ``annotation_f1``, ``annotation_em``,
    ``downstream_bleu``, ``downstream_rouge_l``, ``sft_file``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build conditions: (name, annotated_records)
    conditions: List[tuple] = []
    _N = 5  # total number of conditions

    # -- Oracle conditions (single / 3-oracle majority vote) --
    print(f"\n[1/{_N}] Running condition: Single Oracle …")
    single_oracle = OracleAnnotator(oracle_llm, num_oracles=1, task=oracle_task)
    conditions.append(
        ("Single Oracle", single_oracle.annotate_dataset(dataset))
    )

    print(f"\n[2/{_N}] Running condition: 3-Oracle Majority Vote …")
    three_oracle = OracleAnnotator(oracle_llm, num_oracles=3, task=oracle_task)
    conditions.append(
        ("3-Oracle Majority Vote", three_oracle.annotate_dataset(dataset))
    )

    # -- DataFlow conditions --
    print(f"\n[3/{_N}] Running condition: DataFlow (naive LLM) …")
    conditions.append((
        "DataFlow (naive LLM)",
        run_dataflow_condition(dataset, dataflow_llm, rag=False,
                               confidence_threshold=0.65, task=dataflow_task),
    ))
    print(f"\n[4/{_N}] Running condition: DataFlow (KB + RAG) …")
    conditions.append((
        "DataFlow (KB + RAG)",
        run_dataflow_condition(dataset, dataflow_llm, rag=True,
                               confidence_threshold=0.65, task=dataflow_task),
    ))
    print(f"\n[5/{_N}] Running condition: DataFlow (full pipeline) …")
    conditions.append((
        "DataFlow (full pipeline)",
        run_dataflow_condition(dataset, dataflow_llm, rag=True,
                               confidence_threshold=0.75, task=dataflow_task),
    ))

    results: List[Dict[str, Any]] = []
    for cond_name, annotated in conditions:
        # --- Annotation quality ---
        quality = evaluate_annotation_quality(annotated)

        # --- SFT JSONL ---
        safe_name = re.sub(r"[() +]", "_", cond_name.lower()).strip("_")
        sft_path = os.path.join(output_dir, f"sft_{safe_name}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path, skip_human_review=False)

        # --- Downstream fine-tuning with Qwen ---
        downstream_bleu: Optional[float] = None
        downstream_rouge_l: Optional[float] = None
        if not skip_finetune:
            model_dir = os.path.join(
                output_dir, f"qwen_sft_{safe_name}"
            )
            ft_result = run_downstream_finetune(
                sft_path=sft_path,
                model_name=finetune_model_name,
                model_output_dir=model_dir,
                val_data_path=val_data_path,
            )
            downstream_bleu = ft_result.get("bleu")
            downstream_rouge_l = ft_result.get("rouge_l")

        results.append({
            "condition": cond_name,
            "num_samples": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "downstream_bleu": downstream_bleu,
            "downstream_rouge_l": downstream_rouge_l,
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    has_downstream = any(
        r.get("downstream_bleu") is not None for r in results
    )
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
    if not has_downstream:
        print("  (Downstream metrics skipped; re-run without --skip-finetune for Qwen SFT eval)")
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
        "--samples", type=int, default=500,
        help="Number of QA samples (default: 500)",
    )
    parser.add_argument(
        "--squad-path", default="squad_train.json",
        help="SQuAD training JSON path; falls back to synthetic data if absent",
    )
    parser.add_argument(
        "--oracle-model", default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model name for oracle LLM (default: Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--dataflow-model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for DataFlow LLM (default: Qwen2.5-7B-Instruct)",
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
        "--output-dir", default="/tmp/sft_out",
        help="Directory for SFT JSONL files and model checkpoints (default: /tmp/sft_out)",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip Qwen fine-tuning; report annotation quality only",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic dataset (default: 42)",
    )
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    # --- Instantiate real Qwen LLMs ---
    from misc.llm_provider import LocalLLM

    print(f"Loading oracle LLM: {args.oracle_model}")
    oracle_llm = LocalLLM(args.oracle_model)
    # LocalLLM is stateless after construction (model weights are frozen),
    # so sharing one instance between oracle and DataFlow conditions is safe
    # when both point to the same model name.
    if args.oracle_model == args.dataflow_model:
        dataflow_llm = oracle_llm
    else:
        print(f"Loading DataFlow LLM: {args.dataflow_model}")
        dataflow_llm = LocalLLM(args.dataflow_model)

    print("Running annotation conditions…")
    results = run_experiment(
        dataset=dataset,
        oracle_llm=oracle_llm,
        dataflow_llm=dataflow_llm,
        output_dir=args.output_dir,
        skip_finetune=args.skip_finetune,
        finetune_model_name=args.finetune_model,
        val_data_path=args.val_path,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
