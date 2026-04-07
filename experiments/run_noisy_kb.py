"""Experiment: KB Noise Degradation in RAG Annotation.

This experiment demonstrates that when the knowledge base (KB) used for
Retrieval-Augmented Generation (RAG) contains noisy (incorrect) annotations,
annotation quality **continuously degrades** over time due to a compounding
snowball effect.

How the snowball works
----------------------
1. The KB is pre-seeded with a fraction of **noisy** (incorrect) annotations.
2. At annotation time, RAG retrieves similar KB entries and injects them as
   in-context examples into the prompt.
3. The annotating LLM is influenced by the retrieved examples: it returns the
   **majority-vote** answer from the retrieved KB context.  When most retrieved
   entries are wrong, the LLM outputs a wrong annotation.
4. The wrong annotation is added back to the KB (confidence ≥ threshold),
   increasing the overall noise fraction.
5. As more samples are processed the KB noise fraction grows → more wrong
   examples are retrieved → quality degrades further.

Four conditions are compared, varying only the initial KB noise rate:

1. **Noise 00pct (0%)**  – no noisy entries; quality remains stable.
2. **Noise 25pct (25%)** – mild initial noise; slow degradation.
3. **Noise 50pct (50%)** – moderate noise; visible degradation.
4. **Noise 75pct (75%)** – severe noise; rapid, continuous degradation.

All conditions use the same ``ActiveLearningFilter`` + ``KNNRouter`` +
``Annotator(rag=True)`` pipeline, varying only the pre-seeded KB content.

Usage
-----
    # Offline smoke-test (no GPU / network required)
    python experiments/run_noisy_kb.py --samples 200 --skip-llm

    # Real Qwen annotation (requires GPU)
    python experiments/run_noisy_kb.py \\
        --samples 500 \\
        --models Qwen/Qwen2.5-7B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/noisy_kb_out
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
from routers import CascadeRouter, KNNRouter
from tasks.qa import QATask

# Default candidate LLMs — shared with test.py / HumanLLMAnnotationSystem.
DEFAULT_CANDIDATE_LLMS: List[str] = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# Default KB noise rates for the four conditions.
DEFAULT_NOISE_RATES: List[float] = [0.0, 0.25, 0.50, 0.75]

# Wrong answers injected as noise — plausible-looking but factually incorrect.
_WRONG_ANSWERS: List[str] = [
    "mitosis", "photosynthesis", "democracy", "entropy", "logarithm",
    "centrifuge", "hyperbole", "meridian", "symbiosis", "paradigm",
    "tectonic", "fluorescence", "thermodynamics", "bureaucracy", "metamorphosis",
]


# ---------------------------------------------------------------------------
# Mock LLMs for offline testing
# ---------------------------------------------------------------------------

class MockContextAwareLLM:
    """Mock LLM that propagates KB noise via majority-vote context copying.

    When RAG context is injected into the prompt, this mock LLM extracts all
    ``A: ...`` examples from the context and returns the **majority-vote**
    answer.  This simulates the real-world behaviour where an LLM tends to
    follow in-context examples: when most retrieved entries are noisy, the
    output is noisy too.

    When no RAG context is present (empty KB or no match), the LLM falls back
    to ``answer_map`` — representing a well-calibrated model that answers
    correctly in the absence of in-context examples.

    Parameters
    ----------
    answer_map:
        Mapping ``{question_text: correct_answer}`` used as the no-context
        fallback.
    """

    def __init__(self, answer_map: Optional[Dict[str, str]] = None) -> None:
        self._answer_map: Dict[str, str] = answer_map or {}

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # Try to extract all KB-context answers (RAG section ends at "Answer:")
        context_section = re.search(
            r"Here are some similar QA pairs.*?(?=Answer:)",
            prompt,
            re.S,
        )
        if context_section:
            kb_answers = re.findall(
                r"\bA:\s*(.+?)(?:\n|$)", context_section.group(0)
            )
            if kb_answers:
                # Majority vote: return the most common KB answer.
                # This creates a feedback loop: noisy KB → noisy output →
                # noisy output re-added to KB → more noise (snowball effect).
                vote = Counter(kb_answers).most_common(1)[0][0]
                return f"Answer: {vote} Confidence: 0.85"

        # No KB context — look up the correct answer from the map.
        q_match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt)
        if q_match:
            q_text = q_match.group(1).strip()
            if q_text in self._answer_map:
                return f"Answer: {self._answer_map[q_text]} Confidence: 0.85"

        return "Answer: unknown Confidence: 0.85"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 50):
        return self.generate(prompt), -0.2


class MockJudgeLLM:
    """Judge LLM stub for CascadeRouter bootstrap — always keeps the cheap model."""

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "1"


class MockKNNRouter:
    """Offline-safe KNNRouter stub for ``--skip-llm`` mode.

    Routes all candidates with uniform scores (no model loading, no training).
    """

    def __init__(self, candidate_llms: List[str]) -> None:
        self.candidate_llms = list(candidate_llms)

    def build_from_annotations(self, annotations: Any, out_dir: str = "./") -> None:
        """No-op: mock router does not require training data."""

    def score(self, sample_text: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Return uniform scores for all candidates."""
        n = len(candidate_llms)
        uniform = 1.0 / n if n else 0.0
        return [{"model": c, "score": uniform} for c in candidate_llms]


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
    """Return 1.0 if *prediction* matches *ground_truth* (case-insensitive), else 0.0."""
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def evaluate_annotation_quality(annotated: List[Dict[str, Any]]) -> Dict[str, float]:
    f1s = [
        compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", "")))
        for r in annotated
    ]
    ems = [
        compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", "")))
        for r in annotated
    ]
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
    }


def windowed_f1(annotated: List[Dict[str, Any]], window: int = 50) -> List[Dict[str, Any]]:
    """Return mean token-F1 for successive non-overlapping windows.

    A **declining** trend in the high-noise condition demonstrates the KB
    noise snowball: as more noisy annotations accumulate in the KB, future
    retrievals become increasingly noisy, driving quality down further.
    """
    windows = []
    for start in range(0, len(annotated), window):
        chunk = annotated[start: start + window]
        if not chunk:
            break
        f1s = [
            compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", "")))
            for r in chunk
        ]
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

def _make_synthetic_dataset(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
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


def load_squad_dataset(squad_path: str, max_samples: int = 200) -> List[Dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# KB seeding helpers
# ---------------------------------------------------------------------------

def inject_noise(
    entries: List[Dict[str, Any]],
    noise_rate: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Return a copy of *entries* with *noise_rate* fraction of annotations corrupted.

    Corrupted entries have their ``annotation`` field replaced with a random
    word from :data:`_WRONG_ANSWERS`.  All other fields are preserved.
    """
    noisy = []
    for entry in entries:
        e = dict(entry)
        if rng.random() < noise_rate:
            e["annotation"] = rng.choice(_WRONG_ANSWERS)
        noisy.append(e)
    return noisy


def build_seed_kb(
    dataset: List[Dict[str, Any]],
    seed_size: int,
    noise_rate: float,
    rng: random.Random,
    kb_path: str,
) -> None:
    """Pre-populate a KB JSON file with *seed_size* entries at *noise_rate*.

    Entries are drawn from the **first** *seed_size* records of *dataset*.
    Each entry's ground-truth answer (``answer`` field) is used as the clean
    annotation; ``noise_rate`` fraction are then replaced with wrong answers
    before writing to *kb_path*.
    """
    pool = dataset[:seed_size]
    clean_entries = [
        {**rec, "annotation": rec.get("answer", ""), "confidence": 0.9}
        for rec in pool
    ]
    seeded = inject_noise(clean_entries, noise_rate=noise_rate, rng=rng)
    os.makedirs(os.path.dirname(os.path.abspath(kb_path)), exist_ok=True)
    with open(kb_path, "w", encoding="utf-8") as fh:
        json.dump(seeded, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Condition naming
# ---------------------------------------------------------------------------

def _make_condition_name(noise_rate: float) -> str:
    pct = int(round(noise_rate * 100))
    return f"Noise {pct:02d}pct"


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
    return os.path.join(output_dir, f"sft_noisy_kb_{_safe_name(cond_name)}.jsonl")


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
# Routing helpers (learning-based router support)
# ---------------------------------------------------------------------------

def _route_direct(
    router_obj: Any,
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Route *dataset* by calling ``router_obj.score()`` on each sample.

    Bypasses ``BaseRouter.cold_start`` (triggered by ``BaseRouter.route()``
    when ``if_train`` returns ``True``).  Used for KNNRouter after it has
    been trained via ``build_from_annotations()``.
    """
    candidate_llms = getattr(router_obj, "candidate_llms", [])
    routed: List[Dict[str, Any]] = []
    for d in dataset:
        scores = router_obj.score(d.get("text", ""), candidate_llms)
        if scores:
            best = max(scores, key=lambda x: x.get("score", 0.0))
            chosen = best.get("model")
        else:
            chosen = candidate_llms[0] if candidate_llms else None
        routed.append({**d, "route": chosen, "route_scores": scores})
    return routed


def _ensure_bootstrap_cache(
    filtered_data: List[Dict[str, Any]],
    judge_llm: Any,
    candidate_llms: List[str],
    llm_dict: Dict[str, Any],
    output_dir: str,
    n_bootstrap: int = 50,
) -> List[Dict[str, Any]]:
    """Return bootstrap-routed data for training the KNNRouter.

    Loads from ``{output_dir}/bootstrap_routed.json`` if it already exists;
    otherwise runs ``CascadeRouter`` on the first *n_bootstrap* samples of
    *filtered_data* and caches the result.
    """
    cache_path = os.path.join(output_dir, "bootstrap_routed.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    n_boot = min(n_bootstrap, len(filtered_data))
    print(f"  Creating bootstrap training data ({n_boot} samples via CascadeRouter)…")
    router = CascadeRouter(
        judge_llm=judge_llm,
        candidate_llm=candidate_llms,
        llm_dict=llm_dict,
    )
    routed = _route_direct(router, filtered_data[:n_boot])
    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(routed, f, indent=2, ensure_ascii=False)
    return routed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: List[Dict[str, Any]],
    llm: Any,
    judge_llm: Any,
    output_dir: str = "/tmp/noisy_kb_out",
    noise_rates: Optional[List[float]] = None,
    seed_size: int = 20,
    topk: int = 3,
    window: int = 50,
    seed: int = 42,
    force_fallback: bool = True,
    task: Any = None,
    candidate_llms: Optional[List[str]] = None,
    llm_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run the KB noise degradation experiment.

    For each noise level, the KB is pre-seeded with *seed_size* entries of
    which *noise_rate* fraction have incorrect annotations.  Annotation then
    proceeds on ``dataset[seed_size:]`` using RAG with topk retrieval.

    The ``MockContextAwareLLM`` (or a real LLM in GPU mode) performs
    majority-vote over retrieved KB answers.  This means:

    * **Low noise KB** – most retrieved examples are correct → LLM outputs
      correct annotations → KB stays clean → quality remains high.
    * **High noise KB** – most retrieved examples are wrong → LLM outputs
      wrong annotations → noisy annotations are re-added to KB → KB noise
      fraction grows → quality degrades continuously (snowball).

    The per-window F1 trend (``windowed_f1`` key in each result) directly
    shows the continuous degradation in the high-noise conditions.

    Parameters
    ----------
    dataset:
        Full pool of QA samples.  The first *seed_size* entries are used to
        pre-seed the KB; the rest form the annotation pool.
    llm:
        Primary annotation LLM (or ``MockContextAwareLLM`` in offline mode).
    judge_llm:
        Judge LLM used by ``CascadeRouter`` to generate KNNRouter bootstrap
        training data (real mode only; ignored in offline mode).
    output_dir:
        Directory for KB files, SFT JSONL outputs and per-condition JSONs.
    noise_rates:
        KB noise fractions to test.  Defaults to ``[0.0, 0.25, 0.50, 0.75]``.
    seed_size:
        Number of entries pre-seeded into the KB for each condition.
    topk:
        Number of KB entries retrieved per sample during RAG.
    window:
        Sliding-window size for windowed F1 trend computation.
    seed:
        Base random seed (per-condition seed is ``seed + condition_index``).
    force_fallback:
        Passed to ``ActiveLearningFilter`` — set ``True`` for offline mode.
    task:
        Task object (default: ``QATask``).
    candidate_llms / llm_dict:
        LLM routing configuration.

    Returns
    -------
    List of result dicts, one per condition, with keys:
    ``condition``, ``noise_rate``, ``annotated``, ``annotation_f1``,
    ``annotation_em``, ``final_kb_size``, ``windowed_f1``, ``sft_file``.
    """
    if noise_rates is None:
        noise_rates = DEFAULT_NOISE_RATES
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    if candidate_llms is None or llm_dict is None:
        candidate_llms = ["primary"]
        llm_dict = {"primary": llm}

    conditions = [(_make_condition_name(r), r) for r in noise_rates]
    _N = len(conditions)

    # Annotation pool: samples NOT used for KB seeding.
    annotation_pool = dataset[seed_size:] if len(dataset) > seed_size else dataset[:]

    # Build shared filter+router only when at least one condition still needs to run.
    _needs_run = [not _condition_already_done(cn, output_dir) for cn, _ in conditions]
    if any(_needs_run):
        al_filter = ActiveLearningFilter(
            method="alps",
            budget=len(annotation_pool),
            batch_size=max(2, len(annotation_pool) // 10),
            force_fallback=force_fallback,
        )
        filtered = al_filter.filter(annotation_pool)

        # Use KNNRouter as the default router.
        # In offline mode use a lightweight mock (no transformer loading).
        # In real mode bootstrap training data via CascadeRouter and then
        # train the KNNRouter before routing the full filtered set.
        if force_fallback:
            router: Any = MockKNNRouter(candidate_llms)
        else:
            bootstrap = _ensure_bootstrap_cache(
                filtered_data=filtered,
                judge_llm=judge_llm,
                candidate_llms=candidate_llms,
                llm_dict=llm_dict,
                output_dir=output_dir,
            )
            router = KNNRouter(annotator=None, candidate_llms=candidate_llms)
            if bootstrap:
                train_dir = os.path.join(output_dir, "router_model_knn")
                router.build_from_annotations(bootstrap, out_dir=train_dir)

        routed = _route_direct(router, filtered)
    else:
        routed = []  # unused — all conditions will be loaded from cache

    results: List[Dict[str, Any]] = []
    for i, (cond_name, noise_rate) in enumerate(conditions, 1):
        print(
            f"\n[{i}/{_N}] Running condition: {cond_name}"
            f"  (KB noise = {noise_rate:.0%}) …"
        )
        if _condition_already_done(cond_name, output_dir):
            print("  ↳ Already done — skipping (output file exists).")
            result_path = _condition_result_path(cond_name, output_dir)
            if os.path.exists(result_path):
                results.append(_load_condition_result(cond_name, output_dir))
            else:
                sft_path = _sft_output_path(cond_name, output_dir)
                n_lines = sum(1 for _ in open(sft_path, encoding="utf-8"))
                results.append({
                    "condition": cond_name,
                    "noise_rate": noise_rate,
                    "annotated": n_lines,
                    "annotation_f1": 0.0,
                    "annotation_em": 0.0,
                    "final_kb_size": 0,
                    "windowed_f1": [],
                    "sft_file": sft_path,
                })
            continue

        # Pre-seed this condition's KB (deterministic per-condition seed).
        cond_rng = random.Random(seed + i)
        kb_path = os.path.join(output_dir, f"kb_{_safe_name(cond_name)}.json")
        build_seed_kb(dataset, seed_size, noise_rate, cond_rng, kb_path)

        annotator = Annotator(
            candidate_llms,
            llm_dict,
            task=task,
            rag=True,
            kb_path=kb_path,
        )
        annotator.knowledge_base._topk = topk

        annotated = annotator.annotate_batch(routed)

        quality = evaluate_annotation_quality(annotated)
        w_f1 = windowed_f1(annotated, window=window)
        final_kb_size = len(annotator.knowledge_base.entries)

        sft_path = _sft_output_path(cond_name, output_dir)
        n_written = write_sft_jsonl(annotated, sft_path)

        result: Dict[str, Any] = {
            "condition": cond_name,
            "noise_rate": noise_rate,
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
        f"{'Condition':<16} {'Noise':>6} {'Ann-F1':>7} {'Ann-EM':>7}"
        f" {'KB-Final':>9} {'#Samples':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Noisy KB Degradation — RAG Annotation Quality vs KB Noise Rate")
    print("  (all conditions use RAG; only the KB noise rate varies)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<16} "
            f"{r['noise_rate']:>6.0%} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r['final_kb_size']:>9} "
            f"{r['annotated']:>9}"
        )
    print(sep)

    # Per-window F1 trend
    if any(r.get("windowed_f1") for r in results):
        cond_windows: Dict[str, Dict[int, float]] = {}
        for r in results:
            cond_windows[r["condition"]] = {
                w["window_start"]: w["mean_f1"]
                for w in r.get("windowed_f1", [])
            }

        all_starts = sorted({s for wmap in cond_windows.values() for s in wmap})
        if all_starts:
            col_labels = [f"[{s}-{s + window - 1}]" for s in all_starts]
            col_w = max(8, *(len(c) for c in col_labels))
            hdr = "\n  Per-window token-F1 (declining trend = KB noise snowball):\n"
            hdr += "  " + f"{'Condition':<16} " + " ".join(
                f"{c:>{col_w}}" for c in col_labels
            )
            print(hdr)
            for r in results:
                wmap = cond_windows[r["condition"]]
                row = f"  {r['condition']:<16} "
                row += " ".join(
                    f"{wmap.get(s, 0.0):>{col_w}.4f}" for s in all_starts
                )
                print(row)
    print("")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KB noise degradation experiment: show that noisy RAG KB "
            "causes continuous annotation quality degradation."
        )
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--seed-size", type=int, default=20,
        help="Number of entries to pre-seed in the KB for each condition",
    )
    parser.add_argument("--squad-path", default="squad_train.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANDIDATE_LLMS,
        help="Candidate LLMs to annotate with (default: the 3 standard Qwen models)",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge LLM for CascadeRouter bootstrap (defaults to last --models entry)",
    )
    parser.add_argument(
        "--noise-rates", type=float, nargs="+",
        default=DEFAULT_NOISE_RATES,
        help="KB noise rates to test (default: 0.0 0.25 0.50 0.75)",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use MockContextAwareLLM for offline testing")
    parser.add_argument("--output-dir", default="/tmp/noisy_kb_out")
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    if args.skip_llm:
        print("Using MockContextAwareLLM (--skip-llm)")
        answer_map = {rec["question"]: rec["answer"] for rec in dataset}
        llm: Any = MockContextAwareLLM(answer_map=answer_map)
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
        llm = _llm_dict[models[0]]
        judge_name = args.judge_model or models[-1]
        judge_llm = _llm_dict.get(judge_name) or LocalLLM(judge_name)
        force_fallback = False

    results = run_experiment(
        dataset=dataset,
        llm=llm,
        judge_llm=judge_llm,
        output_dir=args.output_dir,
        noise_rates=args.noise_rates,
        seed_size=args.seed_size,
        topk=args.topk,
        window=args.window,
        seed=args.seed,
        force_fallback=force_fallback,
        candidate_llms=_candidate_llms,
        llm_dict=_llm_dict,
    )

    print_results_table(results, window=args.window)

    summary_path = os.path.join(args.output_dir, "noisy_kb_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
