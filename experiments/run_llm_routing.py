"""Experiment: LLM Routing for Cost-Quality Tradeoff in QA Annotation.

This experiment demonstrates how the repository's routing strategies balance
annotation quality against cost, following the same pipeline pattern from
``test.py`` (``HumanLLMAnnotationSystem``):

    filter → router.route() → annotator.annotate_batch()

Seven routing conditions are compared using the same ``ActiveLearningFilter``
+ ``Annotator``:

1. **All-cheap**    – every sample is annotated by the cheap LLM (no routing).
2. **All-expensive** – every sample is annotated by the expensive LLM (quality
                       upper-bound).
3. **CascadeRouter** – the cheap LLM is tried first; the judge LLM evaluates
                       the answer and escalates to the expensive LLM when the
                       answer does not meet the quality threshold.
4. **LLMRouter**     – a scorer LLM rates each candidate and the highest-rated
                       model is chosen per sample.
5. **KNNRouter**     – k-nearest-neighbour routing: routes by finding the k
                       most similar samples in a prebuilt embedding index
                       trained on bootstrap CascadeRouter annotations.
6. **GraphRouter**   – graph-based routing using Personalised PageRank on a
                       sample-similarity graph trained on bootstrap annotations.
7. **MLPRouter**     – a lightweight MLP trained on (sample, candidate) feature
                       pairs derived from bootstrap annotations.

Conditions 5–7 require an initial training phase.  In offline / mock mode
(``--skip-llm``) lightweight stub routers are used instead and no model is
loaded.

A **resume mechanism** checks whether each condition's per-condition result
JSON already exists in ``output_dir``.  Completed conditions are skipped,
so a partially completed run can be resumed without redundant computation.

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
from routers import CascadeRouter, LLMRouter, KNNRouter, GraphRouter, MLPRouter
from tasks.qa import QATask

# Default candidate LLMs — shared with test.py / HumanLLMAnnotationSystem.
# In this routing experiment, the first entry is the "cheap" model and the
# last entry is the "expensive" model; the middle entry (if present) acts as
# an intermediate tier.
DEFAULT_CANDIDATE_LLMS: List[str] = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


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

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        return '[{"model": "cheap", "score": 0.4}, {"model": "expensive", "score": 0.8}]'


# ---------------------------------------------------------------------------
# Mock learning-based routers for offline testing (no transformer loading)
# ---------------------------------------------------------------------------

class MockKNNRouter:
    """Offline-safe KNNRouter stub for ``--skip-llm`` mode.

    Routes all candidates with uniform scores (no model loading, no training).
    """

    def __init__(self, candidate_llms: List[str]):
        self.candidate_llms = list(candidate_llms)

    def build_from_annotations(self, annotations: Any, out_dir: str = "./") -> None:
        """No-op: mock router does not require training data."""

    def score(self, sample_text: str, candidate_llms: List[str]) -> List[Dict[str, Any]]:
        """Return uniform scores for all candidates."""
        n = len(candidate_llms)
        uniform = 1.0 / n if n else 0.0
        return [{"model": c, "score": uniform} for c in candidate_llms]


class MockGraphRouter(MockKNNRouter):
    """Offline-safe GraphRouter stub for ``--skip-llm`` mode."""


class MockMLPRouter(MockKNNRouter):
    """Offline-safe MLPRouter stub for ``--skip-llm`` mode."""


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
    return os.path.join(output_dir, f"sft_routing_{_safe_name(cond_name)}.jsonl")


def _condition_already_done(cond_name: str, output_dir: str) -> bool:
    """Return ``True`` if *cond_name* has already produced output in *output_dir*.

    The check is intentionally broad: a condition is considered done if
    **either** the per-condition result JSON file **or** the SFT JSONL output
    file already exists on disk.  This lets the resume mechanism work even
    when only the primary output (SFT file) was written in a previous partial
    run.
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
# Routing helpers used by the learning-based router conditions
# ---------------------------------------------------------------------------

def _route_direct(
    router_obj: Any,
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Route *dataset* by calling ``router_obj.score()`` on each sample.

    Bypasses ``BaseRouter.cold_start`` (which is triggered by
    ``BaseRouter.route()`` when ``if_train`` returns ``True``).  We use this
    for learning-based routers that are trained manually via
    ``build_from_annotations()`` before inference.
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
    cascade_threshold: float,
    output_dir: str,
    n_bootstrap: int = 50,
) -> List[Dict[str, Any]]:
    """Return bootstrap-routed data for training learning-based routers.

    Loads from ``{output_dir}/bootstrap_routed.json`` if it exists; otherwise
    creates it by running ``CascadeRouter`` on the first *n_bootstrap* samples
    of *filtered_data* and caching the result.
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
        threshold=cascade_threshold,
    )
    routed = _route_direct(router, filtered_data[:n_boot])
    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(routed, f, indent=2, ensure_ascii=False)
    return routed


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
            os.path.join(_ROOT, "qa_data", "qa_datasets.py"),
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
    candidate_llms: Optional[List[str]] = None,
    llm_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run seven routing conditions following the HumanLLMAnnotationSystem pattern.

    All conditions use the same ``ActiveLearningFilter`` pre-filter; only the
    router changes.  Conditions 5–7 are learning-based and require an
    embedding model (sentence-transformers).  When *force_fallback* is
    ``True`` (offline / mock mode) lightweight stub routers are used instead.

    A **resume mechanism** skips any condition whose per-condition result JSON
    already exists in *output_dir*, allowing partial runs to be resumed.

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
        Directory for SFT JSONL outputs and per-condition result JSON files.
    cascade_threshold:
        Minimum quality score below which ``CascadeRouter`` escalates.
    force_fallback:
        Passed to ``ActiveLearningFilter`` for offline mode.  Also selects
        mock stub routers for conditions 5–7.
    task:
        Task object (default: ``QATask``).
    candidate_llms:
        Ordered list of LLM identifiers (first = cheapest, last = most
        capable).  When provided, ``llm_dict`` must also be supplied.
        Defaults to the two-entry ``["cheap", "expensive"]`` fallback.
    llm_dict:
        Mapping from LLM identifier to LLM instance.

    Returns
    -------
    List of result dicts with keys ``condition``, ``annotated``,
    ``annotation_f1``, ``annotation_em``, ``expensive_call_rate``,
    ``sft_file``.
    """
    if task is None:
        task = QATask()
    os.makedirs(output_dir, exist_ok=True)

    # Use caller-supplied candidate list or fall back to symbolic two-entry mapping.
    if candidate_llms is None or llm_dict is None:
        candidate_llms = ["cheap", "expensive"]
        llm_dict = {"cheap": cheap_llm, "expensive": expensive_llm}

    cheap_name = candidate_llms[0]
    expensive_name = candidate_llms[-1]

    # Shared filter (same for all conditions, as in HumanLLMAnnotationSystem)
    al_filter = ActiveLearningFilter(
        method="alps",
        budget=len(dataset),
        batch_size=max(2, len(dataset) // 10),
        force_fallback=force_fallback,
    )
    filtered = al_filter.filter(dataset)

    # ------------------------------------------------------------------
    # Inner condition functions
    # ------------------------------------------------------------------

    def _annotate_no_router(llm_name: str) -> List[Dict[str, Any]]:
        """Bypass routing: annotate every sample with a fixed LLM."""
        kb_path = os.path.join(output_dir, f"kb_no_router_{_safe_name(llm_name)}.json")
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

    # Learning-based routers: use lightweight mock stubs in offline mode;
    # instantiate real classes (which load sentence-transformers) in GPU mode.
    if force_fallback:
        knn_router: Any = MockKNNRouter(candidate_llms)
        graph_router: Any = MockGraphRouter(candidate_llms)
        mlp_router: Any = MockMLPRouter(candidate_llms)
    else:
        knn_router = KNNRouter(annotator=None, candidate_llms=candidate_llms)
        graph_router = GraphRouter(annotator=None, candidate_llms=candidate_llms)
        mlp_router = MLPRouter(annotator=None, candidate_llms=candidate_llms)

    def _annotate_learning_router(cond_name: str, router_obj: Any) -> List[Dict[str, Any]]:
        """Train router on bootstrap data (real mode only), then route the full
        filtered set via ``_route_direct`` (bypasses ``BaseRouter.cold_start``)
        and annotate with the chosen model.
        """
        kb_path = os.path.join(output_dir, f"kb_{_safe_name(cond_name)}.json")
        annotator = Annotator(candidate_llms, llm_dict, task=task, kb_path=kb_path)
        if not force_fallback:
            bootstrap = _ensure_bootstrap_cache(
                filtered_data=filtered,
                judge_llm=judge_llm,
                candidate_llms=candidate_llms,
                llm_dict=llm_dict,
                cascade_threshold=cascade_threshold,
                output_dir=output_dir,
            )
            if bootstrap:
                train_dir = os.path.join(output_dir, f"router_model_{_safe_name(cond_name)}")
                router_obj.build_from_annotations(bootstrap, out_dir=train_dir)
        routed = _route_direct(router_obj, filtered)
        return annotator.annotate_batch(routed)

    # ------------------------------------------------------------------
    # Seven conditions — iterated with per-condition resume check
    # ------------------------------------------------------------------

    _N = 7
    conditions: List[tuple] = [
        ("All-cheap",     lambda: _annotate_no_router(cheap_name)),
        ("All-expensive", lambda: _annotate_no_router(expensive_name)),
        ("CascadeRouter", _annotate_cascade),
        ("LLMRouter",     _annotate_llm_router),
        ("KNNRouter",     lambda: _annotate_learning_router("KNNRouter", knn_router)),
        ("GraphRouter",   lambda: _annotate_learning_router("GraphRouter", graph_router)),
        ("MLPRouter",     lambda: _annotate_learning_router("MLPRouter", mlp_router)),
    ]

    results: List[Dict[str, Any]] = []
    for i, (cond_name, fn) in enumerate(conditions, 1):
        print(f"\n[{i}/{_N}] Running condition: {cond_name} …")
        if _condition_already_done(cond_name, output_dir):
            print(f"  ↳ Already done — skipping (output file exists).")
            result_path = _condition_result_path(cond_name, output_dir)
            if os.path.exists(result_path):
                results.append(_load_condition_result(cond_name, output_dir))
            else:
                # SFT output file exists but result JSON was never written — reconstruct
                # a minimal result so the summary table can still be printed.
                sft_path = _sft_output_path(cond_name, output_dir)
                n_lines = sum(1 for _ in open(sft_path, encoding="utf-8"))
                results.append({
                    "condition": cond_name,
                    "annotated": n_lines,
                    "annotation_f1": 0.0,
                    "annotation_em": 0.0,
                    "expensive_call_rate": 0.0,
                    "sft_file": sft_path,
                })
            continue
        annotated = fn()
        quality = evaluate_annotation_quality(annotated, expensive_llm_name=expensive_name)
        safe_n = _safe_name(cond_name)
        sft_path = os.path.join(output_dir, f"sft_routing_{safe_n}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path)
        result: Dict[str, Any] = {
            "condition": cond_name,
            "annotated": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "expensive_call_rate": quality["expensive_call_rate"],
            "sft_file": sft_path,
        }
        _save_condition_result(result, output_dir)
        results.append(result)

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
    print("  (✓ = loaded from cache)")
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
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANDIDATE_LLMS,
        help=(
            "Ordered list of candidate LLMs (first = cheapest, last = most capable). "
            "Default: the 3 standard Qwen models."
        ),
    )
    parser.add_argument("--judge-model", default=None,
                        help="Judge LLM for CascadeRouter (defaults to last --models entry)")
    parser.add_argument("--scorer-model", default=None,
                        help="Scorer LLM for LLMRouter (defaults to first --models entry)")
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
        # In mock mode keep the two-entry fallback for lightweight testing
        _candidate_llms = None
        _llm_dict = None
    else:
        from misc.llm_provider import LocalLLM
        models = args.models
        print(f"Loading LLMs: {models}")
        _llm_dict = {m: LocalLLM(m) for m in models}
        _candidate_llms = models
        cheap_llm = _llm_dict[models[0]]
        expensive_llm = _llm_dict[models[-1]]
        judge_name = args.judge_model or models[-1]
        judge_llm = _llm_dict.get(judge_name) or LocalLLM(judge_name)
        scorer_name = args.scorer_model or models[0]
        scorer_llm = _llm_dict.get(scorer_name) or LocalLLM(scorer_name)
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
        candidate_llms=_candidate_llms,
        llm_dict=_llm_dict,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "routing_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
