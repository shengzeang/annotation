"""Experiment: LLM Routing for Cost-Quality Tradeoff in QA Annotation.

This experiment demonstrates how the repository's routing strategies can be
used to balance annotation quality against cost (number of calls to an
expensive high-capability LLM) across four conditions:

1. **All-cheap**   – route every sample to the cheap / fast LLM
2. **All-expensive** – route every sample to the expensive / capable LLM
                       (annotation quality upper-bound)
3. **Cascade**     – try the cheap LLM first; escalate to the expensive LLM
                     only when the cheap answer looks wrong (``CascadeRouter``)
4. **LLM scorer**  – a third "judge" LLM scores each sample's suitability for
                     each candidate and routes accordingly (``LLMRouter``)

In addition to annotation quality (token-F1 / exact-match), the experiment
tracks the **expensive-LLM call rate** for each condition — a proxy for cost.

Real LLMs
---------
When ``--skip-llm`` is *not* set, two real Qwen models must be specified:
  ``--cheap-model``     (e.g. ``Qwen/Qwen2.5-7B-Instruct``)
  ``--expensive-model`` (e.g. ``Qwen/Qwen2.5-72B-Instruct``)
A third model (``--judge-model``) is used by LLMRouter scoring; it defaults
to the cheap model.

Offline / test mode
-------------------
Pass ``--skip-llm`` to substitute all real models with ``MockLLM`` instances
that simulate quality differences deterministically without any GPU or network
access.

Usage
-----
    # Offline smoke-test
    python experiments/run_llm_routing.py --samples 100 --skip-llm

    # Real Qwen routing
    python experiments/run_llm_routing.py \\
        --samples 500 \\
        --cheap-model   Qwen/Qwen2.5-7B-Instruct \\
        --expensive-model Qwen/Qwen2.5-72B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/routing_out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Sys-path fix
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEAP_LLM_ACCURACY: float = 0.65   # simulated accuracy of the cheap LLM
EXPENSIVE_LLM_ACCURACY: float = 0.90  # simulated accuracy of the expensive LLM

# ---------------------------------------------------------------------------
# Lightweight inline QA task (avoids torch-dependent import chain from tasks/)
# ---------------------------------------------------------------------------


class _SimpleQATask:
    """Minimal QA task: build prompt and parse ``Answer: ... Confidence: ...``.

    Used when the repository's ``QATask`` import chain is unavailable (e.g. no
    torch installed).  Produces prompts compatible with Qwen instruction-tuned
    models.
    """

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
    """Return *task* if not None, else attempt to load ``QATask``, falling
    back to ``_SimpleQATask`` if the import chain is unavailable."""
    if task is not None:
        return task
    try:
        from tasks.qa import QATask  # may fail if torch not installed
        return QATask()
    except Exception:
        return _SimpleQATask()


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
    expensive_rate = sum(1 for r in annotated if r.get("routed_to") == "expensive") / max(1, len(annotated))
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
        "expensive_call_rate": round(expensive_rate, 4),
    }


# ---------------------------------------------------------------------------
# Mock LLMs that simulate different quality levels
# ---------------------------------------------------------------------------


class _MockCheapLLM:
    """Cheap LLM: frequently returns the correct answer but sometimes drifts."""

    _NOISE = ["unknown", "various", "some", "multiple", "certain"]

    def __init__(self, accuracy: float = CHEAP_LLM_ACCURACY, seed: int = 0) -> None:
        import random
        self._rng = random.Random(seed)
        self._accuracy = accuracy

    def _answer(self, prompt: str) -> str:
        # Extract the answer from the prompt so we can simulate noise
        import re
        m = re.search(r"Answer:\s*(.*?)(?:\s|$)", prompt)
        gt = m.group(1).strip() if m else "answer"
        if self._rng.random() < self._accuracy:
            return gt
        words = gt.split() if gt else ["answer"]
        words[-1] = self._rng.choice(self._NOISE)
        return " ".join(words)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        ans = self._answer(prompt)
        return f"Answer: {ans} Confidence: 0.60"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 64):
        return self.generate(prompt), -0.8


class _MockExpensiveLLM:
    """Expensive LLM: mostly produces correct answers."""

    _NOISE = ["unknown", "various"]

    def __init__(self, accuracy: float = EXPENSIVE_LLM_ACCURACY, seed: int = 0) -> None:
        import random
        self._rng = random.Random(seed)
        self._accuracy = accuracy

    def _answer(self, prompt: str) -> str:
        import re
        m = re.search(r"Answer:\s*(.*?)(?:\s|$)", prompt)
        gt = m.group(1).strip() if m else "answer"
        if self._rng.random() < self._accuracy:
            return gt
        return self._rng.choice(self._NOISE)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        ans = self._answer(prompt)
        return f"Answer: {ans} Confidence: 0.90"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 64):
        return self.generate(prompt), -0.2


class _MockJudgeLLM:
    """Judge LLM that scores candidate answers (used by CascadeRouter & LLMRouter)."""

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        # CascadeRouter uses: "If correct output 1, else output 0."
        # LLMRouter uses: JSON list of {model, score}
        if "output 1" in prompt.lower() or "output 0" in prompt.lower():
            # CascadeRouter judge — always say answer looks wrong so cascade escalates
            return "0"
        # LLMRouter judge — prefer the expensive model
        return '[{"model": "cheap", "score": 0.4}, {"model": "expensive", "score": 0.8}]'


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------


def _annotate_with_llm(
    sample: Dict[str, Any],
    llm: Any,
    llm_name: str,
    task: Any,
) -> Dict[str, Any]:
    prompt = task.get_prompt(sample)
    raw = llm.generate(prompt, max_new_tokens=64)
    parsed = task.parse_output(raw)
    return {
        **sample,
        "annotation": parsed.get("annotation", ""),
        "confidence": parsed.get("confidence", 0.5),
        "routed_to": llm_name,
    }


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------


def run_all_cheap(
    dataset: List[Dict[str, Any]],
    cheap_llm: Any,
    task: Any,
) -> List[Dict[str, Any]]:
    return [_annotate_with_llm(s, cheap_llm, "cheap", task) for s in dataset]


def run_all_expensive(
    dataset: List[Dict[str, Any]],
    expensive_llm: Any,
    task: Any,
) -> List[Dict[str, Any]]:
    return [_annotate_with_llm(s, expensive_llm, "expensive", task) for s in dataset]


def _load_cascade_router():
    """Load CascadeRouter bypassing routers/__init__.py to avoid torch dependency."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "cascade_router_mod",
        os.path.join(_ROOT, "routers", "cascade_router.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    # base_structure.base_router also cascades to torch; use a stub if needed
    try:
        _spec.loader.exec_module(_mod)
        return _mod.CascadeRouter
    except Exception:
        return None


def _load_llm_router():
    """Load LLMRouter bypassing routers/__init__.py to avoid torch dependency."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "llm_router_mod",
        os.path.join(_ROOT, "routers", "llm_router.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        return _mod.LLMRouter
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inline fallback routing implementations
# (mirrors CascadeRouter / LLMRouter logic without any torch dependency)
# ---------------------------------------------------------------------------


class _InlineCascadeRouter:
    """Inline replica of CascadeRouter for environments without torch.

    Algorithm: call the cheap LLM; ask the judge whether the answer looks
    correct; escalate to the expensive LLM only when the judge says ``"0"``.
    """

    def __init__(self, judge_llm: Any, llm_dict: Dict[str, Any], threshold: float = 0.7):
        self._judge = judge_llm
        self._llm_dict = llm_dict
        self._threshold = threshold

    def route(self, prompt: str) -> str:
        cheap_answer = self._llm_dict["cheap"].generate(prompt, max_new_tokens=64)
        judge_prompt = (
            f"Determine if the answer is correct.\n"
            f"Question: {prompt[:200]}\nAnswer: {cheap_answer}\n"
            "If correct output 1, else output 0."
        )
        verdict = self._judge.generate(judge_prompt).strip()
        if "1" in verdict:
            return "cheap"
        return "expensive"


class _InlineLLMRouter:
    """Inline replica of LLMRouter for environments without torch.

    Algorithm: ask the scorer LLM to output a JSON list of
    ``{"model": ..., "score": ...}`` objects; pick the model with the
    highest score.
    """

    def __init__(self, scorer: Any, candidate_llms: List[str]):
        self._scorer = scorer
        self._candidates = candidate_llms

    def route(self, prompt: str) -> str:
        score_prompt = (
            "Rate which model is better for this text (output JSON array):\n"
            f"Sample: {prompt[:200]}\n"
            f"Candidates: {self._candidates}\n"
            "Output: [{\"model\": \"cheap\", \"score\": 0.4}, {\"model\": \"expensive\", \"score\": 0.8}]"
        )
        raw = self._scorer.generate(score_prompt, max_new_tokens=80)
        try:
            start = raw.index("[")
            end = raw.rindex("]")
            parsed = json.loads(raw[start: end + 1])
            best = max(parsed, key=lambda x: float(x.get("score", 0)))
            return best["model"]
        except Exception:
            return self._candidates[0]


def run_cascade(
    dataset: List[Dict[str, Any]],
    cheap_llm: Any,
    expensive_llm: Any,
    judge_llm: Any,
    task: Any,
    threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """CascadeRouter: annotate with cheap LLM; escalate if judge deems answer wrong.

    Attempts to use the repository's ``CascadeRouter`` class; falls back to an
    inline implementation when the torch-dependent import chain is unavailable.
    """
    llm_dict = {"cheap": cheap_llm, "expensive": expensive_llm}

    CascadeRouter = _load_cascade_router()
    if CascadeRouter is not None:
        router = CascadeRouter(
            judge_llm=judge_llm,
            candidate_llm=["cheap", "expensive"],
            llm_dict=llm_dict,
            threshold=threshold,
        )
        def _get_chosen(prompt):
            scores = router.score(prompt, ["cheap", "expensive"])
            return max(scores, key=lambda x: x["score"])["model"]
    else:
        inline = _InlineCascadeRouter(judge_llm, llm_dict, threshold=threshold)
        def _get_chosen(prompt):
            return inline.route(prompt)

    results = []
    for sample in dataset:
        prompt = task.get_prompt(sample)
        chosen = _get_chosen(prompt)
        llm = cheap_llm if chosen == "cheap" else expensive_llm
        results.append(_annotate_with_llm(sample, llm, chosen, task))
    return results


def run_llm_router(
    dataset: List[Dict[str, Any]],
    cheap_llm: Any,
    expensive_llm: Any,
    judge_llm: Any,
    task: Any,
) -> List[Dict[str, Any]]:
    """LLMRouter: ask the judge LLM to score each candidate and route accordingly.

    Attempts to use the repository's ``LLMRouter`` class; falls back to an
    inline implementation when the torch-dependent import chain is unavailable.
    """
    llm_dict = {"cheap": cheap_llm, "expensive": expensive_llm}

    LLMRouter = _load_llm_router()
    if LLMRouter is not None:
        router = LLMRouter(scorer=judge_llm, candidate_llms=["cheap", "expensive"])
        def _get_chosen(prompt):
            scores = router.score(prompt, ["cheap", "expensive"])
            return max(scores, key=lambda x: x["score"])["model"]
    else:
        inline = _InlineLLMRouter(scorer=judge_llm, candidate_llms=["cheap", "expensive"])
        def _get_chosen(prompt):
            return inline.route(prompt)

    results = []
    for sample in dataset:
        prompt = task.get_prompt(sample)
        chosen = _get_chosen(prompt)
        llm = llm_dict.get(chosen, cheap_llm)
        results.append(_annotate_with_llm(sample, llm, chosen, task))
    return results


# ---------------------------------------------------------------------------
# Write SFT JSONL
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
    import random
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
        spec = __import__("datasets.qa_datasets", fromlist=["SquadDataset"])
        ds = spec.SquadDataset.from_file(squad_path, max_samples=max_samples)
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
    output_dir: str = "/tmp/routing_out",
    cascade_threshold: float = 0.7,
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Run all routing conditions and return result dicts.

    Each result dict contains: ``condition``, ``annotated``,
    ``annotation_f1``, ``annotation_em``, ``expensive_call_rate``,
    ``sft_file``.

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    cheap_llm:
        Fast / inexpensive LLM (lower quality).
    expensive_llm:
        Slow / expensive LLM (higher quality).
    judge_llm:
        LLM used as judge / scorer by CascadeRouter and LLMRouter.
    output_dir:
        Directory for SFT JSONL outputs.
    cascade_threshold:
        Quality threshold for cascade escalation.
    task:
        Task object (default: ``QATask``).
    """
    task = _get_task(task)

    os.makedirs(output_dir, exist_ok=True)

    conditions_data = [
        ("All-cheap",      run_all_cheap(dataset, cheap_llm, task)),
        ("All-expensive",  run_all_expensive(dataset, expensive_llm, task)),
        ("Cascade",        run_cascade(dataset, cheap_llm, expensive_llm, judge_llm, task, cascade_threshold)),
        ("LLM Router",     run_llm_router(dataset, cheap_llm, expensive_llm, judge_llm, task)),
    ]

    results = []
    for cond_name, annotated in conditions_data:
        quality = evaluate_annotation_quality(annotated)
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
    parser.add_argument("--cheap-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name for cheap LLM")
    parser.add_argument("--expensive-model", default="Qwen/Qwen2.5-72B-Instruct",
                        help="HuggingFace model name for expensive LLM")
    parser.add_argument("--judge-model", default=None,
                        help="HuggingFace model name for judge/scorer LLM (defaults to --cheap-model)")
    parser.add_argument("--cascade-threshold", type=float, default=0.7)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use mock LLMs (no GPU required)")
    parser.add_argument("--output-dir", default="/tmp/routing_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    if args.skip_llm:
        print("Using mock LLMs (--skip-llm)")
        cheap_llm: Any = _MockCheapLLM(accuracy=CHEAP_LLM_ACCURACY, seed=args.seed)
        expensive_llm: Any = _MockExpensiveLLM(accuracy=EXPENSIVE_LLM_ACCURACY, seed=args.seed)
        judge_llm: Any = _MockJudgeLLM()
    else:
        from misc.llm_provider import LocalLLM
        print(f"Loading cheap LLM: {args.cheap_model}")
        cheap_llm = LocalLLM(args.cheap_model)
        print(f"Loading expensive LLM: {args.expensive_model}")
        expensive_llm = LocalLLM(args.expensive_model)
        judge_name = args.judge_model or args.cheap_model
        if judge_name == args.cheap_model:
            judge_llm = cheap_llm
        else:
            print(f"Loading judge LLM: {judge_name}")
            judge_llm = LocalLLM(judge_name)

    results = run_experiment(
        dataset=dataset,
        cheap_llm=cheap_llm,
        expensive_llm=expensive_llm,
        judge_llm=judge_llm,
        output_dir=args.output_dir,
        cascade_threshold=args.cascade_threshold,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "routing_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
