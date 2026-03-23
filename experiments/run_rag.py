"""Experiment: RAG-Augmented Annotation for QA.

This experiment demonstrates how the repository's Retrieval-Augmented
Generation (RAG) feature progressively improves annotation quality as the
knowledge base accumulates high-confidence examples.

Four conditions are compared on the same pool of QA samples:

1. **No RAG**          – plain annotation without any retrieval context
2. **RAG (Jaccard)**   – lightweight word-overlap retrieval (no extra deps)
3. **RAG (TF-IDF)**    – scikit-learn TF-IDF cosine retrieval
4. **RAG (semantic)**  – sentence-transformer cosine retrieval (requires
                         ``sentence-transformers``; falls back to TF-IDF)

For each condition the samples are processed in order, and the knowledge base
is built online: each high-confidence annotation is immediately inserted and
made available for subsequent retrievals.  Quality is reported both overall and
in sliding windows of 50 samples so the improvement over time is visible.

Real LLMs
---------
Pass ``--model Qwen/Qwen2.5-7B-Instruct`` to use a real Qwen model.  The
model prompt is augmented with up to ``--topk`` retrieved KB examples when
RAG is enabled.

Offline / test mode
-------------------
Pass ``--skip-llm`` to use a ``MockLLM`` that returns a fixed template,
making the experiment fully GPU-free and deterministic.

Usage
-----
    # Offline smoke-test
    python experiments/run_rag.py --samples 200 --skip-llm

    # Real Qwen annotation
    python experiments/run_rag.py \\
        --samples 500 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --squad-path path/to/train-v1.1.json \\
        --output-dir /tmp/rag_out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sys-path fix
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE: float = 0.85  # mock LLM always produces this confidence
KB_CONFIDENCE_THRESHOLD: float = 0.70  # entries with >= this confidence enter the KB

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
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def evaluate_annotation_quality(annotated: List[Dict[str, Any]]) -> Dict[str, float]:
    f1s = [compute_token_f1(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    ems = [compute_exact_match(str(r.get("annotation", "")), str(r.get("answer", ""))) for r in annotated]
    return {
        "annotation_f1": round(sum(f1s) / len(f1s) if f1s else 0.0, 4),
        "annotation_em": round(sum(ems) / len(ems) if ems else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """LLM stub that returns a deterministic answer.

    When RAG examples are embedded in the prompt the mock picks the most
    recently provided example's annotation as its answer, simulating the
    beneficial effect of in-context retrieval.
    """

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        # If the prompt contains RAG examples, use the last one's answer
        import re as _re
        m = _re.findall(r"A:\s*(\S+)", prompt)
        if m:
            rag_ans = m[-1].strip()
            return f"Answer: {rag_ans} Confidence: {DEFAULT_CONFIDENCE}"
        return f"Answer: test_answer Confidence: {DEFAULT_CONFIDENCE}"

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 64):
        return self.generate(prompt), -0.2


# ---------------------------------------------------------------------------
# In-memory knowledge bases (three retrieval backends)
# ---------------------------------------------------------------------------


class _JaccardKB:
    """Word-overlap (Jaccard) in-memory KB — zero extra dependencies."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)

    def retrieve(self, question: str, topk: int = 3) -> List[Dict[str, Any]]:
        if not self._entries:
            return []
        q_words = set(question.lower().split())
        scored = []
        for e in self._entries:
            q2 = (e.get("question") or e.get("text") or "").lower()
            union = q_words | set(q2.split())
            score = len(q_words & set(q2.split())) / len(union) if union else 0.0
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:topk] if _ > 0]

    def __len__(self) -> int:
        return len(self._entries)


class _TFIDFKb:
    """TF-IDF cosine-similarity in-memory KB (requires scikit-learn)."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._texts: List[str] = []

    def add(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)
        self._texts.append((entry.get("question") or entry.get("text") or "").strip() or " ")

    def retrieve(self, question: str, topk: int = 3) -> List[Dict[str, Any]]:
        if not self._entries:
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            corpus = self._texts + [question]
            vec = TfidfVectorizer(sublinear_tf=True)
            mat = vec.fit_transform(corpus)
            sims = cosine_similarity(mat[-1:], mat[:-1])[0]
            top_idx = sims.argsort()[::-1][:topk]
            return [self._entries[i] for i in top_idx if sims[i] > 0]
        except Exception:
            # fall back to Jaccard
            fb = _JaccardKB()
            fb._entries = self._entries
            return fb.retrieve(question, topk)

    def __len__(self) -> int:
        return len(self._entries)


class _SemanticKb:
    """Sentence-transformer cosine-similarity KB.

    Falls back to :class:`_TFIDFKb` when ``sentence-transformers`` is not installed.
    """

    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._encoder_name = encoder_name
        self._encoder = None
        self._entries: List[Dict[str, Any]] = []
        self._embs = None  # np.ndarray of shape [N, D]
        self._fallback = _TFIDFKb()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._encoder = SentenceTransformer(self._encoder_name)
            except Exception:
                pass
        return self._encoder

    def add(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)
        self._fallback.add(entry)
        enc = self._get_encoder()
        if enc is not None:
            import numpy as np
            text = (entry.get("question") or entry.get("text") or "").strip() or " "
            try:
                emb = enc.encode([text], show_progress_bar=False, convert_to_numpy=True).astype("float32")
                self._embs = emb if self._embs is None else np.vstack([self._embs, emb])
            except Exception:
                pass

    def retrieve(self, question: str, topk: int = 3) -> List[Dict[str, Any]]:
        if not self._entries:
            return []
        enc = self._get_encoder()
        if enc is not None and self._embs is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_emb = enc.encode([question], show_progress_bar=False, convert_to_numpy=True)
                sims = cosine_similarity(q_emb, self._embs)[0]
                import numpy as np
                top_idx = sims.argsort()[::-1][:topk]
                return [self._entries[i] for i in top_idx if sims[i] > 0]
            except Exception:
                pass
        return self._fallback.retrieve(question, topk)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Core annotation function (single-sample, one KB backend)
# ---------------------------------------------------------------------------


def _annotate_one(
    sample: Dict[str, Any],
    llm: Any,
    kb: Any,
    task: Any,
    topk: int = 3,
    use_rag: bool = True,
) -> Dict[str, Any]:
    rag_examples = kb.retrieve(sample.get("question", ""), topk=topk) if (use_rag and kb is not None) else []
    prompt = task.get_prompt(sample, rag_examples if rag_examples else None)
    raw = llm.generate(prompt, max_new_tokens=64)
    parsed = task.parse_output(raw)
    annotation = parsed.get("annotation", "")
    confidence = parsed.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5

    result = {
        **sample,
        "annotation": annotation,
        "confidence": float(confidence),
        "kb_size_at_annotation": len(kb) if kb is not None else 0,
        "rag_examples_used": len(rag_examples),
    }

    # Add to KB if high enough confidence
    if kb is not None and float(confidence) >= KB_CONFIDENCE_THRESHOLD:
        kb.add(result)

    return result


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


def run_no_rag(
    dataset: List[Dict[str, Any]],
    llm: Any,
    task: Any,
) -> List[Dict[str, Any]]:
    """Annotate without any RAG context."""
    return [_annotate_one(s, llm, kb=None, task=task, use_rag=False) for s in dataset]


def run_rag_jaccard(
    dataset: List[Dict[str, Any]],
    llm: Any,
    task: Any,
    topk: int = 3,
) -> List[Dict[str, Any]]:
    """RAG with word-overlap (Jaccard) retrieval."""
    kb = _JaccardKB()
    return [_annotate_one(s, llm, kb, task, topk=topk, use_rag=True) for s in dataset]


def run_rag_tfidf(
    dataset: List[Dict[str, Any]],
    llm: Any,
    task: Any,
    topk: int = 3,
) -> List[Dict[str, Any]]:
    """RAG with TF-IDF cosine retrieval (requires scikit-learn)."""
    kb = _TFIDFKb()
    return [_annotate_one(s, llm, kb, task, topk=topk, use_rag=True) for s in dataset]


def run_rag_semantic(
    dataset: List[Dict[str, Any]],
    llm: Any,
    task: Any,
    topk: int = 3,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """RAG with sentence-transformer semantic retrieval.

    Falls back to TF-IDF when ``sentence-transformers`` is not installed.
    """
    kb = _SemanticKb(encoder_name=encoder_name)
    return [_annotate_one(s, llm, kb, task, topk=topk, use_rag=True) for s in dataset]


# ---------------------------------------------------------------------------
# Windowed quality (shows improvement over time)
# ---------------------------------------------------------------------------


def windowed_f1(
    annotated: List[Dict[str, Any]],
    window: int = 50,
) -> List[Dict[str, Any]]:
    """Return mean token-F1 for successive non-overlapping windows."""
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
        ("Isaac Newton", "Newton formulated the laws of motion and universal gravitation.", "motion"),
        ("William Shakespeare", "Shakespeare wrote plays including Hamlet.", "Hamlet"),
        ("Leonardo da Vinci", "Da Vinci painted the Mona Lisa.", "Mona Lisa"),
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
    llm: Any,
    output_dir: str = "/tmp/rag_out",
    topk: int = 3,
    window: int = 50,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    task: Any = None,
) -> List[Dict[str, Any]]:
    """Run all RAG conditions and return result dicts.

    Each result dict contains: ``condition``, ``annotated``, ``annotation_f1``,
    ``annotation_em``, ``final_kb_size``, ``windowed_f1``, ``sft_file``.

    Parameters
    ----------
    dataset:
        QA samples to annotate.
    llm:
        LLM instance used for annotation.
    output_dir:
        Directory for SFT JSONL outputs.
    topk:
        Number of KB entries to retrieve per sample.
    window:
        Sliding-window size for per-window F1 computation.
    encoder_name:
        Sentence-transformer model name for semantic retrieval.
    task:
        Task object (default: ``QATask``).
    """
    task = _get_task(task)

    os.makedirs(output_dir, exist_ok=True)

    conditions_runners = [
        ("No RAG",        lambda: run_no_rag(dataset, llm, task)),
        ("RAG (Jaccard)", lambda: run_rag_jaccard(dataset, llm, task, topk=topk)),
        ("RAG (TF-IDF)",  lambda: run_rag_tfidf(dataset, llm, task, topk=topk)),
        ("RAG (Semantic)", lambda: run_rag_semantic(dataset, llm, task, topk=topk, encoder_name=encoder_name)),
    ]

    results = []
    for cond_name, runner in conditions_runners:
        annotated = runner()
        quality = evaluate_annotation_quality(annotated)
        w_f1 = windowed_f1(annotated, window=window)
        final_kb_size = max((r.get("kb_size_at_annotation", 0) for r in annotated), default=0)

        safe_name = re.sub(r"[() /\-]", "_", cond_name.lower()).strip("_")
        sft_path = os.path.join(output_dir, f"sft_rag_{safe_name}.jsonl")
        n_written = write_sft_jsonl(annotated, sft_path)

        results.append({
            "condition": cond_name,
            "annotated": n_written,
            "annotation_f1": quality["annotation_f1"],
            "annotation_em": quality["annotation_em"],
            "final_kb_size": final_kb_size,
            "windowed_f1": w_f1,
            "sft_file": sft_path,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Condition':<18} {'Ann-F1':>7} {'Ann-EM':>7} {'KB-Final':>9} {'#Samples':>9}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("  RAG — Retrieval-Augmented QA Annotation Comparison")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['condition']:<18} "
            f"{r['annotation_f1']:>7.4f} "
            f"{r['annotation_em']:>7.4f} "
            f"{r['final_kb_size']:>9} "
            f"{r['annotated']:>9}"
        )
    print(sep)

    # Per-window F1 trend
    print("\n  Per-window token-F1 (window shows KB growth benefit):")
    hdr2 = f"  {'Condition':<18} " + "  ".join(
        f"[{w['window_start']}-{w['window_end']}]" for w in (results[0]["windowed_f1"] if results else [])
    )
    print(hdr2)
    for r in results:
        row = f"  {r['condition']:<18} "
        row += "  ".join(f"{w['mean_f1']:>14.4f}" for w in r["windowed_f1"])
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG experiment: annotation quality vs retrieval strategy"
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--squad-path", default="squad_train.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name for annotation LLM")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use mock LLM (no GPU required)")
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of KB examples to retrieve per sample (default: 3)")
    parser.add_argument("--window", type=int, default=50,
                        help="Sliding-window size for per-window F1 (default: 50)")
    parser.add_argument("--encoder-name", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformer model for semantic RAG")
    parser.add_argument("--output-dir", default="/tmp/rag_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Loading dataset (max {args.samples} samples)…")
    dataset = load_squad_dataset(args.squad_path, max_samples=args.samples)
    print(f"Loaded {len(dataset)} samples.")

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
        output_dir=args.output_dir,
        topk=args.topk,
        window=args.window,
        encoder_name=args.encoder_name,
    )

    print_results_table(results)

    summary_path = os.path.join(args.output_dir, "rag_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
