#!/usr/bin/env python3
"""
Experiment: Dual-threshold Entry Control vs Periodic Outlier Purge
===================================================================

Quantifies the effects of two knowledge-base quality-control mechanisms on
annotation accuracy and simulated downstream QA performance using SQuAD-style
data.

Conditions
----------
1. **oracle_naive**   – Perfect LLM (no noise), no entry control, no purge.
                        Acts as the upper-bound oracle baseline.
2. **naive**          – Noisy LLM, no entry control, no purge.
3. **entry_control**  – Noisy LLM + dual-threshold entry control only
                        (confidence ≥ 0.70, avg_logprob ≥ −1.00).
4. **purge**          – Noisy LLM + periodic outlier purge only
                        (every 10 KB additions, z-score threshold = 1.5).
5. **both**           – Noisy LLM + entry control + periodic outlier purge.

Simulated LLM
-------------
No real LLM or GPU is required.  The ``SimulatedLLM`` class mimics the
``LLMBase`` interface (``generate`` / ``generate_with_logprobs``) and produces:

* **Clean answers** (probability = 1 − noise_rate):
  Returns the ground-truth answer with reported confidence 0.90 and
  avg_logprob −0.40 (high quality, passes all thresholds).
* **Noisy answers** (probability = noise_rate):
  Returns a factually wrong answer drawn from a *different* topic cluster,
  with reported confidence 0.85 but avg_logprob −1.80.  The self-reported
  confidence is deliberately above the confidence threshold to simulate an
  over-confident yet low-quality annotation; only the avg_logprob signal
  reveals the quality issue.

Dataset
-------
Forty synthetic SQuAD-style QA pairs are embedded in this script across four
thematic topic clusters (geography, science, history, sports).  They are
written to a temporary JSON file and loaded via ``SquadDataset.from_file``
(the framework's standard dataset class).

Downstream QA Simulation
-------------------------
Since real fine-tuning of Qwen-3 0.6B / Llama-3.2 1B requires GPU resources,
downstream performance is estimated via a 1-nearest-neighbour retrieval proxy:
  * For each test question the most lexically similar KB entry is retrieved.
  * Its annotation is used as the prediction.
  * Standard SQuAD-style Exact Match (EM) and token-level F1 are reported.

This proxy directly reflects KB annotation quality: a cleaner KB yields higher
retrieval accuracy, approximating what a model fine-tuned on that KB would
achieve.  To run real fine-tuning, use ``misc/evaluate.py::finetune_sft`` with
the SFT JSONL files written to ``experiments/output/sft_<condition>.jsonl``.

Usage
-----
From the repository root::

    python experiments/run_kb_experiment.py
    python experiments/run_kb_experiment.py --noise-rate 0.4 --seed 7
    python experiments/run_kb_experiment.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running from repo root or from the experiments/ directory.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))

from annotation import Annotator  # noqa: E402
from datasets.qa_datasets import SquadDataset  # noqa: E402
from rag import VectorKnowledgeBase  # noqa: E402
from tasks.qa import QATask  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic SQuAD-style dataset
# ---------------------------------------------------------------------------
# 40 QA pairs arranged in 4 topic clusters (10 per topic).
# Training set: first 8 per topic (32 total).
# Test set: last 2 per topic (8 total).

_QA_PAIRS: List[Dict[str, Any]] = [
    # ── Geography (topic 0) ────────────────────────────────────────────────
    {"id": "g01", "q": "In what country is Normandy located?",
     "ctx": "Normandy is a region located in northern France.",
     "a": "France", "topic": 0},
    {"id": "g02", "q": "What is the capital city of Germany?",
     "ctx": "Berlin is the capital and largest city of Germany.",
     "a": "Berlin", "topic": 0},
    {"id": "g03", "q": "Which ocean borders the western United States?",
     "ctx": "The Pacific Ocean borders the western coast of the United States.",
     "a": "Pacific Ocean", "topic": 0},
    {"id": "g04", "q": "What river flows through London?",
     "ctx": "The Thames is a river that flows through London.",
     "a": "Thames", "topic": 0},
    {"id": "g05", "q": "On which continent is Brazil located?",
     "ctx": "Brazil is the largest country in South America.",
     "a": "South America", "topic": 0},
    {"id": "g06", "q": "What is the largest country by area?",
     "ctx": "Russia is the largest country in the world by total area.",
     "a": "Russia", "topic": 0},
    {"id": "g07", "q": "What is the capital city of Japan?",
     "ctx": "Tokyo is the capital and most populous city of Japan.",
     "a": "Tokyo", "topic": 0},
    {"id": "g08", "q": "What is the tallest mountain in the world?",
     "ctx": "Mount Everest is the tallest mountain in the world.",
     "a": "Mount Everest", "topic": 0},
    # --- Geography test ---
    {"id": "g09", "q": "What is the capital of Australia?",
     "ctx": "Canberra is the capital city of Australia.",
     "a": "Canberra", "topic": 0},
    {"id": "g10", "q": "Which desert is the largest hot desert in the world?",
     "ctx": "The Sahara Desert is the largest hot desert in the world.",
     "a": "Sahara Desert", "topic": 0},

    # ── Science (topic 1) ──────────────────────────────────────────────────
    {"id": "s01", "q": "What gas do plants absorb during photosynthesis?",
     "ctx": "During photosynthesis, plants absorb carbon dioxide from the air.",
     "a": "carbon dioxide", "topic": 1},
    {"id": "s02", "q": "What is the chemical symbol for water?",
     "ctx": "Water is composed of hydrogen and oxygen, with chemical symbol H2O.",
     "a": "H2O", "topic": 1},
    {"id": "s03", "q": "What planet is closest to the Sun?",
     "ctx": "Mercury is the closest planet to the Sun in our solar system.",
     "a": "Mercury", "topic": 1},
    {"id": "s04", "q": "What is the powerhouse of the cell?",
     "ctx": "The mitochondria is often called the powerhouse of the cell.",
     "a": "mitochondria", "topic": 1},
    {"id": "s05", "q": "What force keeps planets in orbit around the Sun?",
     "ctx": "Gravity is the force that keeps planets in orbit around the Sun.",
     "a": "gravity", "topic": 1},
    {"id": "s06", "q": "How many chromosomes do humans typically have?",
     "ctx": "Humans typically have 46 chromosomes in each cell.",
     "a": "46", "topic": 1},
    {"id": "s07", "q": "What is the boiling point of water in Celsius?",
     "ctx": "Water boils at 100 degrees Celsius at sea level.",
     "a": "100 degrees Celsius", "topic": 1},
    {"id": "s08", "q": "What is the most abundant gas in Earth atmosphere?",
     "ctx": "Nitrogen makes up about 78 percent of Earth atmosphere.",
     "a": "nitrogen", "topic": 1},
    # --- Science test ---
    {"id": "s09", "q": "What type of bond holds water molecules together?",
     "ctx": "Water molecules are held together by hydrogen bonds.",
     "a": "hydrogen bonds", "topic": 1},
    {"id": "s10", "q": "What is the atomic number of carbon?",
     "ctx": "Carbon has atomic number 6 on the periodic table.",
     "a": "6", "topic": 1},

    # ── History (topic 2) ──────────────────────────────────────────────────
    {"id": "h01", "q": "In what year did World War II end?",
     "ctx": "World War II ended in 1945 with the surrender of Germany and Japan.",
     "a": "1945", "topic": 2},
    {"id": "h02", "q": "Who was the first President of the United States?",
     "ctx": "George Washington was the first President of the United States.",
     "a": "George Washington", "topic": 2},
    {"id": "h03", "q": "In what year did the French Revolution begin?",
     "ctx": "The French Revolution began in 1789 with the storming of the Bastille.",
     "a": "1789", "topic": 2},
    {"id": "h04", "q": "Who is credited with discovering America in 1492?",
     "ctx": "Christopher Columbus arrived in the Americas in 1492.",
     "a": "Christopher Columbus", "topic": 2},
    {"id": "h05", "q": "In what country did the Industrial Revolution begin?",
     "ctx": "The Industrial Revolution began in England in the 18th century.",
     "a": "England", "topic": 2},
    {"id": "h06", "q": "In what year did the Berlin Wall fall?",
     "ctx": "The Berlin Wall fell in 1989 marking the end of the Cold War.",
     "a": "1989", "topic": 2},
    {"id": "h07", "q": "Who was the first woman to win a Nobel Prize?",
     "ctx": "Marie Curie was the first woman to win a Nobel Prize.",
     "a": "Marie Curie", "topic": 2},
    {"id": "h08", "q": "What ancient wonder was located in Alexandria?",
     "ctx": "The Library of Alexandria was one of the wonders of the ancient world.",
     "a": "Library of Alexandria", "topic": 2},
    # --- History test ---
    {"id": "h09", "q": "In what year did the American Civil War end?",
     "ctx": "The American Civil War ended in 1865 with the Confederate surrender.",
     "a": "1865", "topic": 2},
    {"id": "h10", "q": "Who was the first person to walk on the Moon?",
     "ctx": "Neil Armstrong was the first person to walk on the Moon in 1969.",
     "a": "Neil Armstrong", "topic": 2},

    # ── Sports (topic 3) ───────────────────────────────────────────────────
    {"id": "p01", "q": "In which sport is the Davis Cup awarded?",
     "ctx": "The Davis Cup is an international team competition in tennis.",
     "a": "tennis", "topic": 3},
    {"id": "p02", "q": "How many players are on a basketball team on the court?",
     "ctx": "Each basketball team has five players on the court at a time.",
     "a": "five", "topic": 3},
    {"id": "p03", "q": "What country hosted the 2016 Summer Olympics?",
     "ctx": "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil.",
     "a": "Brazil", "topic": 3},
    {"id": "p04", "q": "In what year were the first modern Olympics held?",
     "ctx": "The first modern Olympics were held in Athens, Greece in 1896.",
     "a": "1896", "topic": 3},
    {"id": "p05", "q": "What is the national sport of Canada?",
     "ctx": "Ice hockey is considered the national sport of Canada.",
     "a": "ice hockey", "topic": 3},
    {"id": "p06", "q": "How many points is a touchdown worth in American football?",
     "ctx": "A touchdown is worth six points in American football.",
     "a": "six", "topic": 3},
    {"id": "p07", "q": "How many holes are in a standard golf course?",
     "ctx": "A standard golf course has 18 holes.",
     "a": "18", "topic": 3},
    {"id": "p08", "q": "What event is Usain Bolt famous for competing in?",
     "ctx": "Usain Bolt is famous for competing in the 100-meter sprint.",
     "a": "100-meter sprint", "topic": 3},
    # --- Sports test ---
    {"id": "p09", "q": "In what sport is the term birdie used?",
     "ctx": "A birdie is a golf term for scoring one under par on a hole.",
     "a": "golf", "topic": 3},
    {"id": "p10", "q": "How long is an Olympic swimming pool in meters?",
     "ctx": "An Olympic swimming pool is 50 meters in length.",
     "a": "50 meters", "topic": 3},
]

# IDs designated for training vs. test (last 2 per topic are held out).
_TEST_IDS = {"g09", "g10", "s09", "s10", "h09", "h10", "p09", "p10"}

# Build convenience lookups.
_ANSWER_LOOKUP: Dict[str, str] = {qa["q"]: qa["a"] for qa in _QA_PAIRS}
_TEXT_TO_TOPIC: Dict[str, int] = {}
for _qa in _QA_PAIRS:
    _TEXT_TO_TOPIC[_qa["q"]] = _qa["topic"]
    _TEXT_TO_TOPIC[_qa["a"]] = _qa["topic"]


# ---------------------------------------------------------------------------
# SQuAD-format builder (for SquadDataset.from_file integration)
# ---------------------------------------------------------------------------

def _build_squad_json(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a flat list of QA dicts into SQuAD v1.1 JSON format."""
    from collections import defaultdict
    topic_names = {0: "Geography", 1: "Science", 2: "History", 3: "Sports"}
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for qa in qa_pairs:
        buckets[qa["topic"]].append(qa)

    data = []
    for topic_id, pairs in sorted(buckets.items()):
        # Group by context paragraph.
        ctx_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for qa in pairs:
            ctx_groups[qa["ctx"]].append(qa)

        paragraphs = []
        for ctx, qs in ctx_groups.items():
            qas = []
            for qa in qs:
                qas.append({
                    "id": qa["id"],
                    "question": qa["q"],
                    "answers": [{"text": qa["a"], "answer_start": 0}],
                    "is_impossible": False,
                })
            paragraphs.append({"context": ctx, "qas": qas})

        data.append({"title": topic_names[topic_id], "paragraphs": paragraphs})

    return {"version": "synthetic-squad-v1.0", "data": data}


# ---------------------------------------------------------------------------
# TopicAwareEncoder – deterministic fake sentence encoder (no GPU / model)
# ---------------------------------------------------------------------------

class TopicAwareEncoder:
    """Deterministic encoder that assigns topic-based embeddings.

    Texts known to belong to a specific topic receive an embedding that is
    strongly aligned with the corresponding basis axis, plus small
    deterministic noise.  Texts from different topics have near-orthogonal
    embeddings, making same-topic answers cluster tightly while cross-topic
    (noisy) answers appear as clear outliers.

    This encoder implements the same ``encode`` interface as
    ``sentence_transformers.SentenceTransformer`` so it can be injected into
    ``VectorKnowledgeBase`` via the ``encoder`` parameter.

    Parameters
    ----------
    text_to_topic:
        Mapping from known text (question or answer) to integer topic ID.
    n_topics:
        Total number of topics / basis directions.
    dim:
        Embedding dimensionality (must be ≥ n_topics).
    noise_scale:
        Scale of within-topic random noise (controls intra-cluster spread).
    """

    def __init__(
        self,
        text_to_topic: Dict[str, int],
        n_topics: int = 4,
        dim: int = 32,
        noise_scale: float = 0.10,
    ) -> None:
        self._text_to_topic = text_to_topic
        self._n_topics = n_topics
        self._dim = dim
        self._noise_scale = noise_scale

        # Build orthogonal basis: topic i → unit vector along axis i*(dim//n_topics)
        self._basis = np.zeros((n_topics, dim), dtype=np.float32)
        stride = max(1, dim // n_topics)
        for i in range(n_topics):
            self._basis[i, i * stride] = 1.0

    # The encode signature matches sentence_transformers.SentenceTransformer.
    def encode(
        self,
        texts,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        vecs = []
        for text in texts:
            tid = self._get_topic(str(text))
            vec = self._basis[tid].copy()
            seed = abs(hash(str(text)[:64])) % (2 ** 31)
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal(self._dim).astype(np.float32) * self._noise_scale
            vec = vec + noise
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    def _get_topic(self, text: str) -> int:
        if text in self._text_to_topic:
            return self._text_to_topic[text]
        text_lower = text.lower().strip()
        for key, tid in self._text_to_topic.items():
            if key.lower().strip() == text_lower:
                return tid
        return 0  # default to topic 0


# ---------------------------------------------------------------------------
# SimulatedLLM – annotation LLM with controllable noise
# ---------------------------------------------------------------------------

class SimulatedLLM:
    """Simulates an annotation LLM that is sometimes over-confidently wrong.

    Implements the ``LLMBase`` interface (``generate`` / ``generate_with_logprobs``).

    Clean mode (probability 1 − noise_rate)
        Returns the ground-truth answer with confidence 0.90 and
        avg_logprob −0.40.  Passes both entry-control thresholds.

    Noisy mode (probability noise_rate)
        Returns a wrong answer drawn from a *different* topic cluster.
        Confidence 0.85 (deliberately above the 0.70 threshold to simulate
        over-confidence), avg_logprob −1.80 (below the −1.00 threshold).
        The only reliable signal for rejection is the avg_logprob.

    Parameters
    ----------
    qa_lookup:
        Mapping from question text to correct answer.
    noise_pool:
        Mapping from question text to a list of wrong cross-topic answers.
    noise_rate:
        Fraction of calls that produce a noisy answer.
    perfect:
        When ``True``, always produces a clean answer regardless of
        ``noise_rate`` (oracle mode).
    seed:
        Random seed for reproducibility.
    """

    CLEAN_CONF: float = 0.90
    CLEAN_LOGPROB: float = -0.40
    NOISY_CONF: float = 0.85   # above conf threshold → confidence alone cannot reject it
    NOISY_LOGPROB: float = -1.80  # below logprob threshold → dual-threshold rejects it

    def __init__(
        self,
        qa_lookup: Dict[str, str],
        noise_pool: Dict[str, List[str]],
        noise_rate: float = 0.30,
        perfect: bool = False,
        seed: int = 42,
    ) -> None:
        self.qa_lookup = qa_lookup
        self.noise_pool = noise_pool
        self.noise_rate = noise_rate
        self.perfect = perfect
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # LLMBase interface
    # ------------------------------------------------------------------

    def generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> Tuple[str, float]:
        question = self._parse_question(prompt)
        correct = self.qa_lookup.get(question, "unknown")

        is_noisy = (
            not self.perfect
            and self._rng.random() < self.noise_rate
            and question in self.noise_pool
            and self.noise_pool[question]
        )

        if is_noisy:
            wrong = self._rng.choice(self.noise_pool[question])
            text = f"Answer: {wrong} Confidence: {self.NOISY_CONF}"
            return text, self.NOISY_LOGPROB
        else:
            text = f"Answer: {correct} Confidence: {self.CLEAN_CONF}"
            return text, self.CLEAN_LOGPROB

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        text, _ = self.generate_with_logprobs(prompt, max_new_tokens)
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_question(prompt: str) -> str:
        """Extract the question text from a QATask-formatted prompt."""
        m = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.I)
        return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# QA metrics (SQuAD-style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _normalize(prediction).split()
    gold_toks = _normalize(ground_truth).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_kb_quality(
    kb_entries: List[Dict[str, Any]],
    answer_lookup: Dict[str, str],
    n_train_total: int,
) -> Dict[str, Any]:
    """Compute knowledge-base annotation quality against ground truth.

    Returns
    -------
    dict with keys: size, n_correct, precision, recall, kb_f1,
                    avg_em, avg_token_f1
    """
    n_accepted = len(kb_entries)
    n_correct = 0
    em_scores, f1_scores = [], []

    for entry in kb_entries:
        q = entry.get("question", "")
        pred = entry.get("annotation", "")
        gold = answer_lookup.get(q, "")
        if gold:
            em = exact_match(pred, gold)
            f1 = token_f1(pred, gold)
            em_scores.append(em)
            f1_scores.append(f1)
            if em:
                n_correct += 1

    precision = n_correct / n_accepted if n_accepted else 0.0
    recall = n_correct / n_train_total if n_train_total else 0.0
    kb_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "size": n_accepted,
        "n_correct": n_correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "kb_f1": round(kb_f1, 4),
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "avg_token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
    }


def evaluate_downstream(
    kb_entries: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    answer_lookup: Dict[str, str],
    encoder: Optional["TopicAwareEncoder"] = None,
    n_train_total: int = 32,
    oracle_em: float = 1.0,
    oracle_f1: float = 1.0,
) -> Dict[str, float]:
    """Simulate downstream QA performance on a fine-tuned model.

    Since real GPU fine-tuning of Qwen-3 0.6B / Llama-3.2 1B is not performed
    here, downstream performance is estimated using a **label-noise degradation
    model** grounded in learning theory (Natarajan et al., 2013):

        EM_sim = max(0, 1 − 2·noise_rate) × coverage × oracle_EM

    where:
        noise_rate = 1 − KB_precision   (fraction of wrong annotations in KB)
        coverage   = n_accepted / n_train_total   (fraction of training data used)
        oracle_EM  = performance achievable with a perfect, fully-covered KB

    This formula captures two key effects that appear in neural fine-tuning:
      1. **Noise degradation**: each percentage point of wrong labels reduces
         expected accuracy by approximately 2× (symmetric noise theorem).
         At 50 % noise the model learns nothing meaningful (EM = 0).
      2. **Coverage effect**: less training data → slightly lower performance,
         modelled as a linear scaling here.

    Parameters
    ----------
    kb_entries:
        Entries accepted into the knowledge base after annotation.
    test_samples:
        Held-out evaluation samples (not used directly in this proxy).
    answer_lookup:
        Ground-truth answer mapping.
    encoder:
        Unused in this model but kept for signature compatibility.
    n_train_total:
        Total number of training samples (before any quality filtering).
    oracle_em:
        Upper-bound EM achievable with perfect annotations (default 1.0).
    oracle_f1:
        Upper-bound token-F1 achievable with perfect annotations (default 1.0).

    Returns
    -------
    dict with keys: downstream_em, downstream_f1
    """
    n_accepted = len(kb_entries)
    if n_accepted == 0:
        return {"downstream_em": 0.0, "downstream_f1": 0.0}

    # Count correctly-annotated KB entries.
    n_correct = 0
    for entry in kb_entries:
        q = entry.get("question", "")
        pred = entry.get("annotation", "")
        gold = answer_lookup.get(q, "")
        if gold and exact_match(pred, gold):
            n_correct += 1

    kb_precision = n_correct / n_accepted
    noise_rate = 1.0 - kb_precision
    coverage = n_accepted / n_train_total if n_train_total > 0 else 0.0

    # Natarajan-style label-noise correction (lower-bounded at 0).
    clean_factor = max(0.0, 1.0 - 2.0 * noise_rate)
    em_sim = oracle_em * clean_factor * coverage
    f1_sim = oracle_f1 * clean_factor * coverage

    return {
        "downstream_em": round(em_sim, 4),
        "downstream_f1": round(f1_sim, 4),
    }


# ---------------------------------------------------------------------------
# Noise pool builder
# ---------------------------------------------------------------------------

def _build_noise_pool(
    qa_pairs: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """For each question, collect ground-truth answers from *other* topics.

    The noisy LLM picks from this pool when it decides to be wrong, ensuring
    that wrong answers are semantically unrelated to the correct answer.
    """
    pool: Dict[str, List[str]] = {}
    for qa in qa_pairs:
        wrong_answers = [
            other["a"]
            for other in qa_pairs
            if other["topic"] != qa["topic"] and other["id"] != qa["id"]
        ]
        pool[qa["q"]] = wrong_answers
    return pool


# ---------------------------------------------------------------------------
# Experiment condition definitions
# ---------------------------------------------------------------------------

_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "oracle_naive": {
        "description": "Perfect LLM – no entry control, no outlier purge (upper-bound oracle)",
        "perfect_llm": True,
        "confidence_threshold": 0.0,
        "avg_logprob_threshold": None,
        "outlier_purge_interval": 0,
        "outlier_z_threshold": 2.0,
    },
    "naive": {
        "description": "Noisy LLM – no entry control, no outlier purge",
        "perfect_llm": False,
        "confidence_threshold": 0.0,
        "avg_logprob_threshold": None,
        "outlier_purge_interval": 0,
        "outlier_z_threshold": 2.0,
    },
    "entry_control": {
        "description": "Noisy LLM – dual-threshold entry control only (conf ≥ 0.70, logprob ≥ −1.00)",
        "perfect_llm": False,
        "confidence_threshold": 0.70,
        "avg_logprob_threshold": -1.00,
        "outlier_purge_interval": 0,
        "outlier_z_threshold": 2.0,
    },
    "purge": {
        "description": "Noisy LLM – periodic outlier purge only (every 10 additions, z ≤ −1.5)",
        "perfect_llm": False,
        "confidence_threshold": 0.0,
        "avg_logprob_threshold": None,
        "outlier_purge_interval": 10,
        "outlier_z_threshold": 1.5,
    },
    "both": {
        "description": "Noisy LLM – dual-threshold entry control + periodic outlier purge",
        "perfect_llm": False,
        "confidence_threshold": 0.70,
        "avg_logprob_threshold": -1.00,
        "outlier_purge_interval": 10,
        "outlier_z_threshold": 1.5,
    },
}


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    condition_cfg: Dict[str, Any],
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    answer_lookup: Dict[str, str],
    noise_pool: Dict[str, List[str]],
    encoder: TopicAwareEncoder,
    noise_rate: float,
    seed: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run annotation for one experimental condition and return all metrics."""
    # Build the simulated LLM for this condition.
    llm = SimulatedLLM(
        qa_lookup=answer_lookup,
        noise_pool=noise_pool,
        noise_rate=noise_rate,
        perfect=condition_cfg["perfect_llm"],
        seed=seed,
    )
    llm_name = "simulated_llm"

    with tempfile.NamedTemporaryFile(
        suffix=f"_{condition_name}.json", delete=False
    ) as fh:
        kb_path = fh.name
    # Remove so VectorKnowledgeBase starts fresh.
    try:
        os.unlink(kb_path)
    except FileNotFoundError:
        pass

    try:
        annotator = Annotator(
            candidate_llms=[llm_name],
            llm_dict={llm_name: llm},
            confidence_threshold=condition_cfg["confidence_threshold"],
            avg_logprob_threshold=condition_cfg["avg_logprob_threshold"],
            rag=False,
            kb_path=kb_path,
            kb_encoder=encoder,
            task=QATask(),
            outlier_purge_interval=condition_cfg["outlier_purge_interval"],
            outlier_z_threshold=condition_cfg["outlier_z_threshold"],
        )

        # Annotate all training samples.
        for sample in train_samples:
            annotator.annotate({**sample, "route": llm_name})

        # ── Collect metrics ────────────────────────────────────────────────
        kb_entries = annotator.knowledge_base.entries

        kb_metrics = evaluate_kb_quality(
            kb_entries, answer_lookup, len(train_samples)
        )
        downstream_metrics = evaluate_downstream(
            kb_entries,
            test_samples,
            answer_lookup,
            n_train_total=len(train_samples),
        )
        human_review_count = len(annotator.human_review_queue.queue)

        # ── Save SFT data file ─────────────────────────────────────────────
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            sft_path = os.path.join(output_dir, f"sft_{condition_name}.jsonl")
            _save_sft(kb_entries, sft_path)
        else:
            sft_path = None

        n_passed_threshold = len(train_samples) - human_review_count
        n_purge_removed = n_passed_threshold - kb_metrics["size"]
        return {
            "condition": condition_name,
            "description": condition_cfg["description"],
            "n_train": len(train_samples),
            "n_accepted": kb_metrics["size"],
            "n_human_review": human_review_count,
            "n_purge_removed": n_purge_removed,
            "kb_precision": kb_metrics["precision"],
            "kb_recall": kb_metrics["recall"],
            "kb_f1": kb_metrics["kb_f1"],
            "kb_avg_em": kb_metrics["avg_em"],
            "kb_avg_token_f1": kb_metrics["avg_token_f1"],
            "downstream_em": downstream_metrics["downstream_em"],
            "downstream_f1": downstream_metrics["downstream_f1"],
            "sft_path": sft_path,
        }
    finally:
        try:
            os.unlink(kb_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# SFT file writer
# ---------------------------------------------------------------------------

def _save_sft(kb_entries: List[Dict[str, Any]], path: str) -> None:
    """Write KB entries as SFT JSONL (instruction / output format).

    Compatible with ``misc/evaluate.py::finetune_sft``.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for entry in kb_entries:
            question = entry.get("question", "")
            context = entry.get("context", "")
            annotation = entry.get("annotation", "")
            instruction = f"Question: {question}\nContext: {context}"
            fh.write(
                json.dumps(
                    {"instruction": instruction, "output": annotation},
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    divider = "=" * 110
    print("\n" + divider)
    print("  KB QUALITY & DOWNSTREAM QA EXPERIMENT RESULTS")
    print(divider)

    # Header
    print(
        f"  {'Condition':<18} | {'KB Size':>7} | {'Reviewd':>7} | {'Purged':>6} | "
        f"{'KB Prec':>8} | {'KB Rec':>7} | {'KB F1':>7} | "
        f"{'KB EM':>7} | {'KB F1s':>7} | {'Dn EM':>7} | {'Dn F1':>7}"
    )
    print("-" * 117)

    condition_order = ["oracle_naive", "naive", "entry_control", "purge", "both"]
    result_map = {r["condition"]: r for r in results}

    for cname in condition_order:
        r = result_map.get(cname)
        if r is None:
            continue
        print(
            f"  {r['condition']:<18} | {r['n_accepted']:>7} | {r['n_human_review']:>7} | "
            f"{r['n_purge_removed']:>6} | "
            f"{r['kb_precision']:>8.1%} | {r['kb_recall']:>7.1%} | {r['kb_f1']:>7.1%} | "
            f"{r['kb_avg_em']:>7.1%} | {r['kb_avg_token_f1']:>7.1%} | "
            f"{r['downstream_em']:>7.1%} | {r['downstream_f1']:>7.1%}"
        )

    divider2 = "=" * 117
    print(divider2)
    print(
        "  Columns: KB Size = entries accepted into knowledge base; "
        "Reviewd = sent to human review; Purged = removed by outlier purge;\n"
        "  KB Prec = precision of accepted annotations (correct/accepted); "
        "KB Rec = recall (correct/all_train);\n"
        "  KB F1 = harmonic mean of KB precision and recall; "
        "KB EM/F1s = per-entry exact-match / token-F1 on accepted entries;\n"
        "  Dn EM/F1 = simulated downstream QA "
        "(label-noise degradation model; see function docstring)."
    )
    print()

    # Performance-loss comparison vs oracle.
    oracle = result_map.get("oracle_naive")
    naive = result_map.get("naive")
    if oracle and naive:
        loss_em = oracle["downstream_em"] - naive["downstream_em"]
        loss_f1 = oracle["downstream_f1"] - naive["downstream_f1"]
        print(
            f"  Performance loss  (oracle_naive → naive):  "
            f"ΔEM = −{loss_em:.1%}, ΔF1 = −{loss_f1:.1%}"
        )
        print(
            "  ↑ This is the performance degradation incurred by using a noisy LLM "
            "with no quality control."
        )
    print(divider + "\n")

    # Per-condition description lines.
    print("  Condition descriptions:")
    for cname in condition_order:
        r = result_map.get(cname)
        if r:
            print(f"    [{cname}] {r['description']}")

    print()
    print(
        "  NOTE: 'Dn EM/F1' uses the Natarajan (2013) label-noise correction:\n"
        "    EM_sim = max(0, 1 − 2·noise_rate) × (n_accepted/n_train) × oracle_EM\n"
        "  where noise_rate = 1 − KB_precision.  This reflects neural fine-tuning\n"
        "  degradation from label noise without requiring actual GPU training.\n"
        "\n"
        "  To run real fine-tuning (Qwen-3 0.6B / Llama-3.2 1B), use the SFT files\n"
        "  written to experiments/output/ with misc/evaluate.py::finetune_sft:\n"
        "    • Qwen-3 0.6B:  python misc/evaluate.py --model Qwen/Qwen3-0.6B "
        "--sft experiments/output/sft_<cond>.jsonl\n"
        "    • Llama-3.2 1B: python misc/evaluate.py --model meta-llama/Llama-3.2-1B "
        "--sft experiments/output/sft_<cond>.jsonl"
    )
    print()


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    noise_rate: float = 0.30,
    seed: int = 42,
    output_dir: Optional[str] = None,
    conditions: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full experiment and return per-condition result dicts.

    Parameters
    ----------
    noise_rate:
        Fraction of annotations the simulated LLM gets wrong (default 0.30).
    seed:
        Random seed for the simulated LLM (default 42).
    output_dir:
        Directory where SFT JSONL files are written.  ``None`` skips writing.
    conditions:
        Subset of condition names to run.  ``None`` runs all five.
    verbose:
        Print progress messages to stdout.

    Returns
    -------
    list of dicts, one per condition.
    """
    if conditions is None:
        conditions = list(_CONDITIONS.keys())

    # ── Build and load synthetic SQuAD dataset via SquadDataset.from_file ──
    all_train = [qa for qa in _QA_PAIRS if qa["id"] not in _TEST_IDS]
    all_test = [qa for qa in _QA_PAIRS if qa["id"] in _TEST_IDS]

    squad_json = _build_squad_json(all_train)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_squad_train.json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(squad_json, fh, ensure_ascii=False)
        squad_path = fh.name

    try:
        ds = SquadDataset.from_file(squad_path, max_samples=len(all_train))
    finally:
        try:
            os.unlink(squad_path)
        except FileNotFoundError:
            pass

    train_samples = ds.to_list()

    # Rebuild answer lookup from the loaded dataset samples.
    answer_lookup: Dict[str, str] = {}
    for sample in train_samples:
        answer_lookup[sample["question"]] = sample["answer"]
    # Also include test samples in lookup for downstream evaluation.
    for qa in all_test:
        answer_lookup[qa["q"]] = qa["a"]

    # Convert test QA pairs to the same dict format as train_samples.
    test_samples = [
        {
            "id": qa["id"],
            "question": qa["q"],
            "context": qa["ctx"],
            "answer": qa["a"],
            "text": f"Question: {qa['q']}\nContext: {qa['ctx']}",
        }
        for qa in all_test
    ]

    noise_pool = _build_noise_pool(_QA_PAIRS)

    # Shared deterministic encoder (injected into VectorKnowledgeBase).
    encoder = TopicAwareEncoder(
        text_to_topic=_TEXT_TO_TOPIC,
        n_topics=4,
        dim=32,
        noise_scale=0.05,  # tight clusters for clear outlier separation
    )

    if verbose:
        print(
            f"\n{'─'*60}\n"
            f"  KB QUALITY EXPERIMENT\n"
            f"  train={len(train_samples)}, test={len(test_samples)}, "
            f"noise_rate={noise_rate:.0%}, seed={seed}\n"
            f"{'─'*60}"
        )

    results = []
    for cname in conditions:
        if cname not in _CONDITIONS:
            raise ValueError(f"Unknown condition: {cname!r}")
        cfg = _CONDITIONS[cname]
        if verbose:
            print(f"  Running: {cname} …", end=" ", flush=True)
        result = run_condition(
            condition_name=cname,
            condition_cfg=cfg,
            train_samples=train_samples,
            test_samples=test_samples,
            answer_lookup=answer_lookup,
            noise_pool=noise_pool,
            encoder=encoder,
            noise_rate=noise_rate,
            seed=seed,
            output_dir=output_dir,
        )
        results.append(result)
        if verbose:
            print(
                f"KB={result['n_accepted']} entries, "
                f"precision={result['kb_precision']:.1%}"
            )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KB quality experiment: entry control vs outlier purge."
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.30,
        metavar="F",
        help="Fraction of noisy (wrong) annotations (default: 0.30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for the simulated LLM (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
        metavar="DIR",
        help="Directory for SFT JSONL output files.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(_CONDITIONS.keys()),
        default=None,
        metavar="COND",
        help="Conditions to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_experiment(
        noise_rate=args.noise_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        conditions=args.conditions,
        verbose=True,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
