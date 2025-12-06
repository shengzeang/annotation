"""Additional QA dataset classes built on top of `dataset.Dataset`.

This module provides a flexible `CommonQADataset` that extracts question/context/answer
using a set of heuristics and several thin wrappers for common QA datasets:
- `HotpotDataset`
- `TriviaQADataset`
- `NQDataset` (Natural Questions short-format)

Each class exposes `from_file`, `from_url`, `to_sft` and `save_sft` similar to `SquadDataset`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

from base_structure.dataset import Dataset


def _extract_field(d: Dict[str, Any], keys: Iterable[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


class CommonQADataset(Dataset):
    """A flexible QA dataset that tries multiple heuristics to extract
    question/context/answer from common QA JSON formats.

    The produced examples have keys: `id`, `question`, `context`, `answer`, `text`.
    """

    @classmethod
    def from_file(
        cls,
        path: str,
        max_samples: int = 200,
        shuffle_seed: Optional[int] = None,
    ) -> "CommonQADataset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Accept either a dict with 'data' or a top-level list
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            items = data["data"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # single record
            items = [data]
        else:
            raise ValueError(f"Unsupported dataset format for {path}")

        out: List[Dict[str, Any]] = []
        for rec in items:
            if len(out) >= max_samples:
                break
            # heuristics to find question
            question = _extract_field(rec, ("question", "ques", "query")) or ""
            # heuristics to find context/passages
            context = _extract_field(rec, ("context", "passage", "document_text", "article", "context_text"))
            if context is None:
                # Hotpot-style 'context' can be a list of (title, sentences)
                ctx = _extract_field(rec, ("context", "contexts", "paragraphs"))
                if isinstance(ctx, list):
                    # join sublists/tuples
                    pieces: List[str] = []
                    for c in ctx:
                        if isinstance(c, (list, tuple)):
                            # expect [title, sentences]
                            title = c[0]
                            sents = c[1]
                            if isinstance(sents, list):
                                pieces.append(" ".join(sents))
                            elif isinstance(sents, str):
                                pieces.append(sents)
                        elif isinstance(c, dict):
                            pieces.append(" ".join(c.get("paragraph", "").splitlines()))
                        elif isinstance(c, str):
                            pieces.append(c)
                    context = "\n".join(pieces)
            if context is None:
                context = ""

            # heuristics to find answer(s)
            answer = _extract_field(rec, ("answer", "answers", "gold_answer", "label")) or ""
            if isinstance(answer, list):
                # take first string if list
                answer = next((a for a in answer if isinstance(a, str)), "")

            sid = _extract_field(rec, ("id", "_id", "qid")) or None

            text = f"Question: {question}\nContext: {context}"
            out.append({"id": sid, "question": question, "context": context, "answer": answer, "text": text})

        ds = cls.from_list(out)
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)
        return ds

    @classmethod
    def from_url(
        cls,
        url: str,
        save_path: str,
        overwrite: bool = False,
        **kwargs,
    ) -> "CommonQADataset":
        if not overwrite and os.path.exists(save_path):
            return cls.from_file(save_path, **kwargs)
        urllib.request.urlretrieve(url, save_path)
        return cls.from_file(save_path, **kwargs)

    def to_sft(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for ex in self._data:
            out.append({"instruction": ex.get("text", ""), "output": ex.get("answer", "")})
        return out

    def save_sft(self, path: str, ensure_ascii: bool = False) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.to_sft():
                f.write(json.dumps(rec, ensure_ascii=ensure_ascii) + "\n")


class HotpotDataset(CommonQADataset):
    @classmethod
    def from_file(cls, path: str = "hotpot_train_v1.json", max_samples: int = 200, shuffle_seed: Optional[int] = None) -> "HotpotDataset":
        return super().from_file(path=path, max_samples=max_samples, shuffle_seed=shuffle_seed)


class TriviaQADataset(CommonQADataset):
    @classmethod
    def from_file(cls, path: str = "triviaqa_train.json", max_samples: int = 200, shuffle_seed: Optional[int] = None) -> "TriviaQADataset":
        return super().from_file(path=path, max_samples=max_samples, shuffle_seed=shuffle_seed)


class NQDataset(CommonQADataset):
    @classmethod
    def from_file(cls, path: str = "nq_train.json", max_samples: int = 200, shuffle_seed: Optional[int] = None) -> "NQDataset":
        return super().from_file(path=path, max_samples=max_samples, shuffle_seed=shuffle_seed)


class SquadDataset(CommonQADataset):
    """SQuAD v1.1 specific parser implemented as a thin subclass of CommonQADataset.

    This preserves the previously exposed `SquadDataset` API while reusing the
    shared utilities in `CommonQADataset`.
    """

    @classmethod
    def from_file(
        cls,
        path: str = "squad_train.json",
        max_samples: int = 200,
        skip_initial: int = 0,
        shuffle_seed: Optional[int] = None,
    ) -> "SquadDataset":
        # SQuAD has a top-level 'data' list of articles -> paragraphs -> qas
        with open(path, encoding="utf-8") as f:
            squad = json.load(f)
        qa_list: List[Dict[str, Any]] = []
        i = 0
        for article in squad.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    if qa.get("is_impossible", False):
                        continue
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    answer = answers[0].get("text", "") if answers else ""
                    i += 1
                    if i < skip_initial:
                        continue
                    qa_list.append(
                        {
                            "id": qa.get("id"),
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "text": f"Question: {question}\nContext: {context}",
                        }
                    )
                    if len(qa_list) >= max_samples:
                        break
                if len(qa_list) >= max_samples:
                    break
            if len(qa_list) >= max_samples:
                break
        ds = cls.from_list(qa_list)
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)
        return ds

    @classmethod
    def from_url(
        cls,
        url: str = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        save_path: str = "squad_train.json",
        overwrite: bool = False,
        **kwargs,
    ) -> "SquadDataset":
        if not overwrite and os.path.exists(save_path):
            return cls.from_file(save_path, **kwargs)
        urllib.request.urlretrieve(url, save_path)
        return cls.from_file(save_path, **kwargs)
