"""Lightweight dataset utilities used across the project.

Provides a minimal Dataset class that wraps a list of dicts and offers
convenience methods similar to the HuggingFace `datasets.Dataset` API
enough for local scripts and filters in this repository.

Features:
- `from_list` / `to_list` for conversion
- `from_json` / `save_json` helpers
- `map` to transform examples
- `filter` to keep examples matching a predicate
- `shuffle` to reorder examples
- `train_test_split` to split into two Dataset objects
- basic indexing/iteration and length
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class Dataset:
	"""Simple in-memory dataset wrapper around a list of dicts.

	Example:
		ds = Dataset.from_list([{"text": "a"}, {"text": "b"}])
		ds2 = ds.map(lambda ex: {**ex, "len": len(ex["text"])})
	"""

	def __init__(self, data: Optional[List[Dict[str, Any]]] = None):
		self._data: List[Dict[str, Any]] = list(data or [])

	@classmethod
	def from_list(cls, data: Iterable[Dict[str, Any]]) -> "Dataset":
		return cls(list(data))

	@classmethod
	def from_json(cls, path: str, encoding: str = "utf-8") -> "Dataset":
		with open(path, encoding=encoding) as f:
			data = json.load(f)
		if isinstance(data, dict):
			# assume a top-level dict with 'data' or similar
			# try common keys, otherwise wrap
			for k in ("data", "examples", "items"):
				if k in data and isinstance(data[k], list):
					return cls(data[k])
			# not a list; return single-element dataset
			return cls([data])
		elif isinstance(data, list):
			return cls(data)
		else:
			raise ValueError("Unsupported JSON format for Dataset.from_json")

	def to_list(self) -> List[Dict[str, Any]]:
		return list(self._data)

	def save_json(self, path: str, ensure_ascii: bool = False, indent: Optional[int] = 2):
		with open(path, "w", encoding="utf-8") as f:
			json.dump(self._data, f, ensure_ascii=ensure_ascii, indent=indent)

	def map(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "Dataset":
		"""Apply a function to each example and return a new Dataset.

		The function should accept a single example dict and return a dict (the transformed example).
		"""
		mapped = [fn(dict(ex)) for ex in self._data]
		return Dataset(mapped)

	def filter(self, fn: Callable[[Dict[str, Any]], bool]) -> "Dataset":
		"""Keep only examples where fn(example) is True."""
		filtered = [ex for ex in self._data if fn(ex)]
		return Dataset(filtered)

	def shuffle(self, seed: Optional[int] = None) -> "Dataset":
		data = list(self._data)
		rng = random.Random(seed)
		rng.shuffle(data)
		return Dataset(data)

	def train_test_split(self, test_size: float = 0.2, seed: Optional[int] = None) -> Tuple["Dataset", "Dataset"]:
		"""Split dataset into (train, test).

		`test_size` can be a float fraction or an integer count.
		"""
		n = len(self._data)
		indices = list(range(n))
		rng = random.Random(seed)
		rng.shuffle(indices)
		if 0 < test_size < 1:
			k = int(n * test_size)
		else:
			k = int(test_size)
		test_idx = set(indices[:k])
		train = [self._data[i] for i in indices if i not in test_idx]
		test = [self._data[i] for i in indices if i in test_idx]
		return Dataset(train), Dataset(test)

	def select(self, indices: Iterable[int]) -> "Dataset":
		sel = [self._data[i] for i in indices]
		return Dataset(sel)

	def take(self, k: int) -> "Dataset":
		return Dataset(self._data[:k])

	def __len__(self) -> int:
		return len(self._data)

	def __iter__(self):
		return iter(self._data)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		return self._data[idx]

	def __repr__(self) -> str:
		return f"Dataset(num_examples={len(self)})"

	@classmethod
	def from_squad(
		cls,
		path: str = "squad_train.json",
		max_samples: int = 200,
		skip_initial: int = 500,
		shuffle_seed: Optional[int] = None,
	) -> "Dataset":
		"""Create a Dataset of QA pairs from a SQuAD v1.1 JSON file.

		Args:
			path: path to the SQuAD JSON file.
			max_samples: maximum number of QA examples to include.
			skip_initial: number of examples to skip before collecting (keeps prior behaviour).
			shuffle_seed: optional random seed to shuffle the resulting dataset.

		Returns:
			Dataset of dicts: {"id","question","context","answer","text"}
		"""
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


def load_json_file(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
	with open(path, encoding=encoding) as f:
		return json.load(f)


def save_json_file(path: str, data: List[Dict[str, Any]], ensure_ascii: bool = False, indent: Optional[int] = 2):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

