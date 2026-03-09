"""
Unit tests for base_structure.dataset module:
  - Dataset
  - DatasetStorage
  - SquadDataset / CommonQADataset (file-based)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):

    def _make_ds(self, n=4):
        from base_structure.dataset import Dataset
        return Dataset.from_list([{"id": i, "text": f"item {i}"} for i in range(n)])

    # --- construction ---

    def test_from_list_creates_correct_length(self):
        ds = self._make_ds(n=5)
        self.assertEqual(len(ds), 5)

    def test_from_list_empty(self):
        from base_structure.dataset import Dataset
        ds = Dataset.from_list([])
        self.assertEqual(len(ds), 0)

    def test_to_list_roundtrip(self):
        from base_structure.dataset import Dataset
        data = [{"a": 1}, {"a": 2}]
        ds = Dataset.from_list(data)
        self.assertEqual(ds.to_list(), data)

    # --- indexing and iteration ---

    def test_getitem_returns_correct_item(self):
        ds = self._make_ds(n=3)
        self.assertEqual(ds[0]["id"], 0)
        self.assertEqual(ds[2]["id"], 2)

    def test_iter_covers_all_items(self):
        ds = self._make_ds(n=4)
        ids = [item["id"] for item in ds]
        self.assertEqual(sorted(ids), [0, 1, 2, 3])

    # --- map ---

    def test_map_transforms_examples(self):
        ds = self._make_ds(n=3)
        mapped = ds.map(lambda x: {**x, "doubled": x["id"] * 2})
        for item in mapped:
            self.assertEqual(item["doubled"], item["id"] * 2)

    def test_map_returns_new_dataset(self):
        ds = self._make_ds(n=2)
        mapped = ds.map(lambda x: {**x, "new_field": "v"})
        self.assertIsNot(ds, mapped)

    # --- filter ---

    def test_filter_keeps_matching_items(self):
        ds = self._make_ds(n=5)
        filtered = ds.filter(lambda x: x["id"] % 2 == 0)
        ids = [item["id"] for item in filtered]
        self.assertEqual(sorted(ids), [0, 2, 4])

    def test_filter_returns_empty_when_nothing_matches(self):
        ds = self._make_ds(n=3)
        filtered = ds.filter(lambda x: False)
        self.assertEqual(len(filtered), 0)

    # --- shuffle ---

    def test_shuffle_preserves_length(self):
        ds = self._make_ds(n=10)
        shuffled = ds.shuffle(seed=42)
        self.assertEqual(len(shuffled), len(ds))

    def test_shuffle_preserves_all_elements(self):
        ds = self._make_ds(n=10)
        shuffled = ds.shuffle(seed=42)
        original_ids = sorted(item["id"] for item in ds)
        shuffled_ids = sorted(item["id"] for item in shuffled)
        self.assertEqual(original_ids, shuffled_ids)

    def test_shuffle_deterministic_with_same_seed(self):
        ds = self._make_ds(n=10)
        s1 = ds.shuffle(seed=7).to_list()
        s2 = ds.shuffle(seed=7).to_list()
        self.assertEqual(s1, s2)

    # --- train_test_split ---

    def test_train_test_split_fraction(self):
        from base_structure.dataset import Dataset
        ds = Dataset.from_list([{"i": i} for i in range(10)])
        train, test = ds.train_test_split(test_size=0.2, seed=0)
        self.assertEqual(len(train) + len(test), 10)
        self.assertEqual(len(test), 2)

    def test_train_test_split_no_overlap(self):
        from base_structure.dataset import Dataset
        ds = Dataset.from_list([{"i": i} for i in range(10)])
        train, test = ds.train_test_split(test_size=0.3, seed=0)
        train_ids = {item["i"] for item in train}
        test_ids = {item["i"] for item in test}
        self.assertEqual(train_ids & test_ids, set())

    # --- select / take ---

    def test_select_returns_correct_items(self):
        ds = self._make_ds(n=5)
        sel = ds.select([0, 2, 4])
        ids = [item["id"] for item in sel]
        self.assertEqual(sorted(ids), [0, 2, 4])

    def test_take_returns_first_k(self):
        ds = self._make_ds(n=5)
        taken = ds.take(3)
        self.assertEqual(len(taken), 3)
        self.assertEqual(taken[0]["id"], 0)

    # --- from_json / save_json ---

    def test_from_json_list(self):
        from base_structure.dataset import Dataset
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"x": 1}, {"x": 2}], f)
            fname = f.name
        try:
            ds = Dataset.from_json(fname)
            self.assertEqual(len(ds), 2)
        finally:
            os.unlink(fname)

    def test_save_and_reload_json(self):
        from base_structure.dataset import Dataset
        ds = Dataset.from_list([{"k": "v1"}, {"k": "v2"}])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            ds.save_json(fname)
            ds2 = Dataset.from_json(fname)
            self.assertEqual(ds.to_list(), ds2.to_list())
        finally:
            os.unlink(fname)

    # --- repr ---

    def test_repr_includes_length(self):
        ds = self._make_ds(n=7)
        self.assertIn("7", repr(ds))


# ---------------------------------------------------------------------------
# DatasetStorage
# ---------------------------------------------------------------------------

class TestDatasetStorage(unittest.TestCase):

    def test_write_list_and_read_dataset(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        storage.write([{"x": 1}, {"x": 2}])
        ds = storage.read()
        self.assertEqual(len(ds), 2)

    def test_write_dict_wraps_in_list(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        storage.write({"x": 1})
        ds = storage.read()
        self.assertEqual(len(ds), 1)

    def test_read_before_write_returns_none(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        self.assertIsNone(storage.read())

    def test_read_as_list(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        storage.write([{"a": 1}])
        result = storage.read(output_type="list")
        self.assertIsInstance(result, list)

    def test_get_keys_empty_storage(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        self.assertEqual(storage.get_keys_from_dataframe(), [])

    def test_get_keys_returns_dict_keys(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        storage.write([{"name": "Alice", "age": 30}])
        keys = storage.get_keys_from_dataframe()
        self.assertIn("name", keys)
        self.assertIn("age", keys)

    def test_overwrite_replaces_data(self):
        from base_structure.dataset import DatasetStorage
        storage = DatasetStorage()
        storage.write([{"v": 1}])
        storage.write([{"v": 2}, {"v": 3}])
        ds = storage.read()
        self.assertEqual(len(ds), 2)


# ---------------------------------------------------------------------------
# SquadDataset / CommonQADataset (file-based, no network)
# ---------------------------------------------------------------------------

class TestSquadDataset(unittest.TestCase):

    def _make_squad_file(self):
        """Write a minimal SQuAD-format JSON to a temp file."""
        squad = {
            "data": [
                {
                    "title": "Test Article",
                    "paragraphs": [
                        {
                            "context": "France is a country in Western Europe.",
                            "qas": [
                                {
                                    "id": "qa1",
                                    "question": "Where is France?",
                                    "answers": [{"text": "Western Europe", "answer_start": 26}],
                                    "is_impossible": False,
                                },
                                {
                                    "id": "qa2",
                                    "question": "What is France?",
                                    "answers": [{"text": "a country", "answer_start": 9}],
                                    "is_impossible": False,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(squad, f)
        return path

    def test_from_file_returns_correct_length(self):
        from datasets.qa_datasets import SquadDataset
        path = self._make_squad_file()
        try:
            ds = SquadDataset.from_file(path)
            self.assertEqual(len(ds), 2)
        finally:
            os.unlink(path)

    def test_from_file_max_samples_respected(self):
        from datasets.qa_datasets import SquadDataset
        path = self._make_squad_file()
        try:
            ds = SquadDataset.from_file(path, max_samples=1)
            self.assertEqual(len(ds), 1)
        finally:
            os.unlink(path)

    def test_from_file_has_required_keys(self):
        from datasets.qa_datasets import SquadDataset
        path = self._make_squad_file()
        try:
            ds = SquadDataset.from_file(path)
            for item in ds:
                for key in ("id", "question", "context", "answer", "text"):
                    self.assertIn(key, item)
        finally:
            os.unlink(path)

    def test_from_file_skips_impossible_questions(self):
        from datasets.qa_datasets import SquadDataset
        squad = {
            "data": [{
                "title": "T",
                "paragraphs": [{
                    "context": "ctx",
                    "qas": [
                        {"id": "q1", "question": "Possible?", "answers": [{"text": "yes"}], "is_impossible": False},
                        {"id": "q2", "question": "Impossible?", "answers": [], "is_impossible": True},
                    ]
                }]
            }]
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(squad, f)
        try:
            ds = SquadDataset.from_file(path)
            self.assertEqual(len(ds), 1)
        finally:
            os.unlink(path)

    def test_to_sft_format(self):
        from datasets.qa_datasets import SquadDataset
        path = self._make_squad_file()
        try:
            ds = SquadDataset.from_file(path)
            sft = ds.to_sft()
            self.assertEqual(len(sft), 2)
            for item in sft:
                self.assertIn("instruction", item)
                self.assertIn("output", item)
        finally:
            os.unlink(path)


class TestCommonQADataset(unittest.TestCase):

    def test_from_file_list_format(self):
        from datasets.qa_datasets import CommonQADataset
        data = [
            {"id": "1", "question": "Q1?", "context": "ctx1", "answer": "A1"},
            {"id": "2", "question": "Q2?", "context": "ctx2", "answer": "A2"},
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        try:
            ds = CommonQADataset.from_file(path)
            self.assertEqual(len(ds), 2)
        finally:
            os.unlink(path)

    def test_from_file_single_dict_format(self):
        from datasets.qa_datasets import CommonQADataset
        data = {"question": "Q?", "context": "ctx", "answer": "A"}
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        try:
            ds = CommonQADataset.from_file(path)
            self.assertEqual(len(ds), 1)
        finally:
            os.unlink(path)

    def test_save_sft_writes_jsonl(self):
        from datasets.qa_datasets import CommonQADataset
        data = [{"question": "Q?", "context": "ctx", "answer": "A", "text": "Q? ctx", "id": "1"}]
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        out_fd, out_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(out_fd)
        try:
            ds = CommonQADataset.from_file(path)
            ds.save_sft(out_path)
            with open(out_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertIn("instruction", record)
            self.assertIn("output", record)
        finally:
            os.unlink(path)
            os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()
