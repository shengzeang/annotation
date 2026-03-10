import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_kb_experiment import (
    ConditionConfig,
    OracleLLM,
    SimulatedLLM,
    run_condition,
    run_experiment,
)

PATH_NOT_FOUND = "-1"
FIRST_POSITION = "0"


def _make_samples(n=30):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": str(i),
                "question": f"What is token {i}?",
                "context": f"The correct answer is token-{i}.",
                "answer": f"token-{i}",
                "text": f"Question: What is token {i}?\nContext: The correct answer is token-{i}.",
            }
        )
    return rows


class TestSimulatedLLM(unittest.TestCase):
    def test_noisy_mode_can_emit_high_confidence_low_quality_answer(self):
        key = {"Q1": "A1"}
        llm = SimulatedLLM(answer_key=key, rng_seed=1, error_rate=1.0, high_confidence_error_rate=1.0)
        output, avg_logprob = llm.generate_with_logprobs("Question: Q1\nContext: C1")
        self.assertIn("Confidence: 0.95", output)
        self.assertNotIn("A1", output)
        self.assertLess(avg_logprob, -1.0)

    def test_oracle_mode_is_perfect(self):
        key = {"Q1": "A1"}
        llm = OracleLLM(answer_key=key)
        output, avg_logprob = llm.generate_with_logprobs("Question: Q1\nContext: C1")
        self.assertIn("A1", output)
        self.assertIn("0.99", output)
        self.assertGreater(avg_logprob, -0.1)


class TestExperimentConditions(unittest.TestCase):
    def test_run_kb_experiment_does_not_prepend_project_root_to_sys_path(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.join(repo_root, "experiments", "run_kb_experiment.py")
        script_path_literal = json.dumps(script_path)
        repo_root_literal = json.dumps(repo_root)
        self.assertTrue(os.path.exists(script_path))
        self.assertTrue(os.path.abspath(script_path).startswith(repo_root + os.sep))
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "import importlib.util; "
                f"spec=importlib.util.spec_from_file_location('run_kb_experiment_test_mod', {script_path_literal}); "
                "mod=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod); "
                f"root={repo_root_literal}; "
                "idx=next((index for index, path in enumerate(sys.path) "
                "if os.path.abspath(path or os.getcwd())==root), -1); "
                "print(idx)"
            ),
        ]
        proc = subprocess.run(cmd, cwd=tempfile.gettempdir(), check=True, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(proc.stdout.strip(), PATH_NOT_FOUND, "project root should be added to sys.path")
        self.assertNotEqual(proc.stdout.strip(), FIRST_POSITION, "project root must not be prepended at sys.path[0]")

    def test_run_kb_experiment_does_not_import_local_datasets_package(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.join(repo_root, "experiments", "run_kb_experiment.py")
        script_path_literal = json.dumps(script_path)
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "import importlib.util; "
                f"spec=importlib.util.spec_from_file_location('run_kb_experiment_test_mod', {script_path_literal}); "
                "mod=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod); "
                "datasets_mod = sys.modules.get('datasets'); "
                "print(getattr(datasets_mod, '__file__', ''))"
            ),
        ]
        proc = subprocess.run(cmd, cwd=tempfile.gettempdir(), check=True, capture_output=True, text=True, timeout=30)
        self.assertNotIn(
            f"{os.sep}annotation{os.sep}datasets{os.sep}__init__.py",
            proc.stdout.strip(),
            "run_kb_experiment should not register local datasets package as top-level 'datasets'",
        )

    def test_entry_control_reduces_kb_contamination_vs_naive(self):
        samples = _make_samples(40)
        train, eval_set = samples[:30], samples[30:]
        answer_key = {s["question"]: s["answer"] for s in train}
        noisy = SimulatedLLM(answer_key=answer_key, rng_seed=0, error_rate=0.5, high_confidence_error_rate=1.0)

        with tempfile.TemporaryDirectory() as td:
            naive = run_condition(
                train_set=train,
                eval_set=eval_set,
                llm_obj=noisy,
                cfg=ConditionConfig("naive", 0.0, None, 0),
                output_dir=td,
            )
            entry = run_condition(
                train_set=train,
                eval_set=eval_set,
                llm_obj=noisy,
                cfg=ConditionConfig("entry_control_only", 0.7, -1.0, 0),
                output_dir=td,
            )
        self.assertGreater(naive["kb_contamination_rate"], entry["kb_contamination_rate"])

    def test_run_experiment_outputs_all_conditions(self):
        fake_samples = _make_samples(500)
        with tempfile.TemporaryDirectory() as td:
            with patch("experiments.run_kb_experiment.load_squad_500", return_value=fake_samples):
                summary = run_experiment(output_dir=td, sample_count=500, run_finetune=True, squad_cache_path="unused.json")

            names = [c["condition"] for c in summary["conditions"]]
            self.assertEqual(
                set(names),
                {
                    "naive",
                    "entry_control_only",
                    "outlier_purge_only",
                    "entry_control_and_outlier_purge",
                    "naive_oracle",
                },
            )
            self.assertEqual(summary["sample_count"], 500)
            self.assertTrue(os.path.exists(summary["summary_path"]))
            with open(summary["summary_path"], "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(len(saved["conditions"]), 5)
            self.assertIn("models", saved)


if __name__ == "__main__":
    unittest.main()
