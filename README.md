## Annotation Repository

This repository provides a lightweight Python framework for annotation workflows, routing/ filtering strategies, and active learning experiments. It collects several router and filter implementations plus evaluation utilities to help build and compare annotation pipelines.

### Main use cases
- Research and experiments: compare routing and filtering strategies for annotation efficiency and quality.
- Small-scale annotation runs: serve as a skeleton for task distribution, result collection, and evaluation.

### Key modules (examples)
- `annotation.py`: core annotation logic and entry points.
- `task.py`: task management utilities.
- `routing.py` / `routers/`: routing abstractions and implementations (cascade, knn, mlp, llm, etc.).
- `filtering.py` / `filters/`: filters used to select or prune candidate items.
- `annotators/`: custom annotator implementations.
- `misc/`: helper scripts and evaluation tools (e.g., `evaluate.py`, `llm_provider.py`).
- `utils.py`: shared utility functions.
- `datasets/`: small package with dataset helpers and thin dataset classes for common QA datasets.
	- `SquadDataset`: SQuAD v1.1 parser and helpers (`from_file`, `from_url`, `to_sft`, `save_sft`).
	- `CommonQADataset`: generic QA parser with heuristics to extract `question`, `context`, `answer` and produce examples compatible with the rest of the codebase.
	- `HotpotDataset`, `TriviaQADataset`, `NQDataset`: thin wrappers around `CommonQADataset` for common dataset file formats.
  
Example usage:

```python
from datasets import SquadDataset

# Download and load SQuAD (won't re-download if file exists):
ds = SquadDataset.from_url(save_path="squad_train.json")

# Load local file and convert to SFT JSONL for fine-tuning:
ds = SquadDataset.from_file("squad_train.json", max_samples=1000)
ds.save_sft("squad_sft.jsonl")
```
