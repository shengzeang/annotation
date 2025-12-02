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
- `misc/`: helper scripts and evaluation tools (e.g., `evaluate.py`, `load_squad.py`, `llm_provider.py`).
- `utils.py`: shared utility functions.
