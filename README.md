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

# Repository Structure: 
## Base Structure

### `base_structure/_kmeans_subproc.py`

Subprocess entrypoint to run KMeans safely in a separate process.
This script is invoked with: python -m base_structure._kmeans_subproc <in.npy> <out.npy> <n_clusters> <mb_batch> <use_mini>
It writes the cluster centers to <out.npy> on success and exits non-zero on failure.

### `base_structure/active_learning.py`

Active learning module and sampling strategies.
Allows different text representations and sampling strategies.

**DataPool**

Data pool containing text samples and corresponding IDs

**Embeddings**

Base class for embedding conversion

**Selector**

Base class for active learning samplers. Subclasses must implement select_indices. 
Fall-back to simple selection in subclass.

**BertEmbeddings**

BERT text embedding

**BertKM**

BERT + KMean sampling strategy.
Selects closet samples to each of K cluster centres. 

**SurprisalEmbeddings**

MLM surprisal embeddings

**ALPS**

ALPS Selector.
Ensures exact number of unique samples by supplementing
cluser representatives with random picks when duplicates occur.

### `base_structure/base_filter.py`
**BaseFilter**

Abstract base class for filters.

### `base_structure/base_router.py`
**BaseRouter**

Abstract router interface for scoring candidate LLMs.
Decoupled from specific tasks or datasets with cold-start training mechanism. 

### `base_structure/base_task.py`

Abstraction for annotation tasks.

Defines the base interface for annotation operations.
Includes abstract methods of generating LLMs prompts and parsing results

### `base_structure/dataset.py`

Lightweight dataset utilities used across the project.

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

**Dataset**

Simple in-memory dataset wrapper around a list of dicts.

Example:
        ds = Dataset.from_list([{"text": "a"}, {"text": "b"}])
        ds2 = ds.map(lambda ex: {**ex, "len": len(ex["text"])})

**DatasetStorage**

Simple storage wrapper around `Dataset` that exposes a minimal
`write`/`read` API compatible with dataflow-style operators.

Methods:
  - write(data): accept a `Dataset`, list of dicts, or dict and store as `Dataset`.
  - read(output_type="dataset"): return `Dataset` (default), list, or pandas.DataFrame (lazy).
  - get_keys_from_dataframe(): return list of keys for first example.

## Filters

### `filters/al_filter.py`
**ActiveLearningFilter**

Active Learning filter implementation.
Supported selection methods: "alps", "bertkm".

### `filters/dataflow_filter.py`
**DataFlowFilter**

Adapter that lets repository filters use operators from a DataFlow-like
operator registry, or accept a custom operator class directly.

Usage:
  - Pass `operator_name` to load via `dataflow_operator_import.get_operator`
    (when that module and `dataflow` operators are available).
  - Or pass `operator_class` directly (recommended if you want to avoid
    the dynamic import dependency).

### `filters/llm_filter.py`
**LLMNaiveFilter**

LLM-based naive filter implementation.
Ranks all samples by LLM-rated score and returns top N samples.

## Routers

### `routers/cascade_router.py`
**CascadeRouter**

FrugalGPT-style cascade router.
Calls models until output > confidence threshold.
Pass candidate LLM models to generate answers and judge LLM for evaluation.

### `routers/graph_router.py`
**GraphRouter**

Graph-based router: builds a bipartite graph between samples and models and propagates scores.
Routing decisions are based on semantic similarity and graph-based score propagation

### `routers/knn_router.py`
**KNNRouter**

KNN-based router using historical annotated samples.

### `routers/llm_router.py`
**LLMRouter**

Scores candidate LLMs for each sample using a scoring LLM (which can be local or API-backed).

### `routers/mlp_router.py`
**MLPRouter**

A simple MLP-based router that uses encoded features of (sample, candidate_name)
pairs and predicts a score in [0,1]. Supports training from labeled pairs and
scoring new samples.

### `routers/routerdc_router.py`
**RouterDCRouter**

Router-DC-style

For each LLM, compute a "model identity vector" from our_anno.json
During inference, encode the query and pick the most similar LLM.(comparing the query embedding with the model embedding)

## Tasks

### `tasks/classification.py`

**ClassificationTask**

Classification of texts into predefined categories

### `tasks/ner.py`

**NERTask**

### `tasks/qa.py`
**QATask**

Question answering with confidence score output

### `tasks/summary.py`
**TextSummarization**

Text summary within predefined length.

### `tasks/translation.py`
**Translation**

Translation task from English to Chinese. supporting dictionary hints.