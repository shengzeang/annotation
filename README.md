# DataFlow-Annotator — Unified Data Filtering, Routing, and Annotation Framework

DataFlow-Annotator enables seamless integration of automated data governance and human intervention. In this unified workflow, data progresses through filtering, routing, and annotation stages within the same pipeline, eliminating the need to switch between different systems. Specifically, DataFlow-Annotator provides three key operators: Filter, Router, and Annotator.

- **Filter Operators**: Streamline data filtering through multiple consecutive Filter operators to extract the most valuable samples for the current task, leveraging strategies like active learning or data quality evaluation.
- **Router Operators**: Dynamically allocate samples to the most suitable model within a user-defined set of candidate LLMs using algorithms such as MLP, KNN, Graph, or LLM-based routing, enhancing annotation efficiency and consistency.
- **Annotator Operators**: Integrate with human annotation or review systems (e.g., UniMiner), supporting real-time progress tracking and result collection. Annotator operators route low-confidence samples back to human annotators while high-confidence samples are stored in a knowledge base. This knowledge base serves as backend storage for RAG (Retrieval-Augmented Generation), improving model annotation accuracy over time.

---

## 🌟 Key Features

- **Unified Workflow**: Streamline data filtering, routing, and annotation in a single pipeline without switching systems.
- **Streamlined Filtering**: Extract the most valuable samples using Filter operators based on active learning strategies or data quality evaluation.
- **Dynamic Routing**: Allocate samples dynamically to the optimal model using Router operators powered by MLP, KNN, Graph, or LLM algorithms.
- **Efficient Annotation**: Annotator operators integrate with human annotation systems, enabling real-time progress tracking and result collection.
- **RAG Support**: Built-in knowledge base as backend storage for RAG, enhancing model annotation accuracy over time.
- **Modular Design**: Flexible Router, Filter, and Annotator interfaces for easy extension and experimentation.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/annotation.git
cd annotation
```

### 2. Set Up the Environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the HTTP Server

```bash
python -m api.server
# or
python api/server.py
```

### 4. Run the Frontend Interface

```bash
cd frontend
npm install
npm run dev
```

### 5. Access the API

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to interact with the provided endpoints or run pipeline graphs.

### 6. Explore the Frontend Interface

The frontend of DataFlow-Annotator is built using **React-Flow** for the user interface and **Python Flask** for the backend. It provides an intuitive way to create and manage annotation pipelines. Follow these steps to get started:

1. **Pipeline Editor**: Drag and drop nodes (e.g., Filter, Router, Annotator) to build pipelines and visualize the data flow.
2. **Node Editor**: Click on a node to edit its parameters directly in the interface.
3. **Review Queue**: Access the review queue to manage and process data requiring human intervention. This section displays data that needs manual review and allows users to annotate and track progress in real-time.

---

## 📂 Project Layout

- **`annotation.py`**: Core annotator classes and human review queue.
- **`api/`**: Contains `server.py`, a Flask-based runner for routers, filters, and graph pipelines.
- **`base_structure/`**: Abstract base classes and active-learning implementations.
- **`routers/`**: Router implementations (cascade, knn, mlp, llm, routerdc, graph, etc.).
- **`filters/`**: Filter implementations (active learning, dataflow adapters, LLM-based filters).
- **`tasks/`**: Task definitions and parsing logic (QA, classification, NER, summarization, translation).
- **`datasets/`**: Lightweight dataset adapters and converters (SQuAD helpers, generic QA parsers).
- **`misc/`**: Helper scripts, evaluation tools, and providers (e.g., `evaluate.py`, `llm_provider.py`).
- **`experiments/`**: Self-contained experiment scripts (no GPU required) that compare annotation conditions and estimate downstream fine-tuning performance.

---

## 🧪 Experiments

### Label Studio vs DataFlow-Annotator — QA Annotation Comparison

`experiments/run_label_studio_comparison.py` benchmarks DataFlow-Annotator
against Label Studio–style human annotation using **QA fine-tuning as the
downstream evaluation task**.  It is fully self-contained (no GPU, no network
access required) and uses a Natarajan noise-degradation model to estimate the
performance of a small LLM fine-tuned on each annotated dataset.

**Five conditions are compared:**

| Condition | Description |
|---|---|
| Label Studio (3 annotators) | Majority vote across 3 human annotators (85 % per-annotator accuracy) |
| Label Studio (1 annotator) | Single human annotator (75 % accuracy) |
| DataFlow (naive LLM) | Baseline LLM annotation without KB/RAG |
| DataFlow (KB + RAG) | LLM annotation with knowledge-base retrieval augmentation |
| DataFlow (full pipeline) | KB + RAG + periodic outlier purge |

**Metrics reported:**

- `Ann-F1` / `Ann-EM` — mean token-level F1 and exact-match of annotations vs ground truth
- `DS-EM` / `DS-F1` — estimated downstream exact-match / F1 of a small LLM fine-tuned on the annotated data

**Usage:**

```bash
# Synthetic data (no SQuAD file needed)
python experiments/run_label_studio_comparison.py --samples 500

# With a local SQuAD v1.1 JSON file
python experiments/run_label_studio_comparison.py \
    --samples 500 \
    --squad-path path/to/train-v1.1.json \
    --output-dir /tmp/sft_out \
    --seed 42
```

The script writes one SFT JSONL file per condition (suitable for
instruction-tuning small models such as T5-small or Phi-2) and a
`comparison_summary.json` to `--output-dir`.

---

## 🛠️ Common Operations

- **Human Review Queue**: Pipelines may write `human_review_queue.json` to the repo root. Use the `HumanReviewQueue` helper to export to a custom file.
- **RAG Usage**: Construct `Annotator(..., rag=True, rag_method='bm25')` or `'tfidf'` to enable retrieval-augmented prompts.

---

## 👩‍💻 Developer Notes

- When adding a new `router`, `filter`, or `task`, implement the interfaces in `base_structure/` to ensure compatibility with the runner.
- The `api/server.py` runner dynamically instantiates classes (e.g., `filters.ActiveLearningFilter`) — prefer fully qualified import strings when configuring pipelines.
- There is no enforced test framework in the repo; adding `pytest` tests for core modules is recommended.

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repository and create a new branch for your feature or bugfix.
2. Write clear, concise commit messages.
3. Include usage examples and tests for new features.
4. Submit a pull request with a detailed description of your changes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Verify third-party dependency licenses before using this code in commercial products.

---

## 🙏 Acknowledgements

Special thanks to all contributors and the open-source community for their support and inspiration.

---