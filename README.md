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
- **`experiments/`**: Experiment scripts comparing annotation conditions with real Qwen LLMs and optional downstream fine-tuning.

---

## 🧪 Experiments

### Oracle vs DataFlow-Annotator — QA Annotation Comparison (Qwen LLMs)

`experiments/run_label_studio_comparison.py` benchmarks DataFlow-Annotator
against oracle LLM annotation using **Qwen fine-tuning as the downstream
evaluation task**.  All annotation is performed by real Qwen models (via
`misc/llm_provider.LocalLLM` or compatible API).  Downstream fine-tuning
delegates to `misc/evaluate.py`'s `finetune_sft()` and `evaluate()`
(requires GPU); omit `--skip-finetune` to run the full pipeline.

**Five conditions are compared:**

| Condition | Description |
|---|---|
| Single Oracle | One call to a large Qwen model (e.g. Qwen2.5-72B) per sample |
| 3-Oracle Majority Vote | Three independent Qwen calls; majority vote decides the annotation |
| DataFlow (naive LLM) | Qwen-7B annotation without KB/RAG |
| DataFlow (KB + RAG) | Qwen-7B with in-context KB retrieval augmentation |
| DataFlow (full pipeline) | Qwen-7B with KB + RAG + stricter confidence threshold |

**Metrics:** `Ann-F1` / `Ann-EM`, `DS-BLEU` / `DS-ROUGE-L` (GPU required).

```bash
python experiments/run_label_studio_comparison.py --samples 500 --skip-finetune
```

---

### Active Learning — Annotation Budget Comparison

`experiments/run_active_learning.py` shows how **active learning sampling**
improves data quality within a fixed annotation budget.  Five strategies are
compared: random sampling, TF-IDF diversity k-means, uncertainty-length proxy,
and the repository's `ActiveLearningFilter` (ALPS force-fallback — no BERT
required).  A real Qwen model can be injected with `--model`; pass `--skip-llm`
for a zero-GPU smoke-test.

| Condition | Description |
|---|---|
| Full dataset | Annotate all samples (quality upper-bound) |
| Random sampling | Random budget-sized subset |
| Diversity (TF-IDF) | k-means cluster representatives on TF-IDF features |
| Uncertainty (length) | Samples deviating most from mean text length |
| ALPS (force-fallback) | Repository's `ActiveLearningFilter` (deterministic fallback) |

**Metrics:** `Ann-F1` / `Ann-EM` per condition.

```bash
# Offline smoke-test (no GPU)
python experiments/run_active_learning.py --samples 200 --budget 50 --skip-llm

# Real Qwen annotation
python experiments/run_active_learning.py \
    --samples 500 --budget 100 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --squad-path path/to/train-v1.1.json
```

---

### LLM Routing — Cost-Quality Tradeoff

`experiments/run_llm_routing.py` demonstrates how the repository's routing
strategies balance annotation **quality vs. cost** (fraction of calls to the
expensive LLM).  Two real Qwen models play the role of cheap / expensive
annotator; a third judge model drives `CascadeRouter` and `LLMRouter`.

| Condition | Description |
|---|---|
| All-cheap | Every sample goes to the cheap / fast LLM |
| All-expensive | Every sample goes to the expensive / capable LLM |
| Cascade | Try cheap first; escalate to expensive when judge deems answer wrong |
| LLM Router | Judge LLM scores each candidate and picks the best one |

**Metrics:** `Ann-F1` / `Ann-EM`, `Exp-Rate` (expensive-LLM call fraction).

```bash
# Offline smoke-test (no GPU)
python experiments/run_llm_routing.py --samples 200 --skip-llm

# Real Qwen routing
python experiments/run_llm_routing.py \
    --cheap-model   Qwen/Qwen2.5-7B-Instruct \
    --expensive-model Qwen/Qwen2.5-72B-Instruct
```

---

### RAG — Retrieval-Augmented Annotation

`experiments/run_rag.py` compares four retrieval strategies for knowledge-base
augmentation.  All conditions use the same LLM; the difference is how
in-context examples are fetched from the growing KB.  Quality improves as the
KB fills with high-confidence annotations.

| Condition | Description |
|---|---|
| No RAG | Plain LLM annotation without retrieval |
| RAG (Jaccard) | Word-overlap retrieval (no extra deps) |
| RAG (TF-IDF) | TF-IDF cosine retrieval (scikit-learn) |
| RAG (Semantic) | Sentence-transformer retrieval (falls back to TF-IDF) |

**Metrics:** `Ann-F1` / `Ann-EM`, `KB-Final` (KB size), per-window F1 trend.

```bash
# Offline smoke-test (no GPU)
python experiments/run_rag.py --samples 200 --skip-llm

# Real Qwen annotation with semantic RAG
python experiments/run_rag.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --samples 500 \
    --squad-path path/to/train-v1.1.json
```

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