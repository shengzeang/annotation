**Annotation UI (React Flow) & Backend Executor**

This adds a simple React Flow-based frontend UI to build processing graphs and a Flask endpoint to execute the graph by mapping node types to your repository's filters, routers and annotator.

Quick start

1. Backend (Python)

- Create a Python virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install flask requests
```

- Run the backend API:

```powershell
python -m api.server
```

This starts a server at `http://localhost:5000` and provides `POST /run_graph` to execute a graph JSON.

2. Frontend (React)

- Change to the frontend folder and install:

```powershell
cd frontend
npm install
npm run dev
```

- The UI will open in the browser (Vite default http://localhost:3000). Use the palette to add nodes, connect edges, edit node params, and click "Run Graph" to send the graph to the backend.
- The UI will open in the browser. Note: the project now sets Vite's default dev port to `5173` to avoid common permission errors (visit `http://localhost:5173`). You can override the port by setting the `PORT` environment variable or passing `--port` to Vite.

Notes and limitations

- The frontend nodes store a simple `data.params` object as node parameters — edit as JSON.
- Supported node labels (type) by default: `LoadData`, `Filter`, `Router`, `Annotate`, `Output`, `Task`, `CandidateLLMs`.
- The backend uses dynamic import to instantiate classes. For example to use `ActiveLearningFilter`, set the Filter node's `params.filter_class` to `filters.al_filter.ActiveLearningFilter` and `params.filter_params` to the kwargs dict.
- For `Annotate` nodes, supply `candidate_llms` and `llm_mode` ("local" or "api"). Local LLMs require the models to be available and may be slow.
- This is a minimal integration to get started. Feel free to extend node UIs, validation, and security checks.

If you'd like, I can:
- Expand node editor into typed UI per-node-type
- Add examples / templates of graphs (e.g., standard pipeline)
- Add authentication and job queuing for long-running LLM calls
