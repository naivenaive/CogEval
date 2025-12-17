# CogEval

This repository implements a cognitive function evaluation agent built with LangChain-style abstractions and modular services. The
 design emphasizes:

- **Agentic RAG** for knowledge-grounded dialogue about cognitive assessment (FAISS-backed with a pure-Python fallback) with a two-step agentic retrieval loop.
- **Atom Memory** persisted in SQLite to keep episodic and semantic traces per user session.
- **Multi-model orchestration** (OpenAI, DeepSeek, WanAI, and a Mock client) with a configurable router.
- **Tool calling via an MCP-style server** for interactive assessment tasks (visual reasoning, digit span, word recall).
- **A simple chat UI** with a calm blue background and extensibility hooks for speech/video.
- **Scoring and longitudinal analysis** to derive per-domain scores and trends.

## Architecture overview

```
├── src/
│   ├── agents/            # Agent controllers and orchestration
│   ├── rag/               # Retrieval-augmented generation utilities
│   ├── memory/            # Atom memory implementation
│   ├── server/            # MCP-like task server
│   └── ui/                # Web/CLI UI stubs
├── tests/                 # Unit tests
└── README.md
```

### Modules and communication
- **CognitiveAgent (`src/agents/cognitive_agent.py`)**
  - Orchestrates conversation turns using a chain pipeline.
  - Dispatches to model providers (OpenAI/DeepSeek/WanAI) based on routing policy.
  - Pulls grounding documents from the RAG retriever and annotates responses.
  - Persists turn-level events into Atom Memory.

- **LLM Router (`src/llms/base.py`)**
  - Provides a unified interface for OpenAI, DeepSeek, WanAI, and Mock clients.
  - Uses configurable routing policies per task type (dialogue, retrieval, scoring, longitudinal).

- **Agentic RAG (`src/rag/agentic_rag.py`)**
  - Maintains a FAISS index of assessment guidelines and exemplars (falls back to a lightweight cosine index when FAISS is unavailable).
  - Supports “plan + retrieve + respond” loops: the agent first plans needed context, retrieves, then generates.
  - Provides ingestion helpers for uploading new documents (PDF/text) into the index.

- **Atom Memory (`src/memory/atom_memory.py`)**
  - Stores atomic events (user message, tool result, reflection) with timestamps and importance scores.
  - Supplies summarized context windows to the agent per turn.

- **Scoring and trends (`src/scoring/analyzer.py`, `src/scoring/report.py`)**
  - Aggregates atoms into per-domain scores with interpretation bands.
  - Computes longitudinal slopes across sessions and renders Markdown reports.

- **MCP Task Server (`src/server/mcp_server.py`)**
  - Exposes assessment tasks as tools (e.g., figure reasoning, digit span).
  - Supports async, long-running tasks where the agent must wait for user interaction.
  - Returns structured results that the agent fuses into the final assessment.

- **UI (`src/ui/app.py`)**
  - Minimal FastAPI + HTML chat endpoint with light-blue background.
  - WebSocket channel for streaming responses; placeholders for speech/video capture.

### Call flow per turn
1. Receive user message via UI API.
2. CognitiveAgent builds a turn plan (intent, tool needs) using the planning chain.
3. Agent queries Atom Memory for salient context and RAG for grounding documents.
4. Agent selects a model provider and composes a prompt with context + task.
5. If a task tool is required (e.g., figure reasoning), the agent calls MCP server and pauses until a result is available.
6. Agent produces a grounded response plus interim assessment notes.
7. Atom Memory logs the turn; UI streams the answer back.

### Backend API (FastAPI)
- `POST /api/chat/start` — initialize a chat session.
- `POST /api/chat/message` — send a message and receive the agent reply.
- `POST /api/chat/end` — close a session.
- `POST /api/rag/ingest` — ingest files into the RAG index.
- `GET /api/rag/status` — report document count.
- `GET /api/memory/user/{user_id}/timeline` — fetch recent atoms.
- `POST /api/mcp/task/create` — create an assessment task (visual reasoning, digit span, word recall).
- `GET /api/mcp/task/{task_id}` — retrieve task state.
- `POST /api/mcp/task/{task_id}/answer` — submit an answer and receive a score.

### Code style principles
- Python 3.11+, type hints and dataclasses.
- Small, composable functions; no try/except around imports.
- Configuration via environment variables (see `.env.example`).
- Testing with `pytest` and lightweight fakes for providers.

### Usage
1. **Install dependencies** (example):
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure LLM credentials**:
   - Copy `.env.example` to `.env` and set `COGEVAL_OPENAI_API_KEY` (or standard `OPENAI_API_KEY`).
   - The router will automatically use the real OpenAI Chat model via LangChain when the key is present; otherwise it falls back to the deterministic mock client so tests keep passing.
   - You can also set per-provider models via `COGEVAL_MODEL_OPENAI`, `COGEVAL_MODEL_DEEPSEEK`, or `COGEVAL_MODEL_WANAI`.
2. **Run unit tests**:
   ```bash
   pytest
   ```
3. **Start MCP server stub**:
   ```bash
   uvicorn src.server.mcp_server:app --reload
   ```
4. **Start UI**:
   ```bash
   uvicorn src.ui.app:app --reload --port 8000
   ```
5. **Ingest documents into RAG index** (example):
   ```python
   from src.rag.agentic_rag import AgenticRAG
   rag = AgenticRAG(index_path="./rag_index")
   rag.ingest_documents(["docs/cognitive_guidelines.txt"])
   ```
6. **Interact with the MCP/API** using any HTTP client (example task creation):
   ```bash
   curl -X POST http://localhost:8000/api/mcp/task/create -H 'Content-Type: application/json' -d '{"task_type":"visual_reasoning"}'
   ```

### Dependency notes for FAISS
- The retriever will automatically use FAISS when the `faiss` package is installed and fall back to a pure-Python cosine index otherwise.
- To enable FAISS locally (recommended for larger corpora), install `faiss-cpu` and `numpy`:
  ```bash
  pip install faiss-cpu numpy
  ```

### Ethical and clinical disclaimer
This system is for research and screening support only and does not provide medical diagnosis or treatment recommendations.

### Roadmap
- Replace stub embeddings with production vector DB (FAISS, Chroma, or PGVector).
- Add speech and video capture in the UI with media streaming.
- Add additional assessment tasks and scoring heuristics.
- Tighten safety layers and auditing for clinical workflows.
