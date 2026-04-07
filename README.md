# Lithium

A problem-statement synthesis API powered by the **Orient** LangGraph agent. Send a natural-language description of a problem; the agent conducts structured Q&A and returns a validated `<problem_statement />` XML document.

## Architecture

```text
app/
├── main.py                   # Unified entrypoint: exports `orient` (LangGraph) + `app` (FastAPI)
├── server/                   # FastAPI server package
│   ├── __init__.py           # FastAPI app construction, router inclusion
│   ├── lifespan.py           # AGENT_REGISTRY, checkpointer selection, startup/shutdown
│   ├── models.py             # Pydantic request/response schemas
│   ├── dependencies.py       # FastAPI dependency functions (auth, graph resolution)
│   └── routes.py             # APIRouter at /agents/{agent_name}/...
├── orient/                   # Orient ReAct agent
│   ├── graph.py              # StateGraph: call_llm → tools → call_llm cycle
│   ├── state.py              # InputState / State dataclasses
│   ├── context.py            # Runtime configuration (model, prompts, API key)
│   ├── prompts.py            # XML prompt loading + dynamic Pydantic model generation
│   └── tools.py              # LLM-powered tools: review_user_problem, synthesize_problem_statement
├── converge_diverge/         # Second agent (in progress)
└── utils.py                  # Shared utilities (model loading, XML parsing)

packages/
└── xml-pydantic/             # Standalone XML ↔ Pydantic v2 library

prompts/                      # XML prompt templates (dual-purpose: LLM system prompts + schemas)
tests/
├── unit_tests/               # Offline tests (mocked graph, no API keys required)
└── integration_tests/        # Live tests (require API keys)
```

### Orient Agent

Implements a ReAct (Reasoning + Action) cycle:

```text
START → call_llm → route_model_output → tools → call_llm → ... → END
```

The agent receives a user's problem description and uses two LLM-powered tools to transform it into a validated problem statement:

1. **`review_user_problem`** — validates input against structured Q&A criteria, returns `QAResults` XML
2. **`synthesize_problem_statement`** — synthesises a final `ProblemStatement` XML document

Both tools use structured XML output driven by prompt schemas in `prompts/`. The XML format serves as both the LLM instruction and the Pydantic model definition, enforcing data integrity at each handoff.

### Server

The FastAPI server compiles the LangGraph agent with a checkpointer on startup and exposes it over HTTP. All state is thread-scoped — each conversation has a `thread_id` that maps to a checkpoint.

| Route | Description |
| --- | --- |
| `GET /health` | Liveness probe |
| `POST /agents/{agent}/threads` | Allocate a new thread UUID |
| `POST /agents/{agent}/threads/{id}/runs` | Synchronous run — returns final message |
| `POST /agents/{agent}/threads/{id}/runs/stream` | Streaming run — SSE token stream |
| `GET /agents/{agent}/threads/{id}/state` | Inspect persisted thread state |

**Adding a new agent** requires one change: register it in `app/server/lifespan.py:AGENT_REGISTRY`. The router, auth, and checkpointer are shared automatically.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
uv sync
```

### Environment

```bash
cp .env.example .env
```

Required keys:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Default LLM provider (required) |
| `OPENAI_API_KEY` | Optional alternative provider |
| `LANGSMITH_API_KEY` | Tracing/observability (optional) |

---

## Running

### LangGraph Studio (development)

```bash
langgraph dev
```

Hot-reload is enabled. The `orient` graph is exposed at `./app/main.py:orient` per `langgraph.json`.

### Uvicorn (production)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker (with Postgres persistence)

```bash
docker compose up
```

State is persisted to Postgres and survives restarts. Safe for multi-worker deployments.

**Non-persistent (memory only):**

```bash
CHECKPOINTER=memory docker compose up app
```

State is in-process. Use a single worker only.

**Checkpointer selection** is controlled by the `CHECKPOINTER` environment variable:

- `memory` (default) — `MemorySaver`, in-process, single-worker only
- `postgres` — `AsyncPostgresSaver`, requires `DATABASE_URL`, multi-worker safe

---

## Authentication

Pass your Anthropic API key in the `X-Api-Key` request header. If omitted, the server falls back to the `ANTHROPIC_API_KEY` environment variable.

```bash
curl -X POST http://localhost:8000/agents/orient/threads \
  -H "X-Api-Key: sk-ant-..."

curl -X POST http://localhost:8000/agents/orient/threads/{thread_id}/runs \
  -H "X-Api-Key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Our checkout flow drops ~30% of users on mobile."}'
```

---

## Model Configuration

Default: `anthropic/claude-sonnet-4-5-20250929`

Override per-request via the `config` field:

```json
{
  "message": "...",
  "config": { "model": "anthropic/claude-opus-4-6" }
}
```

Or globally via environment variable:

```bash
MODEL=anthropic/claude-opus-4-6
```

---

## Development

### Commands

```bash
# Setup
uv sync                          # Install all dependencies

# Run
langgraph dev                    # LangGraph dev server with hot reload
uvicorn app.main:app --reload    # FastAPI dev server with hot reload

# Test
uv run pytest                                 # Unit tests (no API keys needed)
cd packages/xml-pydantic && uv run pytest     # xml-pydantic package tests

# Lint & format
uv run ruff check --fix .        # Lint and auto-fix
uv run ruff format .             # Format code

# Type checking
uv run --with mypy mypy --strict app/

# Dependency management
uv add <package>                 # Add runtime dependency
uv add --dev <package>           # Add dev dependency
uv remove <package>              # Remove dependency
uv sync                          # Sync environment with lockfile
```

### Packages

#### `xml-pydantic` (`packages/xml-pydantic/`)

A standalone library for bidirectional conversion between XML, JSON Schema, and Pydantic v2 models. Consumed by the Orient agent to parse XML prompt files into typed Pydantic models for structured LLM output.

See [`packages/xml-pydantic/README.md`](packages/xml-pydantic/README.md) for full documentation.
