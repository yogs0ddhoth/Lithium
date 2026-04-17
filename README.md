# Lithium

A multi-agent problem synthesis API. Send a natural-language description of a problem; the agents conduct structured analysis and return validated XML documents covering problem framing, feature mapping, concept scoring, and prioritised recommendations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Orient Agent](#orient-agent)
  - [Converge/Diverge Agent](#convergediverge-agent)
  - [FastAPI Server](#fastapi-server)
  - [xml-pydantic Package](#xml-pydantic-package)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment](#environment)
- [Running](#running)
- [API Reference](#api-reference)
  - [Routes](#routes)
  - [Authentication](#authentication)
  - [Model Configuration](#model-configuration)
  - [Human-in-the-Loop Resumes](#human-in-the-loop-resumes)
- [Deployment](#deployment)
  - [Fly.io (Prototype)](#flyio-prototype)
  - [CI (GitHub Actions)](#ci-github-actions)
  - [GKE (Future — Azure DevOps)](#gke-future--azure-devops)
- [Development](#development)
  - [Commands](#commands)
  - [Adding a New Agent](#adding-a-new-agent)
- [Architecture Decision Records](#architecture-decision-records)

---

## Overview

Lithium exposes two LangGraph agents over a shared FastAPI server:

| Agent | Purpose | Output |
| --- | --- | --- |
| `orient` | Validates and synthesises a problem statement via structured Q&A | `<problem_statement />` XML |
| `converge-diverge` | Maps features to themes, scores concepts and features, and generates an RTC-EBC prioritisation | `<rtc-ebc />` XML |

Both agents run as persistent, thread-scoped conversations. State is stored in-process (`MemorySaver`) by default, or in Postgres for multi-worker production deployments.

---

## Architecture

```text
app/
├── main.py                        # Entrypoint: exports `orient` (LangGraph) + `app` (FastAPI)
├── server/                        # FastAPI server
│   ├── lifespan.py                # AGENT_REGISTRY, checkpointer selection, startup/shutdown
│   ├── routes.py                  # APIRouter at /agents/{agent_name}/...
│   ├── models.py                  # Pydantic request/response schemas
│   └── dependencies.py            # Auth and graph resolution
├── orient/                        # Orient ReAct agent
│   ├── graph.py                   # StateGraph: call_llm → tools → call_llm cycle
│   ├── state.py                   # InputState / State dataclasses
│   ├── context.py                 # Runtime configuration (model, prompts, API key)
│   ├── prompts.py                 # XML prompt loading + dynamic Pydantic model generation
│   └── tools.py                   # review_user_problem, synthesize_problem_statement
├── converge_diverge/              # Converge/Diverge sequential agent
│   ├── graph.py                   # StateGraph with human_review interrupt nodes
│   ├── state.py                   # InputState / State (features, scores, rtc_ebc fields)
│   ├── context.py                 # Per-tool system prompts and LLM config
│   ├── prompts.py                 # XML prompt loading + dynamic Pydantic model generation
│   └── tools.py                   # map_feature_to_themes, score_concepts, score_features, generate_rct_ebc
└── utils.py                       # Shared utilities (model loading, XmlDto mixin)

packages/
└── xml-pydantic/                  # Standalone XML ↔ Pydantic v2 library

prompts/                           # XML files: dual-purpose LLM prompts + output schemas
docs/adr/                          # Architecture Decision Records
helm/lithium/                      # Helm chart for GKE deployment
scripts/                           # Fly.io lifecycle scripts
tests/
└── unit_tests/                    # Offline tests (mocked, no API keys required)
```

### Orient Agent

[`app/orient/`](app/orient/) implements a ReAct (Reasoning + Action) cycle ([ADR-0002](docs/adr/0002-langgraph-react-over-custom-loop.md)):

```text
START → call_llm → route_model_output → tools → call_llm → … → END
```

Two LLM-powered tools transform a user's problem description into a validated problem statement:

1. **`review_user_problem`** ([tools.py](app/orient/tools.py)) — validates input against Q&A criteria; returns `<qa_results />` XML
2. **`synthesize_problem_statement`** ([tools.py](app/orient/tools.py)) — synthesises the final `<problem_statement />` XML

### Converge/Diverge Agent

[`app/converge_diverge/`](app/converge_diverge/) runs four tools in strict sequence with human-in-the-loop interrupts after the first two ([ADR-0004](docs/adr/0004-human-in-the-loop-interrupts.md)):

```text
START → call_llm ──(tool calls)──→ tools ──(map/score_concepts)──→ human_review → call_llm
                 └──(no tool calls)──→ END        └──(other tools)──────────────→ call_llm
```

| Step | Tool | Input | Output | Interrupt? |
| --- | --- | --- | --- | --- |
| 1 | `map_feature_to_themes` | `<features_and_needs />` | `<features_and_themes />` | Yes — awaits `<user_scores_and_solutions />` |
| 2 | `score_concepts` | `<user_scores_and_solutions />` | `<normalized_concept_scores />` | Yes — awaits `<user_feature_scores />` |
| 3 | `score_features` | `<user_feature_scores />` | `<highest_scoring_features />` | No |
| 4 | `generate_rct_ebc` | themes + scores + features | `<rtc-ebc />` | No |

### FastAPI Server

[`app/server/`](app/server/) is a custom production server replacing the LangGraph platform ([ADR-0003](docs/adr/0003-fastapi-server-over-langgraph-platform.md)). All agents are compiled at startup and registered in [`AGENT_REGISTRY`](app/server/lifespan.py). Routes, auth, and the checkpointer are shared across all agents automatically.

**Checkpointer selection** is controlled by the `CHECKPOINTER` environment variable:

- `memory` (default) — `MemorySaver`, in-process, single-worker only; state lost on restart
- `postgres` — `AsyncPostgresSaver`, requires `DATABASE_URL`, multi-worker safe, persistent

### xml-pydantic Package

[`packages/xml-pydantic/`](packages/xml-pydantic/) is a standalone library for bidirectional conversion between XML, JSON Schema, and Pydantic v2 models. XML files in [`prompts/`](prompts/) serve as both LLM system prompts and output schemas — the `data-*` attributes drive the Pydantic model definition ([ADR-0001](docs/adr/0001-xml-driven-structured-output.md)).

See [`packages/xml-pydantic/README.md`](packages/xml-pydantic/README.md) for full documentation.

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

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Default LLM provider (required) |
| `OPENAI_API_KEY` | Optional alternative provider |
| `LANGSMITH_API_KEY` | Tracing/observability (optional) |
| `CHECKPOINTER` | `memory` (default) or `postgres` |
| `DATABASE_URL` | Required when `CHECKPOINTER=postgres` |

---

## Running

### LangGraph Studio (development)

```bash
langgraph dev
```

Hot-reload is enabled. The `orient` graph is exposed at [`app/main.py:orient`](app/main.py) per [`langgraph.json`](langgraph.json).

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

---

## API Reference

### Routes

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/agents/{agent}/threads` | Allocate a new thread UUID |
| `POST` | `/agents/{agent}/threads/{id}/runs` | Synchronous run or interrupt resume — returns final message |
| `POST` | `/agents/{agent}/threads/{id}/runs/stream` | Streaming run or interrupt resume — SSE token stream |
| `GET` | `/agents/{agent}/threads/{id}/state` | Inspect persisted thread state |

Agents: `orient`, `converge-diverge`

### Authentication

Pass your Anthropic API key in the `X-Api-Key` request header. If omitted, the server falls back to the `ANTHROPIC_API_KEY` environment variable.

```bash
# 1. Create a thread
curl -X POST http://localhost:8000/agents/orient/threads \
  -H "X-Api-Key: sk-ant-..."

# 2. Run the agent
curl -X POST http://localhost:8000/agents/orient/threads/{thread_id}/runs \
  -H "X-Api-Key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Our checkout flow drops ~30% of users on mobile."}'
```

### Model Configuration

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

### Human-in-the-Loop Resumes

The `converge-diverge` agent pauses after `map_feature_to_themes` and `score_concepts` to await structured human feedback. New runs and resumes share the same endpoint, distinguished by the `RunInput` body ([ADR-0004](docs/adr/0004-human-in-the-loop-interrupts.md)):

- Set `message` to start a new run.
- Set `resume` to continue from an interrupt — the payload is validated against the interrupt's expected DTO and serialised to XML before being injected as a `HumanMessage`.

```bash
# Start a new converge-diverge run
curl -X POST http://localhost:8000/agents/converge-diverge/threads/{thread_id}/runs \
  -H "X-Api-Key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -d '{"message": "<features_and_needs>...</features_and_needs>"}'

# Resume after a human-review interrupt
curl -X POST http://localhost:8000/agents/converge-diverge/threads/{thread_id}/runs \
  -H "X-Api-Key: sk-ant-..." \
  -H "Content-Type: application/json" \
  -d '{"resume": {"scores": [...], "solutions": [...]}}'
```

---

## Deployment

### Fly.io (Prototype)

Short-lived prototypes are deployed from a local machine. Each prototype is an isolated named app with its own Postgres cluster.

**Prerequisites:** `flyctl auth login`, Docker daemon running, `ANTHROPIC_API_KEY` set in environment.

**First-time setup** (idempotent — safe to re-run):

```bash
ANTHROPIC_API_KEY=sk-ant-... ./scripts/fly-setup.sh             # default app name: lithium
ANTHROPIC_API_KEY=sk-ant-... ./scripts/fly-setup.sh lithium-v2  # named instance
```

**Deploy** (runs unit tests first, then builds and deploys locally):

```bash
./scripts/fly-deploy.sh             # deploys to 'lithium'
./scripts/fly-deploy.sh lithium-v2  # deploys to a named instance
```

The image is built with the local Docker daemon and pushed directly to Fly's internal registry — no external image registry is involved. Use `--local-only` is enforced by the deploy script.

**Teardown:**

```bash
flyctl apps destroy lithium --yes
flyctl apps destroy lithium-db --yes
```

### CI (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint, format check, and unit tests on every push and pull request. No Docker build or deploy step runs in CI — those remain local operations during the prototype phase.

### GKE (Future — Azure DevOps)

When the project migrates to production on GKE, the pipeline will run in Azure DevOps:

```text
test → build (docker push → JFrog Artifactory) → update helm/lithium/values.yaml with image digest
```

A GitOps system (e.g. ArgoCD) watches the repository and reconciles GKE state when [`helm/lithium/values.yaml`](helm/lithium/values.yaml) changes. The `image.digest` field is the handoff point — never set it by hand.

See [ADR-0005](docs/adr/0005-deployment-strategy.md) for the full rationale and migration path.

---

## Development

### Commands

```bash
# Setup
uv sync

# Run
langgraph dev                        # LangGraph dev server with hot reload
uvicorn app.main:app --reload        # FastAPI dev server with hot reload

# Test
uv run pytest                        # Unit tests with coverage
cd packages/xml-pydantic && uv run pytest   # xml-pydantic package tests

# Lint & format
uv run ruff check --fix .            # Lint and auto-fix
uv run ruff format .                 # Format code

# Type checking
uv run ty check

# Dependency management
uv add <package>                     # Add runtime dependency
uv add --dev <package>               # Add dev dependency
uv remove <package>                  # Remove dependency
uv sync                              # Sync environment with lockfile
```

### Adding a New Agent

Register it in [`app/server/lifespan.py:AGENT_REGISTRY`](app/server/lifespan.py):

```python
from app.my_agent.graph import builder as _my_builder
from app.my_agent.context import Context as _MyContext

AGENT_REGISTRY["my-agent"] = AgentSpec(
    builder=_my_builder,
    name="my-agent",
    context_factory=_make_context_factory(_MyContext),
)
```

The lifespan compiles it with the active checkpointer and exposes it on all routes automatically. For agents with human-in-the-loop interrupts, also populate `interrupt_dtos` — see `converge-diverge` in [`lifespan.py`](app/server/lifespan.py) for the pattern.

---

## Architecture Decision Records

Authoritative records for non-obvious design choices in [`docs/adr/`](docs/adr/):

| ADR | Decision |
| --- | --- |
| [ADR-0001](docs/adr/0001-xml-driven-structured-output.md) | XML as the dual-purpose prompt and schema format; `XmlDto` mixin |
| [ADR-0002](docs/adr/0002-langgraph-react-over-custom-loop.md) | LangGraph ReAct pattern over a custom agent loop |
| [ADR-0003](docs/adr/0003-fastapi-server-over-langgraph-platform.md) | Custom FastAPI server over LangGraph platform; unified run/resume endpoint |
| [ADR-0004](docs/adr/0004-human-in-the-loop-interrupts.md) | Human-in-the-loop interrupts via dedicated `human_review` node |
| [ADR-0005](docs/adr/0005-deployment-strategy.md) | Fly.io prototypes, GitHub Actions CI, GKE production via Azure DevOps + JFrog |
