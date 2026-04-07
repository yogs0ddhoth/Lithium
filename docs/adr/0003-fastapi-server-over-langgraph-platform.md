# ADR-0003: Custom FastAPI Server Over LangGraph Platform Deployment

**Date:** 2026-04-06
**Status:** Accepted
**Deciders:** Ben Lin

---

## Context

LangGraph provides a first-party deployment platform (LangGraph Cloud / `langgraph-api`) that hosts graphs behind a managed HTTP API, handles persistence, auth, and streaming out of the box. The project already uses `langgraph dev` for local development, which runs the same managed server.

As the project moves toward production, the question is whether to continue using the LangGraph platform or host the graph inside a custom application server.

The key constraints driving this decision:

- LangSmith (the LangGraph platform's observability layer) is optional and should not be a hard runtime dependency.
- The deployment target is a containerised environment (Docker / Kubernetes) where the operator controls the persistence backend.
- The API surface must support multiple agents under a single host without running separate services.

---

## Decision

The LangGraph graph is compiled and served by a custom **FastAPI + uvicorn** application (`app/server/`). The graph is recompiled at server startup with an injected checkpointer, and all LangSmith/LangGraph platform dependencies are optional.

The server package is structured as:

```
app/server/
├── lifespan.py     # AGENT_REGISTRY, checkpointer selection
├── models.py       # Pydantic request/response schemas
├── dependencies.py # Auth + graph resolution
└── routes.py       # APIRouter at /agents/{agent_name}/...
```

`app/main.py` exports both `orient` (for `langgraph dev`) and `app` (for uvicorn), so the two deployment modes share the same codebase with no branching.

---

## Rationale

### Alternatives considered

**LangGraph Cloud (managed)**
Fully managed; zero infrastructure. Rejected because it requires LangSmith as a hard dependency and does not support self-hosted Postgres as the persistence backend.

**`langgraph-api` self-hosted**
The open-core server that powers `langgraph dev`. Rejected for the same LangSmith coupling reason, and because its API surface is not stable for programmatic use outside the LangGraph CLI.

**Single-graph FastAPI server (flat `server.py`)**
The initial implementation. Replaced because it tied the server wiring directly to the Orient agent, making it impossible to add a second agent without restructuring the server.

### Design choices within the FastAPI server

**`AGENT_REGISTRY` + `get_graph` dependency**
Rather than hard-coding a single graph, the server resolves the active graph from a registry keyed by agent name. Adding a new agent is a one-line change in `lifespan.py`; all routes, auth, and streaming are shared.

**Checkpointer as a lifespan concern**
The checkpointer is created once during FastAPI's `lifespan` context and shared across all requests. This keeps connection pooling (Postgres) or memory (MemorySaver) from being re-initialised per request. The `CHECKPOINTER` env var selects the backend without any application code change.

**`astream` with `stream_mode="messages"` over `astream_events`**
`astream_events` does not expose a typed `context` parameter in the current LangGraph version. `astream(stream_mode="messages")` emits `(BaseMessageChunk, metadata)` tuples, which carry sufficient information for the SSE event schema and allow the `context` kwarg to be passed correctly.

**`RunnableConfig` for thread binding**
Thread isolation uses `{"configurable": {"thread_id": ...}}` in the `RunnableConfig`. This is the canonical LangGraph pattern and is checkpointer-agnostic — the same config dict works with `MemorySaver` and `AsyncPostgresSaver`.

---

## Consequences

**Positive**
- No LangSmith dependency at runtime; `LANGCHAIN_TRACING_V2=false` in the Dockerfile disables it entirely.
- A single server binary hosts any number of agents; the `orient` and future `converge_diverge` agents share auth, routing, and persistence infrastructure.
- The Postgres checkpointer uses an async connection pool scoped to the lifespan, which is correct for multi-worker uvicorn deployments.
- `app/main.py` dual-exports `orient` and `app`, so `langgraph dev` and `uvicorn app.main:app` both work from the same codebase.

**Negative / trade-offs**
- The managed LangGraph platform UI (Studio thread browser, built-in streaming playground) is not available when running the custom server.
- `CompiledStateGraph` is generic but the server stores and passes it as `CompiledStateGraph[Any, Any, Any, Any]` — the type parameters are erased at the registry boundary.
- `MemorySaver` is not safe with more than one uvicorn worker process; operators must set `CHECKPOINTER=postgres` or `--workers 1` to avoid split-brain state.
