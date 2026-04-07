# ADR-0002: LangGraph ReAct Pattern Over a Custom Agent Loop

**Date:** 2026-03-18
**Status:** Accepted
**Deciders:** Ben Lin

---

## Context

The earliest commit (`894e9cc`) contained a bespoke `app/main.py` with a hand-rolled agent loop and a flat `app/agent_tools/` directory. Commit `0dfd80a` replaced this with the LangGraph ReAct template, and commit `a4b73bb` renamed `react_agent` to `orient` and built the first domain-specific toolset on top of it.

The question at that point was whether to continue with a custom loop or adopt LangGraph as the state-machine substrate.

---

## Decision

The agent is implemented as a `StateGraph` using LangGraph's ReAct pattern:

```
START → call_llm → route_model_output → tools → call_llm → ... → END
```

`StateGraph` manages state transitions, tool dispatch, and the recursion limit. The application code only defines nodes (`call_llm`), edges (`route_model_output`), and tools.

---

## Rationale

### Alternatives considered

**Continue with the hand-rolled loop**
A `while True` loop calling the LLM and dispatching tools. Rejected because it requires re-implementing: thread isolation, state persistence, interrupt/resume, streaming, and the recursion-limit safety valve — all of which LangGraph provides.

**LangChain `AgentExecutor`**
The legacy abstraction. Rejected because it is effectively deprecated in favour of LangGraph and does not support the same degree of state inspection or mid-run interruption.

### Why LangGraph

- **Persistent checkpoints** — `StateGraph.compile(checkpointer=...)` gives every run a replayable, inspectable state history with zero application code.
- **Explicit graph topology** — the call graph is data (nodes + edges), not control flow hidden inside a loop. This makes it possible to inspect, visualise, and test individual nodes in isolation.
- **`ToolNode` prebuilt** — reduces tool-dispatch boilerplate to zero for the common case.
- **`Runtime` context injection** — `ToolRuntime[Context, State]` lets tools access per-run configuration (API keys, prompts) without global state or constructor injection.
- **LangGraph Studio compatibility** — the `StateGraph` + `langgraph.json` convention gives a free visual debugger with hot-reload during development.

---

## Consequences

**Positive**
- Thread-safe state isolation via `thread_id` in `RunnableConfig.configurable` — no shared mutable state.
- Persistence backend is swappable (`MemorySaver` ↔ `AsyncPostgresSaver`) without changing agent code.
- `is_last_step` managed variable provides a clean recursion-limit exit path without defensive `try/except` blocks.

**Negative / trade-offs**
- `StateGraph` is generic over four type parameters; the compiled graph `CompiledStateGraph[Any, Any, Any, Any]` is the practical annotation at the server boundary, which loses specificity.
- The LangGraph API evolves rapidly — `Runtime` replaced `RunnableConfig`-based context injection between the initial template and the current version, requiring a migration.
- LangGraph Studio / LangSmith is the natural observability surface; running without it (as the production server does) requires deliberate effort to expose tracing elsewhere.
