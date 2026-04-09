# ADR-0004: Human-in-the-Loop Interrupts for Sequential Tool Workflows

**Date:** 2026-04-08
**Status:** Accepted
**Deciders:** Ben Lin

---

## Context

The Converge/Diverge agent runs four tools in sequence:

1. `map_feature_to_themes` — synthesises a `<features_and_themes />` map
2. `score_concepts` — normalises scores into `<normalized_concept_scores />`
3. `score_features` — ranks features into `<highest_scoring_features />`
4. `generate_rct_ebc` — produces a `<rtc-ebc />` report

Each tool's output is the human-reviewed input for the next tool. Requiring review at each step is a product requirement, not a performance optimisation — the human must be able to correct or steer the intermediate XML before the workflow continues. Without this, a bad `<features_and_themes />` would silently propagate through all three downstream tools.

The Orient agent's graph — a tight ReAct loop with no pauses — is the wrong shape for this workflow. The question is how to introduce human checkpoints without breaking the LangGraph state model, the checkpointer contract, or the existing streaming API.

---

## Decision

Each tool execution is followed by a dedicated `human_review` node that calls LangGraph's `interrupt()` primitive. The graph topology for Converge/Diverge is:

```
START → call_llm ──(tool calls)──→ tools → human_review → call_llm
                 └──(no tool calls)──→ END
```

`interrupt()` checkpoints the graph at `human_review` and suspends execution. The caller resumes by invoking the graph with `Command(resume=feedback)`. If `feedback` is a non-empty string it is injected as a `HumanMessage` before `call_llm` runs again; an empty string continues without adding a message.

Two resume endpoints are added to the FastAPI server alongside the existing run endpoints:

```
POST /agents/{agent_name}/threads/{thread_id}/runs/resume         → RunResponse
POST /agents/{agent_name}/threads/{thread_id}/runs/resume/stream  → SSE stream
```

The existing `GET /threads/{thread_id}/state` response already surfaces `next: list[str]`. When the graph is paused, `next` contains `["human_review"]`; when it has completed, `next` is empty. Clients use this field to distinguish an interrupted run from a finished one without a separate status endpoint.

### Per-agent context factories

Adding the Converge/Diverge agent exposed a latent problem: `routes.py` was hardcoded to the Orient agent's `Context` class. A per-agent context factory is the minimal fix.

`AgentSpec` in `lifespan.py` gains a `context_factory: Callable[[str, dict], Any]` field. The factory is produced by a generic helper `_make_context_factory(context_cls)` that reflects over the dataclass fields and ignores unknown keys. `app.state.specs` is populated alongside `app.state.graphs` so routes can look up the factory by agent name.

---

## Rationale

### Alternatives considered

**`interrupt_after=["tools"]` at compile time**
Pauses the graph after the `tools` node without requiring a dedicated node. Rejected because it provides no hook for injecting feedback as a message — resuming with `Command(resume=...)` does not automatically create a `HumanMessage`. The `human_review` node handles both the pause and the optional message injection in one place.

**Polling on `next` without `interrupt()`**
The client could call `ainvoke({"messages": [HumanMessage(...)]})` directly to resume, treating the interrupt state as a signal to inject a new human turn. Rejected because this bypasses the checkpoint and loses the semantic contract: LangGraph's `interrupt()` guarantees the graph will not advance until explicitly resumed; a bare `HumanMessage` appended to an active thread would race with the running graph.

**Separate feedback endpoint that calls `aupdate_state` then `ainvoke(None)`**
Call `graph.aupdate_state(config, {"messages": [HumanMessage(feedback)]})` then resume with `ainvoke(None, ...)`. Rejected because it splits the feedback-injection concern across two HTTP calls and two server operations, making failure recovery harder. A single `Command(resume=feedback)` is atomic from the checkpointer's perspective.

**One resume endpoint (sync only)**
The existing run endpoints offer both sync and streaming variants; consistency requires resume to match. Streaming is the expected consumption pattern for long-running LLM steps.

### Why `interrupt()` over a custom node that polls state

`interrupt()` is a first-class LangGraph primitive: it works correctly with both `MemorySaver` and `AsyncPostgresSaver`, is visible in LangGraph Studio, and is the documented pattern for human-in-the-loop workflows. A custom polling mechanism would reimplement the same semantics less reliably.

---

## Consequences

**Positive**

- The interrupt/resume contract is stable across agents — any future agent that needs a human checkpoint uses the same pattern and the same two resume endpoints.
- `GET /state` with `next: ["human_review"]` is sufficient for clients to detect an interrupt; no separate status field or webhook is needed.
- Feedback injected via `Command(resume=feedback)` is visible in the thread's message history, making the conversation auditable.
- The per-agent `context_factory` on `AgentSpec` means adding a third agent with a different `Context` shape is a one-line change in `lifespan.py`.

**Negative / trade-offs**

- The `human_review` node is inserted unconditionally after every tool call. For tools that do not need human review this adds a latency hop (though the interrupt is immediate if the client resumes instantly with empty feedback).
- `Command(resume=...)` requires the compiled graph to have been invoked at least once with the originating `thread_id`; a resume call on an unknown thread will error rather than return 404. The routes layer does not currently guard against this — callers must check `GET /state` first.
- The `context_factory` field on `AgentSpec` defaults to the Orient context factory, which means a registry entry without an explicit factory silently builds the wrong context type. This is an easy mistake to make when adding a third agent.

---

## Addendum — 2026-04-09: Unified run/resume endpoint

### Context

ADR-0003 (2026-04-09 addendum) collapsed the four run/resume × sync/stream endpoints to two. This affects the resume contract described above.

### Decision

The dedicated `POST .../runs/resume` and `POST .../runs/resume/stream` endpoints are removed. Clients resume an interrupted graph by calling the same `POST .../runs` or `POST .../runs/stream` endpoints with a `resume` field in the `RunInput` body instead of a `message` field. The `_resolve_resume` helper in `routes.py` branches on which field is set and returns either a messages dict or `Command(resume=...)` accordingly.

### Rationale

The run/resume distinction is a payload-level concern, not a URL-level one. A single endpoint with a discriminated body is simpler to document, simpler to implement, and removes the client burden of tracking which URL path corresponds to which graph state.

### Consequences

- `GET /threads/{thread_id}` (formerly `/state`) with `next: ["human_review"]` remains the signal for clients to switch from `message` to `resume` in their next request body.
- Clients that previously called `/runs/resume` must migrate to `/runs` with `resume` in the body.

---

## Addendum — 2026-04-09: Selective interrupts for tools 1 and 2 only

### Context

The original decision inserted `human_review` unconditionally after every tool call. In practice only the first two tools (`map_feature_to_themes` and `score_concepts`) produce output that the human must review and score before the workflow can continue. The last two tools (`score_features` and `generate_rct_ebc`) consume their inputs directly from prior tool outputs already in the message thread; pausing them adds a latency hop and an empty resume round-trip with no product value.

### Decision

The unconditional `tools → human_review` edge is replaced with a conditional router, `route_after_tools`, that checks the name of the last-called tool against a fixed set (`map_feature_to_themes`, `score_concepts`). Tools in that set route to `human_review`; all others route directly back to `call_llm`.

```text
tools ──(map_feature_to_themes, score_concepts)──→ human_review → call_llm
      └──(score_features, generate_rct_ebc)────────────────────→ call_llm
```

`interrupt_dtos` in `AgentSpec` already maps only the two interrupting tools; no API layer change is required.

### Rationale

`score_features` and `generate_rct_ebc` take their inputs verbatim from XML already present in the message thread. There is no human scoring step between them; the only reason to pause would be to let the user inspect intermediate output, which is available via `GET /threads/{thread_id}` without needing an interrupt.

### Consequences

- Calls to `score_features` and `generate_rct_ebc` complete without a client resume round-trip, reducing total wall-clock time for a full workflow by two HTTP interactions.
- `GET /threads/{thread_id}` with `next: ["human_review"]` now unambiguously indicates the graph is waiting for scored user input (steps 1 or 2), not merely an acknowledgement.
- The `interrupt_dtos` mapping on `AgentSpec` acts as the canonical declaration of which tools interrupt; `route_after_tools` must stay in sync with it.
