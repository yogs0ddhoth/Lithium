"""API routes for agent thread management."""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any, AsyncIterator, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from pydantic import SecretStr

from app.orient.context import Context
from app.server.dependencies import api_key, get_graph
from app.server.models import (
    RunRequest,
    RunResponse,
    SSEEvent,
    ThreadResponse,
    ThreadStateResponse,
)

# ---------------------------------------------------------------------------
# Context helpers
#
# _make_context is currently tied to the Orient agent's Context dataclass.
# When adding a new agent with a different context schema, extend this by
# accepting the context class as a parameter or adding a per-agent factory to
# AGENT_REGISTRY in lifespan.py.
# ---------------------------------------------------------------------------

_CONTEXT_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(Context))


def _make_context(key: str, overrides: dict[str, Any]) -> Context:
    """Build an Orient ``Context`` from an API key and optional field overrides.

    Only keys that are valid ``Context`` fields are forwarded; unknown keys are
    silently ignored so callers can freely include routing metadata.
    """
    kwargs = {k: v for k, v in overrides.items() if k in _CONTEXT_FIELDS}
    return Context(anthropic_api_key=SecretStr(key), **kwargs)


def _thread_config(thread_id: str) -> RunnableConfig:
    """Return the LangGraph ``RunnableConfig`` that pins a run to a thread."""
    return {"configurable": {"thread_id": thread_id}}


def _serialize_message(msg: BaseMessage) -> dict[str, Any]:
    # model_dump() is typed dict[str, Any] by Pydantic — unavoidable.
    return msg.model_dump()


async def _sse(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    payload: dict[str, list[HumanMessage]],
    config: RunnableConfig,
    ctx: Context,
) -> AsyncIterator[str]:
    """Yield SSE-formatted token chunks from the graph's message stream.

    Uses ``stream_mode="messages"`` which emits ``(BaseMessageChunk, metadata)``
    tuples for every token produced by any node in the graph.
    """
    # astream with stream_mode="messages" yields (BaseMessageChunk, metadata) pairs,
    # but the generic return type is AsyncIterator[dict[str, Any] | Any]. Cast to the
    # concrete tuple type so the type checker can verify attribute access below.
    try:
        async for chunk, metadata in cast(
            AsyncIterator[tuple[BaseMessageChunk, dict[str, Any]]],
            graph.astream(payload, config=config, context=ctx, stream_mode="messages"),
        ):
            event = {
                "type": chunk.type,
                "content": chunk.content,
                "id": chunk.id,
                "node": metadata.get("langgraph_node"),
            }
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_401: dict[int | str, dict[str, Any]] = {
    401: {"description": "Missing or invalid `X-Api-Key` header."}
}
_404: dict[int | str, dict[str, Any]] = {404: {"description": "Thread not found."}}

router = APIRouter(prefix="/agents/{agent_name}", tags=["agents"])


@router.post(
    "/threads",
    response_model=ThreadResponse,
    status_code=201,
    summary="Create a thread",
)
def create_thread(
    graph: CompiledStateGraph[Any, Any, Any, Any] = Depends(get_graph),
) -> ThreadResponse:
    """Allocate a new UUID thread for a conversation.

    Clients may also supply their own UUID — the checkpointer creates the
    thread implicitly on the first run against that ID.
    """
    return ThreadResponse(thread_id=str(uuid.uuid4()))


@router.post(
    "/threads/{thread_id}/runs/stream",
    summary="Stream a run (SSE)",
    response_description=(
        "A ``text/event-stream`` where each ``data:`` line is a JSON-encoded "
        "``SSEEvent`` object. The stream terminates with ``data: [DONE]``."
    ),
    responses={
        **_401,
        200: {
            "content": {"text/event-stream": {"schema": SSEEvent.model_json_schema()}},
            "description": "Server-Sent Event stream of ``SSEEvent`` chunks.",
        },
    },
)
async def stream_run(
    thread_id: str,
    body: RunRequest,
    graph: CompiledStateGraph[Any, Any, Any, Any] = Depends(get_graph),
    key: str = Depends(api_key),
) -> StreamingResponse:
    """Run the agent and stream every token as a Server-Sent Event.

    Each ``data:`` line is a JSON object matching the ``SSEEvent`` schema.
    The stream ends with ``data: [DONE]``.
    """
    ctx = _make_context(key, body.config)
    config = _thread_config(thread_id)
    payload: dict[str, list[HumanMessage]] = {
        "messages": [HumanMessage(content=body.message)]
    }
    return StreamingResponse(
        _sse(graph, payload, config, ctx),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id},
    )


@router.post(
    "/threads/{thread_id}/runs",
    response_model=RunResponse,
    summary="Run synchronously",
    responses=_401,
)
async def sync_run(
    thread_id: str,
    body: RunRequest,
    graph: CompiledStateGraph[Any, Any, Any, Any] = Depends(get_graph),
    key: str = Depends(api_key),
) -> RunResponse:
    """Run the agent and return the final message once the graph completes."""
    ctx = _make_context(key, body.config)
    config = _thread_config(thread_id)
    payload: dict[str, list[HumanMessage]] = {
        "messages": [HumanMessage(content=body.message)]
    }
    result: dict[str, list[BaseMessage]] = await graph.ainvoke(
        payload, config=config, context=ctx
    )
    last: BaseMessage = result["messages"][-1]
    return RunResponse(thread_id=thread_id, type=last.type, content=last.content)


@router.get(
    "/threads/{thread_id}/state",
    response_model=ThreadStateResponse,
    summary="Get thread state",
    responses=_404,
)
async def get_state(
    thread_id: str,
    graph: CompiledStateGraph[Any, Any, Any, Any] = Depends(get_graph),
) -> ThreadStateResponse:
    """Return the persisted state snapshot for a thread.

    Raises **404** if no checkpoint exists for ``thread_id``.
    """
    config = _thread_config(thread_id)
    snapshot: StateSnapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    return ThreadStateResponse(
        thread_id=thread_id,
        next=list(snapshot.next),
        values={
            k: [_serialize_message(m) for m in v] if isinstance(v, list) else v
            for k, v in values.items()
        },
    )
