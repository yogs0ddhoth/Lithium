"""API routes for agent thread management."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from app.server.dependencies import api_key, get_agent
from app.server.lifespan import AgentSpec, CompiledAgentSpec
from app.server.models import (
    RunInput,
    RunResponse,
    SSEEvent,
    ThreadResponse,
    ThreadStateResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_tool_name(snapshot: StateSnapshot) -> str | None:
    """Return the name of the most-recently-called tool from a state snapshot.

    Walks the message list in reverse to find the last ``AIMessage`` that
    contains tool calls, then returns the name of its final tool call.
    Returns ``None`` when no such message exists.
    """
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    for msg in reversed(values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[-1]["name"]
    return None


async def _resolve_resume(
    body: RunInput,
    spec: AgentSpec,
    thread_id: str,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> dict[str, list[HumanMessage]] | Command:
    """Build the graph input from a ``RunInput`` body.

    - ``message`` set → returns a new-run messages dict.
    - ``resume`` set and a plain string or ``None`` → returns ``Command(resume=...)``.
    - ``resume`` set and a ``dict`` / ``list`` → validates against the registered
      interrupt DTO for the last-completed tool, serialises to XML, then returns
      ``Command(resume=<xml>)``.
    """
    if body.message is not None:
        return {"messages": [HumanMessage(content=body.message)]}

    feedback = body.resume
    if isinstance(feedback, (dict, list)):
        snapshot: StateSnapshot = await graph.aget_state(_thread_config(thread_id))
        tool_name = _last_tool_name(snapshot)
        dto_cls = spec.interrupt_dtos.get(tool_name) if tool_name else None
        if dto_cls is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Structured resume is not expected at this interrupt point "
                    f"(last tool: {tool_name!r}). Pass a plain string or null."
                ),
            )
        feedback = dto_cls.model_validate(feedback).model_dump_xml()  # type: ignore[union-attr]

    return Command(resume=feedback or "")


def _thread_config(thread_id: str) -> RunnableConfig:
    """Return the LangGraph RunnableConfig that pins a run to a thread."""
    return {"configurable": {"thread_id": thread_id}}


def _serialize_message(msg: BaseMessage) -> dict[str, Any]:
    return msg.model_dump()


async def _sse(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    payload: Any,
    config: RunnableConfig,
    ctx: Any,
) -> AsyncIterator[str]:
    """Yield SSE-formatted token chunks from the graph's message stream.

    Uses ``stream_mode="messages"`` which emits ``(BaseMessageChunk, metadata)``
    tuples for every token produced by any node in the graph.
    """
    try:
        async for chunk, metadata in cast(
            AsyncIterator[tuple[BaseMessageChunk, dict[str, Any]]],
            graph.astream(
                payload,
                config=config,
                context=ctx,
                stream_mode="messages",
                version="v2",
            ),
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
def create_thread() -> ThreadResponse:
    """Allocate a new UUID thread for a conversation.

    Clients may also supply their own UUID — the checkpointer creates the
    thread implicitly on the first run against that ID.
    """
    return ThreadResponse(thread_id=str(uuid.uuid4()))


@router.post(
    "/threads/{thread_id}/runs/stream",
    summary="Run or resume the agent and stream (SSE)",
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
    body: RunInput,
    agent: CompiledAgentSpec = Depends(get_agent),
    key: str = Depends(api_key),
) -> StreamingResponse:
    """Run the agent and stream every token as a Server-Sent Event.

    Set ``message`` in the body to start a new run.  Set ``resume`` to
    continue from a ``human_review`` interrupt — pass a plain string for
    unstructured guidance, a JSON object or array matching the interrupt
    DTO schema (see ``GET /agents/{agent_name}/interrupt-schemas``), or
    ``null`` to continue without adding a message.

    Each ``data:`` line is a JSON object matching the ``SSEEvent`` schema.
    The stream ends with ``data: [DONE]``.
    """
    ctx = agent.spec.context_factory(key, body.config)
    config = _thread_config(thread_id)
    payload = await _resolve_resume(body, agent.spec, thread_id, agent.graph)
    return StreamingResponse(
        _sse(agent.graph, payload, config, ctx),
        media_type="text/event-stream",
        headers={"X-Thread-Id": thread_id},
    )


@router.post(
    "/threads/{thread_id}/runs",
    response_model=RunResponse,
    summary="Run or resume the agent synchronously",
    responses=_401,
)
async def run(
    thread_id: str,
    body: RunInput,
    agent: CompiledAgentSpec = Depends(get_agent),
    key: str = Depends(api_key),
) -> RunResponse:
    """Run the agent and return the final message once the graph completes or interrupts.

    Set ``message`` in the body to start a new run.  Set ``resume`` to
    continue from a ``human_review`` interrupt — pass a plain string, a JSON
    object or array matching the interrupt DTO schema (see
    ``GET /agents/{agent_name}/interrupt-schemas``), or ``null`` to continue
    without adding a message.

    When the graph pauses at a ``human_review`` interrupt, the response
    reflects the last message before the pause.  Use
    ``GET /threads/{thread_id}`` to inspect the interrupt payload and
    ``POST .../runs`` (with ``resume``) to continue.
    """
    ctx = agent.spec.context_factory(key, body.config)
    config = _thread_config(thread_id)
    payload = await _resolve_resume(body, agent.spec, thread_id, agent.graph)
    result: Any = await agent.graph.ainvoke(
        payload, config=config, context=ctx, version="v2"
    )
    last: BaseMessage = result.messages[-1]
    return RunResponse(thread_id=thread_id, type=last.type, content=last.content)


@router.get(
    "/interrupt-schemas",
    summary="Get interrupt feedback schemas",
    response_model=dict[str, Any],
)
async def get_interrupt_schemas(
    agent: CompiledAgentSpec = Depends(get_agent),
) -> dict[str, Any]:
    """Return JSON Schema definitions for each structured interrupt point.

    Keys are tool names; values are the JSON Schema for the expected
    ``resume`` payload at the interrupt that follows that tool.  Tools not
    listed here expect plain-text (or no) feedback.

    Clients can use these schemas to validate their ``RunInput.resume``
    payload before sending, without keeping a separate copy of the DTO
    definition.
    """
    return {
        tool_name: dto_cls.model_json_schema()
        for tool_name, dto_cls in agent.spec.interrupt_dtos.items()
    }


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadStateResponse,
    summary="Get thread state",
    responses=_404,
)
async def get_thread(
    thread_id: str,
    agent: CompiledAgentSpec = Depends(get_agent),
) -> ThreadStateResponse:
    """Return the persisted state snapshot for a thread.

    When a graph is paused at ``human_review``, ``next`` will be
    ``["human_review"]`` and ``values`` will include the interrupt payload
    under the last tool message.  Use ``POST .../runs`` with ``resume`` to
    continue.

    Raises **404** if no checkpoint exists for ``thread_id``.
    """
    config = _thread_config(thread_id)
    snapshot: StateSnapshot = await agent.graph.aget_state(config)
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
