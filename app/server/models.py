"""Pydantic request and response schemas for the Lithium API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Body for POST /agents/{agent}/threads/{thread_id}/runs (sync and stream)."""

    message: str = Field(
        description="The user's natural-language problem description to send to the agent.",
        examples=["Our checkout flow drops ~30 % of users on mobile."],
    )
    config: dict[str, Any] = Field(
        default={},
        description=(
            "Optional overrides for any ``Context`` field "
            "(`model`, `system_prompt`, `review_prompt`, `synthesis_prompt`, "
            "`max_search_results`). Unknown keys are silently ignored."
        ),
        examples=[{"model": "anthropic/claude-opus-4-6"}],
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(description="Always ``'ok'`` when the service is running.")


class ThreadResponse(BaseModel):
    """Response body for POST /agents/{agent}/threads."""

    thread_id: str = Field(
        description="UUID that identifies the new conversation thread.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )


class RunResponse(BaseModel):
    """Response body for POST /agents/{agent}/threads/{thread_id}/runs."""

    thread_id: str = Field(description="The thread this run belongs to.")
    type: str = Field(
        description="LangChain message type of the final message.",
        examples=["ai"],
    )
    content: Any = Field(description="Text content of the final agent message.")


class SerializedMessage(BaseModel):
    """A single LangChain message serialized via ``model_dump()``."""

    type: str = Field(examples=["human", "ai", "tool"])
    content: Any
    id: str | None = None

    model_config = {"extra": "allow"}


class ThreadStateResponse(BaseModel):
    """Response body for GET /agents/{agent}/threads/{thread_id}/state."""

    thread_id: str
    next: list[str] = Field(
        description="Graph nodes that will execute on the next run, if any."
    )
    values: dict[str, Any] = Field(
        description=(
            "Persisted state snapshot. The ``messages`` key holds a list of "
            "serialized LangChain messages; other keys reflect agent-specific state."
        )
    )


class SSEEvent(BaseModel):
    """A single Server-Sent Event chunk emitted by the stream endpoint.

    The stream ends with the literal line ``data: [DONE]``.
    """

    type: str = Field(
        description="LangChain message chunk type.",
        examples=["AIMessageChunk"],
    )
    content: Any = Field(description="Token or partial content emitted by the model.")
    id: str | None = Field(
        default=None, description="Message ID assigned by the model."
    )
    node: str | None = Field(
        default=None,
        description="Name of the LangGraph node that produced this chunk.",
        examples=["call_llm", "tools"],
    )
