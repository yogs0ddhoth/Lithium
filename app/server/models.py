"""Pydantic request and response schemas for the Lithium API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class RunInput(BaseModel):
    """Body for POST .../runs and .../runs/stream.

    Exactly one of ``message`` or ``resume`` must be provided:

    - Set ``message`` to start a new run.
    - Set ``resume`` to continue a run paused at a ``human_review`` interrupt.
    """

    message: str | None = Field(
        default=None,
        description="User's problem description. Provide for a new run.",
        examples=["Our checkout flow drops ~30 % of users on mobile."],
    )
    resume: str | dict[str, Any] | list[Any] | None = Field(
        default=None,
        description=(
            "Resume value for a paused interrupt. Pass a plain string for "
            "unstructured guidance, a JSON object or array matching the DTO "
            "schema for the current interrupt point (see "
            "``GET /agents/{agent_name}/interrupt-schemas``), or ``null`` to "
            "continue without adding a message. Provide for a resume run."
        ),
        examples=[
            "Focus on mobile-first experiences only.",
            {"concepts": [{"name": "...", "score": 1}]},
        ],
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

    @model_validator(mode="after")
    def _check_exclusive(self) -> RunInput:
        """Ensure exactly one of ``message`` or ``resume`` is provided."""
        has_message = self.message is not None
        has_resume = self.resume is not None
        if has_message and has_resume:
            raise ValueError(
                "Provide 'message' (new run) or 'resume' (interrupt resume), not both."
            )
        if not has_message and not has_resume:
            raise ValueError(
                "Provide either 'message' (new run) or 'resume' (interrupt resume)."
            )
        return self


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
    """Response body for GET /agents/{agent}/threads/{thread_id}."""

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
