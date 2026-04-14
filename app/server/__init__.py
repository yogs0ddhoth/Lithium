"""Lithium FastAPI server package.

Public surface
--------------
- ``app``   — the ``FastAPI`` application (used by uvicorn / ASGI servers)
- ``lifespan``, ``_compile``, ``_thread_config``,
  ``_serialize_message`` — re-exported for use in integration tests.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.server.lifespan import _compile, lifespan
from app.server.models import HealthResponse
from app.server.routes import _serialize_message, _thread_config
from app.server.routes import router as _agent_router

_TAGS: list[dict[str, Any]] = [
    {
        "name": "agents",
        "description": (
            "Create threads and run any registered agent. "
            "Use `agent_name=orient` for the Orient problem-statement agent."
        ),
    },
    {"name": "health", "description": "Service liveness check."},
]

app = FastAPI(
    title="Lithium",
    version="0.1.0",
    description=(
        "Lithium is a problem-statement synthesis API powered by LangGraph agents.\n\n"
        "### Authentication\n"
        "Pass your Anthropic API key in the `X-Api-Key` request header. "
        "If the header is omitted the server falls back to the `ANTHROPIC_API_KEY` "
        "environment variable."
    ),
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(_agent_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
def health() -> HealthResponse:
    """Return ``{"status": "ok"}`` when the service is running."""
    return HealthResponse(status="ok")


__all__ = [
    "app",
    "lifespan",
    "_compile",
    "_thread_config",
    "_serialize_message",
]
