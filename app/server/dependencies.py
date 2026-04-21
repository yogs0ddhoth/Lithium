"""FastAPI dependency functions."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request

from app.server.lifespan import CompiledAgentSpec


def api_key(request: Request) -> str:
    """Return the Anthropic API key from the ``X-Api-Key`` header or env fallback."""
    key = request.headers.get("x-api-key", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    return key


def get_agent(agent_name: str, request: Request) -> CompiledAgentSpec:
    """Resolve an agent name to its compiled graph and spec.

    Raises **404** if the agent name is not registered in ``app.state.agents``.
    """
    agents: dict[str, CompiledAgentSpec] = request.app.state.agents
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agents[agent_name]
