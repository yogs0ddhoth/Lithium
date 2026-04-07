"""FastAPI dependency functions."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Header, HTTPException, Request
from langgraph.graph.state import CompiledStateGraph


def api_key(x_api_key: str = Header(default="")) -> str:
    """Return the Anthropic API key from the request header or env fallback."""
    key = x_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    return key


def get_graph(
    agent_name: str, request: Request
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Resolve an agent name to its compiled LangGraph graph.

    Raises **404** if the agent name is not registered in
    ``app.state.graphs``.
    """
    graphs: dict[str, CompiledStateGraph[Any, Any, Any, Any]] = request.app.state.graphs
    if agent_name not in graphs:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return graphs[agent_name]
