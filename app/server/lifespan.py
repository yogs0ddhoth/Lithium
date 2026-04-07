"""Graph compilation and server lifespan management.

Adding a new agent
------------------
Register it in ``AGENT_REGISTRY``:

    from app.my_agent.graph import builder as _my_builder
    AGENT_REGISTRY["my-agent"] = AgentSpec(builder=_my_builder, name="app.my_agent")

The lifespan will compile it with the active checkpointer automatically.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.orient.graph import builder as _orient_builder


@dataclass(frozen=True)
class AgentSpec:
    """Compile-time specification for a single agent."""

    # StateGraph is generic over (StateT, ContextT, InputT, OutputT); since each agent
    # uses its own concrete types we store the builder as Any and recover the concrete
    # CompiledStateGraph from the compile() return type at the call site.
    builder: Any
    name: str  # passed to builder.compile(); appears in traces


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "orient": AgentSpec(builder=_orient_builder, name="app.orient"),
}


def _compile(
    spec: AgentSpec, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile an agent's StateGraph builder with the given checkpointer."""
    return spec.builder.compile(checkpointer=checkpointer, name=spec.name)  # type: ignore[no-any-return]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compile all registered agents and expose them on ``app.state.graphs``.

    Selects the checkpointer based on the ``CHECKPOINTER`` env var:

    - ``memory`` (default): in-process ``MemorySaver``; state lost on restart;
      must **not** be used with more than one worker process.
    - ``postgres``: persistent ``AsyncPostgresSaver``; requires ``DATABASE_URL``;
      safe for multi-worker deployments.
    """
    mode = os.getenv("CHECKPOINTER", "memory").lower()

    if mode == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        db_url = os.environ["DATABASE_URL"]
        async with AsyncPostgresSaver.from_conn_string(db_url) as saver:
            await saver.setup()  # idempotent: creates tables on first run
            app.state.graphs = {
                name: _compile(spec, saver) for name, spec in AGENT_REGISTRY.items()
            }
            yield
    else:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer: BaseCheckpointSaver = MemorySaver()
        app.state.graphs = {
            name: _compile(spec, checkpointer) for name, spec in AGENT_REGISTRY.items()
        }
        yield
