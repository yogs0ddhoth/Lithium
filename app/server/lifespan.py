"""Graph compilation and server lifespan management.

Adding a new agent
------------------
Register it in ``AGENT_REGISTRY``:

    from app.my_agent.graph import builder as _my_builder
    from app.my_agent.context import Context as _MyContext

    AGENT_REGISTRY["my-agent"] = AgentSpec(
        builder=_my_builder,
        name="app.my_agent",
        context_factory=_make_context_factory(_MyContext),
    )

The lifespan will compile it with the active checkpointer and expose it on
``app.state.agents`` as a ``CompiledAgentSpec`` (graph + spec bundled).
"""

from __future__ import annotations

import dataclasses
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, SecretStr

from app.converge_diverge.context import Context as CDContext
from app.converge_diverge.graph import builder as _cd_builder
from app.converge_diverge.prompts import UserFeatureScores, UserScoresAndSolutions
from app.orient.context import Context as OrientContext
from app.orient.graph import builder as _orient_builder


def _make_context_factory(
    context_cls: type,
) -> Callable[[str, dict[str, Any]], Any]:
    """Return a factory that builds a ``context_cls`` from an API key and overrides.

    Only keys that match declared fields on ``context_cls`` are forwarded;
    unknown keys are silently ignored so callers can include routing metadata.
    """
    valid_fields: frozenset[str] = frozenset(
        f.name for f in dataclasses.fields(context_cls)
    )

    def factory(api_key: str, overrides: dict[str, Any]) -> Any:
        kwargs = {k: v for k, v in overrides.items() if k in valid_fields}
        return context_cls(anthropic_api_key=SecretStr(api_key), **kwargs)

    return factory


@dataclass(frozen=True)
class AgentSpec:
    """Compile-time specification for a single agent."""

    # StateGraph is generic over (StateT, ContextT, InputT, OutputT); since each agent
    # uses its own concrete types we store the builder as Any and recover the concrete
    # CompiledStateGraph from compile() at the call site.
    builder: Any
    name: str
    context_factory: Callable[[str, dict[str, Any]], Any] = field(
        default_factory=lambda: _make_context_factory(OrientContext)
    )
    interrupt_dtos: dict[str, type[BaseModel]] = field(default_factory=dict)
    """Maps the name of the last-completed tool to the expected feedback DTO class.

    When a resume request supplies a ``dict`` or ``list`` for ``feedback``, the
    route looks up the DTO for the tool that triggered the current interrupt,
    validates the payload with Pydantic, and serialises it to XML before
    injecting it as a ``HumanMessage``.  Tools absent from this mapping expect
    plain-text (or no) feedback.
    """


@dataclass(frozen=True)
class CompiledAgentSpec:
    """A compiled agent graph bundled with its compile-time spec."""

    graph: CompiledStateGraph[Any, Any, Any, Any]
    spec: AgentSpec


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "orient": AgentSpec(
        builder=_orient_builder,
        name="orient",
        context_factory=_make_context_factory(OrientContext),
    ),
    "converge-diverge": AgentSpec(
        builder=_cd_builder,
        name="converge-diverge",
        context_factory=_make_context_factory(CDContext),
        interrupt_dtos={
            "map_feature_to_themes": UserScoresAndSolutions,
            "score_concepts": UserFeatureScores,
        },
    ),
}


def _compile(
    spec: AgentSpec, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile an agent's StateGraph builder with the given checkpointer."""
    return spec.builder.compile(checkpointer=checkpointer, name=spec.name)  # type: ignore[no-any-return]


def _build_agents(
    checkpointer: BaseCheckpointSaver,
) -> dict[str, CompiledAgentSpec]:
    """Compile all registered agents and return them as ``CompiledAgentSpec`` instances."""
    return {
        name: CompiledAgentSpec(graph=_compile(spec, checkpointer), spec=spec)
        for name, spec in AGENT_REGISTRY.items()
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compile all registered agents and expose them on ``app.state.agents``.

    Sets ``app.state.agents`` (name → ``CompiledAgentSpec``) so routes can
    resolve both the compiled graph and the agent's context factory and
    interrupt DTOs with a single lookup.

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
            await saver.setup()
            app.state.agents = _build_agents(saver)
            yield
    else:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer: BaseCheckpointSaver = MemorySaver()
        app.state.agents = _build_agents(checkpointer)
        yield
