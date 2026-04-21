"""Unit tests for app/server/lifespan.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.orient.context import Context as OrientContext
from app.server import app
from app.server.lifespan import (
    AGENT_REGISTRY,
    AgentSpec,
    CompiledAgentSpec,
    _make_context_factory,
    lifespan,
)

pytestmark = pytest.mark.anyio

_TEST_KEY = "sk-test-key"


# ---------------------------------------------------------------------------
# _make_context_factory
# ---------------------------------------------------------------------------


class TestContextFactory:
    """Tests for the _make_context_factory helper."""

    def _factory(self) -> Any:
        return _make_context_factory(OrientContext)

    def test_sets_api_key(self) -> None:
        ctx = self._factory()(_TEST_KEY, {})
        assert ctx.anthropic_api_key.get_secret_value() == _TEST_KEY

    def test_valid_field_override(self) -> None:
        ctx = self._factory()(_TEST_KEY, {"model": "openai/gpt-4o"})
        assert ctx.model == "openai/gpt-4o"

    def test_unknown_fields_are_ignored(self) -> None:
        ctx = self._factory()(_TEST_KEY, {"model": "openai/gpt-4o", "unknown_key": 99})
        assert ctx.model == "openai/gpt-4o"

    def test_returns_orient_context_instance(self) -> None:
        assert isinstance(self._factory()(_TEST_KEY, {}), OrientContext)

    def test_default_model_when_not_overridden(self) -> None:
        assert self._factory()(_TEST_KEY, {}).model


# ---------------------------------------------------------------------------
# Lifespan / checkpointer selection
# ---------------------------------------------------------------------------


class TestCheckpointerSelection:
    async def test_memory_env_var_selects_memory_saver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CHECKPOINTER=memory must select MemorySaver (not postgres)."""
        from langgraph.checkpoint.memory import MemorySaver

        monkeypatch.setenv("CHECKPOINTER", "memory")

        captured: list[Any] = []

        def _capture(_spec: Any, cp: Any) -> Any:
            captured.append(cp)
            return MagicMock()

        with patch("app.server.lifespan._compile", side_effect=_capture):
            async with lifespan(app):
                pass

        assert len(captured) >= 1
        assert all(isinstance(cp, MemorySaver) for cp in captured)

    async def test_all_registered_agents_are_compiled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every entry in AGENT_REGISTRY must be compiled on startup."""
        monkeypatch.setenv("CHECKPOINTER", "memory")

        compiled_names: list[str] = []

        def _capture(spec: Any, _cp: Any) -> Any:
            compiled_names.append(spec.name)
            return MagicMock()

        with patch("app.server.lifespan._compile", side_effect=_capture):
            async with lifespan(app):
                pass

        expected = {spec.name for spec in AGENT_REGISTRY.values()}
        assert expected == set(compiled_names)

    async def test_agents_stored_as_compiled_agent_specs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """lifespan must store CompiledAgentSpec instances, not bare graphs."""
        monkeypatch.setenv("CHECKPOINTER", "memory")

        with patch("app.server.lifespan._compile", return_value=MagicMock()):
            async with lifespan(app):
                agents = app.state.agents
                assert all(isinstance(v, CompiledAgentSpec) for v in agents.values())

    async def test_each_spec_carries_context_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every compiled spec must have a callable context_factory."""
        monkeypatch.setenv("CHECKPOINTER", "memory")

        with patch("app.server.lifespan._compile", return_value=MagicMock()):
            async with lifespan(app):
                for compiled in app.state.agents.values():
                    assert callable(compiled.spec.context_factory)
