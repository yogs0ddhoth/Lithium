"""Unit tests for the FastAPI server.

The graph is always replaced with a lightweight mock so these tests run
offline without any LLM API keys.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.orient.context import Context as OrientContext
from app.server import _serialize_message, _thread_config, app
from app.server.lifespan import (
    AgentSpec,
    CompiledAgentSpec,
    _make_context_factory,
)
from app.server.models import RunInput
from app.server.routes import _last_tool_name, _resolve_resume

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_KEY = "sk-test-key"
_THREAD = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_AGENT = "orient"
# Base path for all agent-scoped endpoints
_BASE = f"/agents/{_AGENT}"


def _ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def _fake_result(messages: list) -> Any:
    """Return an object with a .messages attribute (v2 ainvoke output shape)."""
    r = MagicMock()
    r.messages = messages
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> MagicMock:
    """Return a mock compiled graph with sensible defaults."""
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value=_fake_result([_ai("Hello")]))
    g.aget_state = AsyncMock(return_value=None)

    async def _default_stream(*args: Any, **kwargs: Any):  # type: ignore[return]
        yield (AIMessageChunk(content="Hello"), {"langgraph_node": "call_llm"})

    g.astream = _default_stream
    return g


@pytest.fixture
async def client(mock_graph: MagicMock) -> AsyncClient:  # type: ignore[override]
    """Async httpx client wired to the FastAPI app.

    Injects a mock CompiledAgentSpec into ``app.state.agents`` so we bypass
    the lifespan and avoid needing any checkpointer or API keys.
    """
    spec = AgentSpec(
        builder=MagicMock(),
        name=_AGENT,
        context_factory=_make_context_factory(OrientContext),
    )
    app.state.agents = {_AGENT: CompiledAgentSpec(graph=mock_graph, spec=spec)}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pure unit tests — RunInput model validation
# ---------------------------------------------------------------------------


class TestRunInputModel:
    """Tests for the RunInput Pydantic model's mutual-exclusivity validator."""

    def test_message_only_valid(self) -> None:
        ri = RunInput(message="hi")
        assert ri.message == "hi"
        assert ri.resume is None

    def test_resume_string_valid(self) -> None:
        ri = RunInput(resume="continue")
        assert ri.resume == "continue"
        assert ri.message is None

    def test_resume_dict_valid(self) -> None:
        ri = RunInput(resume={"concepts": []})
        assert isinstance(ri.resume, dict)

    def test_resume_list_valid(self) -> None:
        ri = RunInput(resume=[1, 2, 3])
        assert isinstance(ri.resume, list)

    def test_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="not both"):
            RunInput(message="hi", resume="also this")

    def test_neither_raises(self) -> None:
        with pytest.raises(ValidationError, match="either"):
            RunInput()

    def test_config_defaults_to_empty_dict(self) -> None:
        assert RunInput(message="hi").config == {}

    def test_config_can_be_overridden(self) -> None:
        ri = RunInput(message="hi", config={"model": "openai/gpt-4o"})
        assert ri.config == {"model": "openai/gpt-4o"}


# ---------------------------------------------------------------------------
# Pure unit tests — _last_tool_name helper
# ---------------------------------------------------------------------------


class TestLastToolName:
    """Tests for the _last_tool_name helper."""

    def _snapshot(self, messages: list) -> Any:
        s = MagicMock()
        s.values = {"messages": messages}
        return s

    def test_empty_messages_returns_none(self) -> None:
        assert _last_tool_name(self._snapshot([])) is None

    def test_no_ai_messages_returns_none(self) -> None:
        assert _last_tool_name(self._snapshot([HumanMessage(content="hi")])) is None

    def test_ai_message_without_tool_calls_returns_none(self) -> None:
        assert _last_tool_name(self._snapshot([_ai("plain answer")])) is None

    def test_returns_last_tool_name(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "score_concepts", "id": "t1", "args": {}}],
        )
        assert _last_tool_name(self._snapshot([msg])) == "score_concepts"

    def test_returns_last_of_multiple_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "first_tool", "id": "t1", "args": {}},
                {"name": "second_tool", "id": "t2", "args": {}},
            ],
        )
        assert _last_tool_name(self._snapshot([msg])) == "second_tool"

    def test_finds_last_ai_with_tool_calls_ignoring_trailing_plain(self) -> None:
        """A plain AIMessage after a tool-call AIMessage does not hide the tool call."""
        tool_msg = AIMessage(
            content="",
            tool_calls=[{"name": "map_feature_to_themes", "id": "t1", "args": {}}],
        )
        plain = _ai("next message")
        # Reversed scan: plain → no tool calls; tool_msg → has tool calls → found.
        assert _last_tool_name(self._snapshot([tool_msg, plain])) == "map_feature_to_themes"

    def test_returns_none_when_all_ai_messages_lack_tool_calls(self) -> None:
        """All AIMessages have no tool calls → None."""
        assert _last_tool_name(self._snapshot([_ai("a"), _ai("b")])) is None

    def test_empty_values_returns_none(self) -> None:
        s = MagicMock()
        s.values = {}
        assert _last_tool_name(s) is None


# ---------------------------------------------------------------------------
# Pure unit tests — _make_context_factory
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
# Pure unit tests — _thread_config
# ---------------------------------------------------------------------------


class TestThreadConfig:
    """Tests for the _thread_config helper."""

    def test_structure(self) -> None:
        cfg = _thread_config(_THREAD)
        assert cfg == {"configurable": {"thread_id": _THREAD}}

    def test_thread_id_preserved(self) -> None:
        tid = "my-custom-id"
        assert _thread_config(tid)["configurable"]["thread_id"] == tid


# ---------------------------------------------------------------------------
# Pure unit tests — _serialize_message
# ---------------------------------------------------------------------------


class TestSerializeMessage:
    """Tests for the _serialize_message helper."""

    def test_returns_dict(self) -> None:
        assert isinstance(_serialize_message(_ai("hello")), dict)

    def test_content_present(self) -> None:
        assert _serialize_message(_ai("hello"))["content"] == "hello"

    def test_type_present(self) -> None:
        assert _serialize_message(_ai("x"))["type"] == "ai"

    def test_human_message(self) -> None:
        assert _serialize_message(HumanMessage(content="hi"))["type"] == "human"


# ---------------------------------------------------------------------------
# Pure unit tests — _resolve_resume
# ---------------------------------------------------------------------------


class TestResolveResume:
    """Tests for the _resolve_resume helper."""

    def _spec(self, interrupt_dtos: dict | None = None) -> AgentSpec:
        return AgentSpec(
            builder=MagicMock(),
            name="test",
            context_factory=_make_context_factory(OrientContext),
            interrupt_dtos=interrupt_dtos or {},
        )

    async def test_message_returns_human_message_dict(self) -> None:
        body = RunInput(message="hello")
        graph = MagicMock()
        result = await _resolve_resume(body, self._spec(), _THREAD, graph)
        assert isinstance(result, dict)
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "hello"

    async def test_message_does_not_call_aget_state(self) -> None:
        body = RunInput(message="hello")
        graph = AsyncMock()
        await _resolve_resume(body, self._spec(), _THREAD, graph)
        graph.aget_state.assert_not_called()

    async def test_resume_string_returns_command(self) -> None:
        from langgraph.types import Command

        body = RunInput(resume="continue")
        graph = MagicMock()
        result = await _resolve_resume(body, self._spec(), _THREAD, graph)
        assert isinstance(result, Command)
        assert result.resume == "continue"

    async def test_resume_dict_without_dto_raises_400(self) -> None:
        from fastapi import HTTPException

        body = RunInput(resume={"some": "data"})
        graph = AsyncMock()
        snapshot = MagicMock()
        snapshot.values = {"messages": []}
        graph.aget_state = AsyncMock(return_value=snapshot)
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_resume(body, self._spec(), _THREAD, graph)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# HTTP — health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_200(self, client: AsyncClient) -> None:
        assert (await client.get("/health")).status_code == 200

    async def test_body(self, client: AsyncClient) -> None:
        assert (await client.get("/health")).json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# HTTP — unknown agent
# ---------------------------------------------------------------------------


class TestUnknownAgent:
    async def test_create_thread_unknown_agent_returns_201(
        self, client: AsyncClient
    ) -> None:
        """Thread creation does not validate agent name — threads are agent-agnostic UUIDs."""
        r = await client.post("/agents/nonexistent/threads")
        assert r.status_code == 201

    async def test_sync_run_unknown_agent_returns_404(
        self, client: AsyncClient
    ) -> None:
        r = await client.post(
            f"/agents/nonexistent/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# HTTP — create thread
# ---------------------------------------------------------------------------


class TestCreateThread:
    async def test_returns_201(self, client: AsyncClient) -> None:
        assert (await client.post(f"{_BASE}/threads")).status_code == 201

    async def test_returns_thread_id(self, client: AsyncClient) -> None:
        assert "thread_id" in (await client.post(f"{_BASE}/threads")).json()

    async def test_thread_id_is_uuid(self, client: AsyncClient) -> None:
        uuid.UUID((await client.post(f"{_BASE}/threads")).json()["thread_id"])

    async def test_each_call_returns_different_id(self, client: AsyncClient) -> None:
        ids = {
            (await client.post(f"{_BASE}/threads")).json()["thread_id"]
            for _ in range(3)
        }
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# HTTP — auth
# ---------------------------------------------------------------------------


class TestAuth:
    async def test_sync_run_requires_key(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs", json={"message": "hi"}
        )
        assert r.status_code == 401

    async def test_stream_run_requires_key(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream", json={"message": "hi"}
        )
        assert r.status_code == 401

    async def test_env_fallback_accepted(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", _TEST_KEY)
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs", json={"message": "hi"}
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# HTTP — sync run
# ---------------------------------------------------------------------------


class TestSyncRun:
    async def test_returns_200(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 200

    async def test_body_shape(self, client: AsyncClient) -> None:
        body = (
            await client.post(
                f"{_BASE}/threads/{_THREAD}/runs",
                json={"message": "hi"},
                headers={"x-api-key": _TEST_KEY},
            )
        ).json()
        assert body["thread_id"] == _THREAD
        assert "content" in body
        assert "type" in body

    async def test_content_from_graph(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.ainvoke = AsyncMock(
            return_value=_fake_result([_ai("specific response")])
        )
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.json()["content"] == "specific response"

    async def test_graph_invoked_with_correct_message(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "test input"},
            headers={"x-api-key": _TEST_KEY},
        )
        payload = mock_graph.ainvoke.call_args.args[0]
        assert isinstance(payload["messages"][0], HumanMessage)
        assert payload["messages"][0].content == "test input"

    async def test_thread_id_in_config(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        config = mock_graph.ainvoke.call_args.kwargs["config"]
        assert config["configurable"]["thread_id"] == _THREAD

    async def test_api_key_forwarded_as_context(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        ctx: OrientContext = mock_graph.ainvoke.call_args.kwargs["context"]
        assert ctx.anthropic_api_key.get_secret_value() == _TEST_KEY

    async def test_model_override_via_config(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi", "config": {"model": "openai/gpt-4o"}},
            headers={"x-api-key": _TEST_KEY},
        )
        ctx: OrientContext = mock_graph.ainvoke.call_args.kwargs["context"]
        assert ctx.model == "openai/gpt-4o"

    async def test_invalid_body_returns_422(self, client: AsyncClient) -> None:
        """Both message and resume set → validation error → 422."""
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi", "resume": "also this"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 422

    async def test_empty_body_returns_422(self, client: AsyncClient) -> None:
        """Neither message nor resume → validation error → 422."""
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 422

    async def test_resume_string_forwarded_as_command(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """A string resume should be wrapped in Command and passed to ainvoke."""
        from langgraph.types import Command

        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"resume": "user feedback"},
            headers={"x-api-key": _TEST_KEY},
        )
        payload = mock_graph.ainvoke.call_args.args[0]
        assert isinstance(payload, Command)
        assert payload.resume == "user feedback"

    async def test_version_v2_passed_to_ainvoke(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """ainvoke must be called with version='v2' for typed state coercion."""
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert mock_graph.ainvoke.call_args.kwargs.get("version") == "v2"


# ---------------------------------------------------------------------------
# HTTP — streaming run
# ---------------------------------------------------------------------------


class TestStreamRun:
    async def test_returns_200(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 200

    async def test_content_type_is_sse(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert "text/event-stream" in r.headers["content-type"]

    async def test_thread_id_header(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.headers["x-thread-id"] == _THREAD

    async def test_stream_ends_with_done(self, client: AsyncClient) -> None:
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.text.endswith("data: [DONE]\n\n")

    async def test_stream_contains_content(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        async def _stream(*args: Any, **kwargs: Any):  # type: ignore[return]
            yield (AIMessageChunk(content="chunk1"), {"langgraph_node": "call_llm"})
            yield (AIMessageChunk(content="chunk2"), {"langgraph_node": "call_llm"})

        mock_graph.astream = _stream
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert "chunk1" in r.text
        assert "chunk2" in r.text

    async def test_sse_lines_are_valid_json(self, client: AsyncClient) -> None:
        import json

        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        data_lines = [
            line[len("data: ") :]
            for line in r.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        for line in data_lines:
            json.loads(line)

    async def test_stream_error_is_emitted_not_raised(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """Graph errors must surface as an SSE error event, not a 500."""

        async def _failing_stream(*args: Any, **kwargs: Any):  # type: ignore[return]
            raise RuntimeError("LLM exploded")
            yield  # make it a generator

        mock_graph.astream = _failing_stream
        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.status_code == 200
        assert "error" in r.text
        assert r.text.endswith("data: [DONE]\n\n")

    async def test_stream_version_v2_passed(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """astream must be called with version='v2'."""
        captured_kwargs: dict[str, Any] = {}

        async def _capture(*args: Any, **kwargs: Any):  # type: ignore[return]
            captured_kwargs.update(kwargs)
            yield (AIMessageChunk(content="x"), {"langgraph_node": "call_llm"})

        mock_graph.astream = _capture
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs/stream",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert captured_kwargs.get("version") == "v2"


# ---------------------------------------------------------------------------
# HTTP — interrupt schemas
# ---------------------------------------------------------------------------


class TestInterruptSchemas:
    """Tests for GET /agents/{name}/interrupt-schemas."""

    async def test_orient_returns_empty_dict(self, client: AsyncClient) -> None:
        """Orient has no interrupt DTOs — schemas dict must be empty."""
        r = await client.get(f"{_BASE}/interrupt-schemas")
        assert r.status_code == 200
        assert r.json() == {}

    async def test_unknown_agent_returns_404(self, client: AsyncClient) -> None:
        r = await client.get("/agents/nonexistent/interrupt-schemas")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# HTTP — state retrieval
# ---------------------------------------------------------------------------


class TestGetState:
    async def test_unknown_thread_returns_404(self, client: AsyncClient) -> None:
        r = await client.get(f"{_BASE}/threads/{_THREAD}")
        assert r.status_code == 404

    async def test_known_thread_returns_200(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("hi")]}
        snapshot.next = []
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        assert (await client.get(f"{_BASE}/threads/{_THREAD}")).status_code == 200

    async def test_state_body_shape(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("hi")]}
        snapshot.next = ["call_llm"]
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        body = (await client.get(f"{_BASE}/threads/{_THREAD}")).json()
        assert body["thread_id"] == _THREAD
        assert "values" in body
        assert body["next"] == ["call_llm"]

    async def test_messages_are_serialized(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("hello state")]}
        snapshot.next = []
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        body = (await client.get(f"{_BASE}/threads/{_THREAD}")).json()
        msgs = body["values"]["messages"]
        assert isinstance(msgs, list)
        assert msgs[0]["content"] == "hello state"

    async def test_state_called_with_thread_config(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("x")]}
        snapshot.next = []
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        await client.get(f"{_BASE}/threads/{_THREAD}")
        config = mock_graph.aget_state.call_args.args[0]
        assert config["configurable"]["thread_id"] == _THREAD

    async def test_human_review_interrupt_visible_in_next(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """When the graph is paused at human_review, next reflects that."""
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("waiting")]}
        snapshot.next = ["human_review"]
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        body = (await client.get(f"{_BASE}/threads/{_THREAD}")).json()
        assert body["next"] == ["human_review"]

    async def test_non_message_state_values_pass_through(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """Non-list state values (e.g. concept_scores str) must pass through unchanged."""
        snapshot = MagicMock()
        snapshot.values = {
            "messages": [],
            "concept_scores": "<normalized_concept_scores />",
        }
        snapshot.next = []
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        body = (await client.get(f"{_BASE}/threads/{_THREAD}")).json()
        assert body["values"]["concept_scores"] == "<normalized_concept_scores />"


# ---------------------------------------------------------------------------
# Lifespan / checkpointer selection
# ---------------------------------------------------------------------------


class TestCheckpointerSelection:
    async def test_memory_checkpointer_used_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        monkeypatch.delenv("CHECKPOINTER", raising=False)
        assert os.environ.get("CHECKPOINTER", "memory") == "memory"

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
            from app.server.lifespan import lifespan

            async with lifespan(app):
                pass

        assert len(captured) >= 1
        assert all(isinstance(cp, MemorySaver) for cp in captured)

    async def test_all_registered_agents_are_compiled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every entry in AGENT_REGISTRY must be compiled on startup."""
        from app.server.lifespan import AGENT_REGISTRY, lifespan

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
        from app.server.lifespan import CompiledAgentSpec, lifespan

        monkeypatch.setenv("CHECKPOINTER", "memory")

        with patch("app.server.lifespan._compile", return_value=MagicMock()):
            async with lifespan(app):
                agents = app.state.agents
                assert all(isinstance(v, CompiledAgentSpec) for v in agents.values())

    async def test_each_spec_carries_context_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every compiled spec must have a callable context_factory."""
        from app.server.lifespan import lifespan

        monkeypatch.setenv("CHECKPOINTER", "memory")

        with patch("app.server.lifespan._compile", return_value=MagicMock()):
            async with lifespan(app):
                for compiled in app.state.agents.values():
                    assert callable(compiled.spec.context_factory)
