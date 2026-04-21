"""Unit tests for app/server/routes.py — helpers and HTTP integration."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.orient.context import Context as OrientContext
from app.server import app
from app.server.lifespan import AgentSpec, CompiledAgentSpec, _make_context_factory
from app.server.models import RunInput
from app.server.routes import _last_tool_name, _resolve_resume, _serialize_message, _thread_config

pytestmark = pytest.mark.anyio

_TEST_KEY = "sk-test-key"
_THREAD = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_AGENT = "orient"
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
    """Async httpx client wired to the FastAPI app with a mock graph.

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
# _last_tool_name
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
        assert (
            _last_tool_name(self._snapshot([tool_msg, plain]))
            == "map_feature_to_themes"
        )

    def test_returns_none_when_all_ai_messages_lack_tool_calls(self) -> None:
        assert _last_tool_name(self._snapshot([_ai("a"), _ai("b")])) is None

    def test_empty_values_returns_none(self) -> None:
        s = MagicMock()
        s.values = {}
        assert _last_tool_name(s) is None


# ---------------------------------------------------------------------------
# _thread_config
# ---------------------------------------------------------------------------


class TestThreadConfig:
    """Tests for the _thread_config helper."""

    def test_structure(self) -> None:
        cfg = _thread_config(_THREAD)
        assert cfg == {"configurable": {"thread_id": _THREAD}}

    def test_thread_id_preserved(self) -> None:
        tid = "my-custom-id"
        config_dict = _thread_config(tid).get("configurable")
        assert config_dict is not None
        assert config_dict["thread_id"] == tid


# ---------------------------------------------------------------------------
# _serialize_message
# ---------------------------------------------------------------------------


class TestSerializeMessage:
    """Tests for the _serialize_message helper."""

    def test_base_message_returns_dict(self) -> None:
        """BaseMessage inputs must be converted to a dict via model_dump()."""
        assert isinstance(_serialize_message(_ai("hello")), dict)

    def test_non_message_passed_through(self) -> None:
        """Non-BaseMessage values must be returned as-is without transformation."""
        assert _serialize_message({"raw": "data"}) == {"raw": "data"}
        assert _serialize_message(42) == 42
        assert _serialize_message("plain string") == "plain string"


# ---------------------------------------------------------------------------
# _resolve_resume
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

    async def test_resume_dict_with_registered_dto_returns_xml_command(self) -> None:
        """Structured dict resume is validated against the registered DTO and serialised to XML."""
        from langgraph.types import Command

        mock_dto = MagicMock()
        mock_dto.model_validate.return_value.model_dump_xml.return_value = "<user_scores />"
        spec = self._spec(interrupt_dtos={"some_tool": mock_dto})

        snapshot = MagicMock()
        snapshot.values = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "some_tool", "id": "t1", "args": {}}],
                )
            ]
        }
        graph = AsyncMock()
        graph.aget_state = AsyncMock(return_value=snapshot)

        result = await _resolve_resume(body=RunInput(resume={"key": "value"}), spec=spec, thread_id=_THREAD, graph=graph)

        assert isinstance(result, Command)
        assert result.resume == "<user_scores />"
        mock_dto.model_validate.assert_called_once_with({"key": "value"})


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
    # async def test_sync_run_requires_key(self, client: AsyncClient) -> None:
    #     r = await client.post(f"{_BASE}/threads/{_THREAD}/runs", json={"message": "hi"})
    #     assert r.status_code == 401

    # async def test_stream_run_requires_key(self, client: AsyncClient) -> None:
    #     r = await client.post(
    #         f"{_BASE}/threads/{_THREAD}/runs/stream", json={"message": "hi"}
    #     )
    #     assert r.status_code == 401

    async def test_env_fallback_accepted(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", _TEST_KEY)
        r = await client.post(f"{_BASE}/threads/{_THREAD}/runs", json={"message": "hi"})
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

    async def test_interrupt_fallback_reads_last_message_from_snapshot(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """When ainvoke returns no messages, the route falls back to aget_state."""
        mock_graph.ainvoke = AsyncMock(return_value=_fake_result([]))
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("interrupted here")]}
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        assert r.json()["content"] == "interrupted here"

    async def test_interrupt_fallback_empty_snapshot_returns_interrupt_type(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """When both ainvoke and the snapshot have no messages, type must be 'interrupt'."""
        mock_graph.ainvoke = AsyncMock(return_value=_fake_result([]))
        snapshot = MagicMock()
        snapshot.values = {"messages": []}
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        r = await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi"},
            headers={"x-api-key": _TEST_KEY},
        )
        body = r.json()
        assert body["type"] == "interrupt"
        assert body["content"] == ""


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
            line[len("data: "):]
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
