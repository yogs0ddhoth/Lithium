"""Unit tests for the FastAPI server.

The graph is always replaced with a lightweight mock so these tests run
offline without any LLM API keys.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.orient.context import Context
from app.server import _make_context, _serialize_message, _thread_config, app

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> MagicMock:
    """Return a mock compiled graph with sensible defaults."""
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"messages": [_ai("Hello")]})
    g.aget_state = AsyncMock(return_value=None)

    async def _default_stream(*args: Any, **kwargs: Any):  # type: ignore[return]
        yield (AIMessageChunk(content="Hello"), {"langgraph_node": "call_llm"})

    g.astream = _default_stream
    return g


@pytest.fixture
async def client(mock_graph: MagicMock) -> AsyncClient:  # type: ignore[override]
    """Async httpx client wired to the FastAPI app.

    Injects the mock graph directly into ``app.state.graphs`` so we bypass
    the lifespan and avoid needing any checkpointer or API keys.
    """
    app.state.graphs = {_AGENT: mock_graph}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pure unit tests (no HTTP)
# ---------------------------------------------------------------------------


class TestMakeContext:
    """Tests for the _make_context helper."""

    def test_sets_api_key(self) -> None:
        ctx = _make_context(_TEST_KEY, {})
        assert ctx.anthropic_api_key.get_secret_value() == _TEST_KEY

    def test_valid_field_override(self) -> None:
        ctx = _make_context(_TEST_KEY, {"model": "openai/gpt-4o"})
        assert ctx.model == "openai/gpt-4o"

    def test_unknown_fields_are_ignored(self) -> None:
        """Extra keys in the overrides dict must not raise."""
        ctx = _make_context(_TEST_KEY, {"model": "openai/gpt-4o", "unknown_key": 99})
        assert ctx.model == "openai/gpt-4o"

    def test_returns_context_instance(self) -> None:
        ctx = _make_context(_TEST_KEY, {})
        assert isinstance(ctx, Context)

    def test_default_model_when_not_overridden(self) -> None:
        ctx = _make_context(_TEST_KEY, {})
        assert ctx.model


class TestThreadConfig:
    """Tests for the _thread_config helper."""

    def test_structure(self) -> None:
        cfg = _thread_config(_THREAD)
        assert cfg == {"configurable": {"thread_id": _THREAD}}

    def test_thread_id_preserved(self) -> None:
        tid = "my-custom-id"
        assert _thread_config(tid)["configurable"]["thread_id"] == tid


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
    async def test_create_thread_unknown_agent_returns_404(
        self, client: AsyncClient
    ) -> None:
        r = await client.post("/agents/nonexistent/threads")
        assert r.status_code == 404

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
        import uuid

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
            return_value={"messages": [_ai("specific response")]}
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
        ctx: Context = mock_graph.ainvoke.call_args.kwargs["context"]
        assert ctx.anthropic_api_key.get_secret_value() == _TEST_KEY

    async def test_model_override_via_config(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        await client.post(
            f"{_BASE}/threads/{_THREAD}/runs",
            json={"message": "hi", "config": {"model": "openai/gpt-4o"}},
            headers={"x-api-key": _TEST_KEY},
        )
        ctx: Context = mock_graph.ainvoke.call_args.kwargs["context"]
        assert ctx.model == "openai/gpt-4o"


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


# ---------------------------------------------------------------------------
# HTTP — state retrieval
# ---------------------------------------------------------------------------


class TestGetState:
    async def test_unknown_thread_returns_404(self, client: AsyncClient) -> None:
        r = await client.get(f"{_BASE}/threads/{_THREAD}/state")
        assert r.status_code == 404

    async def test_known_thread_returns_200(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("hi")]}
        snapshot.next = []
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        assert (await client.get(f"{_BASE}/threads/{_THREAD}/state")).status_code == 200

    async def test_state_body_shape(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snapshot = MagicMock()
        snapshot.values = {"messages": [_ai("hi")]}
        snapshot.next = ["call_llm"]
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        body = (await client.get(f"{_BASE}/threads/{_THREAD}/state")).json()
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

        body = (await client.get(f"{_BASE}/threads/{_THREAD}/state")).json()
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

        await client.get(f"{_BASE}/threads/{_THREAD}/state")
        config = mock_graph.aget_state.call_args.args[0]
        assert config["configurable"]["thread_id"] == _THREAD


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
