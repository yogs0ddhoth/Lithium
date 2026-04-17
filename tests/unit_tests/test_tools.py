"""Unit tests for app/orient/tools.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.orient.prompts import ProblemStatement, QAResults
from app.orient.tools import review_user_problem, synthesize_problem_statement

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(
    model: str = "anthropic/claude-sonnet-4-5-20250929",
    api_key: str = "sk-test-key",
    tool_call_id: str = "call-abc123",
    review_prompt: str = "Review the problem.",
    synthesis_prompt: str = "Synthesize the problem.",
) -> MagicMock:
    """Build a minimal ToolRuntime mock."""
    ctx = MagicMock()
    ctx.model = model
    ctx.anthropic_api_key.get_secret_value.return_value = api_key
    ctx.review_prompt = review_prompt
    ctx.synthesis_prompt = synthesis_prompt

    runtime = MagicMock()
    runtime.context = ctx
    runtime.tool_call_id = tool_call_id
    return runtime


def _mock_structured_model(return_value: object) -> MagicMock:
    """Return a mock that simulates load_chat_model().with_structured_output()."""
    inner = AsyncMock()
    inner.ainvoke = AsyncMock(return_value=return_value)

    outer = MagicMock()
    outer.with_structured_output.return_value = inner
    return outer


# ---------------------------------------------------------------------------
# synthesize_problem_statement
# ---------------------------------------------------------------------------


async def test_synthesize_problem_statement_returns_xml_string() -> None:
    ps = ProblemStatement(
        executive_summary="Test summary", background="Some background."
    )
    runtime = _make_runtime()

    with patch("orient.tools.load_chat_model", return_value=_mock_structured_model(ps)):
        result = await synthesize_problem_statement.coroutine(
            validated_qa="<validated_qa_results />", runtime=runtime
        )

    assert isinstance(result, str)
    assert "<problem_statement>" in result
    assert "Test summary" in result


async def test_synthesize_problem_statement_uses_synthesis_prompt() -> None:
    ps = ProblemStatement()
    runtime = _make_runtime(synthesis_prompt="Custom synthesis prompt")
    inner = MagicMock()
    inner.ainvoke = AsyncMock(return_value=ps)
    outer = MagicMock()
    outer.with_structured_output.return_value = inner

    with patch("orient.tools.load_chat_model", return_value=outer):
        await synthesize_problem_statement.coroutine(
            validated_qa="<validated_qa_results />", runtime=runtime
        )

    system_msg, human_msg = inner.ainvoke.call_args[0][0]
    assert system_msg.content == "Custom synthesis prompt"
    assert "<validated_qa_results />" in human_msg.content


async def test_synthesize_problem_statement_uses_configured_model() -> None:
    ps = ProblemStatement()
    runtime = _make_runtime(model="openai/gpt-4o", api_key="")
    inner = MagicMock()
    inner.ainvoke = AsyncMock(return_value=ps)

    with patch("orient.tools.load_chat_model") as mock_load:
        mock_load.return_value.with_structured_output.return_value = inner
        await synthesize_problem_statement.coroutine(
            validated_qa="test", runtime=runtime
        )

    mock_load.assert_called_once_with("openai/gpt-4o", anthropic_api_key=None)


async def test_synthesize_problem_statement_raises_on_unexpected_output() -> None:
    runtime = _make_runtime()

    with patch(
        "orient.tools.load_chat_model",
        return_value=_mock_structured_model("not a ProblemStatement"),
    ):
        with pytest.raises(ValueError, match="Expected a results summary"):
            await synthesize_problem_statement.coroutine(
                validated_qa="test", runtime=runtime
            )


# ---------------------------------------------------------------------------
# review_user_problem
# ---------------------------------------------------------------------------


async def test_review_user_problem_returns_command() -> None:
    qa = QAResults(answered_count=3, not_answerable_count=1)
    runtime = _make_runtime()

    with patch("orient.tools.load_chat_model", return_value=_mock_structured_model(qa)):
        result = await review_user_problem.coroutine(
            user_summary="Our app is slow.", runtime=runtime
        )

    assert isinstance(result, Command)


async def test_review_user_problem_command_contains_qa_results() -> None:
    qa = QAResults(answered_count=5, not_answerable_count=2)
    runtime = _make_runtime()

    with patch("orient.tools.load_chat_model", return_value=_mock_structured_model(qa)):
        result = await review_user_problem.coroutine(
            user_summary="test summary", runtime=runtime
        )

    assert "qa_results" in result.update
    assert result.update["qa_results"] == qa.model_dump()


async def test_review_user_problem_command_contains_tool_message() -> None:
    qa = QAResults(answered_count=2)
    runtime = _make_runtime(tool_call_id="call-xyz")

    with patch("orient.tools.load_chat_model", return_value=_mock_structured_model(qa)):
        result = await review_user_problem.coroutine(
            user_summary="test summary", runtime=runtime
        )

    messages = result.update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "call-xyz"
    assert "<validated_qa_results>" in messages[0].content


async def test_review_user_problem_uses_review_prompt() -> None:
    qa = QAResults()
    runtime = _make_runtime(review_prompt="Custom review prompt")
    inner = MagicMock()
    inner.ainvoke = AsyncMock(return_value=qa)
    outer = MagicMock()
    outer.with_structured_output.return_value = inner

    with patch("orient.tools.load_chat_model", return_value=outer):
        await review_user_problem.coroutine(
            user_summary="the user's problem", runtime=runtime
        )

    system_msg, human_msg = inner.ainvoke.call_args[0][0]
    assert system_msg.content == "Custom review prompt"
    assert "the user's problem" in human_msg.content


async def test_review_user_problem_raises_on_unexpected_output() -> None:
    runtime = _make_runtime()

    with patch(
        "orient.tools.load_chat_model",
        return_value=_mock_structured_model(42),
    ):
        with pytest.raises(ValueError, match="Expected a results summary"):
            await review_user_problem.coroutine(user_summary="test", runtime=runtime)
