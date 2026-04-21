"""Unit tests for app/orient/graph.py — pure routing functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.orient.graph import route_model_output


def _state(messages: list) -> SimpleNamespace:
    """Minimal state stub — route_model_output only reads state.messages."""
    return SimpleNamespace(messages=messages)


class TestRouteModelOutput:
    def test_ai_message_with_tool_calls_routes_to_tools(self) -> None:
        state = _state(
            [AIMessage(content="", tool_calls=[{"name": "review_user_problem", "id": "t1", "args": {}}])]
        )
        assert route_model_output(state) == "tools"

    def test_ai_message_without_tool_calls_routes_to_end(self) -> None:
        state = _state([AIMessage(content="Final answer.")])
        assert route_model_output(state) == END

    def test_non_ai_message_raises_value_error(self) -> None:
        state = _state([HumanMessage(content="hi")])
        with pytest.raises(ValueError, match="Expected AIMessage"):
            route_model_output(state)
