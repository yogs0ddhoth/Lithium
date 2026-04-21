"""Unit tests for app/converge_diverge/graph.py — pure routing functions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.converge_diverge.graph import route_after_tools, route_model_output


def _state(messages: list) -> SimpleNamespace:
    """Minimal state stub — routing functions only read state.messages."""
    return SimpleNamespace(messages=messages)


def _ai_with_tools(*tool_names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "id": f"t_{n}", "args": {}} for n in tool_names],
    )


# ---------------------------------------------------------------------------
# route_model_output
# ---------------------------------------------------------------------------


class TestRouteModelOutput:
    def test_ai_message_with_tool_calls_routes_to_tools(self) -> None:
        state = _state([_ai_with_tools("map_feature_to_themes")])
        assert route_model_output(state) == "tools"

    def test_ai_message_without_tool_calls_routes_to_end(self) -> None:
        state = _state([AIMessage(content="done")])
        assert route_model_output(state) == END

    def test_non_ai_message_raises_value_error(self) -> None:
        state = _state([HumanMessage(content="hi")])
        with pytest.raises(ValueError, match="Expected AIMessage"):
            route_model_output(state)


# ---------------------------------------------------------------------------
# route_after_tools
# ---------------------------------------------------------------------------


class TestRouteAfterTools:
    def test_map_feature_to_themes_routes_to_human_review(self) -> None:
        state = _state([_ai_with_tools("map_feature_to_themes")])
        assert route_after_tools(state) == "human_review"

    def test_score_concepts_routes_to_human_review(self) -> None:
        state = _state([_ai_with_tools("score_concepts")])
        assert route_after_tools(state) == "human_review"

    def test_score_features_routes_to_call_llm(self) -> None:
        state = _state([_ai_with_tools("score_features")])
        assert route_after_tools(state) == "call_llm"

    def test_generate_rct_ebc_routes_to_call_llm(self) -> None:
        state = _state([_ai_with_tools("generate_rct_ebc")])
        assert route_after_tools(state) == "call_llm"

    def test_no_ai_message_with_tool_calls_defaults_to_call_llm(self) -> None:
        state = _state([HumanMessage(content="hi"), AIMessage(content="plain")])
        assert route_after_tools(state) == "call_llm"

    def test_last_tool_call_name_determines_route(self) -> None:
        """When an AIMessage has multiple tool calls, only the last name matters."""
        state = _state([_ai_with_tools("score_concepts", "score_features")])
        assert route_after_tools(state) == "call_llm"
