"""Unit tests for app/utils.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.utils import (
    get_message_text,
    load_chat_model,
    load_xml_prompt,
    normalize_whitespace,
    pascal_to_snake,
)

# ---------------------------------------------------------------------------
# get_message_text
# ---------------------------------------------------------------------------


def test_get_message_text_string_content() -> None:
    msg = HumanMessage(content="hello world")
    assert get_message_text(msg) == "hello world"


def test_get_message_text_dict_content() -> None:
    msg = MagicMock()
    msg.content = {"text": "dict content"}
    assert get_message_text(msg) == "dict content"


def test_get_message_text_dict_content_missing_text_key() -> None:
    msg = MagicMock()
    msg.content = {"type": "image"}
    assert get_message_text(msg) == ""


def test_get_message_text_list_content_strings() -> None:
    msg = MagicMock()
    msg.content = ["hello", " ", "world"]
    # The list is joined with "".join then stripped; inner whitespace is preserved.
    assert get_message_text(msg) == "hello world"


def test_get_message_text_list_content_dicts() -> None:
    msg = MagicMock()
    msg.content = [{"text": "foo"}, {"text": "bar"}]
    assert get_message_text(msg) == "foobar"


def test_get_message_text_list_content_mixed() -> None:
    msg = MagicMock()
    msg.content = ["prefix-", {"text": "middle"}, {"type": "image"}, "-suffix"]
    assert get_message_text(msg) == "prefix-middle-suffix"


# ---------------------------------------------------------------------------
# load_chat_model
# ---------------------------------------------------------------------------


def test_load_chat_model_parses_provider_and_model() -> None:
    mock_model = MagicMock()
    with patch("app.utils.init_chat_model", return_value=mock_model) as mock_init:
        result = load_chat_model("anthropic/claude-sonnet-4-5")

    mock_init.assert_called_once_with("claude-sonnet-4-5", model_provider="anthropic")
    assert result is mock_model


def test_load_chat_model_passes_kwargs() -> None:
    mock_model = MagicMock()
    with patch("app.utils.init_chat_model", return_value=mock_model) as mock_init:
        load_chat_model("openai/gpt-4o", temperature=0, max_tokens=512)

    mock_init.assert_called_once_with(
        "gpt-4o", model_provider="openai", temperature=0, max_tokens=512
    )


def test_load_chat_model_handles_model_name_with_slashes() -> None:
    """Provider is split on the first '/' only; model name may contain slashes."""
    mock_model = MagicMock()
    with patch("app.utils.init_chat_model", return_value=mock_model) as mock_init:
        load_chat_model("anthropic/claude-3-5/latest")

    mock_init.assert_called_once_with("claude-3-5/latest", model_provider="anthropic")


# ---------------------------------------------------------------------------
# pascal_to_snake
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("PascalCase", "pascal_case"),
        ("CamelCaseWord", "camel_case_word"),
        ("SimpleWord", "simple_word"),
        ("already_snake", "already_snake"),
        ("lowercase", "lowercase"),
        ("ABC", "abc"),
        ("MyXMLParser", "my_xmlparser"),
        ("GetHTTPResponse", "get_httpresponse"),
        ("Version2Release", "version2_release"),
    ],
)
def test_pascal_to_snake(input_name: str, expected: str) -> None:
    assert pascal_to_snake(input_name) == expected


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("hello   world", "hello world"),
        ("  leading and trailing  ", "leading and trailing"),
        ("tab\there", "tab here"),
        ("newline\nhere", "newline here"),
        ("mixed \t \n whitespace", "mixed whitespace"),
        ("already clean", "already clean"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_whitespace(input_str: str, expected: str) -> None:
    assert normalize_whitespace(input_str) == expected


# ---------------------------------------------------------------------------
# load_xml_prompt
# ---------------------------------------------------------------------------


def test_load_xml_prompt_serializes_children(tmp_path: Path) -> None:
    xml_file = tmp_path / "prompt.xml"
    xml_file.write_text(
        "<root><instruction>Be helpful.</instruction><rule>Be concise.</rule></root>",
        encoding="utf-8",
    )
    result = load_xml_prompt(str(xml_file))
    assert result == "<instruction>Be helpful.</instruction> <rule>Be concise.</rule>"


def test_load_xml_prompt_normalizes_whitespace(tmp_path: Path) -> None:
    xml_file = tmp_path / "prompt.xml"
    xml_file.write_text(
        "<root>\n  <step>  do   this  </step>\n</root>",
        encoding="utf-8",
    )
    result = load_xml_prompt(str(xml_file))
    # ET.tostring preserves inner whitespace, then normalize_whitespace collapses all runs.
    assert result == "<step> do this </step>"


def test_load_xml_prompt_joins_multiple_children_with_space(tmp_path: Path) -> None:
    xml_file = tmp_path / "prompt.xml"
    xml_file.write_text(
        "<root><a>one</a><b>two</b><c>three</c></root>",
        encoding="utf-8",
    )
    result = load_xml_prompt(str(xml_file))
    parts = result.split(" ")
    assert len(parts) == 3


def test_load_xml_prompt_empty_root(tmp_path: Path) -> None:
    xml_file = tmp_path / "prompt.xml"
    xml_file.write_text("<root></root>", encoding="utf-8")
    result = load_xml_prompt(str(xml_file))
    assert result == ""
