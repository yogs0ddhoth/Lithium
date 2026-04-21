"""Unit tests for app/server/models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.server.models import RunInput


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
