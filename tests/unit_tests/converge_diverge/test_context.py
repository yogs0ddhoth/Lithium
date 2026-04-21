"""Unit tests for app/converge_diverge/context.py."""

from __future__ import annotations

import pytest

from app.converge_diverge.context import Context


class TestContextDefaults:
    def test_construction_succeeds(self) -> None:
        ctx = Context()
        assert ctx is not None

    def test_default_model(self) -> None:
        assert Context().model == "anthropic/claude-sonnet-4-5-20250929"

    def test_default_api_key_is_empty(self) -> None:
        assert Context().anthropic_api_key.get_secret_value() == ""


class TestContextEnvVarOverride:
    def test_string_field_overridden_by_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "openai/gpt-4o")
        assert Context().model == "openai/gpt-4o"

    def test_explicit_value_wins_over_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "openai/gpt-4o")
        assert Context(model="anthropic/claude-opus-4-6").model == "anthropic/claude-opus-4-6"

    def test_api_key_env_var_wrapped_in_secret_str(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
        assert Context().anthropic_api_key.get_secret_value() == "sk-env-key"

    def test_unset_env_var_leaves_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODEL", raising=False)
        assert Context().model == "anthropic/claude-sonnet-4-5-20250929"
