"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from pydantic import SecretStr

import app.converge_diverge.prompts as prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    feature_themes_map_synthesis_prompt: str = field(
        default=prompts.FEATURE_THEME_MAP_SYNTHESIS_PROMPT,
        metadata={"description": "System prompt for the map_feature_to_themes tool"},
    )

    concept_scoring_prompt: str = field(
        default=prompts.CONCEPT_SCORING_PROMPT,
        metadata={"description": "System prompt for the score_concepts tool"},
    )
    feature_scoring_prompt: str = field(
        default=prompts.FEATURE_SCORING_PROMPT,
        metadata={"description": "System prompt for the score_features tool"},
    )
    rtc_ebc_prompt: str = field(
        default=prompts.RTC_EBC_SYNTHESIS_PROMPT,
        metadata={
            "description": "System prompt for the synthesize_problem_statement tool"
        },
    )
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    anthropic_api_key: SecretStr = field(
        default=SecretStr(""),
        metadata={
            "description": "Anthropic API key. Injected from X-Anthropic-Api-Key request header; "
            "falls back to ANTHROPIC_API_KEY env var."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            current = getattr(self, f.name)
            if current == f.default:
                env_val = os.environ.get(f.name.upper())
                if env_val is not None:
                    value = (
                        SecretStr(env_val)
                        if isinstance(f.default, SecretStr)
                        else env_val
                    )
                    setattr(self, f.name, value)
