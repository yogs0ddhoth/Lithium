"""Default prompts used by the agent."""

from typing import Any

import xml_pydantic

from app.utils import load_xml_prompt, logger

logger.info("# Loading Prompts...")
FEATURE_THEME_MAP_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/features_and_themes.xml"
)

_THEMES = xml_pydantic.define_model(
    "Themes",
    {
        "$defs": FEATURE_THEME_MAP_SCHEMA["$defs"],
        "type": "array",
        "items": {"$ref": "#/$defs/Theme"},
    },
)

SYNTHESIS_PROMPT = load_xml_prompt("prompts/features_and_themes.xml")
logger.info("# Prompts loaded.")
