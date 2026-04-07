"""Default prompts used by the agent."""

from typing import Any

import xml_pydantic

from app.utils import load_xml_prompt, logger

logger.info("# Loading Prompts...")

FEATURE_THEME_MAP_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/features_and_themes.schema.xml"
)
logger.info(FEATURE_THEME_MAP_SCHEMA)

_FeatureThemesMap = xml_pydantic.define_model(
    "FeatureThemesMap", FEATURE_THEME_MAP_SCHEMA
)


class FeatureThemesMap(_FeatureThemesMap):
    """DTO for the `<features_and_themes />`."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(
            self, root_tag="features_and_themes"
        )


FEATURE_THEME_MAP_SYNTHESIS_PROMPT = load_xml_prompt("prompts/features_and_themes.xml")

CONCEPT_SCORES_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/concept_scores.schema.xml"
)
_ConceptScores = xml_pydantic.define_model("ConceptScores", CONCEPT_SCORES_SCHEMA)


class ConceptScores(_ConceptScores):
    """DTO for the `<concept_scores />`."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(
            self, root_tag="concept_scores"
        )


CONCEPT_SCORING_PROMPT = load_xml_prompt("prompts/concept_scoring.xml")


RTC_EBC_SCHEMA = xml_pydantic.schema.from_file("prompts/rtc-ebc_prompt.schema.xml")

_RtcEbcPrompt = xml_pydantic.define_model("RtcEbcPrompt", RTC_EBC_SCHEMA)


class RtcEbcPrompt(_RtcEbcPrompt):
    """DTO for the `<rtc-ebc />`."""

    def model_dump_xml(self) -> str:
        """Serialize the model to an XML string."""
        return xml_pydantic.serializers.model_to_xml_string(self, root_tag="rtc-ebc")


RTC_EBC_SYNTHESIS_PROMPT = load_xml_prompt("prompts/rtc-ebc_prompt_engineering.xml")


logger.info("# Prompts loaded.")
