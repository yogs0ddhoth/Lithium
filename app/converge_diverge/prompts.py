"""Default prompts used by the agent."""

from typing import Annotated

import xml_pydantic
from pydantic import BaseModel, Field

from app.utils import XmlDto, load_xml_prompt, logger

logger.info("# Loading Prompts...")

SYSTEM_PROMPT = load_xml_prompt("prompts/converge_diverge/system.xml")

# map_feature_to_themes
FEATURE_THEME_MAP_SYNTHESIS_PROMPT = load_xml_prompt(
    "prompts/converge_diverge/features_and_themes.xml"
)

FEATURE_THEME_MAP_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/features_and_themes.schema.xml"
)


class FeatureThemesMap(
    xml_pydantic.define_model("FeatureThemesMap", FEATURE_THEME_MAP_SCHEMA), XmlDto
):
    """DTO for the `<features_and_themes />`."""

    _root_tag = "features_and_themes"


FEATURES_AND_NEEDS_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/features_and_needs.schema.xml"
)


class FeaturesAndNeeds(
    xml_pydantic.define_model("FeaturesAndNeeds", FEATURES_AND_NEEDS_SCHEMA), XmlDto
):
    """DTO for the `<features_and_needs />`."""

    _root_tag = "features_and_needs"


# score_concepts

CONCEPT_SCORING_PROMPT = load_xml_prompt("prompts/converge_diverge/concept_scoring.xml")

CONCEPT_USER_SCORES_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/concept_user_scores.schema.xml"
)


class UserScoresAndSolutions(
    xml_pydantic.define_model("UserScoresAndSolutions", CONCEPT_USER_SCORES_SCHEMA),
    XmlDto,
):
    """DTO for the `<user_scores_and_solutions />`."""

    _root_tag = "user_scores_and_solutions"


CONCEPT_SCORES_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/concept_scores.schema.xml"
)


class ConceptScores(
    xml_pydantic.define_model("ConceptScores", CONCEPT_SCORES_SCHEMA), XmlDto
):
    """DTO for the `<normalized_concept_scores />`."""

    _root_tag = "normalized_concept_scores"


#  score_features
FEATURE_SCORING_PROMPT = load_xml_prompt("prompts/converge_diverge/feature_scoring.xml")

FEATURE_USER_SCORES_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/feature_user_scores.schema.xml"
)


class UserFeatureScores(
    xml_pydantic.define_model("UserFeatureScores", FEATURE_USER_SCORES_SCHEMA), XmlDto
):
    """DTO for the `<user_feature_scores />`."""

    _root_tag = "user_feature_scores"


FEATURE_SCORES_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/feature_scores.schema.xml"
)


class FeatureScores(
    xml_pydantic.define_model("FeatureScores", FEATURE_SCORES_SCHEMA), XmlDto
):
    """DTO for the `<highest_scoring_features />`."""

    _root_tag = "highest_scoring_features"


RTC_EBC_SCHEMA = xml_pydantic.schema.from_file(
    "prompts/converge_diverge/rtc-ebc_prompt.schema.xml"
)


# generate_rct_ebc


class RtcEbcInput(BaseModel):
    """Necessary Input to synthesize `<rtc-ebc />`."""

    features_and_themes: Annotated[
        str, Field(..., description="<features_and_themes />")
    ]
    normalized_concept_scores: Annotated[
        str, Field(..., description="<normalized_concept_scores />")
    ]
    highest_scoring_features: Annotated[
        str, Field(..., description="<highest_scoring_features />")
    ]


class RtcEbcPrompt(xml_pydantic.define_model("RtcEbcPrompt", RTC_EBC_SCHEMA), XmlDto):
    """DTO for the `<rtc-ebc />`."""

    _root_tag = "rtc-ebc"


RTC_EBC_SYNTHESIS_PROMPT = load_xml_prompt(
    "prompts/converge_diverge/rtc-ebc_prompt_engineering.xml"
)

logger.info("# Prompts loaded.")
