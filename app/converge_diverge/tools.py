"""Define the Agent's tools."""

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.types import Command

from app.converge_diverge.context import Context
from app.converge_diverge.prompts import (
    ConceptScores,
    FeatureScores,
    FeatureThemesMap,
    RtcEbcInput,
    RtcEbcPrompt,
)
from app.converge_diverge.state import State
from app.utils import load_chat_model


@tool
async def map_feature_to_themes(
    features_and_needs: str,
    runtime: ToolRuntime[Context, State],
) -> Command:
    """Review the <features_and_needs /> and generate synthesize a <features_and_themes /> for the user to create <user_scores_and_solutions />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(FeatureThemesMap)

    # Get the model's response
    match await model.ainvoke(
        [
            SystemMessage(runtime.context.feature_themes_map_synthesis_prompt),
            HumanMessage(features_and_needs),
        ],
    ):
        case FeatureThemesMap() as parsed:
            # Return the model's response as a list to be added to existing messages
            return Command(
                update={
                    "features_and_themes": parsed.model_dump(),
                    "messages": [
                        ToolMessage(
                            content=parsed.model_dump_xml(),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case unknown:
            raise ValueError(f"Expected a results summary, but got {unknown}")


@tool
async def score_concepts(
    user_scores_and_solutions: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Synthesize the <user_scores_and_solutions /> into <normalized_concept_scores /> so that the user can create <user_feature_scores />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(ConceptScores)

    match await model.ainvoke(
        [
            SystemMessage(runtime.context.concept_scoring_prompt),
            HumanMessage(user_scores_and_solutions),
        ]
    ):
        case ConceptScores() as parsed:
            return Command(
                update={
                    "concept_scores": parsed.model_dump(),
                    "messages": [
                        ToolMessage(
                            content=parsed.model_dump_xml(),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case unknown:
            raise ValueError(f"Expected a results summary, but got {unknown}")


@tool
async def score_features(
    user_feature_scores: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Synthesize the <user_feature_scores /> into <highest_scoring_features />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(FeatureScores)

    match await model.ainvoke(
        [
            SystemMessage(runtime.context.feature_scoring_prompt),
            HumanMessage(user_feature_scores),
        ]
    ):
        case FeatureScores() as parsed:
            return Command(
                update={
                    "feature_scores": parsed.model_dump(),
                    "messages": [
                        ToolMessage(
                            content=parsed.model_dump_xml(),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case unknown:
            raise ValueError(f"Expected a results summary, but got {unknown}")


@tool(args_schema=RtcEbcInput)
async def generate_rct_ebc(
    features_and_themes: str,
    normalized_concept_scores: str,
    highest_scoring_features: str,
    runtime: ToolRuntime[Context, State],
) -> Command:
    """Synthesize <features_and_themes>, <normalized_concept_scores />, <highest_scoring_features /> into <rtc-ebc />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(RtcEbcPrompt)

    match await model.ainvoke(
        [
            SystemMessage(runtime.context.rtc_ebc_prompt),
            HumanMessage(
                features_and_themes
                + normalized_concept_scores
                + highest_scoring_features
            ),
        ]
    ):
        case RtcEbcPrompt() as parsed:
            return Command(
                update={
                    "rtc_ebc": parsed.model_dump(),
                    "messages": [
                        ToolMessage(
                            content=parsed.model_dump_xml(),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        case unknown:
            raise ValueError(f"Expected a results summary, but got {unknown}")


TOOLS: list[BaseTool] = [
    map_feature_to_themes,
    score_concepts,
    score_features,
    generate_rct_ebc,
]
