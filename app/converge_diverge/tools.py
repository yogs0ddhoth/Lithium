"""Define the Agent's tools."""

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.types import Command

from app.converge_diverge.context import Context
from app.converge_diverge.prompts import ConceptScores, FeatureThemesMap, RtcEbcPrompt
from app.converge_diverge.state import State
from app.utils import load_chat_model


@tool
async def score_concepts(
    validated_qa: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Synthesize the <validated_qa /> into <concept_scores />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(ConceptScores)

    match await model.ainvoke(
        [
            SystemMessage(runtime.context.concept_scoring_prompt),
            HumanMessage(validated_qa),
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
async def map_feature_to_themes(
    user_summary: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Review the summary of the user's problem and generate the <validated_qa /> necessary to synthesize a <features_and_themes /> list."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(FeatureThemesMap)

    # Get the model's response
    match await model.ainvoke(
        [
            SystemMessage(runtime.context.feature_themes_map_synthesis_prompt),
            HumanMessage(user_summary),
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
async def generate_rct_ebc(
    validated_qa: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Synthesize the <validated_qa /> into <rtc-ebc />."""
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value() or None,
    ).with_structured_output(RtcEbcPrompt)

    match await model.ainvoke(
        [
            SystemMessage(runtime.context.rtc_ebc_prompt),
            HumanMessage(validated_qa),
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


TOOLS: list[BaseTool] = [score_concepts, map_feature_to_themes, generate_rct_ebc]
