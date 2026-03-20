"""Define the Agent's tools."""

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.types import Command

from app.orient.context import Context
from app.orient.models import QAResults, SynthesisResults
from app.orient.state import State
from app.utils import load_chat_model


@tool
async def synthesize_problem_statement(
    validated_qa: str, runtime: ToolRuntime[Context, State]
) -> str:
    """Synthesize the <validated_qa /> into a <problem_statement />."""
    model = load_chat_model(runtime.context.model).with_structured_output(
        SynthesisResults
    )

    match await model.ainvoke(
        [SystemMessage(runtime.context.synthesis_prompt), HumanMessage(validated_qa)]
    ):
        case SynthesisResults() as parsed:
            return parsed.model_dump_xml()
        case unknown:
            raise ValueError(f"Expected a results summary, but got {unknown}")


@tool
async def review_user_problem(
    user_summary: str, runtime: ToolRuntime[Context, State]
) -> Command:
    """Review the summary of the user's problem and generate the <validated_qa /> necessary to synthesize a <problem_statement />."""
    model = load_chat_model(runtime.context.model).with_structured_output(QAResults)

    # Get the model's response
    match await model.ainvoke(
        [SystemMessage(runtime.context.review_prompt), HumanMessage(user_summary)],
    ):
        case QAResults() as parsed:
            # Return the model's response as a list to be added to existing messages
            return Command(
                update={
                    "qa_results": parsed.model_dump(),
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


TOOLS: list[BaseTool] = [synthesize_problem_statement, review_user_problem]
