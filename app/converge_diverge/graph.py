"""Converge/Diverge agent graph.

Implements a ReAct loop with selective human-in-the-loop interrupts.  The
graph pauses at ``human_review`` only after ``map_feature_to_themes`` and
``score_concepts``; ``score_features`` and ``generate_rct_ebc`` route directly
back to ``call_llm`` without a pause.

Flow::

    START → call_llm ──(tool calls)──→ tools ──(map/score_concepts)──→ human_review → call_llm
                     └──(no tool calls)──→ END        └──(other tools)──────────────→ call_llm
"""

from typing import Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.converge_diverge.context import Context
from app.converge_diverge.state import InputState, State
from app.converge_diverge.tools import TOOLS
from app.utils import load_chat_model


async def call_llm(
    state: State,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering the agent.

    Args:
        state: The current conversation state.
        runtime: Provides access to ``context``, ``store``, and ``stream_writer``.

    Returns:
        A dict with a single ``messages`` key containing the model's response.
    """
    model = load_chat_model(
        runtime.context.model,
        anthropic_api_key=runtime.context.anthropic_api_key.get_secret_value(),
    ).bind_tools(TOOLS)

    response = await model.ainvoke(
        [SystemMessage(runtime.context.system_prompt), *state.messages]
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not complete the workflow in the allowed number of steps.",
                )
            ]
        }
    return {"messages": [response]}


def route_model_output(state: State) -> Literal[END, "tools"]:  # pyright: ignore[reportInvalidTypeForm]
    """Route after ``call_llm``: go to tools if there are tool calls, else end.

    Args:
        state: The current conversation state.

    Returns:
        ``"tools"`` when the last message contains tool calls, ``END`` otherwise.
    """
    match state.messages[-1]:
        case AIMessage(tool_calls=tool_calls):
            if tool_calls:
                return "tools"
            else:
                return END
        case last_message:
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )


async def human_review(state: State) -> Dict[str, list]:
    """Pause the graph and surface the last tool output for human inspection.

    Calls ``interrupt()`` so LangGraph checkpoints the graph here and waits for
    a ``Command(resume=...)`` from the caller.  The interrupted value is a dict
    that API consumers can read from the thread state.

    If the caller resumes with a non-empty string, it is injected as a
    ``HumanMessage`` so the LLM sees the feedback on its next turn.

    Args:
        state: The current conversation state.

    Returns:
        A dict with an updated ``messages`` list (empty if no feedback given).
    """
    last_content = state.messages[-1].content if state.messages else ""
    feedback: str = interrupt(
        {
            "question": (
                "Review the tool output. Provide feedback to guide the next step, "
                "or resume with an empty string to continue."
            ),
            "last_tool_output": last_content,
        }
    )
    if feedback:
        return {"messages": [HumanMessage(content=feedback)]}
    return {}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

_INTERRUPT_AFTER: frozenset[str] = frozenset({"map_feature_to_themes", "score_concepts"})


def route_after_tools(state: State) -> Literal["human_review", "call_llm"]:
    """Route to human_review only for tools that require human feedback.

    Walks the message list in reverse to find the last ``AIMessage`` with tool
    calls and checks whether the final tool name is in ``_INTERRUPT_AFTER``.
    All other tools bypass ``human_review`` and return directly to ``call_llm``.

    Args:
        state: The current conversation state.

    Returns:
        ``"human_review"`` for ``map_feature_to_themes`` and ``score_concepts``;
        ``"call_llm"`` for all other tools.
    """
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if msg.tool_calls[-1]["name"] in _INTERRUPT_AFTER:
                return "human_review"
            return "call_llm"
    return "call_llm"


builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node(call_llm)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(human_review)

builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", route_model_output)
builder.add_conditional_edges("tools", route_after_tools)
builder.add_edge("human_review", "call_llm")
