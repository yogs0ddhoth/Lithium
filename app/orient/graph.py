"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from typing import Dict, List, Literal

from langchain.messages import SystemMessage
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from app.orient.context import Context
from app.orient.state import InputState, State
from app.orient.tools import TOOLS
from app.utils import load_chat_model


async def call_llm(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our Agent.

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    # Get the model's response
    response = await model.ainvoke(
        [SystemMessage(runtime.context.system_prompt), *state.messages]
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    return {"messages": [response]}


#
def route_model_output(state: State) -> Literal[END, "tools"]:  # pyright: ignore[reportInvalidTypeForm]
    """Determine the next node based on the llm's output.

    This function implements the ReAct Tool loop with explicit control of state.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    # langgraph.prebuilt.tools_condition does the same as below. Defining it explicitly gives finer control over state
    match state.messages[-1]:
        case AIMessage(tool_calls=tool_calls):
            if tool_calls:
                return "tools"  # Execute the requested actions
            else:
                return END  # If there is no tool call, then we finish
        case last_message:
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )


# Define a new graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_llm).add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_llm`
builder.add_edge(
    START, "call_llm"
).add_conditional_edges(
    "call_llm",
    # After call_llm finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output, # State -> ("tools"|END)
).add_edge(
    "tools", "call_llm"
)  # This creates the cycle: after using tools, we always return to the model

# Compile the builder into an executable graph
graph = builder.compile(name="app.orient")
