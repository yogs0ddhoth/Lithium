import os
import pprint
from typing import Literal

from agent_tools.calculator import add, divide, multiply

# from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from models import MessagesState
from models.config import Settings

app_config = Settings()
os.environ["ANTHROPIC_API_KEY"] = app_config.anthropic_api_key
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)
# Augment the LLM with tools
calculator = [add, multiply, divide]
tools_by_name = {method.name: method for method in calculator}
model_with_tools = model.bind_tools(calculator)  # pyright: ignore[reportUnknownMemberType]


def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState) -> MessagesState:
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:  # type: ignore
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])  # type: ignore
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))  # type: ignore
    return {"messages": result, "llm_calls": state.get("llm_calls", 0)}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:  # pyright: ignore[reportInvalidTypeForm]
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if getattr(last_message, "tool_calls", []):
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent

# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


# Invoke
def main():
    messages = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0}
    )
    pprint.pprint(messages)
    for m in messages["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    print("Running...")
    main()
    main()
