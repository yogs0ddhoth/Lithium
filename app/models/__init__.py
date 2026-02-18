import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
