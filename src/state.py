from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class LoopGuardrails(TypedDict):
    max_llm_calls: int


class UserInputState(TypedDict):
    user_input: str
    in_scope: bool
    query_type: str
    user_profile: str


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int = 0


class GraphState(UserInputState, MessagesState, LoopGuardrails):
    plan: list[str]
