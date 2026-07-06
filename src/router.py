"""Router nodes: classify a query as structured / unstructured / out_of_scope.

Plain nodes and conditionals wired into the main graph (src/graph.py).
"""

from typing import Annotated, Literal

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from .common import get_model
from .prompts import router_prompt
from .state import GraphState
from .tools import get_dataset_description_tool

model = get_model("router")


class EmitRouterNodeResponse(BaseModel):
    reasoning: str = Field(description="The reasoning for your decision")
    query_type: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="The classification of the user's query"
    )
    tool_call_id: Annotated[str, InjectedToolCallId()] = Field(
        description="The tool call id of the tool that called this tool"
    )


@tool(args_schema=EmitRouterNodeResponse)
def emit_router_node_response(
    reasoning: str,
    query_type: Literal["structured", "unstructured", "out_of_scope"],
    tool_call_id: Annotated[str, InjectedToolCallId()],
) -> Command[GraphState]:
    """
    Create a router node response based upon the reasoning and query_type input.
    Args:
        reasoning: The reasoning for your decision
        query_type: The classification of the user's query
        tool_call_id: The tool call id of the tool that called this tool
    """
    return Command(
        update={
            "messages": [
                ToolMessage(content=reasoning, tool_call_id=tool_call_id),
            ],
            "query_type": query_type,
            "in_scope": query_type != "out_of_scope",
        },
    )


router_tools = [
    emit_router_node_response,
    get_dataset_description_tool,
]
# tool_choice enforced at the API level: prompt-only "you MUST call the
# tool" is not reliable (measured: the router answered meta-statements in
# plain text, which fell through to the decline path)
model_with_tools = model.bind_tools(router_tools, tool_choice="required")
model_forced_emit = model.bind_tools(
    [emit_router_node_response], tool_choice="emit_router_node_response"
)

router_tools_node = ToolNode(tools=router_tools, name="router_tools_node")


def _description_fetched_this_turn(messages) -> bool:
    """Did get_dataset_description_tool run since the last user message?"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return False
        if (
            isinstance(message, ToolMessage)
            and message.name == "get_dataset_description_tool"
        ):
            return True
    return False


def router_node(state: GraphState) -> GraphState:
    # Route the user input to the appropriate node.
    # The description is static, so one fetch per turn is all the
    # information the router can ever get - after that, force the
    # classification (measured: the model otherwise re-fetches it in a
    # loop until max_llm_calls, and the turn ends in a spurious decline).
    bound = (
        model_forced_emit
        if _description_fetched_this_turn(state["messages"])
        else model_with_tools
    )
    messages = bound.invoke(
        [SystemMessage(content=router_prompt)] + state["messages"]
    )

    return {
        "messages": [messages],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_after_router_llm(
    state: GraphState,
) -> Literal["tools", "no_decision"]:
    """Run the router's tool calls, or bail out if it stopped without deciding."""
    if state["llm_calls"] > state["max_llm_calls"]:
        return "no_decision"
    if state["messages"][-1].tool_calls:
        return "tools"
    return "no_decision"


def route_after_router_tools(
    state: GraphState,
) -> Literal["in_scope", "out_of_scope", "continue"]:
    """After tools ran: decision reached -> route by scope, otherwise loop the router."""
    if state.get("in_scope") is None:
        return "continue"
    return "in_scope" if state["in_scope"] else "out_of_scope"
