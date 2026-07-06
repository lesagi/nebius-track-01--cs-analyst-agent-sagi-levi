"""The main agent graph: route -> plan -> execute (ReAct) -> answer."""

import os
from typing import Literal

from langchain.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from .common import CONFIG, get_model
from .prompts import (
    DECLINE_MESSAGE,
    FALLBACK_MESSAGE,
    executor_prompt,
    planner_prompt,
    profile_update_prompt,
    synthesizer_prompt,
)
from .router import (
    route_after_router_llm,
    route_after_router_tools,
    router_node,
    router_tools_node,
)
from .state import GraphState
from .tools import scan_model, tools

# Per-role models (config.yaml): small fast executor for the tool loop,
# sophisticated planner and synthesizer.
planner_model = get_model("planner")
executor_model = get_model("executor").bind_tools(tools)
synthesizer_model = get_model("synthesizer")


class Plan(BaseModel):
    steps: list[str] = Field(
        description="The fewest short concrete steps that answer the question, each achievable with the available tools"
    )


def profile_path(config: RunnableConfig) -> str:
    user = config.get("configurable", {}).get("user_id", "default")
    return os.path.join("profiles", f"{user}.md")


def load_profile(config: RunnableConfig) -> str:
    try:
        with open(profile_path(config)) as f:
            return f.read()
    except OSError:
        return ""


def current_turn_scratch(messages) -> list[str]:
    """Ids of the working messages after this turn's HumanMessage (tool calls, observations)."""
    ids = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        ids.append(message.id)
    return ids


def start_node(state: GraphState, config: RunnableConfig):
    return {
        "user_input": state["messages"][-1].content,
        "max_llm_calls": int(
            os.getenv("MAX_LLM_CALLS") or CONFIG["max_llm_calls"]
        ),
        "user_profile": load_profile(config),
        # reset per-turn decisions - stale values from the previous turn
        # would short-circuit the router
        "in_scope": None,
        "query_type": None,
    }


def planner(state: GraphState):
    scratch = current_turn_scratch(state["messages"])
    # ponytail: plan from the clean conversation - the router's tool-call
    # transcript confuses structured output, but prior turns are needed
    # for follow-up questions ("show me 3 more")
    history = [m for m in state["messages"] if m.id not in scratch]
    # method="function_calling", not the default json_schema: json_schema
    # relies on the server enforcing the schema and Nebius doesn't for
    # gpt-oss-120b (measured: it returned a bare tool-invocation object,
    # crashing the turn); a forced tool call is the same mechanism the
    # executor already exercises reliably.
    try:
        plan = planner_model.with_structured_output(
            Plan, method="function_calling"
        ).invoke(
            [
                SystemMessage(
                    content=planner_prompt.format(
                        query_type=state["query_type"],
                        user_profile=state.get("user_profile")
                        or "(nothing yet)",
                    )
                )
            ]
            + history
        )
        steps = plan.steps if plan else []
    except (OutputParserException, ValidationError) as e:
        # model-fault only (malformed output despite a healthy API): schema
        # violations (ValidationError) and calls to unbound tools - Nebius
        # enforces neither json_schema nor tool_choice for gpt-oss-120b
        # (OutputParserException). Infra errors propagate - a fallback plan
        # would just mask an outage that kills the executor's calls one
        # node later anyway
        print(f"[planner] structured output failed ({e}); using fallback")
        steps = []
    steps = steps or ["Answer the user's question using the tools."]
    return {
        "plan": steps,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "messages": [RemoveMessage(id=i) for i in scratch],
    }


def _clean(ai: AIMessage) -> AIMessage:
    """Strip whitespace-padded content before it enters stored history
    (hosted thinking models pad answers with a leading '\\n\\n')."""
    if isinstance(ai.content, str) and ai.content != ai.content.strip():
        return ai.model_copy(update={"content": ai.content.strip()})
    return ai


def llm_call(state: GraphState):
    plan_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(state["plan"])
    )
    prompt = executor_prompt.format(
        plan=plan_text,
        query_type=state["query_type"],
        user_profile=state.get("user_profile") or "(nothing yet)",
    )
    return {
        "messages": [
            _clean(
                executor_model.invoke(
                    [SystemMessage(content=prompt)] + state["messages"]
                )
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def should_use_tool(
    state: GraphState,
) -> Literal["tool", "done", "fallback"]:
    if state["llm_calls"] >= state["max_llm_calls"]:
        return "fallback"
    if state["messages"][-1].tool_calls:
        return "tool"
    return "done"


def synthesizer(state: GraphState):
    """Write the user-facing answer from the executor's gathered evidence."""
    # the transcript ends with an AI message; models won't reliably generate
    # another AI turn on top of it, so close with an explicit human turn
    request = HumanMessage(
        content=f"Write the final answer to my question now: {state['user_input']}"
    )
    return {
        "messages": [
            _clean(
                synthesizer_model.invoke(
                    [SystemMessage(content=synthesizer_prompt)]
                    + state["messages"]
                    + [request]
                )
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def decline(state: GraphState):
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def fallback(state: GraphState):
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def end_node(state: GraphState, config: RunnableConfig):
    # keep clean question -> answer pairs in history: drop this turn's
    # scratch (tool calls, observations) so the context doesn't bloat
    # across turns; the final answer carries the numbers follow-ups need
    scratch = current_turn_scratch(state["messages"][:-1])
    result = {
        "llm_calls": 0,
        "messages": [RemoveMessage(id=i) for i in scratch],
    }

    # distill new user facts from this exchange (scanner model - cheap);
    # runs on every path so even declined turns can teach us the user's name
    updated = scan_model.invoke(
        [
            HumanMessage(
                content=profile_update_prompt.format(
                    profile=state.get("user_profile") or "(empty)",
                    question=state["user_input"],
                    answer=state["messages"][-1].content,
                )
            )
        ]
    ).content
    # keep only the bullet lines - small models wrap them in chatter
    profile = "\n".join(
        line.strip()
        for line in updated.splitlines()
        if line.strip().startswith("- ")
    )
    if profile and "NO_CHANGE" not in updated:
        os.makedirs("profiles", exist_ok=True)
        with open(profile_path(config), "w") as f:
            f.write(profile)
        result["user_profile"] = profile
    return result


builder = StateGraph(GraphState)
builder.add_node("start_node", start_node)
builder.add_node("router_node", router_node)
builder.add_node("router_tools_node", router_tools_node)
builder.add_node("planner", planner)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", ToolNode(tools=tools, name="tool_node"))
builder.add_node("synthesizer", synthesizer)
builder.add_node("decline", decline)
builder.add_node("fallback", fallback)
builder.add_node("end_node", end_node)

builder.add_edge(START, "start_node")
builder.add_edge("start_node", "router_node")
builder.add_conditional_edges(
    "router_node",
    route_after_router_llm,
    {"tools": "router_tools_node", "no_decision": "decline"},
)
builder.add_conditional_edges(
    "router_tools_node",
    route_after_router_tools,
    {
        "in_scope": "planner",
        "out_of_scope": "decline",
        "continue": "router_node",
    },
)
builder.add_edge("decline", "end_node")
builder.add_edge("planner", "llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_use_tool,
    {"tool": "tool_node", "done": "synthesizer", "fallback": "fallback"},
)
builder.add_edge("tool_node", "llm_call")
builder.add_edge("synthesizer", "end_node")
builder.add_edge("fallback", "end_node")
builder.add_edge("end_node", END)

graph = builder.compile()
