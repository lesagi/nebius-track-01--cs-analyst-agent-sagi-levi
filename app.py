"""Streamlit chat UI for the customer service data analyst agent.

Same agent as main.py: reasoning steps (router, plan, tool calls and
observations) stream into an expandable status box, then the final answer.
The sidebar session ID switches/resumes conversations (SQLite checkpoints).

Run: uv run streamlit run app.py
"""

import sqlite3

import streamlit as st
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from load_data import ensure_db
from src.graph import builder

OBSERVATION_LIMIT = 2000  # same display cap as main.py


@st.cache_resource
def get_graph():
    ensure_db()
    connection = sqlite3.connect(
        "checkpoints.sqlite", check_same_thread=False
    )
    return builder.compile(checkpointer=SqliteSaver(connection))


def render_steps(update: dict) -> str | None:
    """Render one node update's reasoning; return final answer if present."""
    if update.get("query_type"):
        st.markdown(f"**router** → {update['query_type']}")
    if update.get("plan"):
        st.markdown(
            "**plan**\n"
            + "\n".join(
                f"{i}. {step}" for i, step in enumerate(update["plan"], 1)
            )
        )
    answer = None
    for message in update.get("messages") or []:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                st.markdown(f"**tool call** `{call['name']}` {call['args']}")
        elif isinstance(message, ToolMessage):
            content = str(message.content)
            if len(content) > OBSERVATION_LIMIT:
                content = (
                    content[:OBSERVATION_LIMIT]
                    + "\n... [truncated for display]"
                )
            st.code(content)
        elif isinstance(message, AIMessage) and message.content:
            answer = message.content
    return answer


st.set_page_config(page_title="CS Data Analyst Agent", page_icon="📊")
st.title("Customer Service Data Analyst Agent")

with st.sidebar:
    session = st.text_input("Session ID", "default")
    user = st.text_input("User ID", "default")

graph = get_graph()
config = {"configurable": {"thread_id": session, "user_id": user}}

# Replay the persisted conversation for this session (survives restarts).
state = graph.get_state(config)
for message in (state.values or {}).get("messages", []):
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        st.chat_message("assistant").write(message.content)

if question := st.chat_input("Ask about the Bitext dataset"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        answer = None
        with st.status("Thinking...", expanded=True) as status:
            stream = graph.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
                stream_mode="updates",
            )
            for chunk in stream:
                for update in chunk.values():
                    if update:
                        answer = render_steps(update) or answer
            status.update(label="Reasoning", state="complete", expanded=False)
        st.write((answer or "").strip() or "I could not produce an answer.")
