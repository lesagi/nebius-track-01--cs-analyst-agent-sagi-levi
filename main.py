"""Interactive CLI for the customer service data analyst agent.

Prints the agent's reasoning steps (tool calls and observations) as they
happen, then the final answer. Conversations persist per --session across
restarts (SQLite checkpoints).
"""

import argparse
import sqlite3

from langchain.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from load_data import ensure_db
from src.graph import builder

BANNER = """Customer Service Data Analyst Agent
Ask questions about the Bitext customer service dataset.
Type 'exit' (or Ctrl-D) to leave."""


OBSERVATION_LIMIT = 2000  # generous - a full query_db result is ~12K chars


def print_update(update: dict) -> str | None:
    """Print the reasoning inside one node update; return final answer text if present."""
    if not update:
        return None
    if update.get("query_type"):
        print(
            f"\n[router] classified the question as: {update['query_type']}"
        )
    if update.get("plan"):
        print("\n[plan]")
        for i, step in enumerate(update["plan"], 1):
            print(f"  {i}. {step}")
    answer = None
    for message in update.get("messages") or []:
        if isinstance(message, AIMessage) and message.tool_calls:
            print(message.pretty_repr())
        elif isinstance(message, ToolMessage):
            content = str(message.content)
            if len(content) > OBSERVATION_LIMIT:
                message.content = (
                    content[:OBSERVATION_LIMIT]
                    + "\n... [truncated for display]"
                )
            print(message.pretty_repr())
        elif isinstance(message, AIMessage) and message.content:
            answer = message.content
    return answer


def ask(graph, question: str, config: dict) -> None:
    answer = None
    stream = graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    )
    for chunk in stream:
        for update in chunk.values():
            answer = print_update(update) or answer
    answer = (answer or "").strip()
    print(f"\nagent> {answer or 'I could not produce an answer.'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default="default",
        help="Session id - the same id restores the same conversation, even after a restart.",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="User id - the profile of distilled facts (profiles/<user>.md) persists across sessions.",
    )
    args = parser.parse_args()

    ensure_db()  # first run: build bitext.sqlite from HuggingFace
    connection = sqlite3.connect(
        "checkpoints.sqlite", check_same_thread=False
    )
    graph = builder.compile(checkpointer=SqliteSaver(connection))
    config = {
        "configurable": {"thread_id": args.session, "user_id": args.user}
    }

    print(BANNER)
    print(f"(session: {args.session}, user: {args.user})")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(graph, question, config)
    print("bye")


if __name__ == "__main__":
    main()
