"""FastMCP server exposing the data analyst's tools. Run: uv run python mcp_server.py

Serves over stdio - point an MCP client at this script (see README).
"""

from fastmcp import FastMCP

from src.tools import (
    aggregate_db,
    calculate,
    get_dataset_description_tool,
    query_db,
    read_texts,
)

mcp = FastMCP("cs-data-analyst")

# The langchain tools wrap plain functions (.func) with full type hints and
# docstrings - FastMCP derives the MCP schemas from those directly.
# ponytail: scan_texts is not exposed - it needs a running Ollama and takes
# minutes, a poor fit for MCP client timeouts.
for lc_tool in [
    get_dataset_description_tool,
    query_db,
    aggregate_db,
    calculate,
    read_texts,
]:
    mcp.tool(lc_tool.func, name=lc_tool.name, description=lc_tool.description)

if __name__ == "__main__":
    mcp.run()
