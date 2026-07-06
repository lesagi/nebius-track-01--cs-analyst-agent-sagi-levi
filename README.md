# Customer Service Data Analyst Agent

A LangGraph agent that answers questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — structured ("How many refund requests did we get?"), open-ended ("Summarize the FEEDBACK category"), and follow-ups across restarts ("What is the total count of the last two?"). Out-of-scope questions are politely declined. Tools are also exposed over MCP.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a [Nebius Token Factory](https://tokenfactory.nebius.com) API key.

```bash
git clone <this repo> && cd 03-cs-data-analyst-agent
uv sync
cp .env.example .env    # put your NEBIUS_API_KEY in it
```

On first run the agent downloads the dataset from HuggingFace into a local SQLite database (`bitext.sqlite`) that the tools query; subsequent runs skip the download. Already have the CSV? Point `BITEXT_DATASET_PATH` at it. To (re)build manually: `uv run python load_data.py`.

> **Configuration:** everything non-secret (dataset source, provider, per-role models, limits) lives in `config.yaml`. Secrets (API keys) go in `.env` — copy `.env.example`. Models are instantiated in one place (`get_model(role)` in `src/common.py`), so no other code knows which provider is active.

### Choosing a provider

`config.yaml` holds one models block per provider (base endpoint + one model per role); `provider:` selects the active block, `MODEL_PROVIDER` env overrides it:

- **nebius** (default; [Nebius Token Factory](https://tokenfactory.nebius.com), OpenAI-compatible API): needs only `NEBIUS_API_KEY` in `.env`. The configured models were picked by a measured bake-off: `gpt-oss-120b` for the sophisticated roles, `Qwen3-30B-A3B-Instruct-2507` for the mechanical ones (see [Model choice](#model-choice)).
- **ollama** (local alternative, same role split): requires a running [Ollama](https://ollama.com/) server with `ollama pull qwen3.5:9b` and `ollama pull llama3.1:8b-instruct-q4_K_M`, then `MODEL_PROVIDER=ollama` (or flip `provider:` in `config.yaml`).

## Run the CLI

```bash
uv run cs-agent --session my_session --user sagi
# (equivalent: uv run python main.py ...)
```

- The agent prints its **reasoning steps** as they happen — router classification, plan, every tool call with arguments, every observation — then the final answer.
- `--session`: the same id restores the same conversation, **including after a restart** (LangGraph checkpoints in `checkpoints.sqlite`). Follow-ups work: *"How many complaints did we get?" → "What about refunds?" → "What is the total count of the last two?"*
- `--user`: a persistent profile of distilled facts (`profiles/<user>.md`) — name, preferences, topics you ask about — separate from conversation history. Ask *"What do you remember about me?"*.

Example:

```
you> How many refund requests did we get?
[router] classified the question as: structured
[plan]
  1. aggregate_db(column='category', filters={'category': 'REFUND'})
================================== Ai Message ==================================
Tool Calls:
  aggregate_db  Args: group_by: category, categories: ['REFUND']
================================= Tool Message =================================
{"total": 2992, "counts": {"REFUND": 2992}, "percentages": {"REFUND": 100.0}}

agent> We received **2,992 refund requests** in total.
```

## Run the Streamlit UI (Bonus A)

```bash
uv run streamlit run app.py
```

Same agent in a chat interface: reasoning steps stream into an expandable "Thinking..." box above each answer, and the sidebar **Session ID** / **User ID** inputs switch or resume conversations (changing the session replays its persisted history).

## MCP server

```bash
uv run python mcp_server.py    # serves over stdio
```

Exposes 5 tools: `get_dataset_description_tool`, `query_db`, `aggregate_db`, `calculate`, `read_texts`.

**Connect a client** — e.g. Claude Desktop / any MCP client config:

```json
{
  "mcpServers": {
    "cs-data-analyst": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/this/repo"
    }
  }
}
```

Or call a tool programmatically:

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_server.py") as client:
        result = await client.call_tool(
            "aggregate_db", {"group_by": "intent", "categories": ["REFUND"]}
        )
        print(result.content[0].text)
        # {"total": 2992, "counts": {"track_refund": 998, ...}, "percentages": {...}}

asyncio.run(main())
```

## Architecture

One flat LangGraph (`src/graph.py`, visualizable with `uv run langgraph dev`):

```
START → start_node → router_node ⇄ router_tools_node
                          │ out_of_scope        │ in_scope
                          ▼                     ▼
                       decline               planner → llm_call ⇄ tool_node
                          │                     done ↓        ↓ max iterations
                          └──→ end_node ←─ synthesizer     fallback
                                  ↓                            │
                                 END ←─────────────────────────┘
```

- **Router** — classifies each question `structured` / `unstructured` / `out_of_scope` before any analysis; out-of-scope gets a polite decline, never an answer from general knowledge.
- **Planner** — one structured-output call producing the fewest concrete steps (tool + arguments). It knows the dataset's semantics (what's labeled vs. free text) and structure (categories/intents), so user terms ground against the real schema.
- **Executor** — a ReAct loop over the tools, guided by the plan. Numbers are never computed by the model: aggregations return counts/percentages precomputed, and cross-call arithmetic goes through the `calculate` tool. Tool schemas absorb harmless arg slop (a scalar where a list belongs) and reject unknown arguments instead of silently ignoring them.
- **Synthesizer** — writes the final user-facing answer from the executor's gathered evidence.
- **Memory** — conversations persist per session via `SqliteSaver`; each finished turn is pruned to a clean question→answer pair so context doesn't bloat. The user profile is distilled after every turn by the scanner model.
- **Guardrails** — `MAX_LLM_CALLS` caps the loop with a graceful fallback message; context budgets derive from the configured model's context window at runtime.

### Tools (Pydantic schemas in `src/tools.py`)

| Tool | Purpose |
|------|---------|
| `get_dataset_description_tool` | Dataset semantics, categories/intents, flags |
| `query_db` | Sample example rows by flags/categories/intents (seeded random, capped) |
| `aggregate_db` | Grouped counts + percentages, precomputed — no model math |
| `calculate` | Safe AST arithmetic for combining numbers across calls |
| `read_texts` | Deduplicated texts of a slice as one block, for summarizing |
| `scan_texts` | Row-by-row LLM labeling of texts against user-defined criteria, with deterministic reduce (CLI only — excluded from MCP due to latency) |

### Model choice

One model per role per provider (`config.yaml` → `models`), instantiated through a single factory (`get_model(role)` in `src/common.py`). The role split was derived by measurement on the local block (shown below); the default Nebius block follows the same split and was picked by its own bake-off (next subsection):

| Role | Model | Why |
|------|-------|-----|
| Router | `llama3.1:8b-instruct` | Classification needs little context: measured 4/4 correct at 0.5–0.9s vs 3–10s on the thinking model (~10×) |
| Planner | `qwen3.5:9b` (thinking) | One call per question; plan quality and schema grounding matter more than latency |
| Executor | `qwen3.5:9b` (thinking) | We tried the small model here: it produced wrong counts and malformed tool args once real conversation history piled up — reverted |
| Synthesizer | `qwen3.5:9b` (thinking) | The user-facing answer, especially summaries, is where sophistication shows |
| Scanner (`scan_texts`) | `llama3.1:8b-instruct` | Batch-labeling short texts is mechanical; thinking wastes minutes (17s vs 56s per 40-text chunk) at equal label quality |

Smaller candidates were tested and rejected: `llama3.2:3b` passes simple router probes but hallucinates a data answer instead of classifying once a tool result enters its context, and over-labels wildly as scanner (50/80 false positives vs 11/80) with no latency gain; `llama3.2:1b` never emits a well-formed tool call; `qwen3.5:4b` spirals in its thinking phase under forced structured output (no plan after 7 minutes vs seconds on 9B).

Temperature is pinned to 0 throughout: routing and planning must not flip-flop between runs.

#### Nebius models: tested and disqualified

The hosted block was picked by a bake-off, not by catalog reputation. Each candidate had to pass three gates:

1. **Planner probe** — 4 canonical questions through the planner with forced structured output; plans must parse, name real tools with sensible arguments, and the deliberately ambiguous question ("cancellations due to product dissatisfaction" — not a labeled field) must plan a text scan and acknowledge the limitation rather than invent a count.
2. **Structured 3-turn conversation** — "how many complaints?" → "and refund requests?" → "total of the last two?" through the full CLI. The answers are verifiable (1,000 / 2,992 / 3,992) and the last turn tests conversation memory plus tool-based arithmetic.
3. **Open-ended summary** — "Summarize the feedback we received from customers": must read the actual texts and produce a summary grounded in them (the correct insight here is that the FEEDBACK rows are procedural "how do I complain?" requests, not opinions).

| Model | Result | Why |
|-------|--------|-----|
| `openai/gpt-oss-120b` | **Winner — sophisticated roles** | Passed all three gates at default settings; cheapest and fastest of the full passers ($0.15/$0.60 per 1M, 5B active params) |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | **Winner — router + scanner** | Router 4/4 probes at 0.8–3.1s; scanned 80 texts in 8s with 0 false positives (the local 8B had 11) |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Passed, not picked | Full passer, but strictly dominated by `gpt-oss-120b` on price and speed; kept as the documented fallback |
| `meta-llama/Llama-3.3-70B-Instruct` | Disqualified | Failed the 3-turn conversation twice: it compresses answers to a bare number ("1000"), and that degenerate history destabilized the next turn's routing; a second run hit a planner timeout |
| `Qwen/Qwen3.5-397B-A17B` | Rejected | A hybrid-thinking flagship: at temperature 0 it falls into an infinite reasoning loop ("I will write: 3992. Wait, I'll write: 3992. Okay." — forever), surfacing as client timeouts. Making it work required model-specific template hacks — the wrong trade for agent plumbing |

Thinking/hybrid models are avoided across the board: the pipeline pins temperature to 0, which is exactly the regime where they loop, and at ~5 LLM calls per user turn their 10–100× output-token overhead multiplies cost for no measured quality gain on this workload.

## Tests

```bash
uv run python test_tools.py
```

Self-checks for the filter logic, sampling, aggregation math, the calculator (including rejection of non-arithmetic input), text-tool coverage headers, and enum↔dataset consistency.

## Known limitations: breaking the guardrails

Exploring the agent adversarially turned up ways around the out-of-scope guardrail. Asked directly, an out-of-scope question is declined as designed:

![Direct out-of-scope question is declined](docs/decline-direct.png)

But the same question smuggled inside a statement about the user slips through. Preference statements are legitimately in-scope (they feed the user profile), and once the turn is classified in-scope, nothing stops the analyst from answering the question embedded in it from general knowledge:

![A preference statement smuggles an out-of-scope question through](docs/bypass-preference-smuggling.png)

Two distinct gaps, both downstream of a *correct* router decision:

1. **Answering instead of acknowledging** — the "never answer from general knowledge" rule binds only the router's decline path; the executor and synthesizer trust anything classified in-scope, including content embedded in a statement they should merely acknowledge.
2. **The profile is an unvalidated side-channel** — whatever the user declares is distilled into the persistent profile (note "Current preferred category: Product" above), survives across sessions, and is injected into every future prompt — effectively stored prompt injection.

Fix directions: the acknowledgment path should acknowledge only, never execute or answer content embedded in a statement; and profile updates should be validated against the small set of fact types the profile is meant to hold.

## Development notes & AI assistance

This project started before I joined the supervised track, so the original version was built as a playground rather than for submission. It had a more complex multi-agent architecture — an orchestrator agent delegating to router, analyst and synthesizer sub-agents — driven through LangGraph Studio rather than a CLI, with the emphasis on making it resilient rather than clean or performant (so parts of it may still read as over-engineered). Tool design and session support got less attention at that stage.

For submission, I used Claude (Anthropic's coding agent) to adapt that original work to the assignment's requirements: consolidating the sub-agent architecture into a single flat graph, adding the interactive CLI with streamed reasoning output (printing the intermediate steps was the sticking point), tightening the tools, sessions and profile memory, and general refactoring and cleanup.

Part of that adaptation was forced by the move to Nebius Token Factory models. The original build ran on local Ollama `qwen3.5` models, which honestly worked better out of the box for this pipeline; the hosted models needed real adjustments — the model bake-off documented above, switching the planner's structured output to `function_calling` (the server doesn't enforce `json_schema` for `gpt-oss-120b`), and enforcing the router's tool use with `tool_choice` instead of prompt instructions.

## Future improvements

- **Clarification loop in the planner**: when the planner judges that no plan over the available tools answers the question *precisely*, it should ask the user a refining question instead of guessing — and keep refining through conversation until the request is either executable with the tools we have, or identified as infeasible (and say so). LangGraph's `interrupt()` makes this a natural extension of the existing planner node.
- **Tautology guardrail in `aggregate_db`**: when asked to group by a column the filters already pin to a single value (e.g. `group_by=category` filtered to `ORDER`), append a hint pointing at the informative breakdown (`group_by=intent`) so the model self-corrects in one step.
- **Parallel `scan_texts` fan-out**: chunks are labeled sequentially because the local Ollama serves one request at a time; with `OLLAMA_NUM_PARALLEL` or a hosted API, `asyncio.gather` over chunks cuts a full-census scan from minutes to seconds and lets the scan budget grow.
- **Verify-pass for scan labels**: the small scanner over-labels subtle semantics (e.g. "problem with cancelling" ≠ product complaint); a second cheap pass verifying only the positive labels would tighten precision.
- **Group-sampling in `query_db`**: "show me one example of each category" currently costs a call per category; a per-group sample parameter would collapse it to one.
