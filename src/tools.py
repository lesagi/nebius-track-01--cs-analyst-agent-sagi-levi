"""The analyst's tools: dataset filtering, aggregation, arithmetic and text analysis."""

import ast
import json
import operator
import os
import random
import sqlite3
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import CONTEXT_CHARS, DB_PATH, get_model
from .db_attributes import (
    CATEGORIES_AND_INTENTS,
    FLAG_DESCRIPTIONS,
    Category,
    Flags,
    IntentType,
)
from .prompts import DATA_SEMANTICS, scan_chunk_prompt

# Scanner model for scan_texts chunk labeling: mechanical work a small fast
# model handles well (see config.yaml models.scanner).
scan_model = get_model("scanner")

# Budgets derived from the configured model's context window, never from a
# specific model. Hard caps because huge prompts stall local inference
# regardless of the declared window.
READ_BUDGET_CHARS = min(CONTEXT_CHARS // 4, 60_000)
CHUNK_CHARS = min(8_000, CONTEXT_CHARS // 8)
# ponytail: scan cap is latency-driven (~17s per 2K-char chunk on this box,
# sequential - OLLAMA_NUM_PARALLEL=1); raise it when chunks can run in parallel
SCAN_BUDGET_CHARS = min(CONTEXT_CHARS // 2, 30_000)


class StrictArgs(BaseModel):
    """Absorb harmless LLM arg slop (scalar for list), reject dangerous slop
    (unknown args would otherwise be silently dropped -> unfiltered queries)."""

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "flags",
        "categories",
        "intents",
        "labels",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def _listify(cls, value):
        if not isinstance(value, str):
            return value
        # some hosted models emit list args as JSON-encoded strings
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
        return [value]


class GetRows(StrictArgs):
    flags: Annotated[list[str], list[Flags]] | None = Field(
        description="The flags to filter the rows by.",
        default=None,
    )
    categories: Annotated[list[str], list[Category]] | None = Field(
        description="The categories to filter the rows by.",
        default=None,
    )
    intents: Annotated[list[str], list[IntentType]] | None = Field(
        description="The intents to filter the rows by.",
        default=None,
    )
    limit: int = Field(
        description="The maximum number of rows to return (at most 20).",
        default=10,
    )
    sample_seed: int = Field(
        description="Sampling seed. Use a different seed to get different examples than the ones already shown.",
        default=0,
    )


class AggregateRows(StrictArgs):
    group_by: Literal["category", "intent", "flags"] = Field(
        description="The column to group the row counts by.",
    )
    flags: Annotated[list[str], list[Flags]] | None = Field(
        description="The flags to filter the rows by.",
        default=None,
    )
    categories: Annotated[list[str], list[Category]] | None = Field(
        description="The categories to filter the rows by.",
        default=None,
    )
    intents: Annotated[list[str], list[IntentType]] | None = Field(
        description="The intents to filter the rows by.",
        default=None,
    )


class ReadTexts(StrictArgs):
    column: Literal["instruction", "response"] = Field(
        description="The text column to read: customer requests (instruction) or agent replies (response).",
    )
    flags: Annotated[list[str], list[Flags]] | None = Field(
        description="The flags to filter the rows by.",
        default=None,
    )
    categories: Annotated[list[str], list[Category]] | None = Field(
        description="The categories to filter the rows by.",
        default=None,
    )
    intents: Annotated[list[str], list[IntentType]] | None = Field(
        description="The intents to filter the rows by.",
        default=None,
    )


class ScanTexts(ReadTexts):
    task: str = Field(
        description="The labeling instruction. Define each label precisely, including what does NOT qualify, e.g. 'label product_dissatisfaction only if the text explicitly complains about the product itself'.",
    )
    labels: list[str] = Field(
        description="The allowed labels, e.g. ['dissatisfaction', 'affordability', 'no_reason_given'].",
    )


class CalculateInput(StrictArgs):
    expression: str = Field(
        description="An arithmetic expression using numbers and + - * / % ** parentheses only, e.g. '2992 + 1997'"
    )


class TextLabel(BaseModel):
    i: int = Field(description="The index of the text")
    label: str = Field(description="The label assigned to that text")


class TextLabels(BaseModel):
    labels: list[TextLabel] = Field(
        description="One entry per text, covering every index"
    )


def get_flags_mask(df: pd.DataFrame, flags: list[str]) -> np.ndarray:
    masks = [
        df["flags"].str.contains(flag, case=False, na=False) for flag in flags
    ]
    return np.any(masks, axis=0)


def get_column_mask(
    df: pd.DataFrame, column: str, values: list[str]
) -> np.ndarray:
    return df[column].str.lower().isin([v.lower() for v in values]).to_numpy()


def load_df() -> pd.DataFrame:
    """Load the dataset from the local SQLite database (built by load_data.py)."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"{DB_PATH} not found - build it first: uv run python load_data.py"
        )
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("SELECT * FROM interactions", connection)


def filter_df(
    flags: list[str] | None = None,
    categories: list[str] | None = None,
    intents: list[str] | None = None,
) -> pd.DataFrame:
    """Load the dataset and filter it: values within a filter are OR-ed, filters are AND-ed."""
    df = load_df()
    masks = []
    if flags:
        masks.append(get_flags_mask(df, flags))
    if categories:
        masks.append(get_column_mask(df, "category", categories))
    if intents:
        masks.append(get_column_mask(df, "intent", intents))
    if masks:
        df = df[np.all(masks, axis=0)]
    return df


@tool()
def get_dataset_description_tool():
    """
    Retrieve the description of the dataset.
    Args:
        None
    Returns:
        string: A string containing the description of the dataset.
    """
    headline = f"The dataset is a collection of customer service interactions that have been labeled with categories and intents.\n{DATA_SEMANTICS}"

    categories_and_intents = "# Categories and Intents\n\n"
    for item in CATEGORIES_AND_INTENTS:
        categories_and_intents += f"## Category: {item['category'].value}\nIntents: {', '.join(map(lambda x: x.value, item['intent_types']))}\n\n"
    categories_and_intents = categories_and_intents.strip()

    flags = "The flags are: \n" + "\n".join(
        f"flag: '{flag.value}' - description: '{description}'"
        for flag, description in FLAG_DESCRIPTIONS.items()
    )
    flags = flags.strip()

    return f"{headline}\n\n{categories_and_intents}\n\n{flags}"


@tool(args_schema=GetRows)
def query_db(
    flags: list[str] | None = None,
    categories: list[str] | None = None,
    intents: list[str] | None = None,
    limit: int = 10,
    sample_seed: int = 0,
) -> str:
    """
    Get example rows matching the given filters. Values within a filter are
    OR-ed, different filters are AND-ed together.
    Use this to show examples; for counting or distributions use aggregate_db.
    When asked for more/different examples, pass a new sample_seed.
    """
    df = filter_df(flags, categories, intents)
    # ponytail: hard cap - a 100-row dump of full response texts stalls a local model for minutes
    n = min(limit, 20, len(df))
    # deterministic random sample - head() always showed the same first CSV rows
    return json.dumps(
        df.sample(n=n, random_state=sample_seed).to_dict(orient="records")
    )


@tool(args_schema=AggregateRows)
def aggregate_db(
    group_by: Literal["category", "intent", "flags"],
    flags: list[str] | None = None,
    categories: list[str] | None = None,
    intents: list[str] | None = None,
) -> str:
    """
    Count the rows matching the given filters, grouped by the given column.
    Returns the total count, per-group counts (largest first) and per-group
    percentages, all precomputed - report them as-is, never recompute them.
    Use this for "how many" and distribution questions.
    """
    df = filter_df(flags, categories, intents)
    # ponytail: flags grouped by their raw combo string (e.g. "BQZ"); per-letter explode if ever needed
    counts = df.groupby(group_by).size().sort_values(ascending=False)
    total = int(counts.sum())
    percentages = {
        k: round(v * 100 / total, 1) for k, v in counts.items() if total
    }
    return json.dumps(
        {
            "total": total,
            "counts": counts.to_dict(),
            "percentages": percentages,
        }
    )


_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _calc_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](
            _calc_eval(node.left), _calc_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@tool(args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """
    Evaluate an arithmetic expression of plain numbers, e.g. '2992 + 1997'.
    Use this for any arithmetic on numbers you got from other tools -
    never compute sums, differences or percentages yourself.
    """
    return str(_calc_eval(ast.parse(expression, mode="eval").body))


@tool(args_schema=ReadTexts)
def read_texts(
    column: Literal["instruction", "response"],
    flags: list[str] | None = None,
    categories: list[str] | None = None,
    intents: list[str] | None = None,
) -> str:
    """
    Read the deduplicated texts of a column for the matching rows as one
    block, each line prefixed with how many rows share that text (x12 ...).
    Use this to summarize or characterize what texts contain.
    Not for counting per row - use scan_texts for that.
    """
    counts = filter_df(flags, categories, intents)[column].value_counts()
    total_rows = int(counts.sum())
    # +8 approximates the "xN " prefixes so the estimate errs toward shuffling
    truncating = (
        sum(map(len, counts.index)) + 8 * len(counts) > READ_BUDGET_CHARS
    )
    if truncating:
        # random order so the slice isn't just the most duplicated
        # templates; each line still carries its xN row count
        counts = counts.sample(frac=1, random_state=0)
    lines, chars, rows_covered = [], 0, 0
    for text, n in counts.items():
        line = f"x{n} {text}"
        if chars + len(line) > READ_BUDGET_CHARS:
            break
        lines.append(line)
        chars += len(line) + 1
        rows_covered += int(n)
    header = (
        f"{len(lines)} unique texts covering {rows_covered}/{total_rows} rows"
    )
    if truncating:
        header += " (TRUNCATED at budget - representative random sample)"
    return header + "\n" + "\n".join(lines)


def _label_texts(
    task: str, labels: list[str], texts: list[str], retry: bool = True
) -> dict[int, str]:
    """One structured-output call labeling a numbered chunk; retries missing indices once."""
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(texts))
    try:
        out = scan_model.with_structured_output(TextLabels).invoke(
            [
                HumanMessage(
                    content=scan_chunk_prompt.format(
                        task=task,
                        labels=", ".join(labels),
                        last=len(texts) - 1,
                        texts=numbered,
                    )
                )
            ]
        )
        entries = out.labels
    except (
        Exception
    ):  # malformed structured output - the retry pass gets another shot
        entries = []
    assigned = {
        entry.i: entry.label for entry in entries if 0 <= entry.i < len(texts)
    }
    missing = [i for i in range(len(texts)) if i not in assigned]
    if missing and retry:
        recovered = _label_texts(
            task, labels, [texts[i] for i in missing], retry=False
        )
        for j, label in recovered.items():
            assigned[missing[j]] = label
    return assigned


def merge_scan_labels(
    labeled: list[tuple[str, int, str | None]], allowed: list[str]
) -> dict:
    """Deterministic reduce of (text, multiplicity, label) triples: weighted counts, percentages, examples. Pure - no LLM."""
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for text, n, label in labeled:
        if label is None:
            label = "unlabeled"
        elif label not in allowed:
            label = "other"
        counts[label] = counts.get(label, 0) + n
        if len(examples.setdefault(label, [])) < 3:
            examples[label].append(f"x{n} {text}")
    counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
    total = sum(counts.values())
    percentages = {
        k: round(v * 100 / total, 1) for k, v in counts.items() if total
    }
    return {
        "counts": counts,
        "percentages": percentages,
        "examples": examples,
    }


@tool(args_schema=ScanTexts)
def scan_texts(
    column: Literal["instruction", "response"],
    task: str,
    labels: list[str],
    flags: list[str] | None = None,
    categories: list[str] | None = None,
    intents: list[str] | None = None,
) -> str:
    """
    Classify the text of every matching row against the given task and
    labels, and count the rows per label. Use this when a question requires
    reading free text row by row (e.g. counting texts that express a
    specific reason or sentiment). Counts, percentages and examples come
    back precomputed - report them as-is.
    """
    value_counts = filter_df(flags, categories, intents)[
        column
    ].value_counts()
    total_rows = int(value_counts.sum())
    items = [(str(text), int(n)) for text, n in value_counts.items()]

    coverage = "full"
    if sum(len(text) for text, _ in items) > SCAN_BUDGET_CHARS:
        # ponytail: sample unique texts to budget; good enough as an estimate, exact census on demand later
        random.Random(0).shuffle(items)
        picked, chars = [], 0
        for text, n in items:
            if chars + len(text) > SCAN_BUDGET_CHARS:
                break
            picked.append((text, n))
            chars += len(text)
        items = picked
        coverage = "sampled - treat counts as an estimate"

    chunks: list[list[tuple[str, int]]] = [[]]
    chars = 0
    for text, n in items:
        if chunks[-1] and chars + len(text) > CHUNK_CHARS:
            chunks.append([])
            chars = 0
        chunks[-1].append((text, n))
        chars += len(text)

    # ponytail: chunks run sequentially - asyncio.gather them when on a hosted API
    labeled: list[tuple[str, int, str | None]] = []
    for chunk in chunks:
        assigned = _label_texts(task, labels, [text for text, _ in chunk])
        labeled += [
            (text, n, assigned.get(i)) for i, (text, n) in enumerate(chunk)
        ]

    return json.dumps(
        {
            "total_rows": total_rows,
            "scanned_rows": sum(n for _, n, _ in labeled),
            "coverage": coverage,
            **merge_scan_labels(labeled, labels),
        }
    )


tools = [
    query_db,
    aggregate_db,
    calculate,
    read_texts,
    scan_texts,
    get_dataset_description_tool,
]
