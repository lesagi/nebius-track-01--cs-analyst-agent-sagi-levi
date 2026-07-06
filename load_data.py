"""Download the Bitext dataset from HuggingFace into a local SQLite database.

Runs automatically on the agent's first start; rerun manually to rebuild:
uv run python load_data.py
"""

import os
import sqlite3

import pandas as pd

from src.common import CONFIG, DB_PATH


def main() -> None:
    source = os.getenv("BITEXT_DATASET_PATH") or CONFIG["dataset"]["hf_csv"]
    print(f"loading {source} ...")
    df = pd.read_csv(source)
    with sqlite3.connect(DB_PATH) as connection:
        df.to_sql(
            "interactions", connection, index=False, if_exists="replace"
        )
    print(f"{len(df)} rows -> {DB_PATH} (table: interactions)")


def ensure_db() -> None:
    """Build the database only if it doesn't exist yet."""
    if not os.path.exists(DB_PATH):
        main()


if __name__ == "__main__":
    main()
