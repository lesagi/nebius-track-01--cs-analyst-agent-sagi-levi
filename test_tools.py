"""Self-checks for the query tools. Run: uv run python test_tools.py"""

import json

import load_data

load_data.ensure_db()

from src.tools import (  # noqa: E402
    aggregate_db,
    calculate,
    filter_df,
    load_df,
    merge_scan_labels,
    query_db,
    read_texts,
)


def test_filters_combine():
    df = filter_df(categories=["REFUND"], intents=["track_refund"])
    assert len(df) > 0
    assert set(df["category"]) == {"REFUND"}
    assert set(df["intent"]) == {"track_refund"}


def test_multi_value_is_or():
    df = filter_df(categories=["REFUND", "ORDER"])
    assert set(df["category"]) == {"REFUND", "ORDER"}


def test_query_db_limit():
    rows = json.loads(query_db.invoke({"categories": ["REFUND"]}))
    assert len(rows) == 10
    assert all(row["category"] == "REFUND" for row in rows)


def test_arg_slop_handling():
    # scalar where a list belongs is absorbed
    rows = json.loads(query_db.invoke({"categories": "REFUND", "limit": 2}))
    assert all(row["category"] == "REFUND" for row in rows)
    # JSON-stringified list (seen from hosted models) is parsed, not wrapped
    rows = json.loads(
        query_db.invoke({"categories": '["REFUND"]', "limit": 2})
    )
    assert all(row["category"] == "REFUND" for row in rows)
    # invented args are rejected instead of silently ignored
    try:
        query_db.invoke({"filters": "category=REFUND"})
        raise AssertionError("unknown arg should have been rejected")
    except Exception:
        pass


def test_query_db_samples_are_representative():
    rows = json.loads(query_db.invoke({"limit": 10}))
    # head() used to return 10x cancel_order; a random sample spans categories
    assert len({row["category"] for row in rows}) > 1


def test_aggregate_db():
    out = json.loads(
        aggregate_db.invoke({"group_by": "intent", "categories": ["ACCOUNT"]})
    )
    assert out["total"] > 0
    assert out["total"] == sum(out["counts"].values())
    counts = list(out["counts"].values())
    assert counts == sorted(counts, reverse=True)
    assert abs(sum(out["percentages"].values()) - 100) < 1


def test_calculate():
    assert calculate.invoke({"expression": "2992 + 1997"}) == "4989"
    assert calculate.invoke({"expression": "100 * 3 / 6"}) == "50.0"
    try:
        calculate.invoke({"expression": "__import__('os').system('id')"})
        raise AssertionError("should have rejected non-arithmetic input")
    except Exception:
        pass


def test_read_texts():
    out = read_texts.invoke(
        {"column": "instruction", "intents": ["cancel_order"]}
    )
    header = out.split("\n")[0]
    assert "493 unique texts covering 998/998 rows" in header
    assert "TRUNCATED" not in header

    out = read_texts.invoke(
        {"column": "response", "categories": ["FEEDBACK"]}
    )
    assert "TRUNCATED" in out.split("\n")[0]


def test_merge_scan_labels():
    result = merge_scan_labels(
        [("a", 5, "x"), ("b", 3, "bogus"), ("c", 2, None), ("d", 1, "x")],
        allowed=["x", "y"],
    )
    assert result["counts"] == {"x": 6, "other": 3, "unlabeled": 2}
    assert abs(sum(result["percentages"].values()) - 100) < 1
    assert result["examples"]["x"] == ["x5 a", "x1 d"]


def test_enums_match_dataset():
    from src.db_attributes import Category, IntentType

    df = load_df()
    assert set(df["category"]) == {c.value for c in Category}
    assert set(df["intent"]) == {i.value for i in IntentType}


if __name__ == "__main__":
    test_filters_combine()
    test_multi_value_is_or()
    test_query_db_limit()
    test_arg_slop_handling()
    test_query_db_samples_are_representative()
    test_aggregate_db()
    test_calculate()
    test_read_texts()
    test_merge_scan_labels()
    test_enums_match_dataset()
    print("ok")
