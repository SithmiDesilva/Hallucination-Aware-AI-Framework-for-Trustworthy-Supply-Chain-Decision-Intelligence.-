"""Tests for the dataset-wide questions (src/bench/dataset_questions.py).

Helpers; the shipped file (complete, unique, dataset scope, no answer leaking into its own question, in sync
with the generator); the user's own example answers; and brute-force checks straight from the raw CSVs that use
none of the generator's grouping code (boolean masks and explicit column picks). Skipped without raw data.
"""
import re

import numpy as np
import pandas as pd
import pytest

from src.bench import dataset_questions as dq
from src.bench.dataset_questions import (
    OUT, PERIOD_DAYS, build, disagreements, fmt, m5_check_facts, m5_facts, mismatches, month_label, pct, same, tidy, unique_top,
)
from src.cards.schemas import Question, read_jsonl
from src.config import RAW

FILE = OUT / "b0_dataset_wide_v1.jsonl"
needs_data = pytest.mark.skipif(
    not (FILE.exists() and (RAW / "m5" / "sales_train_evaluation.csv").exists() and (RAW / "dataco" / "DataCoSupplyChainDataset.csv").exists()),
    reason="dataset-wide file or raw data not available")


# ----------------------------------------------------------------------------- helpers
def test_formatting_helpers():
    assert fmt(11188180) == "11,188,180" and fmt(np.int64(5)) == "5" and pct(1, 4) == "25.0%"
    assert month_label("2012-07") == "July 2012"
    assert tidy("Most: CA_3, 1 units.\nLeast: X, 1 days .") == "Most: CA_3, 1 unit.\nLeast: X, 1 day."  # line breaks survive
    assert tidy("11 units, 1.1 units, 21 days") == "11 units, 1.1 units, 21 days"


def test_unique_top_rejects_ties():
    s = pd.Series({"a": 5, "b": 9, "c": 7})
    assert list(unique_top(s, 2).index) == ["b", "c"]
    with pytest.raises(AssertionError):
        unique_top(pd.Series({"a": 5, "b": 9, "c": 9}))  # a tied winner is ambiguous, so it must not become a question


def test_disagreements_catches_any_difference():
    a = {"x": pd.Series([1, 2]), "n": 3, "t": pd.DataFrame({"c": [1.0]})}
    assert disagreements(a, {k: v for k, v in a.items()}) == []
    assert disagreements(a, {**a, "x": pd.Series([1, 3])}) == ["x"]
    assert disagreements(a, {**a, "n": 4, "t": pd.DataFrame({"c": [2.0]})}) == ["n", "t"]
    assert disagreements(a, {**a, "extra": 1}) == ["extra"]
    assert same(1, 1) and not same(pd.Series([1]), pd.Series([2]))


def tiny_m5(n_days=70):
    rng = np.random.default_rng(5)
    rows = []
    for store, state in (("CA_1", "CA"), ("CA_2", "CA"), ("TX_1", "TX")):
        for cat, dept in (("FOODS", "FOODS_1"), ("FOODS", "FOODS_2"), ("HOBBIES", "HOBBIES_1")):
            for i in range(3):
                rows.append({"item_id": f"{dept}_{i:03d}", "dept_id": dept, "cat_id": cat, "store_id": store, "state_id": state,
                             **{f"d_{d}": int(v) for d, v in enumerate(rng.poisson(3, n_days), 1)}})
    dates = pd.date_range("2011-01-29", periods=n_days)
    cal = pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)], "date": dates.strftime("%Y-%m-%d"),
                        "event_name_1": [None] * n_days, "event_name_2": [None] * n_days, "snap_CA": [0] * n_days})
    return pd.DataFrame(rows), cal


def test_the_two_m5_methods_agree_and_a_changed_cell_is_caught(monkeypatch):
    monkeypatch.setattr(dq, "PERIOD_DAYS", 70)
    sales, cal = tiny_m5()
    full, check = dq.m5_facts(sales, cal), dq.m5_check_facts(sales, cal)
    assert set(check) <= set(full) and "store_month" in check and len(check["store_month"]) >= 2  # Feb and Mar are complete months
    assert dq.mismatches(full, check) == []
    altered = sales.copy()
    altered.loc[0, "d_30"] += 1  # one extra unit on one day of one series
    bad = dq.mismatches(full, dq.m5_check_facts(altered, cal))
    assert {"store_total", "state_total", "cat_total", "dept_total", "item_total", "daily"} <= set(bad)
    assert dq.mismatches({"a": 1}, {"b": 2}) == ["b"]


# ----------------------------------------------------------------------------- the shipped file
@pytest.fixture(scope="module")
def qs():
    return list(read_jsonl(FILE, Question))


@needs_data
def test_file_is_large_unique_and_dataset_scoped(qs):
    assert len(qs) >= 60
    assert len({q.qid for q in qs}) == len(qs) and len({q.question for q in qs}) == len(qs)
    assert all(q.scope == "dataset" and q.answerable and q.gold_card_ids == [] and q.gold_facts and q.split == "dev" for q in qs)
    assert {q.domain for q in qs} == {"inventory", "delivery"}
    fams = {q.template_id.split("/")[0] for q in qs}
    assert fams == {"Dataset-wide aggregation", "Dataset-wide ranking and comparison", "Dataset-wide temporal", "Dataset-wide item-level", "Dataset facts"}


@needs_data
def test_period_dependent_questions_state_their_period(qs):
    for q in qs:
        if q.qid.startswith("DW-M5") and "/" in q.template_id and q.template_id.split("/")[1] not in (
                "items_per_category", "counts", "stores_in_state", "stores_per_state", "series_count", "period"):
            assert "29 January 2011 to 24 April 2016" in q.question, q.qid
        if q.qid.startswith("DW-DC") and q.template_id.split("/")[1] != "period":
            assert "1 January 2015 to 31 January 2018" in q.question, q.qid


@needs_data
def test_no_answer_leaks_into_its_own_question(qs):
    for q in qs:
        for f in q.gold_facts:
            v = f.value
            # large numbers and long strings only: a candidate year or store named as an option is not a leak
            shown = {fmt(v), str(v)} if isinstance(v, int) and abs(v) >= 10_000 else {v} if isinstance(v, str) and len(v) >= 6 else set()
            assert not any(x in q.question for x in shown), (q.qid, f.attribute, v)


@needs_data
def test_the_users_own_example_answers(qs):
    by = {q.question: q.reference_answer for q in qs}
    pre = "In the M5 sales data (29 January 2011 to 24 April 2016), "
    assert by[pre + "which store sold the most units over the whole period, and which sold the least?"] == \
        "Most: CA_3, 11,188,180 units.\nLeast: CA_4, 4,103,676 units."
    assert by[pre + "how many total units did the FOODS category sell?"] == \
        "45,089,939 units, against 14,480,670 for HOUSEHOLD and 6,124,800 for HOBBIES."
    assert by[pre + "which state sold the most units overall?"] == "CA, with 28,675,547, ahead of TX (18,899,006) and WI (18,120,856)."
    assert by[pre + "how many units did store CA_1 sell in July 2012?"] == "126,739 units."


@needs_data
def test_the_file_is_in_sync_with_the_generator(qs):
    assert [q.model_dump() for q in build()] == [q.model_dump() for q in qs]  # same seed, same questions and answers


# ----------------------------------------------------------------------------- brute force from the raw CSVs
@pytest.fixture(scope="module")
def m5():
    sales = pd.read_csv(RAW / "m5" / "sales_train_evaluation.csv")
    cal = pd.read_csv(RAW / "m5" / "calendar.csv")
    return sales, cal, [f"d_{i}" for i in range(1, PERIOD_DAYS + 1)]


def answer(qs, text):
    return next(q for q in qs if q.question.endswith(text)).reference_answer


@needs_data
def test_m5_totals_by_masks_and_slices(qs, m5):
    sales, cal, days = m5
    grid = sales[days].to_numpy()
    store, state, cat = sales["store_id"].to_numpy(), sales["state_id"].to_numpy(), sales["cat_id"].to_numpy()
    for s in ("CA_1", "CA_2", "TX_1", "WI_3"):
        assert f"{int(grid[store == s].sum()):,} units." == answer(qs, f"how many units did store {s} sell over the whole period?")
    assert answer(qs, "which state sold the most units overall?").startswith(f"CA, with {int(grid[state == 'CA'].sum()):,}")
    assert answer(qs, "how many total units did the FOODS category sell?").startswith(f"{int(grid[cat == 'FOODS'].sum()):,} units")
    assert int(grid.sum()) == int(grid[state == "CA"].sum() + grid[state == "TX"].sum() + grid[state == "WI"].sum())


@needs_data
def test_m5_month_and_date_by_explicit_column_picks(qs, m5):
    sales, cal, days = m5
    cal = cal[cal["d"].isin(days)]
    july = cal[(cal["date"] >= "2012-07-01") & (cal["date"] <= "2012-07-31")]["d"].tolist()
    assert len(july) == 31
    ca1 = sales[sales["store_id"] == "CA_1"]
    assert answer(qs, "how many units did store CA_1 sell in July 2012?") == f"{int(ca1[july].to_numpy().sum()):,} units."
    xmas = cal[cal["date"] == "2013-12-25"]["d"].iloc[0]
    assert answer(qs, "how many units were sold across all stores on 2013-12-25?") == f"{int(sales[xmas].sum()):,} units."
    assert answer(qs, "how many days does it cover, and what are its first and last dates?").startswith(f"{PERIOD_DAYS:,} days, from {cal['date'].min()}")
    ev = cal[(cal["event_name_1"] == "Easter") & (cal["date"].str.startswith("2015"))]["date"].tolist()
    assert answer(qs, "on what date was Easter in 2015 according to the M5 calendar?") == f"{ev[0]}."


@needs_data
def test_m5_items_and_structure(qs, m5):
    sales, _, days = m5
    item_total = pd.Series(sales[days].to_numpy().sum(axis=1), index=sales["item_id"].to_numpy())
    best = item_total.groupby(level=0).sum().idxmax()
    assert answer(qs, "which item sold the most units across all stores over the whole period, and how many?").startswith(f"{best},")
    assert answer(qs, "how many stores, items and states does the data cover?") == \
        f"{sales['store_id'].nunique()} stores, {sales['item_id'].nunique():,} items and {sales['state_id'].nunique()} states."
    assert answer(qs, "how many item-store combinations (rows) does the sales table have?") == f"{len(sales):,}."


@needs_data
def test_dataco_counts_by_set_logic(qs):
    raw = pd.read_csv(RAW / "dataco" / "DataCoSupplyChainDataset.csv", encoding="latin-1")
    ids = set(raw["Order Id"])
    cancelled = set(raw.loc[raw["Delivery Status"] == "Shipping canceled", "Order Id"])
    delivered = ids - cancelled
    late = set(raw.loc[(raw["Late_delivery_risk"] == 1) & (raw["Delivery Status"] != "Shipping canceled"), "Order Id"])
    pre = "In the DataCo orders data (1 January 2015 to 31 January 2018), "
    assert answer(qs, "how many orders are there, and how many order lines?") == f"{len(ids):,} orders and {len(raw):,} order lines."
    assert answer(qs, "how many orders were cancelled (delivery status 'Shipping canceled'), and what share of all orders is that?").startswith(f"{len(cancelled):,} orders")
    assert answer(qs, "among the delivered (not cancelled) orders, what share arrived later than scheduled?") == \
        f"{len(late) / len(delivered) * 100:.1f}% ({len(late):,} of {len(delivered):,} delivered orders)."
    assert answer(qs, "how many orders used Same Day shipping?") == f"{raw.loc[raw['Shipping Mode'] == 'Same Day', 'Order Id'].nunique():,} orders."
    assert all(pre in q.question for q in qs if "among the delivered" in q.question)
    assert re.fullmatch(r"\d+\.\d%\.", answer(qs, "among delivered First Class orders, what share arrived later than scheduled?"))
