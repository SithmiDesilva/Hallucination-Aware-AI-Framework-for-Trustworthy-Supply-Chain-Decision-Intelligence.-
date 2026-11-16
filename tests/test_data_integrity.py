"""Integrity checks on the real processed tables in data/processed/ (skipped until `make data` has run).

M5: the declared scope (3 stores x 300 items), every row priced, no duplicate store/item/date rows,
no missing dates inside a series, row count consistent with each series' date span.
DataCo: exactly one row per order, exactly the configured sample size, no personal columns, and the
cancelled-order rule applied (no delay or lateness for cancelled orders; for all others the late flag
equals real > scheduled).
"""
import pandas as pd
import pytest

from src.config import PROCESSED, load_config
from src.data.load_dataco import PII_COLS

CFG = load_config()
M5 = PROCESSED / "m5_long.parquet"
DATACO = PROCESSED / "dataco_orders.parquet"


@pytest.fixture(scope="module")
def m5():
    if not M5.exists():
        pytest.skip("data/processed/m5_long.parquet not built")
    return pd.read_parquet(M5)


@pytest.fixture(scope="module")
def dataco():
    if not DATACO.exists():
        pytest.skip("data/processed/dataco_orders.parquet not built")
    return pd.read_parquet(DATACO)


def test_m5_scope_matches_config(m5):
    series = m5[["store_id", "item_id"]].drop_duplicates()
    assert sorted(series["store_id"].unique()) == sorted(CFG["data"]["m5_stores"])
    assert series["item_id"].nunique() == CFG["data"]["m5_items"]
    assert len(series) == len(CFG["data"]["m5_stores"]) * CFG["data"]["m5_items"]


def test_m5_every_row_has_a_price(m5):
    assert m5["sell_price"].notna().all() and (m5["sell_price"] > 0).all()


def test_m5_no_duplicate_series_days(m5):
    assert not m5.duplicated(["store_id", "item_id", "date"]).any()


def test_m5_no_missing_dates_inside_a_series(m5):
    span = m5.groupby(["store_id", "item_id"])["date"].agg(["min", "max", "count"])
    assert (((span["max"] - span["min"]).dt.days + 1) == span["count"]).all()
    assert len(m5) == span["count"].sum()


def test_dataco_one_row_per_order_and_configured_size(dataco):
    assert dataco["Order Id"].is_unique
    assert len(dataco) == CFG["data"]["dataco_orders"]


def test_dataco_has_no_personal_columns(dataco):
    assert not set(PII_COLS) & set(dataco.columns)


def test_dataco_cancelled_rule_applied(dataco):
    cancelled = dataco["is_cancelled"]
    assert cancelled.any()
    assert dataco.loc[cancelled, "shipping_delay_days"].isna().all()
    assert dataco.loc[cancelled, "late"].isna().all()
    assert dataco.loc[cancelled, "raw_delay_days"].notna().all()  # raw value kept
    ok = dataco[~cancelled]
    assert ok["late"].notna().all()
    assert (ok["late"].astype(bool) == (ok["raw_delay_days"] > 0)).all()
    assert (ok["shipping_delay_days"] == ok["raw_delay_days"]).all()
