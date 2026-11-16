"""Tests for src/data/describe.py: the summary tables' totals must match the underlying counts.

Synthetic tables check the arithmetic (group rows add up to the All row; late rate and mean delay
exclude cancelled orders). The raw-data tests recompute the headline totals straight from the downloaded
CSVs and are skipped when data/raw or data/processed is not available.
"""
import pandas as pd
import pytest

from src.config import PROCESSED, RAW, load_config
from src.data.describe import dataco_summary, m5_summary
from src.data.load_dataco import CANCELLED_STATUS


def make_orders():
    return pd.DataFrame({
        "Order Id": range(1, 7),
        "Market": ["A", "A", "A", "B", "B", "B"],
        "Shipping Mode": ["X", "X", "Y", "X", "X", "X"],
        "is_cancelled": [False, False, True, False, True, False],
        "late": pd.array([True, False, pd.NA, True, pd.NA, True], dtype="boolean"),
        "shipping_delay_days": pd.array([2, -1, pd.NA, 1, pd.NA, 3], dtype="Int64"),
    })


def test_dataco_summary_totals_and_cancelled_exclusion():
    s = dataco_summary(make_orders())
    groups, total = s[s["Market"] != "All"], s[s["Market"] == "All"].iloc[0]
    assert groups["orders"].sum() == total["orders"] == 6
    assert groups["cancelled"].sum() == total["cancelled"] == 2
    assert total["late_rate"] == pytest.approx(3 / 4)            # 4 non-cancelled orders, 3 late
    assert total["mean_delay_days"] == pytest.approx((2 - 1 + 1 + 3) / 4)
    ay = s[(s["Market"] == "A") & (s["Shipping Mode"] == "Y")].iloc[0]
    assert ay["orders"] == 1 and ay["cancelled"] == 1 and pd.isna(ay["late_rate"])  # nothing delivered


def make_m5():
    rows = []
    for store, item, cat, units in [("CA_1", "F1", "FOODS", [0, 2, 4]), ("CA_1", "H1", "HOBBIES", [1, 1, 1]),
                                    ("TX_2", "F1", "FOODS", [3, 0, 3])]:
        for i, u in enumerate(units):
            rows.append({"store_id": store, "item_id": item, "cat_id": cat, "date": pd.Timestamp("2011-01-29") + pd.Timedelta(days=i),
                         "units": u, "sell_price": 2.0})
    long = pd.DataFrame(rows)
    sim = long.rename(columns={"units": "demand"})[["store_id", "item_id", "date", "demand"]].copy()
    sim["sim_units_sold"] = sim["demand"]
    sim.loc[(sim["store_id"] == "TX_2") & (sim["demand"] == 3), "sim_units_sold"] = 2  # a shortfall of 1 on two days
    sim["shortfall"] = sim["demand"] - sim["sim_units_sold"]
    return long, sim


def test_m5_summary_totals():
    long, sim = make_m5()
    s = m5_summary(long, sim)
    groups, total = s[s["store_id"] != "All"], s[s["store_id"] == "All"].iloc[0]
    assert groups["series"].sum() == total["series"] == 3
    assert groups["days"].sum() == total["days"] == len(long) == 9
    assert groups["units"].sum() == total["units"] == long["units"].sum() == 15
    assert groups["sim_shortfall_units"].sum() == total["sim_shortfall_units"] == 2
    assert total["zero_sales_share"] == pytest.approx(2 / 9, abs=1e-3)  # tables are rounded to 3-4 decimals
    assert total["sim_fill_rate"] == pytest.approx(13 / 15, abs=1e-4)


def _raw_dataco():
    path = RAW / "dataco" / "DataCoSupplyChainDataset.csv"
    if not path.exists():
        pytest.skip("raw DataCo file not available")
    return pd.read_csv(path, encoding="latin-1")


def test_dataco_full_summary_matches_raw_counts():
    from src.data.load_dataco import clean, to_orders
    raw = _raw_dataco()
    s = dataco_summary(to_orders(clean(raw)))
    total = s[s["Market"] == "All"].iloc[0]
    assert total["orders"] == raw["Order Id"].nunique()
    assert total["cancelled"] == raw.loc[raw["Delivery Status"] == CANCELLED_STATUS, "Order Id"].nunique()
    assert s[s["Market"] != "All"]["orders"].sum() == total["orders"]


def test_m5_summary_matches_raw_sales():
    from src.data.load_m5 import load_raw, sample_scope
    long_path, sim_path = PROCESSED / "m5_long.parquet", PROCESSED / "m5_sim.parquet"
    if not (long_path.exists() and sim_path.exists() and (RAW / "m5" / "sales_train_evaluation.csv").exists()):
        pytest.skip("M5 raw or processed files not available")
    cfg = load_config()
    long, sim = pd.read_parquet(long_path), pd.read_parquet(sim_path)
    total = m5_summary(long, sim).query("store_id == 'All'").iloc[0]
    sales = sample_scope(load_raw()[0], stores=cfg["data"]["m5_stores"], n_items=cfg["data"]["m5_items"], seed=cfg["seed"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    assert total["series"] == len(sales)
    assert total["units"] == int(sales[day_cols].to_numpy().sum())  # unpriced (pre-launch) days carry no sales
    assert total["days"] == len(long)
