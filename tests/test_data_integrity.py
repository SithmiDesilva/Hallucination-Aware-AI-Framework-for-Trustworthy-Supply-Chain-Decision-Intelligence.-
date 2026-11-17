"""Integrity checks on the real processed tables in data/processed/ (skipped until `make data` has run).

M5: the declared scope (all 10 stores x 300 items), every row priced, no duplicate store/item/date rows,
no missing dates inside a series, row count consistent with each series' date span.
DataCo: exactly one row per order, exactly the configured sample size, no personal or unused columns, snake_case
names, nothing filled (the only nulls are the delay and late of cancelled orders), text categories left as text, values
not scaled or capped, the cancelled-order rule applied, and late (delay > 0) cross-checked against Late_delivery_risk
and Delivery Status. Raw-file checks use the downloaded CSV.
"""
import pandas as pd
import pytest

from src.config import PROCESSED, RAW, load_config
from src.data.load_dataco import PERSONAL_COLS, UNUSED_COLS, snake_case

CFG = load_config()
M5 = PROCESSED / "m5_long.parquet"
DATACO = PROCESSED / "dataco_orders.parquet"
DATACO_ALL = PROCESSED / "dataco_orders_all.parquet"
RAW_DATACO = RAW / "dataco" / "DataCoSupplyChainDataset.csv"


@pytest.fixture(scope="module")
def m5():
    if not M5.exists():
        pytest.skip("data/processed/m5_long.parquet not built")
    return pd.read_parquet(M5)


@pytest.fixture(scope="module")
def dataco_all():
    if not DATACO_ALL.exists():
        pytest.skip("data/processed/dataco_orders_all.parquet not built")
    return pd.read_parquet(DATACO_ALL)


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
    assert dataco["order_id"].is_unique
    assert len(dataco) == CFG["data"]["dataco_orders"]


def test_dataco_has_no_personal_or_unused_columns(dataco, dataco_all):
    dropped = {snake_case(c) for c in PERSONAL_COLS + UNUSED_COLS}
    assert not dropped & set(dataco.columns) and not dropped & set(dataco_all.columns)


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


# ----------------------------------------------------------------------------- M5 preprocessing rules, checked against the raw CSVs
RAW_M5 = [RAW / "m5" / "sales_train_evaluation.csv", RAW / "m5" / "calendar.csv", RAW / "m5" / "sell_prices.csv"]
needs_raw_m5 = pytest.mark.skipif(not all(p.exists() for p in RAW_M5), reason="raw M5 files not available")


@pytest.fixture(scope="module")
def raw_m5():
    from src.data.load_m5 import load_raw, sample_scope
    sales, calendar, prices = load_raw()
    sampled = sample_scope(sales, stores=CFG["data"]["m5_stores"], n_items=CFG["data"]["m5_items"], seed=CFG["seed"])
    keys = sampled[["store_id", "item_id"]].drop_duplicates()
    return sampled, calendar, prices.merge(keys, on=["store_id", "item_id"])


def test_m5_keeps_300_items_100_per_category_in_all_10_stores(m5):
    per_store = m5.groupby("store_id")["item_id"].apply(frozenset)
    assert len(per_store) == 10 and len(set(per_store)) == 1  # the same 300 items in every store
    items = m5.drop_duplicates("item_id")
    assert len(items) == 300 and items["cat_id"].value_counts().to_dict() == {"FOODS": 100, "HOBBIES": 100, "HOUSEHOLD": 100}


def test_m5_keys_come_from_the_columns_and_are_consistent(m5):
    ids = m5[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
    assert ids.groupby("item_id")["dept_id"].nunique().eq(1).all()
    assert all(i.startswith(d + "_") for i, d in zip(ids["item_id"], ids["dept_id"]))
    assert all(d.startswith(c + "_") for d, c in zip(ids["dept_id"], ids["cat_id"]))
    assert all(s.startswith(st + "_") for s, st in zip(ids["store_id"], ids["state_id"]))
    assert ids["store_id"].nunique() == 10 and ids["state_id"].nunique() == 3 and ids["dept_id"].nunique() <= 7


def test_m5_event_names_are_filled_with_none(m5):
    for col in ("event_name_1", "event_name_2", "event_type_1", "event_type_2"):
        assert m5[col].notna().all()
    assert (m5["event_name_1"] == "None").any() and (m5["event_name_1"] != "None").any()


@needs_raw_m5
def test_m5_sales_are_not_capped_or_altered(m5, raw_m5):
    sampled, _, _ = raw_m5
    days = [c for c in sampled.columns if c.startswith("d_")]
    grid = sampled[days].to_numpy()
    assert int(m5["units"].max()) == int(grid.max())  # no outlier capping
    assert int(m5["units"].sum()) == int(grid.sum())  # days before the first price carry no sales, so nothing is lost
    assert m5["units"].dtype.kind in "iu" and (m5["units"] >= 0).all()


@needs_raw_m5
def test_m5_prices_are_the_raw_prices(m5, raw_m5):
    _, _, prices = raw_m5
    got = m5[["store_id", "item_id", "wm_yr_wk", "sell_price"]].drop_duplicates().merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", suffixes=("", "_raw"), validate="1:1")
    assert (got["sell_price"] == got["sell_price_raw"]).all()  # joined on wm_yr_wk, and never capped


@needs_raw_m5
def test_m5_no_day_before_a_series_first_price_record_and_none_missing_after(m5, raw_m5):
    _, _, prices = raw_m5
    first = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    kept_first = m5.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    assert (kept_first == first.reindex(kept_first.index)).all() and len(kept_first) == len(first) == 3000
    assert m5["sell_price"].notna().all()


@needs_raw_m5
def test_m5_dates_come_from_the_calendar_with_walmart_weeks(m5, raw_m5):
    _, calendar, _ = raw_m5
    cal = calendar.assign(date=pd.to_datetime(calendar["date"])).set_index("d")
    d = m5.drop_duplicates("d").set_index("d")
    assert (d["date"] == cal.loc[d.index, "date"]).all() and (d["wm_yr_wk"] == cal.loc[d.index, "wm_yr_wk"]).all()
    wk = m5.drop_duplicates("date").groupby("wm_yr_wk")["date"].agg(["min", "max", "count"])
    full = wk[wk["count"] == 7]
    assert (full["min"].dt.day_name() == "Saturday").all() and (full["max"].dt.day_name() == "Friday").all()  # Saturday-Friday weeks


# ----------------------------------------------------------------------------- DataCo preprocessing rules
needs_raw_dataco = pytest.mark.skipif(not RAW_DATACO.exists(), reason="raw DataCo file not available")


def test_dataco_all_orders_table_keeps_every_order_and_the_sample_is_a_subset(dataco, dataco_all):
    assert dataco_all["order_id"].is_unique and len(dataco_all) > len(dataco)
    assert set(dataco["order_id"]) <= set(dataco_all["order_id"])
    assert dataco_all["is_cancelled"].sum() > dataco["is_cancelled"].sum() > 0  # cancelled orders stay in the index


def test_dataco_columns_are_snake_case(dataco, dataco_all):
    for table in (dataco, dataco_all):
        assert all(c == c.lower() and " " not in c and "(" not in c for c in table.columns)
        assert len(set(table.columns)) == len(table.columns)


def test_dataco_nothing_is_filled(dataco, dataco_all):
    for table in (dataco, dataco_all):
        nulls = table.isna().sum()
        assert set(nulls[nulls > 0].index) == {"shipping_delay_days", "late"}  # only the cancelled orders' delay and lateness
        assert nulls["late"] == nulls["shipping_delay_days"] == table["is_cancelled"].sum()


def test_dataco_categories_stay_text_and_numbers_are_unscaled(dataco_all):
    for col in ("delivery_status", "shipping_mode", "market", "order_region", "order_status"):
        assert dataco_all[col].dtype == object and dataco_all[col].map(type).eq(str).all()
    assert set(dataco_all["delivery_status"]) == {"Late delivery", "Advance shipping", "Shipping on time", "Shipping canceled"}
    assert dataco_all["days_for_shipping_real"].dtype.kind == "i" and dataco_all["days_for_shipping_real"].min() >= 0  # real days, not z-scores
    assert dataco_all["days_for_shipment_scheduled"].isin([0, 1, 2, 4]).all()


def test_dataco_late_is_delay_above_zero_and_agrees_with_the_datasets_own_flags(dataco_all):
    ok = dataco_all[~dataco_all["is_cancelled"]]
    assert (ok["late"].astype(bool) == (ok["raw_delay_days"] > 0)).all()
    assert (ok["late"].astype(bool) == (ok["late_delivery_risk"] == 1)).all()
    sign = {"Late delivery": 1, "Advance shipping": -1, "Shipping on time": 0}
    assert (ok["delivery_status"].map(sign) == ok["raw_delay_days"].apply(lambda d: (d > 0) - (d < 0))).all()
    assert (dataco_all["raw_delay_days"] == dataco_all["days_for_shipping_real"] - dataco_all["days_for_shipment_scheduled"]).all()


@needs_raw_dataco
def test_dataco_values_match_the_raw_file_with_no_capping_scaling_or_filling(dataco_all):
    from src.data.load_dataco import clean, load_raw
    raw = load_raw()
    items = clean(raw)
    assert items["sales"].equals(raw["Sales"]) and items["order_item_product_price"].equals(raw["Order Item Product Price"])  # no capping
    assert items["sales"].max() == raw["Sales"].max() == 1999.98999
    first = raw.drop_duplicates("Order Id").set_index("Order Id").loc[dataco_all["order_id"]]
    assert (dataco_all["days_for_shipping_real"].to_numpy() == first["Days for shipping (real)"].to_numpy()).all()
    assert len(items) == len(raw) and items.columns.is_unique
    assert not any(k in c for c in items.columns for k in ("rfm", "recency", "frequency", "monetary", "segment_score"))  # no RFM features
    assert "Customer Lname" in raw.columns and raw["Customer Lname"].isna().any() and "customer_lname" not in items.columns  # dropped, not filled


def test_the_access_logs_file_is_never_read():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src"
    offenders = [str(f) for f in root.rglob("*.py") if "tokenized_access_logs" in f.read_text(encoding="utf-8") or "access_logs" in f.read_text(encoding="utf-8")]
    assert offenders == []
