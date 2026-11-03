"""Unit tests for the data loaders (src/data/load_m5.py, src/data/load_dataco.py).

Cover seeded/stratified sampling of M5 series and DataCo orders, M5 wide-to-long reshaping
(unpriced days dropped, state-specific SNAP flag), and DataCo order aggregation: delay definition,
the lateness rule, cancelled orders keeping raw values but no delay/lateness, and rejection of
conflicting item rows.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.load_dataco import REQUIRED, sample_orders, to_orders
from src.data.load_m5 import sample_series, to_long


def make_sales():
    rows = []
    for state, stores in {"CA": ["CA_1", "CA_2"], "TX": ["TX_1"], "WI": ["WI_1"]}.items():
        for store in stores:
            for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
                for i in range(20):
                    rows.append({"id": f"{cat}_{i}_{store}", "item_id": f"{cat}_1_{i:03d}", "dept_id": f"{cat}_1",
                                 "cat_id": cat, "store_id": store, "state_id": state,
                                 "d_1": 1, "d_2": 0})
    return pd.DataFrame(rows)


def test_sample_series_is_stratified_and_seeded():
    sales = make_sales()
    a = sample_series(sales, n_stores=3, n_items=9, seed=42)
    b = sample_series(sales, n_stores=3, n_items=9, seed=42)
    assert a.equals(b)
    assert a["store_id"].nunique() == 3
    assert set(a["state_id"]) == {"CA", "TX", "WI"}
    assert a.groupby("cat_id")["item_id"].nunique().tolist() == [3, 3, 3]


def test_to_long_drops_unpriced_days_and_picks_state_snap():
    sales = make_sales().iloc[:1]
    cal = pd.DataFrame({
        "d": ["d_1", "d_2"], "date": ["2011-01-29", "2011-01-30"], "wm_yr_wk": [11101, 11101],
        "weekday": ["Saturday", "Sunday"], "event_name_1": [None, None], "event_type_1": [None, None],
        "event_name_2": [None, None], "event_type_2": [None, None],
        "snap_CA": [1, 0], "snap_TX": [0, 0], "snap_WI": [0, 0],
    })
    prices = pd.DataFrame({"store_id": ["CA_1"], "item_id": [sales.iloc[0]["item_id"]],
                           "wm_yr_wk": [11101], "sell_price": [1.25]})
    long = to_long(sales, cal, prices)
    assert len(long) == 2 and long["snap"].tolist() == [1, 0]
    long_unpriced = to_long(sales, cal, prices.assign(wm_yr_wk=99999))
    assert long_unpriced.empty


def make_items():
    base = {c: 1 for c in REQUIRED}
    rows = []
    for oid, mode in [(1, "Standard Class"), (1, "Standard Class"), (2, "First Class")]:
        rows.append({**base, "Order Id": oid, "Shipping Mode": mode, "Market": "LATAM",
                     "Order Region": "R", "Order Country": "C", "Delivery Status": "Late delivery",
                     "order date (DateOrders)": "2016-01-01", "shipping date (DateOrders)": "2016-01-03",
                     "Days for shipping (real)": 5, "Days for shipment (scheduled)": 3,
                     "Late_delivery_risk": 1})
    return pd.DataFrame(rows)


def test_to_orders_aggregates_and_defines_delay():
    orders = to_orders(make_items())
    assert len(orders) == 2
    assert orders["n_items"].tolist() == [2, 1]
    assert orders["shipping_delay_days"].tolist() == [2, 2] and orders["late"].all()
    assert not orders["is_cancelled"].any()


def test_cancelled_orders_have_no_delay_or_lateness_but_keep_raw_values():
    items = make_items()
    items.loc[2, "Delivery Status"] = "Shipping canceled"
    items.loc[2, "Late_delivery_risk"] = 0
    orders = to_orders(items).set_index("Order Id")
    assert orders.loc[2, "is_cancelled"] and not orders.loc[1, "is_cancelled"]
    assert pd.isna(orders.loc[2, "shipping_delay_days"]) and pd.isna(orders.loc[2, "late"])
    assert orders.loc[2, "raw_delay_days"] == 2 and orders.loc[2, "Days for shipping (real)"] == 5
    assert orders.loc[1, "shipping_delay_days"] == 2


def test_lateness_gold_rule():
    from src.data.dataco_rules import CANCELLED_LATE_ANSWER, lateness_gold
    items = make_items()
    items.loc[2, "Delivery Status"] = "Shipping canceled"
    orders = to_orders(items).set_index("Order Id")
    cancelled = lateness_gold(orders.loc[2])
    assert cancelled["outcome"] == "cancelled" and cancelled["answer"] == CANCELLED_LATE_ANSWER
    assert cancelled["late"] is None and cancelled["delay_days"] is None
    late = lateness_gold(orders.loc[1])
    assert late["outcome"] == "late" and late["delay_days"] == 2


def test_to_orders_rejects_conflicting_items():
    items = make_items()
    items.loc[1, "Days for shipping (real)"] = 9
    with pytest.raises(ValueError):
        to_orders(items)


def test_sample_orders_size_and_seed():
    orders = pd.DataFrame({"Order Id": range(1000), "Market": ["A", "B"] * 500,
                           "Shipping Mode": ["X", "Y", "Z", "W"] * 250})
    s = sample_orders(orders, 100, seed=1)
    assert abs(len(s) - 100) <= 8 and s.equals(sample_orders(orders, 100, seed=1))
