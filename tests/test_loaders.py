"""Unit tests for the data loaders (src/data/load_m5.py, src/data/load_dataco.py) on small synthetic tables.

M5: the declared store scope and seeded, category-stratified item sampling; the wide-to-long
reshape (rows = series x days); the calendar/price join (unpriced days dropped, state-specific SNAP flag).
DataCo: personal columns dropped; order aggregation and the delay definition; the lateness rule;
cancelled orders keeping raw values but no delay/lateness; rejection of conflicting item rows;
a stratified sample of exactly n orders.
Checks on the real processed tables are in test_data_integrity.py.
"""
import pandas as pd
import pytest

from src.data.load_dataco import PERSONAL_COLS, REQUIRED, UNUSED_COLS, clean, sample_orders, snake_case, to_orders
from src.data.load_m5 import join_calendar_prices, prepare_calendar, sample_scope, to_long, validate_sales


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


def test_sample_scope_keeps_declared_stores_and_stratifies_items():
    sales = make_sales()
    a = sample_scope(sales, stores=["CA_1", "TX_1", "WI_1"], n_items=9, seed=42)
    b = sample_scope(sales, stores=["CA_1", "TX_1", "WI_1"], n_items=9, seed=42)
    assert a.equals(b)
    assert sorted(a["store_id"].unique()) == ["CA_1", "TX_1", "WI_1"]
    assert a.groupby("cat_id")["item_id"].nunique().tolist() == [3, 3, 3]
    assert len(a) == 3 * 9  # every picked item appears in every picked store


def test_sample_scope_defaults_to_all_stores_with_the_same_items():
    sales = make_sales()
    every = sample_scope(sales, n_items=9, seed=42)
    named = sample_scope(sales, stores=sorted(sales["store_id"].unique()), n_items=9, seed=42)
    assert every.equals(named) and every["store_id"].nunique() == 4
    assert every.groupby("store_id")["item_id"].nunique().tolist() == [9, 9, 9, 9]  # the same 9 items in every store
    assert every.groupby("cat_id")["item_id"].nunique().tolist() == [3, 3, 3]
    # the items do not depend on which stores are kept, so shrinking the scope never changes the item sample
    few = sample_scope(sales, stores=["CA_1"], n_items=9, seed=42)
    assert set(few["item_id"]) == set(every["item_id"])


def test_sample_scope_rejects_unknown_store():
    with pytest.raises(ValueError):
        sample_scope(make_sales(), stores=["CA_1", "ZZ_9"], n_items=9)


def test_to_long_row_count_is_series_times_days():
    sales = make_sales()
    long = to_long(sales)
    assert len(long) == len(sales) * 2
    assert long["units"].dtype == "int32" and set(long["d"]) == {"d_1", "d_2"}


def make_calendar(n=3):
    """n consecutive days from Saturday 2011-01-29, one Walmart week per day for the tests; M5's wday: Saturday = 1."""
    dates = pd.date_range("2011-01-29", periods=n)
    return pd.DataFrame({
        "d": [f"d_{i + 1}" for i in range(n)], "date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": [11101 + i for i in range(n)],
        "weekday": dates.day_name(), "wday": [(d.weekday() - 5) % 7 + 1 for d in dates],
        "event_name_1": [None, "SuperBowl", None][:n], "event_type_1": [None, "Sporting", None][:n],
        "event_name_2": [None] * n, "event_type_2": [None] * n,
        "snap_CA": [1, 0, 1][:n], "snap_TX": [0] * n, "snap_WI": [0] * n,
    })


def test_join_fills_events_with_none_and_picks_state_snap():
    sales = make_sales().iloc[:1].assign(d_3=2)
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": sales.iloc[0]["item_id"], "wm_yr_wk": [11101, 11102, 11103], "sell_price": [1.25, 1.5, 1.5]})
    joined = join_calendar_prices(to_long(sales), make_calendar(), prices)
    assert joined["event_name_1"].tolist() == ["None", "SuperBowl", "None"] and joined["event_type_1"].tolist() == ["None", "Sporting", "None"]
    assert not joined[["event_name_1", "event_name_2", "event_type_1", "event_type_2"]].isna().any().any()
    assert joined["snap"].tolist() == [1, 0, 1] and joined["sell_price"].tolist() == [1.25, 1.5, 1.5]
    assert joined["date"].dt.day_name().tolist() == ["Saturday", "Sunday", "Monday"]


def test_days_before_the_first_price_record_are_removed():
    sales = make_sales().iloc[:1].assign(d_3=2)
    item = sales.iloc[0]["item_id"]
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": item, "wm_yr_wk": [11102, 11103], "sell_price": [1.5, 1.5]})
    joined = join_calendar_prices(to_long(sales), make_calendar(), prices)
    assert joined["d"].tolist() == ["d_2", "d_3"]  # d_1 is before the first price record: not on sale yet


def test_a_missing_price_after_the_first_record_is_an_error_not_a_guess():
    sales = make_sales().iloc[:1].assign(d_3=2)
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": sales.iloc[0]["item_id"], "wm_yr_wk": [11101, 11103], "sell_price": [1.5, 1.5]})
    with pytest.raises(ValueError, match="no price AFTER"):
        join_calendar_prices(to_long(sales), make_calendar(), prices)


def test_a_series_with_no_price_at_all_is_an_error():
    sales = make_sales().iloc[:1].assign(d_3=2)
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": "FOODS_1_999", "wm_yr_wk": [11101], "sell_price": [1.5]})
    with pytest.raises(ValueError, match="no price record at all"):
        join_calendar_prices(to_long(sales), make_calendar(), prices)


# ----------------------------------------------------------------------------- the checks the preprocessing must run
def valid_sales(n_days=2):
    s = make_sales()
    s["id"] = s["item_id"] + "_" + s["store_id"] + "_evaluation"
    return s[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *[f"d_{i}" for i in range(1, n_days + 1)]]]


def test_valid_sales_pass_validation():
    validate_sales(valid_sales(), full=False)


@pytest.mark.parametrize("break_it, message", [
    (lambda s: s.__setitem__("id", s["id"].iloc[0]), "id is not unique"),
    (lambda s: s.__setitem__("item_id", ["FOODS-1-000"] + list(s["item_id"].iloc[1:])), "item_id has 1 malformed"),
    (lambda s: s.__setitem__("store_id", ["CA_X"] + list(s["store_id"].iloc[1:])), "store_id has 1 malformed"),
    (lambda s: s.__setitem__("cat_id", ["HOBBIES"] + list(s["cat_id"].iloc[1:])), "dept_id does not start with its cat_id"),
    (lambda s: s.__setitem__("state_id", ["TX"] + list(s["state_id"].iloc[1:])), "store_id does not start with its state_id"),
    (lambda s: s.__setitem__("id", s["id"].str.replace("_evaluation", "_validation")), "id is not item_id_store_id_evaluation"),
    (lambda s: s.__setitem__("d_1", -1), "negative"),
    (lambda s: s.__setitem__("d_1", 1.5), "not all integers"),
    (lambda s: s.drop(columns="d_1", inplace=True), "day columns"),
])
def test_bad_sales_are_rejected(break_it, message):
    s = valid_sales().copy()
    break_it(s)
    with pytest.raises(ValueError, match=message):
        validate_sales(s, full=False)


def test_a_duplicated_item_store_pair_is_rejected():
    s = pd.concat([valid_sales(), valid_sales().iloc[:1].assign(id="OTHER_ID")], ignore_index=True)
    with pytest.raises(ValueError, match=r"\(item_id, store_id\) is not unique"):
        validate_sales(s, full=False)


def test_the_full_file_counts_are_checked():
    with pytest.raises(ValueError, match="expected 30,490"):
        validate_sales(valid_sales(), full=True)


def test_splitting_the_id_string_gives_wrong_keys_which_is_why_we_never_do_it():
    """On the real file it gives 3 departments, 4 stores and 824 items instead of 7, 10 and 3,049: the item number
    alone (001..) is not an item, and the store number alone (1..4) is not a store. Keys come from the columns."""
    s = valid_sales()
    parts = s["id"].str.split("_", expand=True)  # FOODS_1_000_CA_1_evaluation -> FOODS, 1, 000, CA, 1, evaluation
    assert parts[2].nunique() < s["item_id"].nunique()
    assert parts[4].nunique() < s["store_id"].nunique()


def test_the_calendar_check_accepts_the_real_convention_and_fills_events():
    cal = prepare_calendar(make_calendar(), n_days=3)
    assert cal["event_name_1"].tolist() == ["None", "SuperBowl", "None"] and cal["date"].dt.day_name()[0] == "Saturday"
    assert int(make_calendar()["wday"].iloc[0]) == 1  # M5 starts at Saturday = 1


@pytest.mark.parametrize("break_it, message", [
    (lambda c: c.__setitem__("wday", c["date"].map(lambda d: pd.Timestamp(d).weekday() + 1)), "wday does not follow Saturday = 1"),  # Monday-based
    (lambda c: c.__setitem__("date", ["2011-01-29", "2011-01-31", "2011-02-01"]), "dates are not consecutive"),
    (lambda c: c.__setitem__("weekday", ["Monday"] * 3), "weekday names do not match"),
    (lambda c: c.__setitem__("snap_CA", [2, 0, 1]), "SNAP flags"),
    (lambda c: c.__setitem__("wm_yr_wk", [11101, None, 11103]), "wm_yr_wk has missing"),
])
def test_bad_calendars_are_rejected(break_it, message):
    cal = make_calendar()
    break_it(cal)
    with pytest.raises(ValueError, match=message):
        prepare_calendar(cal, n_days=3)


def test_a_calendar_missing_days_is_rejected():
    with pytest.raises(ValueError, match="calendar has 3 of the 5 days"):
        prepare_calendar(make_calendar(3), n_days=5)


# ----------------------------------------------------------------------------- DataCo
def make_raw():
    """Item rows with the RAW column names, including the personal and unused columns (some with missing values)."""
    rows = []
    for i, (oid, mode) in enumerate([(1, "Standard Class"), (1, "Standard Class"), (2, "First Class")], 1):
        rows.append({
            "Order Id": oid, "Order Item Id": i, "Order Status": "COMPLETE", "Shipping Mode": mode, "Market": "LATAM",
            "Order Region": "R", "Order Country": "C", "Delivery Status": "Late delivery", "Late_delivery_risk": 1,
            "order date (DateOrders)": "2016-01-01 10:00:00", "shipping date (DateOrders)": "2016-01-06 10:00:00",
            "Days for shipping (real)": 5, "Days for shipment (scheduled)": 3, "Sales": 1999.99, "Order Item Product Price": 50.0,
            "Customer Fname": "Ann", "Customer Lname": None, "Customer Email": "a@b.c", "Customer Password": "XXXXXXXXX",
            "Customer Street": "1 Road", "Customer Zipcode": None, "Product Description": None, "Order Zipcode": None,
            "Product Image": "http://img",
        })
    return pd.DataFrame(rows)


def make_items():
    """Cleaned item rows (snake_case names) as to_orders expects them."""
    base = {c: 1 for c in REQUIRED}
    rows = []
    for oid, mode in [(1, "Standard Class"), (1, "Standard Class"), (2, "First Class")]:
        rows.append({**base, "order_id": oid, "shipping_mode": mode, "market": "LATAM", "order_region": "R", "order_country": "C",
                     "delivery_status": "Late delivery", "order_date": "2016-01-01", "shipping_date": "2016-01-03",
                     "days_for_shipping_real": 5, "days_for_shipment_scheduled": 3, "late_delivery_risk": 1})
    return pd.DataFrame(rows)


def test_snake_case_names():
    assert snake_case("Days for shipping (real)") == "days_for_shipping_real"
    assert snake_case("Late_delivery_risk") == "late_delivery_risk" and snake_case("Order Item Product Price") == "order_item_product_price"
    assert snake_case("order date (DateOrders)") == "order_date" and snake_case("shipping date (DateOrders)") == "shipping_date"


def test_clean_drops_personal_and_unused_columns_before_anything_else():
    cleaned = clean(make_raw())  # would raise on the missing last names / zip codes if they were not dropped first
    assert not {"customer_fname", "customer_lname", "customer_email", "customer_password", "customer_street", "customer_zipcode"} & set(cleaned.columns)
    assert not {"product_description", "order_zipcode", "product_image"} & set(cleaned.columns)
    assert not set(PERSONAL_COLS + UNUSED_COLS) & set(cleaned.columns)


def test_clean_renames_to_snake_case_parses_dates_and_leaves_values_untouched():
    cleaned = clean(make_raw())
    assert all(c == c.lower() and " " not in c and "(" not in c for c in cleaned.columns)
    assert {"order_id", "delivery_status", "late_delivery_risk", "days_for_shipping_real", "order_date", "shipping_date"} <= set(cleaned.columns)
    assert str(cleaned["order_date"].dtype).startswith("datetime64") and str(cleaned["shipping_date"].dtype).startswith("datetime64")
    assert cleaned["sales"].tolist() == [1999.99] * 3 and cleaned["order_item_product_price"].tolist() == [50.0] * 3  # no scaling, no capping
    assert cleaned["delivery_status"].dtype == object and cleaned["shipping_mode"].dtype == object and cleaned["market"].dtype == object  # text, not one-hot


def test_clean_fails_on_a_name_collision():
    raw = make_raw().assign(**{"order id": 7})  # snake_case of "order id" is order_id, like "Order Id"
    with pytest.raises(ValueError, match="collide"):
        clean(raw)


def test_clean_never_fills_a_missing_value():
    raw = make_raw().assign(**{"Order State": [None, "TX", "TX"]})  # a kept column with a gap: the column must go, not be filled
    with pytest.raises(ValueError, match="drop the column, do not fill it"):
        clean(raw)


def test_clean_rejects_duplicates():
    dup_id = make_raw()
    dup_id["Order Item Id"] = 1
    with pytest.raises(ValueError, match="order_item_id is not unique"):
        clean(dup_id)
    with pytest.raises(ValueError, match="duplicated rows"):
        clean(pd.concat([make_raw().drop(columns="Order Item Id"), make_raw().drop(columns="Order Item Id").iloc[:1]], ignore_index=True))


def test_to_orders_aggregates_and_defines_delay_and_late():
    orders = to_orders(make_items())
    assert len(orders) == 2
    assert orders["n_items"].tolist() == [2, 1]
    assert orders["shipping_delay_days"].tolist() == [2, 2] and orders["late"].all()
    assert not orders["is_cancelled"].any()


def test_late_is_delay_above_zero_for_early_on_time_and_late_orders():
    items = make_items()
    items.loc[[0, 1], ["days_for_shipping_real", "delivery_status", "late_delivery_risk"]] = [2, "Advance shipping", 0]   # 2 vs 3 scheduled
    items.loc[2, ["days_for_shipping_real", "delivery_status", "late_delivery_risk"]] = [3, "Shipping on time", 0]
    orders = to_orders(items).set_index("order_id")
    assert orders.loc[1, "shipping_delay_days"] == -1 and not orders.loc[1, "late"]
    assert orders.loc[2, "shipping_delay_days"] == 0 and not orders.loc[2, "late"]


def test_the_late_flag_is_cross_checked_against_late_delivery_risk_and_delivery_status():
    items = make_items()
    items["late_delivery_risk"] = 0  # delay is +2 but the dataset's own flag says not at risk
    with pytest.raises(ValueError, match="2 disagree with Late_delivery_risk"):
        to_orders(items)
    items = make_items()
    items["delivery_status"] = "Advance shipping"  # delay +2 but the status says early
    with pytest.raises(ValueError, match="2 disagree with Delivery Status"):
        to_orders(items)
    items = make_items()
    items["delivery_status"] = "Mystery status"
    with pytest.raises(ValueError, match="unknown status"):
        to_orders(items)


def test_cancelled_orders_have_no_delay_or_lateness_but_keep_raw_values():
    items = make_items()
    items.loc[2, "delivery_status"] = "Shipping canceled"
    items.loc[2, "late_delivery_risk"] = 0  # the cross-check skips cancelled orders: a recorded delay means nothing there
    orders = to_orders(items).set_index("order_id")
    assert orders.loc[2, "is_cancelled"] and not orders.loc[1, "is_cancelled"]
    assert pd.isna(orders.loc[2, "shipping_delay_days"]) and pd.isna(orders.loc[2, "late"])
    assert orders.loc[2, "raw_delay_days"] == 2 and orders.loc[2, "days_for_shipping_real"] == 5
    assert orders.loc[1, "shipping_delay_days"] == 2


def test_lateness_gold_rule():
    from src.data.dataco_rules import CANCELLED_LATE_ANSWER, lateness_gold
    items = make_items()
    items.loc[2, "delivery_status"] = "Shipping canceled"
    orders = to_orders(items).set_index("order_id")
    cancelled = lateness_gold(orders.loc[2])
    assert cancelled["outcome"] == "cancelled" and cancelled["answer"] == CANCELLED_LATE_ANSWER
    assert cancelled["late"] is None and cancelled["delay_days"] is None
    late = lateness_gold(orders.loc[1])
    assert late["outcome"] == "late" and late["delay_days"] == 2


def test_to_orders_rejects_conflicting_items():
    items = make_items()
    items.loc[1, "days_for_shipping_real"] = 9
    with pytest.raises(ValueError, match="disagree within orders"):
        to_orders(items)


@pytest.mark.parametrize("n", [100, 333, 999])
def test_sample_orders_is_exactly_n_stratified_and_seeded(n):
    orders = pd.DataFrame({"order_id": range(1000), "market": ["A", "B", "B"] * 333 + ["A"],
                           "shipping_mode": ["X", "Y", "Z", "W"] * 250})
    s = sample_orders(orders, n, seed=1)
    assert len(s) == n and s["order_id"].is_unique
    assert s.equals(sample_orders(orders, n, seed=1))
    full = orders.groupby(["market", "shipping_mode"]).size()
    got = s.groupby(["market", "shipping_mode"]).size().reindex(full.index, fill_value=0)
    assert ((got - full * n / len(orders)).abs() < 1).all()  # proportional within rounding
