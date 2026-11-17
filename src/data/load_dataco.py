"""DataCo loader.

Done: read as latin-1; drop the personal columns FIRST (names, email, password, street, zip code) and the empty or
unused ones (Product Description, Order Zipcode, Product Image); rename every column to snake_case and fail on a name
collision; parse the order and shipping dates; delay = real - scheduled shipping days and late = delay > 0, cross-checked
against Delivery Status and Late_delivery_risk; cancelled orders stay in the table (the index) but are excluded from
lateness (their shipping_delay_days and late are null); roll order items up to orders and fail if the shipping details
differ within an order; sample 20,000 orders stratified by market x shipping mode with a fixed seed. The region x month
aggregates are computed from ALL orders (src/cards/build_cards.py), not only the sample.

Not done on purpose: filling missing values (the columns that have any are dropped instead), scaling (no z-scores),
one-hot encoding (delivery status, shipping mode and market stay text), capping sales or prices, customer-segmentation
(RFM) features, and the access-logs file (not relevant; it holds IP addresses) is never read.

Definitions (also in data/README.md); proven by src/data/dataco_checks.py:
  is_cancelled        = delivery_status == "Shipping canceled"  (identical to order_status CANCELED / SUSPECTED_FRAUD)
  raw_delay_days      = days_for_shipping_real - days_for_shipment_scheduled, kept for every order
  shipping_delay_days = raw_delay_days, NULL for cancelled orders (nothing was delivered)
  late                = raw_delay_days > 0, NULL for cancelled orders
"""
import re

import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, load_config

CANCELLED_STATUS = "Shipping canceled"
PERSONAL_COLS = ["Customer Fname", "Customer Lname", "Customer Email", "Customer Password", "Customer Street", "Customer Zipcode"]
UNUSED_COLS = ["Product Description", "Order Zipcode", "Product Image"]  # empty or not needed
RENAMES = {"order date (DateOrders)": "order_date", "shipping date (DateOrders)": "shipping_date"}
ORDER_LEVEL_COLS = [
    "order_status", "order_date", "shipping_date", "days_for_shipping_real", "days_for_shipment_scheduled",
    "delivery_status", "late_delivery_risk", "shipping_mode", "market", "order_region", "order_country",
]
REQUIRED = ["order_id"] + ORDER_LEVEL_COLS
# Delivery Status for a delivered order is the sign of its delay: Late delivery > 0, Advance shipping < 0, on time = 0
STATUS_SIGN = {"Late delivery": 1, "Advance shipping": -1, "Shipping on time": 0}


def snake_case(name: str) -> str:
    return RENAMES.get(name) or re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def load_raw() -> pd.DataFrame:
    """The DataCo item-level file. It is not UTF-8."""
    return pd.read_csv(RAW / "dataco" / "DataCoSupplyChainDataset.csv", encoding="latin-1")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop personal and unused columns first, then snake_case the names (failing on collisions), parse the dates, and
    refuse to continue if any value is missing: missing values are never filled, their columns are dropped instead."""
    df = raw.drop(columns=[c for c in PERSONAL_COLS + UNUSED_COLS if c in raw.columns])
    names = {c: snake_case(c) for c in df.columns}
    clashes = sorted({n for n in names.values() if list(names.values()).count(n) > 1})
    if clashes:
        raise ValueError(f"column names collide after snake_case: {clashes}")
    df = df.rename(columns=names)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"DataCo file is missing expected columns: {missing}")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise ValueError(f"missing values remain after dropping the personal and unused columns: {nulls.to_dict()}; "
                         "drop the column, do not fill it")
    if "order_item_id" in df and not df["order_item_id"].is_unique:
        raise ValueError("order_item_id is not unique")
    if df.duplicated().any():
        raise ValueError(f"{int(df.duplicated().sum())} duplicated rows")
    for col in ("order_date", "shipping_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def cross_check_late(orders: pd.DataFrame) -> None:
    """For delivered orders, late (delay > 0) must agree with Late_delivery_risk, and Delivery Status must be the sign
    of the delay (Late delivery > 0, Advance shipping < 0, Shipping on time = 0). Raises with the counts if not."""
    ok = orders[~orders["is_cancelled"]]
    bad_flag = int(((ok["raw_delay_days"] > 0) != (ok["late_delivery_risk"] == 1)).sum())
    expected = ok["delivery_status"].map(STATUS_SIGN)
    unknown = int(expected.isna().sum())
    bad_status = int((expected.notna() & (expected != np.sign(ok["raw_delay_days"]))).sum())
    if bad_flag or bad_status or unknown:
        raise ValueError(f"late-flag cross-check failed on delivered orders: {bad_flag} disagree with Late_delivery_risk, "
                         f"{bad_status} disagree with Delivery Status, {unknown} have an unknown status")


def apply_cancelled_rule(orders: pd.DataFrame) -> pd.DataFrame:
    """The one rule: a cancelled order has no delivery, so no delay and no lateness."""
    orders = orders.copy()
    orders["is_cancelled"] = orders["delivery_status"] == CANCELLED_STATUS
    orders["raw_delay_days"] = orders["days_for_shipping_real"] - orders["days_for_shipment_scheduled"]
    orders["shipping_delay_days"] = orders["raw_delay_days"].where(~orders["is_cancelled"]).astype("Int64")
    orders["late"] = (orders["raw_delay_days"] > 0).astype("boolean").mask(orders["is_cancelled"])
    return orders


def to_orders(items: pd.DataFrame) -> pd.DataFrame:
    """One row per order. Shipping fields must agree within an order (raise if not); delay, late and the cross-check."""
    grouped = items.groupby("order_id")
    conflicts = {c: int((grouped[c].nunique(dropna=False) > 1).sum()) for c in ORDER_LEVEL_COLS}
    conflicts = {c: n for c, n in conflicts.items() if n}
    if conflicts:
        raise ValueError(f"Order-level fields disagree within orders: {conflicts}")
    orders = grouped[ORDER_LEVEL_COLS].first().reset_index()
    orders["n_items"] = grouped.size().to_numpy()
    orders = apply_cancelled_rule(orders)
    cross_check_late(orders)
    return orders


def sample_orders(orders: pd.DataFrame, n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Stratified by market x shipping mode, proportional allocation summing to exactly n
    (largest-remainder rounding)."""
    if n >= len(orders):
        return orders.sort_values("order_id").reset_index(drop=True)
    keys = ["market", "shipping_mode"]
    sizes = orders.groupby(keys).size()
    exact = sizes * n / len(orders)
    alloc = np.floor(exact).astype(int)
    for key in (exact - alloc).sort_values(ascending=False, kind="stable").index[: n - int(alloc.sum())]:
        alloc[key] += 1
    parts = [g.sample(n=int(alloc[key]), random_state=seed) for key, g in orders.groupby(keys)]
    return pd.concat(parts).sort_values("order_id").reset_index(drop=True)


def build_dataco(cfg: dict | None = None) -> pd.DataFrame:
    """Writes dataco_orders_all.parquet (every order: the source of the region x month aggregates) and
    dataco_orders.parquet (the 20,000-order sample that gets one evidence card each); returns the sample."""
    cfg = cfg or load_config()
    orders = to_orders(clean(load_raw()))
    sampled = sample_orders(orders, cfg["data"]["dataco_orders"], cfg["seed"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(PROCESSED / "dataco_orders_all.parquet", index=False)
    sampled.to_parquet(PROCESSED / "dataco_orders.parquet", index=False)
    return sampled


if __name__ == "__main__":
    df = build_dataco()
    print(f"dataco_orders.parquet: {len(df):,} sampled orders (dataco_orders_all.parquet holds all orders)")
