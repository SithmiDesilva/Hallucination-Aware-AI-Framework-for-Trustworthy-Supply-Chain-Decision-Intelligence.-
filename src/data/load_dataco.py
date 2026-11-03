"""DataCo loader: clean, aggregate item rows to orders, sample 20,000 orders.

Definitions (also in data/README.md); proven by src/data/dataco_checks.py:
  is_cancelled        = Delivery Status == "Shipping canceled"
  raw_delay_days      = Days for shipping (real) - Days for shipment (scheduled), kept for every order
  shipping_delay_days = raw_delay_days, NULL for cancelled orders (nothing was delivered)
  late                = Late_delivery_risk == 1, NULL for cancelled orders
  status              = Delivery Status as given
The raw "Days for shipping (real)" and "Late_delivery_risk" columns are kept untouched.
"""
import pandas as pd

from src.config import PROCESSED, RAW, load_config

CANCELLED_STATUS = "Shipping canceled"
PII_COLS = [
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Product Image",
]
ORDER_LEVEL_COLS = [
    "Order Status", "order date (DateOrders)", "shipping date (DateOrders)", "Days for shipping (real)",
    "Days for shipment (scheduled)", "Delivery Status", "Late_delivery_risk",
    "Shipping Mode", "Market", "Order Region", "Order Country",
]
REQUIRED = ["Order Id"] + ORDER_LEVEL_COLS


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise KeyError(f"DataCo file is missing expected columns: {missing}")
    df = raw.drop(columns=[c for c in PII_COLS if c in raw.columns])
    for col in ("order date (DateOrders)", "shipping date (DateOrders)"):
        df[col] = pd.to_datetime(df[col])
    return df


def apply_cancelled_rule(orders: pd.DataFrame) -> pd.DataFrame:
    """The one rule: a cancelled order has no delivery, so no delay and no lateness."""
    orders = orders.copy()
    orders["is_cancelled"] = orders["Delivery Status"] == CANCELLED_STATUS
    orders["raw_delay_days"] = orders["Days for shipping (real)"] - orders["Days for shipment (scheduled)"]
    orders["shipping_delay_days"] = orders["raw_delay_days"].where(~orders["is_cancelled"]).astype("Int64")
    orders["late"] = (orders["Late_delivery_risk"] == 1).astype("boolean").mask(orders["is_cancelled"])
    return orders


def to_orders(items: pd.DataFrame) -> pd.DataFrame:
    """One row per order. Shipping fields must agree within an order; raise if they do not."""
    grouped = items.groupby("Order Id")
    conflicts = {c: int((grouped[c].nunique(dropna=False) > 1).sum()) for c in ORDER_LEVEL_COLS}
    conflicts = {c: n for c, n in conflicts.items() if n}
    if conflicts:
        raise ValueError(f"Order-level fields disagree within orders: {conflicts}")
    orders = grouped[ORDER_LEVEL_COLS].first().reset_index()
    orders["n_items"] = grouped.size().to_numpy()
    return apply_cancelled_rule(orders)


def sample_orders(orders: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified by market x shipping mode, proportional allocation."""
    strata = orders.groupby(["Market", "Shipping Mode"], group_keys=False)
    frac = min(1.0, n / len(orders))
    out = strata.sample(frac=frac, random_state=seed)
    return out.sort_values("Order Id").reset_index(drop=True)


def descriptive_table(orders: pd.DataFrame) -> pd.DataFrame:
    return (
        orders.groupby(["Market", "Shipping Mode"])
        .agg(orders=("Order Id", "size"), cancelled=("is_cancelled", "sum"),
             late_rate=("late", "mean"), mean_delay=("shipping_delay_days", "mean"))
        .round(3).reset_index()
    )


def load_items() -> pd.DataFrame:
    return clean(pd.read_csv(RAW / "dataco" / "DataCoSupplyChainDataset.csv", encoding="latin-1"))


def build_dataco(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    orders = to_orders(load_items())
    sampled = sample_orders(orders, cfg["dataco"]["n_orders"], cfg["seed"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(PROCESSED / "dataco_orders.parquet", index=False)
    descriptive_table(sampled).to_csv(PROCESSED / "dataco_descriptive.csv", index=False)
    return sampled


if __name__ == "__main__":
    df = build_dataco()
    print(f"dataco_orders.parquet: {len(df):,} orders")
