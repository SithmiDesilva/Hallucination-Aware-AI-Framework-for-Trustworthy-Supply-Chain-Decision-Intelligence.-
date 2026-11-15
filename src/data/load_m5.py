"""M5 loader: pick the 900 item-store series, then reshape wide -> long and join calendar and prices.

Order matters: sample_scope runs on the WIDE table, before to_long, because the full long table is
~59M rows and does not fit in 16 GB.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, load_config

DAY_PREFIX = "d_"
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLS = [
    "d", "date", "wm_yr_wk", "weekday", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three M5 CSVs: wide sales (evaluation file, d_1..d_1941), calendar, sell prices."""
    d = RAW / "m5"
    return (pd.read_csv(d / "sales_train_evaluation.csv"), pd.read_csv(d / "calendar.csv"),
            pd.read_csv(d / "sell_prices.csv"))


def sample_scope(sales: pd.DataFrame, stores: list[str] | None = None, n_stores: int = 3,
                 n_items: int = 300, seed: int = 42) -> pd.DataFrame:
    """Keep the chosen stores and n_items items, stratified evenly across the item categories.

    `stores` is the declared scope (configs: data.m5_stores). Without it, one store per state is drawn.
    """
    rng = np.random.default_rng(seed)
    if stores is None:
        by_state = sales[["state_id", "store_id"]].drop_duplicates().sort_values("store_id")
        stores = sorted(rng.choice(g["store_id"].to_numpy()) for _, g in by_state.groupby("state_id", sort=True))[:n_stores]
    unknown = set(stores) - set(sales["store_id"].unique())
    if unknown:
        raise ValueError(f"stores not in the M5 sales table: {sorted(unknown)}")

    items = sales[["cat_id", "item_id"]].drop_duplicates().sort_values("item_id")
    cats = sorted(items["cat_id"].unique())
    base, extra = divmod(n_items, len(cats))
    picked_items = []
    for i, cat in enumerate(cats):
        pool = items.loc[items["cat_id"] == cat, "item_id"].to_numpy()
        take = min(len(pool), base + (1 if i < extra else 0))
        picked_items.extend(rng.choice(pool, size=take, replace=False))

    out = sales[sales["store_id"].isin(stores) & sales["item_id"].isin(picked_items)]
    return out.reset_index(drop=True)


def to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Wide -> long: one row per series x day (item, store, d, units)."""
    day_cols = [c for c in sales.columns if c.startswith(DAY_PREFIX)]
    long = sales.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="units")
    long["units"] = long["units"].astype("int32")
    return long


def join_calendar_prices(long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add date, Walmart week, events, the state's SNAP flag and price; keep on-sale days only."""
    cal = calendar[CALENDAR_COLS].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    long = long.merge(cal, on="d", how="left", validate="m:1")
    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]], default=0,
    ).astype("int8")
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="m:1")
    # a missing price means the item was not on sale that week
    long = long.dropna(subset=["sell_price"])
    return long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def build_m5(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    sales, calendar, prices = load_raw()
    sampled = sample_scope(sales, stores=cfg["data"]["m5_stores"], n_items=cfg["data"]["m5_items"], seed=cfg["seed"])
    keys = sampled[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(keys, on=["store_id", "item_id"])
    long = join_calendar_prices(to_long(sampled), calendar, prices)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    long.to_parquet(PROCESSED / "m5_long.parquet", index=False)
    return long


if __name__ == "__main__":
    df = build_m5()
    print(f"m5_long.parquet: {len(df):,} rows, {df[['store_id', 'item_id']].drop_duplicates().shape[0]} series")
