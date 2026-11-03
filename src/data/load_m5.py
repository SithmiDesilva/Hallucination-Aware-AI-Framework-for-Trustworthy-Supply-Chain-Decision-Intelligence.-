"""M5 loader: sample 900 item-store series, then reshape wide -> long.

Filtering happens BEFORE the reshape: the full long table is ~59M rows and does not fit in 16 GB.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, load_config

DAY_PREFIX = "d_"


def sample_series(sales: pd.DataFrame, n_stores: int, n_items: int, seed: int) -> pd.DataFrame:
    """One store per state where possible, items stratified across the three categories."""
    rng = np.random.default_rng(seed)
    stores = (
        sales[["state_id", "store_id"]].drop_duplicates().sort_values("store_id")
    )
    per_state = [
        rng.choice(g["store_id"].to_numpy())
        for _, g in stores.groupby("state_id", sort=True)
    ]
    picked_stores = sorted(per_state)[:n_stores]

    items = sales[["cat_id", "item_id"]].drop_duplicates().sort_values("item_id")
    cats = sorted(items["cat_id"].unique())
    base, extra = divmod(n_items, len(cats))
    picked_items = []
    for i, cat in enumerate(cats):
        pool = items.loc[items["cat_id"] == cat, "item_id"].to_numpy()
        take = min(len(pool), base + (1 if i < extra else 0))
        picked_items.extend(rng.choice(pool, size=take, replace=False))

    out = sales[sales["store_id"].isin(picked_stores) & sales["item_id"].isin(picked_items)]
    return out.reset_index(drop=True)


def to_long(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in sales.columns if c.startswith(DAY_PREFIX)]
    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols, var_name="d", value_name="units",
    )
    long["units"] = long["units"].astype("int32")
    cal = calendar[[
        "d", "date", "wm_yr_wk", "weekday", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
    ]].copy()
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
    sales = pd.read_csv(RAW / "m5" / "sales_train_evaluation.csv")
    calendar = pd.read_csv(RAW / "m5" / "calendar.csv")
    prices = pd.read_csv(RAW / "m5" / "sell_prices.csv")
    sampled = sample_series(sales, cfg["m5"]["n_stores"], cfg["m5"]["n_items"], cfg["seed"])
    keys = sampled[["store_id", "item_id"]].drop_duplicates()
    prices = prices.merge(keys, on=["store_id", "item_id"])
    long = to_long(sampled, calendar, prices)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    long.to_parquet(PROCESSED / "m5_long.parquet", index=False)
    return long


if __name__ == "__main__":
    df = build_m5()
    print(f"m5_long.parquet: {len(df):,} rows, {df[['store_id', 'item_id']].drop_duplicates().shape[0]} series")
