"""M5 loader. Uses sales_train_evaluation.csv ONLY (it already contains the validation file) and its own
item_id, dept_id, cat_id, store_id and state_id columns as keys. The `id` string is never split on "_": that
gives 3 departments, 4 stores and 824 items instead of 7, 10 and 3,049.

Steps: validate the sales table -> pick 300 items (100 per category, fixed seed) and keep them in ALL 10 stores
-> wide-to-long -> join the calendar (dates, Walmart week wm_yr_wk, events with "None" for no event, SNAP) and the
prices on wm_yr_wk -> drop each series' days before its first price record (not yet on sale).

Done on purpose: week = wm_yr_wk (Saturday to Friday), never an ISO week number; no capping of sales (the maximum
stays 763) or of prices; no lag or rolling forecasting features in the data (the simulation computes its own trailing
mean and standard deviation inside the reorder-point policy only).
sample_scope runs on the WIDE table, before to_long: the full long table is ~59M rows and does not fit in 16 GB.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, load_config

DAY_PREFIX = "d_"
N_DAYS = 1941  # d_1 .. d_1941 in sales_train_evaluation.csv
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLS = [
    "d", "date", "wm_yr_wk", "weekday", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
NO_EVENT = "None"
CATEGORIES = {"FOODS", "HOBBIES", "HOUSEHOLD"}
STATES = {"CA", "TX", "WI"}
FULL_FILE = {"series": 30490, "items": 3049, "stores": 10, "departments": 7, "states": 3, "categories": 3}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three M5 CSVs: wide sales (evaluation file, d_1..d_1941), calendar, sell prices."""
    d = RAW / "m5"
    return (pd.read_csv(d / "sales_train_evaluation.csv"), pd.read_csv(d / "calendar.csv"),
            pd.read_csv(d / "sell_prices.csv"))


# ----------------------------------------------------------------------------- checks
def validate_sales(sales: pd.DataFrame, full: bool = True) -> None:
    """IDs unique and correctly formatted, hierarchy consistent, sales non-negative integers. Raises ValueError
    listing every problem. `full=True` also checks the counts of the complete evaluation file."""
    problems: list[str] = []
    day_cols = [c for c in sales.columns if c.startswith(DAY_PREFIX)]
    missing = [c for c in ["id", *ID_COLS] if c not in sales.columns]
    if missing:
        raise ValueError(f"sales table is missing columns: {missing}")
    if day_cols != [f"{DAY_PREFIX}{i}" for i in range(1, len(day_cols) + 1)] or (full and len(day_cols) != N_DAYS):
        problems.append(f"day columns are not d_1..d_{N_DAYS if full else len(day_cols)} in order")
    if not sales["id"].is_unique:
        problems.append("id is not unique")
    if sales.duplicated(["item_id", "store_id"]).any():
        problems.append("(item_id, store_id) is not unique")
    if not (sales["id"] == sales["item_id"] + "_" + sales["store_id"] + "_evaluation").all():
        problems.append("id is not item_id_store_id_evaluation")
    formats = {"item_id": r"(FOODS|HOBBIES|HOUSEHOLD)_\d_\d{3}", "dept_id": r"(FOODS|HOBBIES|HOUSEHOLD)_\d",
               "store_id": r"(CA|TX|WI)_\d", "cat_id": "|".join(sorted(CATEGORIES)), "state_id": "|".join(sorted(STATES))}
    for col, pattern in formats.items():
        bad = ~sales[col].astype(str).str.fullmatch(pattern)
        if bad.any():
            problems.append(f"{col} has {int(bad.sum())} malformed values, e.g. {sales.loc[bad, col].iloc[0]!r}")
    for child, parent in (("item_id", "dept_id"), ("dept_id", "cat_id"), ("store_id", "state_id")):
        if not all(c.startswith(p + "_") for c, p in zip(sales[child], sales[parent])):
            problems.append(f"{child} does not start with its {parent}")
    if (sales.groupby("item_id")["dept_id"].nunique() > 1).any():
        problems.append("an item belongs to more than one department")
    grid = sales[day_cols]
    if not all(t.kind in "iu" for t in grid.dtypes.unique()):
        problems.append("sales are not all integers")
    elif int(grid.to_numpy().min()) < 0:
        problems.append("sales contain negative values")
    if full:
        got = {"series": len(sales), "items": sales["item_id"].nunique(), "stores": sales["store_id"].nunique(),
               "departments": sales["dept_id"].nunique(), "states": sales["state_id"].nunique(), "categories": sales["cat_id"].nunique()}
        problems += [f"{k}: expected {v:,}, found {got[k]:,}" for k, v in FULL_FILE.items() if got[k] != v]
    if problems:
        raise ValueError("sales table failed validation: " + "; ".join(problems))


def prepare_calendar(calendar: pd.DataFrame, n_days: int | None = None) -> pd.DataFrame:
    """Parse the dates, fill missing event names and types with "None", and check the first n_days rows: d_1..d_n map to
    consecutive days, wday starts at Saturday = 1 (M5's convention, not Monday-based), SNAP flags are 0/1."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    for col in EVENT_COLS:
        cal[col] = cal[col].fillna(NO_EVENT)
    head = cal if n_days is None else cal[cal["d"].isin([f"{DAY_PREFIX}{i}" for i in range(1, n_days + 1)])]
    problems = []
    if n_days is not None and len(head) != n_days:
        problems.append(f"calendar has {len(head)} of the {n_days} days d_1..d_{n_days}")
    head = head.assign(n=head["d"].str.removeprefix(DAY_PREFIX).astype(int)).sort_values("n")
    if not (head["n"].diff().dropna() == 1).all() or (head["n"].iloc[0] != 1):
        problems.append("d_ numbers are not consecutive from d_1")
    if not (head["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        problems.append("dates are not consecutive days")
    expected_wday = (head["date"].dt.weekday - 5) % 7 + 1  # Saturday -> 1 ... Friday -> 7
    if not (head["wday"] == expected_wday).all():
        problems.append("wday does not follow Saturday = 1")
    if not (head["weekday"] == head["date"].dt.day_name()).all():
        problems.append("weekday names do not match the dates")
    if head["wm_yr_wk"].isna().any():
        problems.append("wm_yr_wk has missing values")
    if not head[["snap_CA", "snap_TX", "snap_WI"]].isin([0, 1]).all().all():
        problems.append("SNAP flags are not 0/1")
    if problems:
        raise ValueError("calendar failed validation: " + "; ".join(problems))
    return cal


# ----------------------------------------------------------------------------- build
def sample_scope(sales: pd.DataFrame, stores: list[str] | None = None, n_items: int = 300, seed: int = 42) -> pd.DataFrame:
    """Keep n_items items, stratified evenly across the item categories (100 per category for 300), in the given
    stores (configs: data.m5_stores) or, without `stores`, in ALL stores. Every item appears in every store."""
    rng = np.random.default_rng(seed)
    if stores is None:
        stores = sorted(sales["store_id"].unique())
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
    """Add date, Walmart week, events, the state's SNAP flag and the price (joined on store, item and wm_yr_wk), then
    drop each series' days before its first price record: the item was not on sale yet."""
    cal = prepare_calendar(calendar)[CALENDAR_COLS]
    long = long.merge(cal, on="d", how="left", validate="m:1")
    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]], default=0,
    ).astype("int8")
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    first = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min().rename("first_priced_wk").reset_index()
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="m:1")
    long = long.merge(first, on=["store_id", "item_id"], how="left", validate="m:1")
    no_price = long.loc[long["first_priced_wk"].isna(), ["store_id", "item_id"]].drop_duplicates()
    if len(no_price):
        raise ValueError(f"{len(no_price)} series have no price record at all, e.g. {tuple(no_price.iloc[0])}")
    long = long[long["wm_yr_wk"] >= long["first_priced_wk"]].drop(columns="first_priced_wk")
    gaps = int(long["sell_price"].isna().sum())
    if gaps:  # none in the data today; if it ever happens the rule needs a decision, not a silent guess
        raise ValueError(f"{gaps} days have no price AFTER their series' first price record")
    return long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def build_m5(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    sales, calendar, prices = load_raw()
    validate_sales(sales)
    prepare_calendar(calendar, N_DAYS)
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
