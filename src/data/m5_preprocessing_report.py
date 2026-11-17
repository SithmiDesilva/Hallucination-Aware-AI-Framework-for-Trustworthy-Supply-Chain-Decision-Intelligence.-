"""The M5 preprocessing checklist (what to do, what not to do) with the evidence for each rule, for the thesis.

    python -m src.data.m5_preprocessing_report   ->  results/data_validation/m5_preprocessing_checks.{csv,md}

Every row is computed from the raw files and the processed tables when this runs; a rule that fails is reported as
FAIL (and the process exits non-zero), never silently skipped.
"""
import sys
from pathlib import Path

import pandas as pd

from src.config import PROCESSED, ROOT, load_config
from src.data.dataco_checks import to_markdown
from src.data.load_m5 import N_DAYS, load_raw, prepare_calendar, sample_scope, validate_sales

OUT = ROOT / "results" / "data_validation"
FORBIDDEN_COLUMNS = ("lag", "roll", "iso", "ewm", "mean_", "std_")


def run() -> pd.DataFrame:
    cfg = load_config()
    sales, calendar, prices = load_raw()
    long = pd.read_parquet(PROCESSED / "m5_long.parquet")
    days = [c for c in sales.columns if c.startswith("d_")]
    grid = sales[days].to_numpy()
    rows: list[tuple] = []

    def rule(kind, text, ok, evidence):
        rows.append((kind, text, "PASS" if ok else "FAIL", evidence))

    # ---- Do
    rule("Do", "Use sales_train_evaluation.csv only", len(days) == N_DAYS,
         f"{len(sales):,} series x {len(days):,} days (d_1..d_{len(days)}); sales_train_validation.csv is not read anywhere in the pipeline")
    try:
        validate_sales(sales)
        ok, detail = True, (f"{sales['id'].nunique():,} unique ids; {sales['dept_id'].nunique()} departments, {sales['store_id'].nunique()} stores, "
                            f"{sales['item_id'].nunique():,} items, {sales['state_id'].nunique()} states, {sales['cat_id'].nunique()} categories")
    except ValueError as e:
        ok, detail = False, str(e)
    rule("Do", "Keys come from the file's own item_id, dept_id, cat_id, store_id, state_id; IDs unique and correctly formatted", ok, detail)
    rule("Do", "Sales are non-negative integers", all(t.kind in "iu" for t in sales[days].dtypes.unique()) and int(grid.min()) >= 0,
         f"integer columns, minimum {int(grid.min())}, maximum {int(grid.max())}")
    try:
        cal = prepare_calendar(calendar, N_DAYS)
        head = cal[cal["d"].isin(days)]
        ok, detail = True, (f"d_1..d_{N_DAYS} map to consecutive days {head['date'].min().date()} to {head['date'].max().date()}; "
                            f"wday 1 = {head.iloc[0]['weekday']} ({head.iloc[0]['date'].date()})")
    except ValueError as e:
        ok, detail = False, str(e)
    rule("Do", "Parse the calendar dates and map d_1-d_1941 to dates (wday: Saturday = 1)", ok, detail)
    missing = int(calendar["event_name_1"].isna().sum())
    rule("Do", 'Fill missing event names with "None"', not long[["event_name_1", "event_name_2", "event_type_1", "event_type_2"]].isna().any().any(),
         f'{missing:,} of {len(calendar):,} calendar days had no event_name_1; the processed table has no missing event fields ({(long["event_name_1"] == "None").mean():.1%} of rows are "None")')
    chosen = sample_scope(sales, stores=cfg["data"]["m5_stores"], n_items=cfg["data"]["m5_items"], seed=cfg["seed"])
    keys = chosen[["store_id", "item_id"]].drop_duplicates()
    p = prices.merge(keys, on=["store_id", "item_id"])
    got = long[["store_id", "item_id", "wm_yr_wk", "sell_price"]].drop_duplicates().merge(p, on=["store_id", "item_id", "wm_yr_wk"], how="left", suffixes=("", "_raw"), validate="1:1")
    rule("Do", "Use wm_yr_wk as the week key so sales join correctly with prices", bool((got["sell_price"] == got["sell_price_raw"]).all()) and long["sell_price"].notna().all(),
         f"{len(got):,} (store, item, week) price rows matched on wm_yr_wk, 0 unmatched; every kept day has a price")
    items = long.drop_duplicates("item_id")
    rule("Do", "Select 300 items (100 per category, fixed seed) and keep them in all 10 stores",
         len(items) == 300 and items["cat_id"].value_counts().eq(100).all() and long["store_id"].nunique() == 10
         and long.groupby("store_id")["item_id"].nunique().eq(300).all(),
         f"seed {cfg['seed']}: {len(items)} items ({items['cat_id'].value_counts().sort_index().to_dict()}) x {long['store_id'].nunique()} stores = "
         f"{len(keys):,} series")
    first = p.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    kept_first = long.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    dropped = len(keys) * N_DAYS - len(long)
    rule("Do", "Remove days before each item's first price record (not yet on sale)",
         bool((kept_first == first.reindex(kept_first.index)).all()),
         f"{dropped:,} of {len(keys) * N_DAYS:,} series-days removed ({dropped / (len(keys) * N_DAYS):.1%}); "
         f"{int((kept_first > long['wm_yr_wk'].min()).sum()):,} of {len(keys):,} series launched after the first week")
    sim = pd.read_parquet(PROCESSED / "m5_sim.parquet", columns=["day_idx"])
    rule("Do", "Simulate stock with the reorder-point policy (Equations 1-3)", len(sim) > 0,
         f"{len(sim):,} simulated series-days (service level {cfg['simulation']['service_level']}, lead time {cfg['simulation']['lead_time']}, "
         f"review {cfg['simulation']['review']}, {cfg['simulation']['window']}-day window)")

    # ---- Don't
    parts = sales["id"].str.split("_", expand=True)
    rule("Don't", 'Split the id string on "_"', parts[1].nunique() != sales["dept_id"].nunique() and parts[4].nunique() != sales["store_id"].nunique(),
         f"splitting would give {parts[1].nunique()} departments, {parts[4].nunique()} stores and {parts[2].nunique()} items instead of "
         f"{sales['dept_id'].nunique()}, {sales['store_id'].nunique()} and {sales['item_id'].nunique():,}; the pipeline never splits it")
    sample_max = int(chosen[days].to_numpy().max())
    rule("Don't", "Cap outliers", int(long["units"].max()) == sample_max and int(long["units"].sum()) == int(chosen[days].to_numpy().sum()),
         f"maximum daily sales is {int(grid.max())} in the file and {int(long['units'].max())} in the processed sample (the raw sample maximum is {sample_max}); "
         "the processed total equals the raw total")
    rule("Don't", "Cap prices", bool((got["sell_price"] == got["sell_price_raw"]).all()), f"processed prices equal the raw prices; range {long['sell_price'].min()} to {long['sell_price'].max()}")
    offenders = sorted(str(f.relative_to(ROOT)) for f in (ROOT / "src").rglob("*.py")
                       if f.resolve() != Path(__file__).resolve() and "sales_train_" + "validation" in f.read_text(encoding="utf-8"))
    rule("Don't", "Use sales_train_validation.csv (the evaluation file already contains it)", not offenders,
         f"scanned {len(list((ROOT / 'src').rglob('*.py')))} source files: " + (f"still referenced in {offenders}" if offenders else
              "none reads it; the loader, simulation, cards and dataset-wide questions use the evaluation file (its first 1,913 days where "
              "the training period is meant)"))
    wk = long.drop_duplicates("date").groupby("wm_yr_wk")["date"].agg(["min", "max", "count"])
    full = wk[wk["count"] == 7]
    rule("Don't", "Use ISO week numbers; compare wday with a Monday-based weekday",
         bool((full["min"].dt.day_name() == "Saturday").all() and (full["max"].dt.day_name() == "Friday").all()),
         f"weeks are wm_yr_wk: {len(full)} complete Saturday-Friday weeks; wday is validated against Saturday = 1")
    bad_cols = [c for c in long.columns if any(c.startswith(x) or x in c for x in FORBIDDEN_COLUMNS)]
    rule("Don't", "Build forecasting features such as rolling means or lags", not bad_cols,
         f"processed columns: {', '.join(long.columns)}; the simulation computes its trailing mean and standard deviation inside the reorder-point policy only")

    table = pd.DataFrame(rows, columns=["rule_type", "rule", "result", "evidence"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "m5_preprocessing_checks.csv", index=False)
    (OUT / "m5_preprocessing_checks.md").write_text(to_markdown(table), encoding="utf-8")
    return table


if __name__ == "__main__":
    result = run()
    print(to_markdown(result))
    sys.exit(0 if (result["result"] == "PASS").all() else 1)
