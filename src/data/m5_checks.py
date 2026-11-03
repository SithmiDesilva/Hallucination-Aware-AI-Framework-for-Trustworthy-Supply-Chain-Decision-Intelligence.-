"""M5 counterpart of the DataCo "two definitions disagree" check: where does the same observation
admit two readings, or where do the observed data and the simulation disagree?

    python -m src.data.m5_checks

Writes results/data_validation/m5_ambiguity_checks.{csv,md,json}. Each row is a candidate
conflicting-evidence question type for the evaluation slice.
"""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED, ROOT
from src.data.dataco_checks import to_markdown

OUT = ROOT / "results" / "data_validation"
LONG_ZERO_RUN = 28  # days; matches the demand window: after this the simulation's mean demand is 0
N_CALENDAR_DAYS = 1941


def _row(check, claim, n, total, note):
    share = f"{n / total * 100:.2f}%" if total else ""
    return {"check": check, "claim": claim, "n": int(n), "of": int(total), "share": share, "note": note}


def zero_runs(units: np.ndarray) -> np.ndarray:
    """Length of the zero-run each day belongs to (0 for days with sales)."""
    out = np.zeros(len(units), dtype=int)
    i = 0
    while i < len(units):
        if units[i] == 0:
            j = i
            while j < len(units) and units[j] == 0:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def run() -> pd.DataFrame:
    long = pd.read_parquet(PROCESSED / "m5_long.parquet", columns=["store_id", "item_id", "date", "wm_yr_wk", "units"])
    sim = pd.read_parquet(PROCESSED / "m5_sim.parquet")
    long = long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    rows: list[dict] = []

    # 1. A zero-sales day: no demand, or a real stockout? Observed data cannot tell; the simulation
    #    treats every zero as zero demand.
    zero = long["units"] == 0
    run_len = np.concatenate([zero_runs(g["units"].to_numpy()) for _, g in long.groupby(["store_id", "item_id"], sort=True)])
    long_run = zero.to_numpy() & (run_len >= LONG_ZERO_RUN)
    rows.append(_row("1a", "zero-sales days (no demand, or out of stock in reality?)", zero.sum(), len(long),
                     "observed data cannot distinguish the two readings"))
    rows.append(_row("1b", f"zero-sales days inside a run of >= {LONG_ZERO_RUN} zero days", long_run.sum(), zero.sum(),
                     "likely real stockout or delisting rather than lack of demand"))
    rows.append(_row("1c", "days with zero demand while simulated stock was 0 (stock 0, shortfall 0)",
                     ((sim["demand"] == 0) & (sim["on_hand_end"] == 0)).sum(), len(sim),
                     "simulated 'out of stock' that no customer noticed"))

    # 2. Observed sales vs simulated sales: the same word ("units sold") means two different numbers.
    short = sim["shortfall"] > 0
    wk = sim.assign(short=short).groupby(["store_id", "item_id", "wm_yr_wk"])["short"].any()
    rows.append(_row("2a", "days where simulated sales < observed sales (simulated shortfall)", short.sum(), len(sim),
                     "observed units_sold vs simulated sim_units_sold disagree"))
    rows.append(_row("2b", "item-weeks containing at least one such day", wk.sum(), len(wk),
                     "weekly cards: observed units_sold and shortfall appear side by side"))

    # 3. Policy collapse: after a long zero stretch mean demand is 0, ROP is 0, nothing is ordered, and the
    #    first sales afterwards become simulated shortfall. An artefact of the policy, not of the item.
    collapse = short & (sim["rop"] == 0)
    rows.append(_row("3a", "simulated shortfall days with trailing-28-day mean demand = 0 (ROP = 0)", collapse.sum(), short.sum(),
                     "shortfall caused by the policy collapsing after a zero stretch"))
    rows.append(_row("3b", "simulated shortfall units on those days", sim.loc[collapse, "shortfall"].sum(),
                     sim["shortfall"].sum(), "share of all simulated shortfall units"))

    # 4. Not on sale vs sold zero: unpriced days are dropped, so an absent week means 'not sold here', not '0 units'.
    first = long.groupby(["store_id", "item_id"])["date"].min()
    day0 = pd.Timestamp("2011-01-29")
    late_start = (first > day0).sum()
    dropped = long["store_id"].nunique() * 0 + (long.groupby(["store_id", "item_id"]).ngroups * N_CALENDAR_DAYS - len(long))
    rows.append(_row("4a", "series whose first priced day is after the first calendar day (launched late)", late_start, len(first),
                     "questions before launch are unanswerable (not on sale), not 0 units"))
    rows.append(_row("4b", "series-days dropped because the item had no price", dropped, long.groupby(["store_id", "item_id"]).ngroups * N_CALENDAR_DAYS,
                     "no gaps inside a series (checked): all dropped days are before launch"))

    # 5. 'Week' vs 'month': a Walmart week (Sat-Fri) can straddle two calendar months.
    weeks = long.drop_duplicates("wm_yr_wk").copy()
    span = long.groupby("wm_yr_wk")["date"].agg(["min", "max"])
    straddle = (span["min"].dt.to_period("M") != span["max"].dt.to_period("M")).sum()
    rows.append(_row("5a", "Walmart weeks that straddle two calendar months", straddle, len(span),
                     "monthly total != sum of weekly cards; week and month questions need explicit dates"))

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "m5_ambiguity_checks.csv", index=False)
    (OUT / "m5_ambiguity_checks.md").write_text(to_markdown(table), encoding="utf-8")
    (OUT / "m5_ambiguity_checks.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return table


if __name__ == "__main__":
    print(to_markdown(run()))
