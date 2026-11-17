"""Dataset-wide questions: whole-period totals, rankings, calendar and structure facts, with real answers.

    python -m src.bench.dataset_questions   ->  benchmark/manual/b0_dataset_wide_v1.{jsonl,csv,md}

These ask about the FULL raw data (all 10 M5 stores, 3,049 items, every day; all 65,752 DataCo orders), not the
sampled evidence cards, so the answer is in no card: scope="dataset". B0 (no evidence at all) can be tried on them,
and a RAG system is expected to abstain. M5 uses sales_train_evaluation.csv ONLY (it already contains the validation
file), restricted to its first 1,913 days, 29 Jan 2011 to 24 Apr 2016, the competition's training period. The date
range is stated in every question because the file has 28 more days and "the whole period" would otherwise be ambiguous.

Verification: every fact is computed twice by independent methods and must be identical.
  M5      grouped sums (m5_facts)       vs   boolean masks and explicit column picks (m5_check_facts), same file
  DataCo  the raw file read directly   vs   the same file read through the cleaning pipeline (load_dataco)
Questions and answers are then generated from the facts, so one check covers every answer.
"""
import csv
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.cards.schemas import GoldFact, Question, Tolerance, write_jsonl
from src.config import RAW, ROOT
from src.data.load_dataco import CANCELLED_STATUS, clean, to_orders

SEED = 42
OUT = ROOT / "benchmark" / "manual"
PERIOD_DAYS = 1913
M5_PERIOD = "29 January 2011 to 24 April 2016"
DC_PERIOD = "1 January 2015 to 31 January 2018"
STATE_NAME = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
F_AGG, F_RANK, F_TEMP, F_ITEM, F_FACT = ("Dataset-wide aggregation", "Dataset-wide ranking and comparison", "Dataset-wide temporal",
                                         "Dataset-wide item-level", "Dataset facts")
_SINGULAR = re.compile(r"(?<![\d.])1 (units|days|orders|stores|items|states|regions|markets)\b")


def fmt(n) -> str:
    return f"{int(n):,}"


def pct(part, whole) -> str:
    return f"{part / whole * 100:.1f}%"


def month_label(ym: str) -> str:
    return f"{MONTHS[int(ym[5:7]) - 1]} {ym[:4]}"


def tidy(s: str) -> str:
    s = re.sub(r"\s+([,.;:)])", r"\1", " ".join(s.replace("\n", "\x00").split())).replace("\x00 ", "\n").replace("\x00", "\n")
    return _SINGULAR.sub(lambda m: "1 " + m.group(1).removesuffix("s"), s)


# ----------------------------------------------------------------------------- facts: M5
def load_m5_sales() -> pd.DataFrame:
    path = RAW / "m5" / "sales_train_evaluation.csv"
    cols = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: "int32" for c in cols if c.startswith("d_")})


def m5_facts(sales: pd.DataFrame, cal: pd.DataFrame) -> dict:
    days = [f"d_{i}" for i in range(1, PERIOD_DAYS + 1)]
    ids = sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].reset_index(drop=True)
    units = sales[days].to_numpy()
    total = units.sum(axis=1, dtype=np.int64)
    c = cal.set_index("d").loc[days]
    dates = pd.DatetimeIndex(pd.to_datetime(c["date"]))
    series = ids.assign(total=total)
    stores = sorted(ids.store_id.unique())
    daily_store = pd.DataFrame({s: units[(ids.store_id == s).to_numpy()].sum(axis=0, dtype=np.int64) for s in stores}, index=dates)
    daily = daily_store.sum(axis=1)
    month_key = dates.strftime("%Y-%m")
    store_month = daily_store.groupby(month_key).sum()
    days_in_month = pd.Series(1, index=dates).groupby(month_key).sum()
    complete = [m for m in store_month.index if days_in_month[m] == pd.Period(m).days_in_month]
    events: dict[str, list[str]] = {}
    for col in ("event_name_1", "event_name_2"):
        for d, name in zip(dates, c[col]):
            if isinstance(name, str):
                events.setdefault(name, []).append(str(d.date()))
    store_state = ids.drop_duplicates("store_id").set_index("store_id").state_id
    return {
        "n_series": len(ids), "n_stores": len(stores), "n_items": int(ids.item_id.nunique()), "n_states": int(ids.state_id.nunique()),
        "n_days": len(days), "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
        "first_weekday": dates[0].day_name(), "last_weekday": dates[-1].day_name(),
        "store_total": series.groupby("store_id").total.sum(), "state_total": series.groupby("state_id").total.sum(),
        "cat_total": series.groupby("cat_id").total.sum(), "dept_total": series.groupby("dept_id").total.sum(),
        "items_per_cat": ids.drop_duplicates("item_id").cat_id.value_counts().sort_index(),
        "stores_per_state": store_state.value_counts().sort_index(),
        "state_stores": {st: sorted(store_state[store_state == st].index) for st in sorted(store_state.unique())},
        "store_month": store_month.loc[complete], "all_month": store_month.loc[complete].sum(axis=1),
        "daily": daily, "weekday_mean": daily.groupby(dates.day_name()).mean(), "year_total": daily.groupby(dates.year).sum(),
        "item_total": series.groupby("item_id").total.sum(), "store_item_total": series.set_index(["store_id", "item_id"]).total,
        "item_cat": ids.drop_duplicates("item_id").set_index("item_id").cat_id, "zero_series": int((total == 0).sum()),
        "events": events, "snap_ca_2015": int(((dates.year == 2015) & (c["snap_CA"].to_numpy() == 1)).sum()),
    }


def m5_check_facts(sales: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """The main totals computed a second way (boolean masks and explicit column picks, no groupby, no shared helper) to
    cross-check m5_facts on the same file. Returns a subset of m5_facts' keys."""
    days = [f"d_{i}" for i in range(1, PERIOD_DAYS + 1)]
    grid = sales[days].to_numpy(dtype=np.int64)
    by_row = grid.sum(axis=1)
    col = {c: sales[c].to_numpy() for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id")}
    dates = pd.DatetimeIndex(pd.to_datetime(cal.set_index("d").loc[days]["date"]))

    def totals(name: str) -> pd.Series:
        values = sorted(set(col[name]))
        return pd.Series([int(by_row[col[name] == v].sum()) for v in values], index=values)

    stores = sorted(set(col["store_id"]))
    day_by_store = {s: grid[col["store_id"] == s].sum(axis=0) for s in stores}
    month = np.array(dates.strftime("%Y-%m"))
    complete = [m for m in sorted(set(month)) if (month == m).sum() == pd.Period(m).days_in_month]
    store_month = pd.DataFrame({s: [int(day_by_store[s][month == m].sum()) for m in complete] for s in stores}, index=complete)
    codes, items = pd.factorize(col["item_id"], sort=True)
    item_total = np.zeros(len(items), dtype=np.int64)
    np.add.at(item_total, codes, by_row)
    daily = pd.Series(grid.sum(axis=0), index=dates)
    names, years = np.array(dates.day_name()), np.array(dates.year)
    return {
        "store_total": totals("store_id"), "state_total": totals("state_id"), "cat_total": totals("cat_id"), "dept_total": totals("dept_id"),
        "item_total": pd.Series(item_total, index=items), "store_month": store_month, "all_month": store_month.sum(axis=1),
        "daily": daily, "weekday_mean": pd.Series({n: daily.to_numpy()[names == n].mean() for n in sorted(set(names))}),
        "year_total": pd.Series({int(y): int(daily.to_numpy()[years == y].sum()) for y in sorted(set(years))}),
        "zero_series": int((by_row == 0).sum()), "n_series": len(by_row), "n_items": len(items), "n_stores": len(stores),
    }


# ----------------------------------------------------------------------------- facts: DataCo
def _region(s) -> str:
    return " ".join(str(s).split())


def dc_frame_raw(raw: pd.DataFrame) -> pd.DataFrame:
    o = raw.drop_duplicates("Order Id")
    return pd.DataFrame({
        "order_id": o["Order Id"].to_numpy(), "date": pd.to_datetime(o["order date (DateOrders)"]).dt.normalize().to_numpy(),
        "market": o["Market"].to_numpy(), "region": [_region(x) for x in o["Order Region"]], "mode": o["Shipping Mode"].to_numpy(),
        "cancelled": (o["Delivery Status"] == CANCELLED_STATUS).to_numpy(), "late": (o["Late_delivery_risk"] == 1).to_numpy(),
        "delay": (o["Days for shipping (real)"] - o["Days for shipment (scheduled)"]).to_numpy(),
    }).sort_values("order_id").reset_index(drop=True)


def dc_frame_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    o = to_orders(clean(raw))
    return pd.DataFrame({
        "order_id": o["order_id"].to_numpy(), "date": o["order_date"].dt.normalize().to_numpy(),
        "market": o["market"].to_numpy(), "region": [_region(x) for x in o["order_region"]], "mode": o["shipping_mode"].to_numpy(),
        "cancelled": o["is_cancelled"].to_numpy(), "late": o["late"].fillna(False).astype(bool).to_numpy(),
        "delay": o["raw_delay_days"].to_numpy(),
    }).sort_values("order_id").reset_index(drop=True)


def _ranked(s: pd.Series) -> pd.Series:
    return s.sort_index().sort_values(ascending=False, kind="stable")


def dc_facts(frame: pd.DataFrame, n_lines: int) -> dict:
    ok = frame[~frame.cancelled]
    return {
        "n_orders": len(frame), "n_lines": n_lines, "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
        "n_markets": int(frame.market.nunique()), "n_regions": int(frame.region.nunique()),
        "market_orders": _ranked(frame.market.value_counts()), "region_orders": _ranked(frame.region.value_counts()),
        "mode_orders": _ranked(frame["mode"].value_counts()),
        "cancelled": int(frame.cancelled.sum()), "delivered": len(ok), "late": int(ok.late.sum()),
        "mode_delay": ok.groupby("mode").delay.mean().round(2), "mode_late_rate": ok.groupby("mode").late.mean().round(4),
        "month_orders": frame.groupby(frame.date.dt.strftime("%Y-%m")).size(), "year_orders": frame.groupby(frame.date.dt.year).size(),
    }


def same(a, b) -> bool:
    """Equal labels and values (floats to 1e-12, so two valid summation orders do not disagree)."""
    if isinstance(a, (pd.Series, pd.DataFrame)):
        if not isinstance(b, type(a)) or not a.index.equals(b.index) or (isinstance(a, pd.DataFrame) and not a.columns.equals(b.columns)):
            return False
        return bool(np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-12, atol=0))
    return a == b


def disagreements(a: dict, b: dict) -> list[str]:
    return [k for k in a if not same(a[k], b[k])] + [k for k in b if k not in a]


def mismatches(full: dict, check: dict) -> list[str]:
    """Keys where the cross-check (which covers a subset of the facts) disagrees with the full computation."""
    return [k for k in check if k not in full or not same(full[k], check[k])]


# ----------------------------------------------------------------------------- questions
def G(ds: str, attr: str, value, typ: str = "numerical", **key) -> GoldFact:
    value = value.item() if hasattr(value, "item") else value
    tol = None if typ != "numerical" else Tolerance(abs=0) if isinstance(value, int) else Tolerance(abs=0.05 if attr == "share_percent" else 0.005)
    return GoldFact(key={"dataset": ds, **{k: str(v) for k, v in key.items()}}, attribute=attr, value=value, type=typ, tolerance=tol)


def unique_top(s: pd.Series, k: int = 1):
    """The k largest entries; fails if the k-th ties with the next one (every ranking here must be unambiguous)."""
    srt = s.sort_values(ascending=False, kind="stable")
    assert len(srt) <= k or srt.iloc[k - 1] != srt.iloc[k], f"tie at rank {k} in {list(srt.index[:k + 1])}"
    return srt.iloc[:k]


class Bank:
    def __init__(self):
        self.qs: list[Question] = []
        self.n = Counter()

    def add(self, ds: str, family: str, template: str, qtype: str, text: str, ref: str, facts: list[GoldFact]):
        prefix = "M5" if ds == "m5" else "DC"
        self.n[prefix] += 1
        self.qs.append(Question(
            qid=f"DW-{prefix}-{self.n[prefix]:03d}", split="dev", domain="inventory" if ds == "m5" else "delivery",
            template_id=f"{family}/{template}", question_type=qtype, answerable=True, question=tidy(text), reference_answer=tidy(ref),
            gold_facts=facts, scope="dataset"))


def m5_questions(b: Bank, f: dict, rng: random.Random) -> None:
    PRE = f"In the M5 sales data ({M5_PERIOD}), "
    PRE0 = "In the M5 sales data, "  # for facts that do not depend on the period
    g = lambda *a, **k: G("m5", *a, **k)
    st, state, cat, dept = f["store_total"], f["state_total"], f["cat_total"], f["dept_total"]
    grand = int(st.sum())
    assert grand == int(state.sum()) == int(cat.sum()) == int(dept.sum()), "totals disagree across groupings"

    # ---- stores
    top, bot = unique_top(st).index[0], unique_top(-st).index[0]
    b.add("m5", F_RANK, "store_most_least", "comparison", PRE + "which store sold the most units over the whole period, and which sold the least?",
          f"Most: {top}, {fmt(st[top])} units.\nLeast: {bot}, {fmt(st[bot])} units.",
          [g("total_units", st[top], store_id=top, rank="most"), g("total_units", st[bot], store_id=bot, rank="least"),
           g("store_most", top, "entity"), g("store_least", bot, "entity")])
    for s in rng.sample(sorted(st.index), 4):
        b.add("m5", F_AGG, "store_total", "aggregation", PRE + f"how many units did store {s} sell over the whole period?",
              f"{fmt(st[s])} units.", [g("total_units", st[s], store_id=s)])
    t3 = unique_top(st, 3)
    b.add("m5", F_RANK, "top_three_stores", "comparison", PRE + "which three stores sold the most units over the whole period, and how many did each sell?",
          "; ".join(f"{s}: {fmt(v)} units" for s, v in t3.items()) + ".", [g("total_units", v, store_id=s, rank=i + 1) for i, (s, v) in enumerate(t3.items())])

    # ---- states
    ts = unique_top(state, 3)
    b.add("m5", F_RANK, "state_most", "comparison", PRE + "which state sold the most units overall?",
          f"{ts.index[0]}, with {fmt(ts.iloc[0])}, ahead of {ts.index[1]} ({fmt(ts.iloc[1])}) and {ts.index[2]} ({fmt(ts.iloc[2])}).",
          [g("state_most", ts.index[0], "entity")] + [g("total_units", v, state_id=s) for s, v in ts.items()])
    for s in ("TX", "WI"):
        b.add("m5", F_AGG, "state_total", "aggregation",
              PRE + f"how many units were sold by the {STATE_NAME[s]} stores ({', '.join(f['state_stores'][s])}) over the whole period?",
              f"{fmt(state[s])} units.", [g("total_units", state[s], state_id=s)])
    diff = int(state["CA"] - state["TX"])
    b.add("m5", F_RANK, "state_difference", "comparison", PRE + "how many more units did the California stores sell than the Texas stores over the whole period?",
          f"{fmt(diff)} more units (California {fmt(state['CA'])}, Texas {fmt(state['TX'])}).",
          [g("total_units", state["CA"], state_id="CA"), g("total_units", state["TX"], state_id="TX"), g("difference", diff, state_id="CA-TX")])
    b.add("m5", F_AGG, "state_share", "aggregation", PRE + "what share of all units sold came from the California stores?",
          f"{pct(state['CA'], grand)} ({fmt(state['CA'])} of {fmt(grand)} units).", [g("share_percent", round(state["CA"] / grand * 100, 1), state_id="CA")])

    # ---- categories and departments
    others = [c for c in cat.sort_values(ascending=False).index if c != "FOODS"]
    b.add("m5", F_AGG, "category_total", "aggregation", PRE + "how many total units did the FOODS category sell?",
          f"{fmt(cat['FOODS'])} units, against {fmt(cat[others[0]])} for {others[0]} and {fmt(cat[others[1]])} for {others[1]}.",
          [g("total_units", cat[c], cat_id=c) for c in ("FOODS", *others)])
    low = unique_top(-cat).index[0]
    b.add("m5", F_RANK, "category_fewest", "comparison", PRE + "which product category sold the fewest units over the whole period, and how many?",
          f"{low}, {fmt(cat[low])} units.", [g("category_fewest", low, "entity"), g("total_units", cat[low], cat_id=low)])
    for d in ("FOODS_3", "HOUSEHOLD_1", "HOBBIES_1"):
        b.add("m5", F_AGG, "department_total", "aggregation", PRE + f"how many units did the {d} department sell over the whole period?",
              f"{fmt(dept[d])} units.", [g("total_units", dept[d], dept_id=d)])
    td = unique_top(dept).index[0]
    b.add("m5", F_RANK, "department_most", "comparison", PRE + "which department sold the most units over the whole period, and what share of all units was that?",
          f"{td}, {fmt(dept[td])} units ({pct(dept[td], grand)} of all units).",
          [g("department_most", td, "entity"), g("total_units", dept[td], dept_id=td), g("share_percent", round(dept[td] / grand * 100, 1), dept_id=td)])
    b.add("m5", F_AGG, "category_share", "aggregation", PRE + "what percentage of all units sold were in the FOODS category?",
          f"{pct(cat['FOODS'], grand)} ({fmt(cat['FOODS'])} of {fmt(grand)} units).", [g("share_percent", round(cat["FOODS"] / grand * 100, 1), cat_id="FOODS")])
    ipc = f["items_per_cat"]
    b.add("m5", F_FACT, "items_per_category", "aggregation", PRE0 + "how many distinct items does each product category have?",
          "; ".join(f"{c}: {fmt(n)} items" for c, n in ipc.sort_values(ascending=False).items()) + ".", [g("n_items", n, cat_id=c) for c, n in ipc.items()])

    # ---- store x month
    sm = f["store_month"]
    first = ("CA_1", "2012-07")  # the example you gave: 126,739 units
    pairs = [(s, m) for s in sm.columns for m in sm.index if (s, m) != first]
    for s, m in [first] + rng.sample(pairs, 7):
        b.add("m5", F_AGG, "store_month_total", "aggregation", PRE + f"how many units did store {s} sell in {month_label(m)}?",
              f"{fmt(sm.loc[m, s])} units.", [g("total_units", sm.loc[m, s], store_id=s, month=m)])
    for s in rng.sample(sorted(sm.columns), 2):
        col = sm[s]
        m = unique_top(col).index[0]
        b.add("m5", F_TEMP, "store_peak_month", "comparison", PRE + f"in which month (counting complete calendar months only) did store {s} sell the most units, and how many?",
              f"{month_label(m)}, {fmt(col[m])} units.", [g("peak_month", m, "temporal", store_id=s), g("total_units", col[m], store_id=s, month=m)])
    am = f["all_month"]
    for m in rng.sample(list(am.index), 2):
        b.add("m5", F_AGG, "all_stores_month_total", "aggregation", PRE + f"how many units did all stores together sell in {month_label(m)}?",
              f"{fmt(am[m])} units.", [g("total_units", am[m], month=m)])

    # ---- temporal and calendar
    daily = f["daily"]
    best = unique_top(daily).index[0]
    b.add("m5", F_TEMP, "busiest_day", "comparison", PRE + "on which date were the most units sold across all stores, and how many?",
          f"{best.date()}, {fmt(daily[best])} units.", [g("busiest_date", str(best.date()), "temporal"), g("total_units", daily[best], date=best.date())])
    yt = f["year_total"].loc[2012:2015]
    y = unique_top(yt).index[0]
    b.add("m5", F_TEMP, "best_year", "comparison", PRE + "which calendar year had the most units sold among 2012, 2013, 2014 and 2015?",
          f"{y}, {fmt(yt[y])} units (" + ", ".join(f"{k}: {fmt(v)}" for k, v in yt.items() if k != y) + ").",
          [g("best_year", int(y), "temporal")] + [g("total_units", v, year=k) for k, v in yt.items()])
    wm = f["weekday_mean"].round(0).astype(int)
    hi, lo = unique_top(wm).index[0], unique_top(-wm).index[0]
    b.add("m5", F_TEMP, "weekday_extremes", "comparison", PRE + "which day of the week had the highest average daily sales across all stores, and which the lowest?",
          f"Highest: {hi}, about {fmt(wm[hi])} units a day.\nLowest: {lo}, about {fmt(wm[lo])} units a day.",
          [g("weekday_highest", hi, "entity"), g("weekday_lowest", lo, "entity")])
    for d in ("2012-12-25", "2013-12-25", "2014-11-27", "2015-07-04"):
        day = pd.Timestamp(d)
        b.add("m5", F_TEMP, "units_on_date", "lookup", PRE + f"how many units were sold across all stores on {d}?",
              f"{fmt(daily[day])} units.", [g("total_units", daily[day], date=d)])
    ev = f["events"]
    for name, label, year in (("SuperBowl", "the Super Bowl", 2016), ("Easter", "Easter", 2014), ("Easter", "Easter", 2015), ("Thanksgiving", "Thanksgiving", 2013)):
        dates = [d for d in ev[name] if d.startswith(str(year))]
        assert len(dates) == 1, (name, year, dates)
        b.add("m5", F_TEMP, "calendar_event_date", "lookup", PRE + f"on what date was {label} in {year} according to the M5 calendar?",
              f"{dates[0]}.", [g("event_date", dates[0], "temporal", event=name, year=year)])
    b.add("m5", F_TEMP, "snap_days", "aggregation", PRE + "on how many days in 2015 did SNAP benefits apply in California?",
          f"{f['snap_ca_2015']} days.", [g("snap_days", f["snap_ca_2015"], state_id="CA", year=2015)])

    # ---- items
    it = f["item_total"]
    t1 = unique_top(it).index[0]
    b.add("m5", F_ITEM, "best_selling_item", "comparison", PRE + "which item sold the most units across all stores over the whole period, and how many?",
          f"{t1}, {fmt(it[t1])} units.", [g("best_item", t1, "entity"), g("total_units", it[t1], item_id=t1)])
    sit = f["store_item_total"]
    for s in rng.sample(sorted(st.index), 2):
        col = sit.loc[s]
        i = unique_top(col).index[0]
        b.add("m5", F_ITEM, "best_item_in_store", "comparison", PRE + f"which item sold the most units at store {s} over the whole period, and how many?",
              f"{i}, {fmt(col[i])} units.", [g("best_item", i, "entity", store_id=s), g("total_units", col[i], store_id=s, item_id=i)])
    hob = it[f["item_cat"].reindex(it.index) == "HOBBIES"]
    h3 = unique_top(hob, 3)
    b.add("m5", F_ITEM, "top_three_hobbies", "comparison", PRE + "which three items in the HOBBIES category sold the most units across all stores over the whole period, and how many each?",
          "; ".join(f"{i}: {fmt(v)} units" for i, v in h3.items()) + ".", [g("total_units", v, item_id=i, rank=k + 1) for k, (i, v) in enumerate(h3.items())])
    for i in rng.sample(sorted(it.index), 3):
        b.add("m5", F_ITEM, "item_total", "aggregation", PRE + f"how many units of item {i} were sold across all stores over the whole period?",
              f"{fmt(it[i])} units.", [g("total_units", it[i], item_id=i)])
    b.add("m5", F_ITEM, "zero_sales_series", "aggregation",
          PRE + f"how many of the {fmt(f['n_series'])} item-store combinations recorded no sales at all over the whole period?",
          f"{fmt(f['zero_series'])} combinations.", [g("zero_series", f["zero_series"])])

    # ---- structure
    b.add("m5", F_FACT, "counts", "lookup", PRE0 + "how many stores, items and states does the data cover?",
          f"{f['n_stores']} stores, {fmt(f['n_items'])} items and {f['n_states']} states.",
          [g("n_stores", f["n_stores"]), g("n_items", f["n_items"]), g("n_states", f["n_states"])])
    b.add("m5", F_FACT, "stores_in_state", "lookup", PRE0 + "which stores are in Wisconsin?", ", ".join(f["state_stores"]["WI"]) + ".",
          [g("stores", ",".join(f["state_stores"]["WI"]), "categorical", state_id="WI")])
    sps = f["stores_per_state"]
    b.add("m5", F_FACT, "stores_per_state", "aggregation", PRE0 + "how many stores does each state have?",
          "; ".join(f"{s}: {n}" for s, n in sps.items()) + ".", [g("n_stores", n, state_id=s) for s, n in sps.items()])
    b.add("m5", F_FACT, "series_count", "lookup", PRE0 + "how many item-store combinations (rows) does the sales table have?",
          f"{fmt(f['n_series'])}.", [g("n_series", f["n_series"])])
    b.add("m5", F_FACT, "period", "lookup", "In the M5 competition's training data (the sales before the final 28-day evaluation period), how many days does it cover, and what are its first and last dates?",
          f"{fmt(f['n_days'])} days, from {f['first_date']} (a {f['first_weekday']}) to {f['last_date']} (a {f['last_weekday']}).",
          [g("n_days", f["n_days"]), g("first_date", f["first_date"], "temporal"), g("last_date", f["last_date"], "temporal")])


def dc_questions(b: Bank, f: dict, rng: random.Random) -> None:
    PRE = f"In the DataCo orders data ({DC_PERIOD}), "
    g = lambda *a, **k: G("dataco", *a, **k)
    n, mk, rg, md = f["n_orders"], f["market_orders"], f["region_orders"], f["mode_orders"]
    b.add("dataco", F_FACT, "order_counts", "lookup", PRE + "how many orders are there, and how many order lines?",
          f"{fmt(n)} orders and {fmt(f['n_lines'])} order lines.", [g("n_orders", n), g("n_lines", f["n_lines"])])
    b.add("dataco", F_FACT, "period", "lookup", "In the DataCo orders data, what period do the order dates cover?",
          f"From {f['first_date']} to {f['last_date']}.", [g("first_date", f["first_date"], "temporal"), g("last_date", f["last_date"], "temporal")])
    top, bot = unique_top(mk).index[0], unique_top(-mk).index[0]
    b.add("dataco", F_RANK, "market_most_least", "comparison", PRE + "which market had the most orders, and which the fewest?",
          f"Most: {top}, {fmt(mk[top])} orders.\nLeast: {bot}, {fmt(mk[bot])} orders.",
          [g("market_most", top, "entity"), g("market_least", bot, "entity"), g("orders", mk[top], market=top), g("orders", mk[bot], market=bot)])
    for m in rng.sample(sorted(mk.index), 2):
        b.add("dataco", F_AGG, "market_orders", "aggregation", PRE + f"how many orders were placed in the {m} market?", f"{fmt(mk[m])} orders.", [g("orders", mk[m], market=m)])
    tr = unique_top(rg).index[0]
    b.add("dataco", F_RANK, "region_most", "comparison", PRE + "which region had the most orders, and how many?", f"{tr}, {fmt(rg[tr])} orders.",
          [g("region_most", tr, "entity"), g("orders", rg[tr], region=tr)])
    b.add("dataco", F_FACT, "regions_markets", "lookup", PRE + "how many regions and how many markets are there?", f"{f['n_regions']} regions and {f['n_markets']} markets.",
          [g("n_regions", f["n_regions"]), g("n_markets", f["n_markets"])])
    tm = unique_top(md).index[0]
    b.add("dataco", F_RANK, "mode_most_used", "comparison", PRE + "which shipping mode was used for the most orders, and for what share of all orders?",
          f"{tm}, {fmt(md[tm])} orders ({pct(md[tm], n)} of all orders).", [g("mode_most", tm, "entity"), g("orders", md[tm], shipping_mode=tm),
                                                                           g("share_percent", round(md[tm] / n * 100, 1), shipping_mode=tm)])
    b.add("dataco", F_AGG, "same_day_orders", "aggregation", PRE + "how many orders used Same Day shipping?", f"{fmt(md['Same Day'])} orders.", [g("orders", md["Same Day"], shipping_mode="Same Day")])
    b.add("dataco", F_AGG, "cancelled_orders", "aggregation", PRE + "how many orders were cancelled (delivery status 'Shipping canceled'), and what share of all orders is that?",
          f"{fmt(f['cancelled'])} orders ({pct(f['cancelled'], n)} of all orders).", [g("cancelled_orders", f["cancelled"]), g("share_percent", round(f["cancelled"] / n * 100, 1), what="cancelled")])
    b.add("dataco", F_AGG, "late_share", "aggregation", PRE + "among the delivered (not cancelled) orders, what share arrived later than scheduled?",
          f"{pct(f['late'], f['delivered'])} ({fmt(f['late'])} of {fmt(f['delivered'])} delivered orders).", [g("late_orders", f["late"]), g("delivered_orders", f["delivered"]),
                                                                                                               g("share_percent", round(f["late"] / f["delivered"] * 100, 1), what="late")])
    delay = f["mode_delay"]
    hd = unique_top(delay).index[0]
    b.add("dataco", F_RANK, "mode_highest_delay", "comparison", PRE + "among delivered orders, which shipping mode had the highest mean delay (actual minus scheduled shipping days), and what was it?",
          f"{hd}, {delay[hd]} days on average.", [g("mode_highest_delay", hd, "entity"), g("mean_delay", delay[hd], shipping_mode=hd)])
    b.add("dataco", F_AGG, "first_class_late_rate", "aggregation", PRE + "among delivered First Class orders, what share arrived later than scheduled?",
          f"{f['mode_late_rate']['First Class'] * 100:.1f}%.", [g("late_rate", f["mode_late_rate"]["First Class"], shipping_mode="First Class")])
    mo = f["month_orders"]
    tmo = unique_top(mo).index[0]
    b.add("dataco", F_TEMP, "busiest_month", "comparison", PRE + "in which month were the most orders placed, and how many?",
          f"{month_label(tmo)}, {fmt(mo[tmo])} orders.", [g("busiest_month", tmo, "temporal"), g("orders", mo[tmo], month=tmo)])
    for m in ("2015-06", "2016-03", "2017-09"):
        b.add("dataco", F_TEMP, "month_orders", "aggregation", PRE + f"how many orders were placed in {month_label(m)}?", f"{fmt(mo[m])} orders.", [g("orders", mo[m], month=m)])
    yo = f["year_orders"]
    b.add("dataco", F_TEMP, "orders_per_year", "aggregation", PRE + "how many orders were placed in each of 2015, 2016 and 2017?",
          f"2015: {fmt(yo[2015])}; 2016: {fmt(yo[2016])}; 2017: {fmt(yo[2017])}.", [g("orders", yo[y], year=y) for y in (2015, 2016, 2017)])


# ----------------------------------------------------------------------------- build and write
def build() -> list[Question]:
    cal = pd.read_csv(RAW / "m5" / "calendar.csv")
    sales = load_m5_sales()
    a, b = m5_facts(sales, cal), m5_check_facts(sales, cal)  # two independent methods on the same (evaluation) file
    del sales
    bad = mismatches(a, b)
    if bad:
        raise SystemExit(f"the two M5 calculation methods disagree on: {bad}")
    raw = pd.read_csv(RAW / "dataco" / "DataCoSupplyChainDataset.csv", encoding="latin-1")
    da, db = dc_facts(dc_frame_raw(raw), len(raw)), dc_facts(dc_frame_pipeline(raw), len(raw))
    bad = disagreements(da, db)
    if bad:
        raise SystemExit(f"DataCo raw and pipeline disagree on: {bad}")
    bank, rng = Bank(), random.Random(SEED)
    m5_questions(bank, a, rng)
    dc_questions(bank, da, rng)
    return bank.qs


def write_outputs(qs: list[Question]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "b0_dataset_wide_v1.jsonl", qs)
    src = {"inventory": "M5: sales_train_evaluation.csv, first 1,913 days + calendar.csv (full data)", "delivery": "DataCo: DataCoSupplyChainDataset.csv (full data)"}
    with open(OUT / "b0_dataset_wide_v1.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "category", "template", "domain", "question_type", "question", "real_answer", "source"])
        for q in qs:
            fam, tpl = q.template_id.split("/")
            w.writerow([q.qid, fam, tpl, q.domain, q.question_type, q.question, q.reference_answer.replace("\n", " | "), src[q.domain]])
    lines = ["# Dataset-wide questions v1", "",
             f"{len(qs)} questions about the FULL data (not the sampled evidence cards), with real answers verified from two independent "
             "sources. For trying B0 by hand; a RAG system should abstain on them.", "", "| Category | Questions |", "|---|---|"]
    fams = list(dict.fromkeys(q.template_id.split("/")[0] for q in qs))
    lines += [f"| {fam} | {sum(q.template_id.startswith(fam + '/') for q in qs)} |" for fam in fams] + [f"| **Total** | **{len(qs)}** |", ""]
    for fam in fams:
        lines += [f"## {fam}", "", "| ID | Question | Real answer |", "|---|---|---|"]
        lines += [f"| {q.qid} | {q.question} | {q.reference_answer.replace(chr(10), '<br>')} |" for q in qs if q.template_id.startswith(fam + "/")]
        lines.append("")
    (OUT / "b0_dataset_wide_v1.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    questions = build()
    write_outputs(questions)
    print(f"{len(questions)} dataset-wide questions, answers identical across independent sources -> {OUT}")
    print(pd.Series([q.template_id.split("/")[0] for q in questions]).value_counts().to_string())
