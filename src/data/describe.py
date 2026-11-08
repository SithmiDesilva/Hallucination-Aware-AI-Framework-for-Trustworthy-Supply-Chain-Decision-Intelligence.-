"""Descriptive tables for the methodology chapter.

    python -m src.data.describe     ->  results/descriptive/

  dataco_summary(orders)   orders, cancelled, late rate and mean delay by market x shipping mode (+ an All row).
                           Late rate and mean delay are over NON-cancelled orders (the cancelled-order rule).
  m5_summary(long, sim)    series, days, units, zero-sales share, price, simulated shortfall and fill rate by store x category.
Every table ends with an `All` row, and its totals are checked against the raw counts in tests/test_describe.py.
"""
import pandas as pd

from src.config import PROCESSED, ROOT
from src.data.dataco_checks import to_markdown
from src.data.load_dataco import clean, load_raw as load_dataco_raw, to_orders

OUT = ROOT / "results" / "descriptive"


def dataco_summary(orders: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "orders": len(g),
            "cancelled": int(g["is_cancelled"].sum()),
            "late_rate": g["late"].mean(),
            "mean_delay_days": g["shipping_delay_days"].mean(),
        })

    rows = [{"Market": m, "Shipping Mode": s, **agg(g)} for (m, s), g in orders.groupby(["Market", "Shipping Mode"])]
    rows.append({"Market": "All", "Shipping Mode": "All", **agg(orders)})
    out = pd.DataFrame(rows)
    out[["orders", "cancelled"]] = out[["orders", "cancelled"]].astype(int)
    return out.round({"late_rate": 3, "mean_delay_days": 3})


def m5_summary(long: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    cat = long[["store_id", "item_id", "cat_id"]].drop_duplicates()
    sim = sim.merge(cat, on=["store_id", "item_id"], how="left", validate="m:1")

    def agg(l: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "series": l[["store_id", "item_id"]].drop_duplicates().shape[0],
            "days": len(l),
            "units": int(l["units"].sum()),
            "zero_sales_share": (l["units"] == 0).mean(),
            "mean_price": l["sell_price"].mean(),
            "sim_shortfall_units": int(s["shortfall"].sum()),
            "sim_fill_rate": s["sim_units_sold"].sum() / s["demand"].sum(),
        })

    rows = []
    for (store, c), g in long.groupby(["store_id", "cat_id"]):
        rows.append({"store_id": store, "cat_id": c,
                     **agg(g, sim[(sim["store_id"] == store) & (sim["cat_id"] == c)])})
    rows.append({"store_id": "All", "cat_id": "All", **agg(long, sim)})
    out = pd.DataFrame(rows)
    out[["series", "days", "units", "sim_shortfall_units"]] = out[["series", "days", "units", "sim_shortfall_units"]].astype(int)
    return out.round({"zero_sales_share": 3, "mean_price": 3, "sim_fill_rate": 4})


def write(name: str, table: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(to_markdown(table), encoding="utf-8")


if __name__ == "__main__":
    write("dataco_sample_summary", dataco_summary(pd.read_parquet(PROCESSED / "dataco_orders.parquet")))
    write("dataco_full_summary", dataco_summary(to_orders(clean(load_dataco_raw()))))
    write("m5_summary", m5_summary(pd.read_parquet(PROCESSED / "m5_long.parquet"), pd.read_parquet(PROCESSED / "m5_sim.parquet")))
    for f in sorted(OUT.glob("*.md")):
        print(f"--- {f.name}\n{f.read_text(encoding='utf-8')}")
