"""The DataCo preprocessing checklist (what to do, what not to do) with the evidence for each rule, for the thesis.

    python -m src.data.dataco_preprocessing_report   ->  results/data_validation/dataco_preprocessing_checks.{csv,md}

Every row is computed from the raw file and the processed tables when this runs; a rule that fails is reported as FAIL
(and the process exits non-zero), never silently skipped.
"""
import sys
from pathlib import Path

import pandas as pd

from src.cards.schemas import Card, read_jsonl
from src.config import CARDS, PROCESSED, RAW, ROOT, load_config
from src.data.dataco_checks import to_markdown
from src.data.load_dataco import PERSONAL_COLS, UNUSED_COLS, clean, load_raw, snake_case, to_orders

OUT = ROOT / "results" / "data_validation"


def run() -> pd.DataFrame:
    cfg = load_config()
    raw = load_raw()
    items = clean(raw)
    orders = to_orders(items)
    all_orders = pd.read_parquet(PROCESSED / "dataco_orders_all.parquet")
    sample = pd.read_parquet(PROCESSED / "dataco_orders.parquet")
    rows: list[tuple] = []

    def rule(kind, text, ok, evidence):
        rows.append((kind, text, "PASS" if ok else "FAIL", evidence))

    ok_orders = all_orders[~all_orders["is_cancelled"]]

    # ---- Do
    rule("Do", "Load with latin-1 and parse the order and shipping dates", str(items["order_date"].dtype).startswith("datetime64") and str(items["shipping_date"].dtype).startswith("datetime64"),
         f"{len(raw):,} item lines x {raw.shape[1]} columns read as latin-1; order_date {items['order_date'].min().date()} to {items['order_date'].max().date()}, "
         f"shipping_date {items['shipping_date'].min().date()} to {items['shipping_date'].max().date()}")
    personal_missing = {c: int(raw[c].isna().sum()) for c in PERSONAL_COLS if raw[c].isna().any()}
    rule("Do", "Drop the personal columns first: names, email, password, street and zip code", not {snake_case(c) for c in PERSONAL_COLS} & set(items.columns),
         f"dropped {', '.join(PERSONAL_COLS)}; dropped before any value check, so their gaps {personal_missing} never mattered")
    empties = {c: f"{raw[c].isna().mean():.1%} missing" for c in UNUSED_COLS}
    rule("Do", "Drop the empty or unused columns: Product Description, Order Zipcode and Product Image", not {snake_case(c) for c in UNUSED_COLS} & set(items.columns),
         f"dropped {', '.join(UNUSED_COLS)} ({'; '.join(f'{c}: {v}' for c, v in empties.items())})")
    rule("Do", "Rename columns to snake_case and check for duplicates",
         items.columns.is_unique and all(c == c.lower() and " " not in c and "(" not in c for c in items.columns) and not items.duplicated().any(),
         f"{raw.shape[1]} raw columns -> {items.shape[1]} after the drops, all snake_case with no name collisions; {int(raw.duplicated().sum())} duplicated rows; "
         f"order_item_id unique: {items['order_item_id'].is_unique}")
    delay_ok = bool((all_orders["raw_delay_days"] == all_orders["days_for_shipping_real"] - all_orders["days_for_shipment_scheduled"]).all()
                    and (ok_orders["late"].astype(bool) == (ok_orders["raw_delay_days"] > 0)).all())
    rule("Do", "Compute delay = real - scheduled shipping days, and late = delay > 0", delay_ok,
         f"raw_delay_days = days_for_shipping_real - days_for_shipment_scheduled for all {len(all_orders):,} orders; late = delay > 0 for the "
         f"{len(ok_orders):,} delivered orders ({int(ok_orders['late'].sum()):,} late)")
    sign = {"Late delivery": 1, "Advance shipping": -1, "Shipping on time": 0}
    flag_ok = bool((ok_orders["late"].astype(bool) == (ok_orders["late_delivery_risk"] == 1)).all())
    status_ok = bool((ok_orders["delivery_status"].map(sign) == ok_orders["raw_delay_days"].apply(lambda d: (d > 0) - (d < 0))).all())
    rule("Do", "Cross-check the late flag against Delivery Status and Late_delivery_risk", flag_ok and status_ok,
         f"{len(ok_orders):,} delivered orders: 0 disagree with Late_delivery_risk, 0 disagree with Delivery Status "
         "(Late delivery = delay > 0, Advance shipping = delay < 0, Shipping on time = delay 0)")
    rule("Do", "Exclude cancelled orders from lateness rates, but keep them in the index",
         all_orders["late"].isna().sum() == all_orders["is_cancelled"].sum() and all_orders["is_cancelled"].sum() > 0 and all_orders["order_id"].is_unique,
         f"{int(all_orders['is_cancelled'].sum()):,} cancelled orders stay in the order table and get order cards (null late and delay); "
         f"the late rate is over the {len(ok_orders):,} delivered orders: {ok_orders['late'].astype(bool).mean():.1%}")
    rule("Do", "Roll order items up to orders, checking that the shipping details match", len(all_orders) == items["order_id"].nunique() and len(all_orders) == len(orders),
         f"{len(items):,} item lines -> {len(all_orders):,} orders; the build fails if any order-level field differs between an order's lines (none do)")
    strata = all_orders.groupby(["market", "shipping_mode"]).size()
    got = sample.groupby(["market", "shipping_mode"]).size().reindex(strata.index, fill_value=0)
    dev = float((got - strata * len(sample) / len(all_orders)).abs().max())
    rule("Do", "Sample 20,000 orders, stratified by market and shipping mode with a fixed seed",
         len(sample) == cfg["data"]["dataco_orders"] and dev < 1 and set(sample["order_id"]) <= set(all_orders["order_id"]),
         f"seed {cfg['seed']}: {len(sample):,} orders over {len(strata)} market x shipping-mode strata, each within {dev:.2f} of its proportional share")
    cards = [c for c in read_jsonl(CARDS / "cards.jsonl", Card) if c.card_type == "region_month"]
    month = all_orders["order_date"].dt.strftime("%Y-%m")
    cells = all_orders.groupby(["order_region", month]).size()
    rule("Do", "Compute the region x month aggregates from all non-cancelled orders",
         len(cards) == int((cells >= cfg["cards"]["region_month_min_orders"]).sum()) and sum(c.fields["orders"] for c in cards) > len(sample),
         f"{len(cards)} region-month cards from all {len(all_orders):,} orders (rates and mean delays over the {len(ok_orders):,} delivered ones), not the "
         f"{len(sample):,}-order sample; only {len(cells)} of {all_orders['order_region'].nunique() * month.nunique()} possible region-month cells have any order "
         f"(a region is active in about {int(all_orders.groupby('order_region')['order_date'].apply(lambda s: s.dt.strftime('%Y-%m').nunique()).median())} of {month.nunique()} months)")

    # ---- Don't
    nulls = all_orders.isna().sum()
    nulls = nulls[nulls > 0]
    rule("Don't", "Fill in missing values (drop the columns instead)", set(nulls.index) == {"shipping_delay_days", "late"} and items.isna().sum().sum() == 0,
         f"the raw file has missing values in {int((raw.isna().sum() > 0).sum())} columns, all of them dropped; 0 missing values remain in the cleaned item table; "
         f"the only nulls in the order table are the {int(nulls.iloc[0]):,} cancelled orders' delay and late")
    rule("Don't", "Scale values with StandardScaler", bool(all_orders["days_for_shipping_real"].dtype.kind == "i" and all_orders["days_for_shipping_real"].min() >= 0
                                                       and (all_orders["days_for_shipping_real"].to_numpy() == raw.drop_duplicates("Order Id").set_index("Order Id").loc[all_orders["order_id"], "Days for shipping (real)"].to_numpy()).all()),
         f"shipping days are the raw integers {int(all_orders['days_for_shipping_real'].min())} to {int(all_orders['days_for_shipping_real'].max())}, equal to the raw file for every order (no z-scores)")
    text = ["delivery_status", "shipping_mode", "market"]
    rule("Don't", "One-hot encode delivery status, shipping mode or market", all(all_orders[c].dtype == object for c in text) and not [c for c in all_orders.columns if c.startswith(tuple(f"{t}_" for t in text))],
         "kept as text: " + "; ".join(f"{c} = {sorted(all_orders[c].unique())}" for c in ("delivery_status", "shipping_mode")) + f"; market has {all_orders['market'].nunique()} values")
    rule("Don't", "Cap sales or prices", bool(items["sales"].equals(raw["Sales"]) and items["order_item_product_price"].equals(raw["Order Item Product Price"]) and items["product_price"].equals(raw["Product Price"])),
         f"sales, order_item_product_price and product_price equal the raw columns; maximum {items['sales'].max()}")
    rule("Don't", "Build customer segmentation (RFM) features", not [c for c in all_orders.columns if any(k in c for k in ("rfm", "recency", "frequency", "monetary"))],
         f"the order table has only: {', '.join(all_orders.columns)}")
    offenders = sorted(str(f.relative_to(ROOT)) for f in (ROOT / "src").rglob("*.py")
                       if f.resolve() != Path(__file__).resolve() and "access" + "_logs" in f.read_text(encoding="utf-8"))
    logs = [p.name for p in (RAW / "dataco").iterdir() if "access" in p.name.lower()]
    rule("Don't", "Use the access-logs file (not relevant; it contains IP addresses)", not offenders and not logs,
         f"scanned {len(list((ROOT / 'src').rglob('*.py')))} source files: none refers to it; files in data/raw/dataco: {', '.join(p.name for p in (RAW / 'dataco').iterdir() if p.is_file())}")

    table = pd.DataFrame(rows, columns=["rule_type", "rule", "result", "evidence"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "dataco_preprocessing_checks.csv", index=False)
    (OUT / "dataco_preprocessing_checks.md").write_text(to_markdown(table), encoding="utf-8")
    return table


if __name__ == "__main__":
    result = run()
    print(to_markdown(result))
    sys.exit(0 if (result["result"] == "PASS").all() else 1)
