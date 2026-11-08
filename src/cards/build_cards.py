"""Evidence cards: one retrievable unit with a stable ID.

    python -m src.cards.build_cards      ->  data/cards/cards.jsonl (+ cards.jsonl.sha256)

Every field is computed in pandas from data/processed/ (an LLM never writes a card value). `text` is
rendered from one fixed sentence template per card type, so wording is identical across cards and only the
values differ. Near-duplicates (same item in another store, a neighbouring item or week) are kept on
purpose: they are the hard negatives that test entity misattribution.

Card types and keys                                                      id
  item_store_week   store_id, item_id, wm_yr_wk  (Walmart week, Sat-Fri)  M5-<store>-<item>-W<wm_yr_wk>
  item_store_month  store_id, item_id, month     (calendar month)         M5-<store>-<item>-M<yyyy-mm>
  order             order_id                                              DC-ORD-<order_id>
  region_month      region, month                                         DC-REG-<region>-<yyyy-mm>

Naming: `units_sold` is OBSERVED sales; `shortfall` and everything about stock are SIMULATED (the
reorder-point policy of simulate_inventory.py). The text says so explicitly.
Cancelled DataCo orders show the raw recorded shipping days next to the cancelled status (the conflicting
evidence of the conflict slice); their `delay_days` and `late` are null.
"""
import calendar as cal
import hashlib
import re

import pandas as pd

from src.cards.schemas import Card, write_jsonl
from src.config import CARDS, PROCESSED, load_config

SHIPPING_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
M5_ID = {"week": "M5-{store_id}-{item_id}-W{wm_yr_wk}", "month": "M5-{store_id}-{item_id}-M{month}"}


# ----------------------------------------------------------------------------- ids and text
def card_id(key: dict) -> str:
    """Deterministic ID built from the key alone (the key set identifies the card type)."""
    k = set(key)
    if k == {"store_id", "item_id", "wm_yr_wk"}:
        return M5_ID["week"].format(**key)
    if k == {"store_id", "item_id", "month"}:
        return M5_ID["month"].format(**key)
    if k == {"order_id"}:
        return f"DC-ORD-{key['order_id']}"
    if k == {"region", "month"}:
        return f"DC-REG-{re.sub(r'[^A-Za-z0-9]+', '_', key['region']).strip('_')}-{key['month']}"
    raise ValueError(f"unrecognised card key: {key}")


def fmt(x) -> str:
    """Plain number: no thousands separator, at most 4 decimals, no trailing zeros."""
    if isinstance(x, bool):
        raise TypeError("booleans are rendered by the templates")
    if isinstance(x, int):
        return str(x)
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _pct(rate) -> str:
    return f"{fmt(rate)} ({rate * 100:.1f}%)"


def pieces(card_type: str, key: dict, f: dict) -> dict[str, str]:
    """The exact text fragment that states each field; every non-null field must have one."""
    if card_type == "item_store_week":
        return {
            "units_sold": f"observed sales of {fmt(f['units_sold'])} units",
            "mean_price": f"mean price ${f['mean_price']:.2f}",
            "snap_days": f"{fmt(f['snap_days'])} SNAP days",
            "events": f"events: {f['events']}",
            "week_start": f"{f['week_start']}",
            "week_end": f"{f['week_end']}",
            "on_hand_end": f"end-of-week stock {fmt(f['on_hand_end'])} units",
            "rop": f"reorder point {fmt(f['rop'])}",
            "safety_stock": f"safety stock {fmt(f['safety_stock'])}",
            "order_placed": (f"an order of {fmt(f['order_qty'])} units was placed" if f["order_placed"]
                             else "no order was placed"),
            "order_qty": (f"an order of {fmt(f['order_qty'])} units was placed" if f["order_placed"]
                          else "no order was placed"),
            "shortfall": f"simulated shortfall {fmt(f['shortfall'])} units",
        }
    if card_type == "item_store_month":
        return {
            "total_units": f"observed sales totalled {fmt(f['total_units'])} units",
            "mean_daily_units": f"mean daily sales {fmt(f['mean_daily_units'])}",
            "sd_daily_units": f"standard deviation {fmt(f['sd_daily_units'])}",
            "shortfall_days": f"simulated shortfall on {fmt(f['shortfall_days'])} days",
        }
    if card_type == "order":
        p = {
            "order_date": f"placed {f['order_date']}",
            "shipping_date": f"shipping date {f['shipping_date']}",
            "shipping_mode": f"{f['shipping_mode']}",
            "market": f"market {f['market']}",
            "region": f"region {f['region']}",
            "delivery_status": f"delivery status {f['delivery_status']}",
            "scheduled_days": f"{fmt(f['scheduled_days'])} days scheduled",
            "is_cancelled": "the order was cancelled" if f["is_cancelled"] else "the order was not cancelled",
        }
        if f["is_cancelled"]:
            # neutral on purpose: the raw recorded values as in the source row, with no hint about how to read them
            p["recorded_real_days"] = f"recorded shipping time {fmt(f['recorded_real_days'])} days"
            p["recorded_delay_days"] = f"recorded difference {fmt(f['recorded_delay_days'])} days against schedule"
        else:
            p["real_days"] = f"{fmt(f['real_days'])} days actual"
            p["delay_days"] = (f"{fmt(f['delay_days'])} days late" if f["delay_days"] > 0 else
                               f"{fmt(-f['delay_days'])} days early" if f["delay_days"] < 0 else "on time")
            p["late"] = "flagged late" if f["late"] else "not flagged late"
        return p
    if card_type == "region_month":
        p = {
            "orders": f"{fmt(f['orders'])} orders",
            "cancelled_orders": f"{fmt(f['cancelled_orders'])} cancelled",
            "late_orders": f"{fmt(f['late_orders'])} late",
            "late_rate": f"late rate {_pct(f['late_rate'])}" if f["late_rate"] is not None else "late rate not defined",
        }
        for mode in SHIPPING_MODES:
            v = f[_mode_field(mode)]
            p[_mode_field(mode)] = f"{mode} {fmt(v)}" if v is not None else f"{mode} none"
        return p
    raise ValueError(card_type)


def _mode_field(mode: str) -> str:
    return "mean_delay_days_" + mode.lower().replace(" ", "_")


def render_text(card_type: str, key: dict, fields: dict) -> str:
    f = {k: v for k, v in fields.items()}
    p = pieces(card_type, key, f)
    if card_type == "item_store_week":
        return (f"Item {key['item_id']} at store {key['store_id']}, Walmart week {key['wm_yr_wk']} "
                f"({p['week_start']} to {p['week_end']}): {p['units_sold']} at a {p['mean_price']}; "
                f"{p['snap_days']}; {p['events']}. Simulated inventory (reorder-point policy): "
                f"{p['on_hand_end']}; {p['rop']}; {p['safety_stock']}. This week {p['order_placed']}; "
                f"{p['shortfall']}.")
    if card_type == "item_store_month":
        return (f"Item {key['item_id']} at store {key['store_id']}, month {key['month']}: "
                f"{p['total_units']}; {p['mean_daily_units']}, {p['sd_daily_units']}; {p['shortfall_days']}.")
    if card_type == "order":
        base = (f"Order {key['order_id']} ({p['order_date']}, {p['shipping_date']}), {p['shipping_mode']}, "
                f"{p['market']}, {p['region']}: {p['delivery_status']}; {p['scheduled_days']}; ")
        if f["is_cancelled"]:
            return base + f"{p['is_cancelled']}; {p['recorded_real_days']}, {p['recorded_delay_days']}."
        return base + f"{p['real_days']}, {p['delay_days']}, {p['late']}; {p['is_cancelled']}."
    if card_type == "region_month":
        modes = ", ".join(p[_mode_field(m)] for m in SHIPPING_MODES)
        return (f"Region {key['region']}, month {key['month']}: {p['orders']}, {p['cancelled_orders']}, "
                f"{p['late_orders']}; {p['late_rate']} of non-cancelled orders. "
                f"Mean delay in days by shipping mode (non-cancelled orders): {modes}.")
    raise ValueError(card_type)


def _card(source, card_type, key, fields) -> Card:
    return Card(card_id=card_id(key), source=source, card_type=card_type, key=key, fields=fields,
                text=render_text(card_type, key, fields))


# ----------------------------------------------------------------------------- M5
def _window_weeks(long: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """The last n complete Walmart weeks (all 7 dates present), with their first and last date."""
    days = long.drop_duplicates("date")[["date", "wm_yr_wk", "event_name_1", "event_name_2"]]
    wk = days.groupby("wm_yr_wk").agg(week_start=("date", "min"), week_end=("date", "max"), n=("date", "size"))
    wk = wk[wk["n"] == 7].sort_values("week_start").tail(n_weeks)

    def events(g: pd.DataFrame) -> str:
        names = sorted(set(g["event_name_1"].dropna()) | set(g["event_name_2"].dropna()))
        return "; ".join(names) if names else "none"

    wk["events"] = days.groupby("wm_yr_wk").apply(events, include_groups=False)
    return wk


def _sim_with_price(long: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    px = long[["store_id", "item_id", "date", "sell_price", "snap"]]
    return sim.merge(px, on=["store_id", "item_id", "date"], how="left", validate="1:1")


def m5_week_cards(long: pd.DataFrame, sim: pd.DataFrame, n_weeks: int = 52) -> list[Card]:
    wk = _window_weeks(long, n_weeks)
    d = _sim_with_price(long, sim)
    d = d[d["wm_yr_wk"].isin(wk.index)].sort_values(["store_id", "item_id", "date"])
    g = d.groupby(["store_id", "item_id", "wm_yr_wk"], sort=True)
    agg = g.agg(n=("date", "size"), units_sold=("demand", "sum"), mean_price=("sell_price", "mean"),
                snap_days=("snap", "sum"), on_hand_end=("on_hand_end", "last"), rop=("rop", "last"),
                safety_stock=("safety_stock", "last"), order_placed=("order_placed", "any"),
                order_qty=("order_qty", "sum"), shortfall=("shortfall", "sum")).reset_index()
    agg = agg[agg["n"] == 7]  # a week needs all 7 simulated days (the first 28 days of a series are warm-up)
    cards = []
    for r in agg.itertuples(index=False):
        w = wk.loc[r.wm_yr_wk]
        key = {"store_id": r.store_id, "item_id": r.item_id, "wm_yr_wk": str(r.wm_yr_wk)}
        fields = {
            "week_start": w["week_start"].strftime("%Y-%m-%d"), "week_end": w["week_end"].strftime("%Y-%m-%d"),
            "units_sold": int(r.units_sold), "mean_price": round(float(r.mean_price), 2),
            "snap_days": int(r.snap_days), "events": w["events"],
            "on_hand_end": int(r.on_hand_end), "rop": round(float(r.rop), 1),
            "safety_stock": round(float(r.safety_stock), 1),
            "order_placed": bool(r.order_placed), "order_qty": int(r.order_qty), "shortfall": int(r.shortfall),
        }
        cards.append(_card("m5", "item_store_week", key, fields))
    return cards


def _window_months(long: pd.DataFrame, n_months: int) -> list[pd.Period]:
    first, last = long["date"].min(), long["date"].max()
    months = pd.period_range(first, last, freq="M")
    complete = [m for m in months if m.start_time >= first and m.end_time.normalize() <= last]
    return complete[-n_months:]


def m5_month_cards(long: pd.DataFrame, sim: pd.DataFrame, n_months: int = 12) -> list[Card]:
    months = _window_months(long, n_months)
    d = sim.assign(month=sim["date"].dt.to_period("M"))
    d = d[d["month"].isin(months)]
    g = d.groupby(["store_id", "item_id", "month"], sort=True)
    agg = g.agg(n=("date", "size"), total_units=("demand", "sum"), mean_daily_units=("demand", "mean"),
                sd_daily_units=("demand", "std"),
                shortfall_days=("shortfall", lambda s: int((s > 0).sum()))).reset_index()
    agg = agg[agg["n"] == agg["month"].map(lambda m: m.days_in_month)]  # complete months only
    cards = []
    for r in agg.itertuples(index=False):
        key = {"store_id": r.store_id, "item_id": r.item_id, "month": str(r.month)}
        fields = {"total_units": int(r.total_units), "mean_daily_units": round(float(r.mean_daily_units), 2),
                  "sd_daily_units": round(float(r.sd_daily_units), 2), "shortfall_days": int(r.shortfall_days)}
        cards.append(_card("m5", "item_store_month", key, fields))
    return cards


# ----------------------------------------------------------------------------- DataCo
def dataco_order_cards(orders: pd.DataFrame) -> list[Card]:
    cards = []
    for _, r in orders.sort_values("Order Id").iterrows():
        key = {"order_id": str(int(r["Order Id"]))}
        cancelled = bool(r["is_cancelled"])
        fields = {
            "order_date": r["order date (DateOrders)"].strftime("%Y-%m-%d"),
            "shipping_date": r["shipping date (DateOrders)"].strftime("%Y-%m-%d"),
            "shipping_mode": r["Shipping Mode"], "market": r["Market"], "region": r["Order Region"],
            "delivery_status": r["Delivery Status"], "scheduled_days": int(r["Days for shipment (scheduled)"]),
            "is_cancelled": cancelled,
        }
        if cancelled:
            fields["recorded_real_days"] = int(r["Days for shipping (real)"])
            fields["recorded_delay_days"] = int(r["raw_delay_days"])
            fields["delay_days"] = None
            fields["late"] = None
        else:
            fields["real_days"] = int(r["Days for shipping (real)"])
            fields["delay_days"] = int(r["shipping_delay_days"])
            fields["late"] = bool(r["late"])
        cards.append(_card("dataco", "order", key, fields))
    return cards


def dataco_region_month_cards(orders: pd.DataFrame, min_orders: int = 10) -> list[Card]:
    d = orders.assign(month=orders["order date (DateOrders)"].dt.to_period("M").astype(str))
    cards = []
    for (region, month), g in d.groupby(["Order Region", "month"], sort=True):
        if len(g) < min_orders:
            continue
        ok = g[~g["is_cancelled"]]
        late_orders = int(ok["late"].sum())
        fields = {"orders": len(g), "cancelled_orders": int(g["is_cancelled"].sum()), "late_orders": late_orders,
                  "late_rate": round(late_orders / len(ok), 4) if len(ok) else None}
        for mode in SHIPPING_MODES:
            m = ok.loc[ok["Shipping Mode"] == mode, "shipping_delay_days"]
            fields[_mode_field(mode)] = round(float(m.mean()), 2) if len(m) else None
        cards.append(_card("dataco", "region_month", {"region": region, "month": month}, fields))
    return cards


# ----------------------------------------------------------------------------- build
def build_cards(cfg: dict | None = None) -> list[Card]:
    cfg = cfg or load_config()
    c = cfg["cards"]
    long = pd.read_parquet(PROCESSED / "m5_long.parquet")
    sim = pd.read_parquet(PROCESSED / "m5_sim.parquet")
    orders = pd.read_parquet(PROCESSED / "dataco_orders.parquet")
    cards = (m5_week_cards(long, sim, c["m5_weeks"]) + m5_month_cards(long, sim, c["m5_months"])
             + dataco_order_cards(orders) + dataco_region_month_cards(orders, c["region_month_min_orders"]))
    ids = [x.card_id for x in cards]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate card IDs")
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / "cards.jsonl"
    write_jsonl(path, cards)
    (CARDS / "cards.jsonl.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  cards.jsonl\n")
    return cards


if __name__ == "__main__":
    cards = build_cards()
    counts = pd.Series([x.card_type for x in cards]).value_counts()
    print(f"{len(cards):,} cards -> {CARDS / 'cards.jsonl'}\n{counts.to_string()}")
