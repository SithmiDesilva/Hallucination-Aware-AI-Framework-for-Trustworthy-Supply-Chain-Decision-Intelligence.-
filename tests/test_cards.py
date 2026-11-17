"""Tests for src/cards/build_cards.py.

Synthetic: deterministic IDs; the text states every non-null field; week and month cards equal hand
calculations (incomplete weeks and warm-up weeks get no card); order cards (cancelled orders keep the raw
recorded days, with null delay and lateness); region-month cards (late rate over non-cancelled orders only).
Real data (skipped until `make cards` has run): for 50 random cards of each type every field equals a
direct pandas query, IDs are unique, every card validates, the text states every field, and the checksum matches.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.cards.build_cards import (
    SHIPPING_MODES, _mode_field, card_id, dataco_order_cards, dataco_region_month_cards, m5_month_cards,
    m5_week_cards, pieces, render_text,
)
from src.cards.schemas import Card, read_jsonl
from src.config import CARDS, PROCESSED
from src.data.simulate_inventory import SimParams, simulate_all


# ----------------------------------------------------------------------------- ids and text
def test_card_ids_are_deterministic_and_type_specific():
    wk = {"store_id": "CA_1", "item_id": "FOODS_3_090", "wm_yr_wk": "11512"}
    mo = {"store_id": "CA_1", "item_id": "FOODS_3_090", "month": "2015-05"}
    assert card_id(wk) == "M5-CA_1-FOODS_3_090-W11512" == card_id(dict(reversed(list(wk.items()))))
    assert card_id(mo) == "M5-CA_1-FOODS_3_090-M2015-05"
    assert card_id({"order_id": "320"}) == "DC-ORD-320"
    assert card_id({"region": "Western Europe", "month": "2016-03"}) == "DC-REG-Western_Europe-2016-03"
    with pytest.raises(ValueError):
        card_id({"foo": "bar"})


WEEK = {"week_start": "2015-03-14", "week_end": "2015-03-20", "units_sold": 412, "mean_price": 1.25, "snap_days": 3,
        "events": "none", "on_hand_end": 58, "rop": 96.4, "safety_stock": 31.2, "order_placed": True, "order_qty": 180,
        "shortfall": 0}
MONTH = {"total_units": 30, "mean_daily_units": 1.0, "sd_daily_units": 1.5, "shortfall_days": 2}
ORDER_OK = {"order_date": "2016-03-01", "shipping_date": "2016-03-04", "shipping_mode": "Standard Class", "market": "LATAM",
            "region": "South America", "delivery_status": "Late delivery", "scheduled_days": 3, "is_cancelled": False,
            "real_days": 5, "delay_days": 2, "late": True}
ORDER_CANCELLED = {**{k: v for k, v in ORDER_OK.items() if k not in ("real_days",)}, "delivery_status": "Shipping canceled",
                   "is_cancelled": True, "recorded_real_days": 5, "recorded_delay_days": 2, "delay_days": None, "late": None}
REGION = {"orders": 36, "cancelled_orders": 2, "late_orders": 19, "late_rate": 0.5588,
          **{_mode_field(m): v for m, v in zip(SHIPPING_MODES, [-0.12, 1.83, 1.0, None])}}


@pytest.mark.parametrize("card_type, key, fields", [
    ("item_store_week", {"store_id": "CA_1", "item_id": "F_1", "wm_yr_wk": "11512"}, WEEK),
    ("item_store_week", {"store_id": "CA_1", "item_id": "F_1", "wm_yr_wk": "11512"}, {**WEEK, "order_placed": False, "order_qty": 0}),
    ("item_store_month", {"store_id": "CA_1", "item_id": "F_1", "month": "2015-05"}, MONTH),
    ("order", {"order_id": "320"}, ORDER_OK),
    ("order", {"order_id": "320"}, {**ORDER_OK, "delay_days": -2, "late": False}),
    ("order", {"order_id": "321"}, ORDER_CANCELLED),
    ("region_month", {"region": "North Africa", "month": "2017-01"}, REGION),
])
def test_text_states_every_non_null_field(card_type, key, fields):
    text = render_text(card_type, key, fields)
    p = pieces(card_type, key, fields)
    missing = [k for k, v in fields.items() if v is not None and k not in p]
    assert not missing, f"fields with no sentence in the template: {missing}"
    assert all(piece in text for piece in p.values())
    assert text == render_text(card_type, key, dict(fields))  # fixed template: same input, same text


def test_cancelled_order_text_gives_raw_values_without_hints():
    text = render_text("order", {"order_id": "321"}, ORDER_CANCELLED)
    assert "Shipping canceled" in text and "the order was cancelled" in text
    assert "recorded shipping time 5 days" in text and "recorded difference 2 days" in text
    assert "late" not in text.lower().replace("flagged late", "")  # nothing calls it late or early
    assert "early" not in text and "nothing was delivered" not in text


# ----------------------------------------------------------------------------- M5 (synthetic series)
@pytest.fixture(scope="module")
def m5_synth():
    n = 150
    dates = pd.date_range("2011-01-29", periods=n)  # starts on a Saturday: Walmart weeks are Sat-Fri
    rows = []
    rng = np.random.default_rng(3)
    for store, item, scale in [("CA_1", "F_1", 6.0), ("TX_2", "F_1", 2.0)]:
        units = rng.poisson(scale, n)
        for i, d in enumerate(dates):
            rows.append({"store_id": store, "item_id": item, "date": d, "wm_yr_wk": 11101 + i // 7, "units": int(units[i]),
                         "sell_price": 1.5 + 0.25 * (i // 7 % 3), "snap": int(i % 5 == 0),
                         "event_name_1": "Cinco" if i == 100 else None, "event_name_2": None})
    long = pd.DataFrame(rows)
    sim = simulate_all(long, SimParams())
    return long, sim


def test_week_cards_match_hand_calculation(m5_synth):
    long, sim = m5_synth
    cards = m5_week_cards(long, sim, n_weeks=6)
    assert len(cards) == 12 and len({c.card_id for c in cards}) == 12  # 6 weeks x 2 series
    for c in cards:
        s, e = pd.Timestamp(c.fields["week_start"]), pd.Timestamp(c.fields["week_end"])
        assert (e - s).days == 6 and s.day_name() == "Saturday" and e.day_name() == "Friday"
        k = c.key
        L = long[(long.store_id == k["store_id"]) & (long.date >= s) & (long.date <= e)]
        S = sim[(sim.store_id == k["store_id"]) & (sim.date >= s) & (sim.date <= e)].sort_values("date")
        assert c.fields["units_sold"] == L.units.sum()
        assert c.fields["mean_price"] == pytest.approx(L.sell_price.mean())
        assert c.fields["snap_days"] == L.snap.sum()
        assert c.fields["on_hand_end"] == S.on_hand_end.iloc[-1]
        assert c.fields["rop"] == pytest.approx(S.rop.iloc[-1], abs=0.05)
        assert c.fields["order_qty"] == S.order_qty.sum() and c.fields["order_placed"] == bool(S.order_placed.any())
        assert c.fields["shortfall"] == S.shortfall.sum()
    assert any(c.fields["events"] == "Cinco" for c in m5_week_cards(long, sim, n_weeks=30))


def test_the_string_none_means_no_event(m5_synth):
    """The preprocessed calendar fills missing events with the string "None"; the card must not call that an event."""
    long, sim = m5_synth
    filled = long.assign(event_name_1=long["event_name_1"].fillna("None"), event_name_2=long["event_name_2"].fillna("None"))
    assert [c.fields["events"] for c in m5_week_cards(filled, sim, n_weeks=30)] == [c.fields["events"] for c in m5_week_cards(long, sim, n_weeks=30)]
    assert not any("None" in c.fields["events"] for c in m5_week_cards(filled, sim, n_weeks=30))
    assert any(c.fields["events"] == "none" for c in m5_week_cards(filled, sim, n_weeks=30))


def test_warmup_weeks_and_partial_weeks_get_no_card(m5_synth):
    long, sim = m5_synth
    starts = {c.fields["week_start"] for c in m5_week_cards(long, sim, n_weeks=100)}
    first_sim_day = sim.date.min()
    assert all(pd.Timestamp(s) >= first_sim_day for s in starts)       # warm-up weeks have no simulated stock
    assert all(pd.Timestamp(s).day_name() == "Saturday" for s in starts)
    last_complete_end = max(pd.Timestamp(c.fields["week_end"]) for c in m5_week_cards(long, sim, n_weeks=100))
    assert last_complete_end <= long.date.max()                          # a trailing part-week is excluded


def test_month_cards_match_hand_calculation_and_skip_incomplete_months(m5_synth):
    long, sim = m5_synth
    cards = m5_month_cards(long, sim, n_months=12)
    months = {c.key["month"] for c in cards}
    # data spans 2011-01-29..2011-06-27; the sim starts 2011-02-26, so Feb (incomplete in the sim), Jan and Jun get no card
    assert months == {"2011-03", "2011-04", "2011-05"}
    for c in cards:
        k = c.key
        S = sim[(sim.store_id == k["store_id"]) & (sim.date.dt.to_period("M").astype(str) == k["month"])]
        assert c.fields["total_units"] == S.demand.sum()
        assert c.fields["mean_daily_units"] == pytest.approx(S.demand.mean(), abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass
        assert c.fields["sd_daily_units"] == pytest.approx(S.demand.std(ddof=1), abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass
        assert c.fields["shortfall_days"] == (S.shortfall > 0).sum()


# ----------------------------------------------------------------------------- DataCo (synthetic)
def make_orders():
    rows = []
    for oid, region, month, mode, cancelled, delay in [
        (1, "R1", "2016-01", "Standard Class", False, 2), (2, "R1", "2016-01", "Standard Class", False, -1),
        (3, "R1", "2016-01", "First Class", False, 1), (4, "R1", "2016-01", "Standard Class", True, 3),
        (5, "R2", "2016-01", "Same Day", False, 0), (6, "R1", "2016-02", "Standard Class", False, 2),
    ]:
        rows.append({
            "order_id": oid, "order_region": region, "market": "M", "shipping_mode": mode,
            "order_date": pd.Timestamp(f"{month}-10 12:00"), "shipping_date": pd.Timestamp(f"{month}-13"),
            "delivery_status": "Shipping canceled" if cancelled else "Late delivery",
            "days_for_shipment_scheduled": 3, "days_for_shipping_real": 3 + delay, "raw_delay_days": delay,
            "is_cancelled": cancelled, "shipping_delay_days": pd.NA if cancelled else delay,
            "late": pd.NA if cancelled else delay > 0,
        })
    o = pd.DataFrame(rows)
    o["shipping_delay_days"] = o["shipping_delay_days"].astype("Int64")
    o["late"] = o["late"].astype("boolean")
    return o


def test_order_cards_cancelled_vs_delivered():
    cards = {c.key["order_id"]: c for c in dataco_order_cards(make_orders())}
    ok, cancelled = cards["1"], cards["4"]
    assert ok.fields["delay_days"] == 2 and ok.fields["late"] is True and "real_days" in ok.fields
    assert cancelled.fields["delay_days"] is None and cancelled.fields["late"] is None
    assert cancelled.fields["recorded_real_days"] == 6 and cancelled.fields["recorded_delay_days"] == 3
    assert "real_days" not in cancelled.fields
    assert cards["2"].fields["delay_days"] == -1 and cards["2"].fields["late"] is False


def test_region_month_cards_exclude_cancelled_orders_from_rates():
    cards = {c.key["region"] + c.key["month"]: c for c in dataco_region_month_cards(make_orders(), min_orders=2)}
    assert set(cards) == {"R12016-01"}  # R1/2016-02 and R2/2016-01 have fewer than 2 orders
    f = cards["R12016-01"].fields
    assert f["orders"] == 4 and f["cancelled_orders"] == 1 and f["late_orders"] == 2   # orders 1 and 3 are late
    assert f["late_rate"] == pytest.approx(2 / 3, abs=1e-4)                              # 3 non-cancelled orders
    assert f[_mode_field("Standard Class")] == pytest.approx(0.5)                        # (2 + -1) / 2; cancelled order 4 excluded
    assert f[_mode_field("First Class")] == 1.0 and f[_mode_field("Same Day")] is None


# ----------------------------------------------------------------------------- real data
CARDS_FILE = CARDS / "cards.jsonl"
needs_cards = pytest.mark.skipif(not (CARDS_FILE.exists() and (PROCESSED / "m5_sim.parquet").exists()),
                                 reason="cards not built (python -m src.cards.build_cards)")


@pytest.fixture(scope="module")
def real():
    cards = list(read_jsonl(CARDS_FILE, Card))  # also validates every card against the schema
    long, sim = pd.read_parquet(PROCESSED / "m5_long.parquet"), pd.read_parquet(PROCESSED / "m5_sim.parquet")
    return {
        "cards": cards,
        # one frame per series, so each check filters ~2k rows instead of 1.4M (the filters themselves stay independent)
        "long_by": dict(tuple(long.groupby(["store_id", "item_id"]))),
        "sim_by": dict(tuple(sim.groupby(["store_id", "item_id"]))),
        "days": long.drop_duplicates("date")[["date", "event_name_1", "event_name_2"]],
        "orders": pd.read_parquet(PROCESSED / "dataco_orders.parquet"),          # the sample: one order card each
        "all_orders": pd.read_parquet(PROCESSED / "dataco_orders_all.parquet"),  # every order: the region x month aggregates
    }


def _sample(cards, card_type, n=50, seed=7):
    pool = [c for c in cards if c.card_type == card_type]
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


@needs_cards
def test_ids_unique_deterministic_and_text_complete(real):
    cards = real["cards"]
    assert len({c.card_id for c in cards}) == len(cards)
    assert all(c.card_id == card_id(c.key) for c in cards)
    for c in cards:
        p = pieces(c.card_type, c.key, c.fields)
        assert all(k in p for k, v in c.fields.items() if v is not None)
        assert all(piece in c.text for piece in p.values())
    assert {c.card_type for c in cards} == {"item_store_week", "item_store_month", "order", "region_month"}


@needs_cards
def test_checksum_matches(real):
    recorded = (CARDS / "cards.jsonl.sha256").read_text().split()[0]
    assert recorded == hashlib.sha256(CARDS_FILE.read_bytes()).hexdigest()


@needs_cards
def test_50_random_week_cards_match_direct_queries(real):
    for c in _sample(real["cards"], "item_store_week"):
        k, f = c.key, c.fields
        s, e = pd.Timestamp(f["week_start"]), pd.Timestamp(f["week_end"])
        long, sim = real["long_by"][(k["store_id"], k["item_id"])], real["sim_by"][(k["store_id"], k["item_id"])]
        L = long[long.wm_yr_wk == int(k["wm_yr_wk"])]
        S = sim[(sim.date >= s) & (sim.date <= e)].sort_values("date")
        assert len(L) == 7 and len(S) == 7 and L.date.min() == s and L.date.max() == e
        assert f["units_sold"] == L.units.sum() and f["snap_days"] == L.snap.sum()
        assert f["mean_price"] == pytest.approx(L.sell_price.mean(), abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass
        assert f["on_hand_end"] == S.on_hand_end.iloc[-1]
        assert f["rop"] == pytest.approx(S.rop.iloc[-1], abs=0.05) and f["safety_stock"] == pytest.approx(S.safety_stock.iloc[-1], abs=0.05)
        assert f["order_qty"] == S.order_qty.sum() and f["order_placed"] == bool(S.order_placed.any())
        assert f["shortfall"] == S.shortfall.sum()
        d = real["days"][real["days"].date.between(s, e)]
        names = sorted({n for n in (*d.event_name_1, *d.event_name_2) if isinstance(n, str) and n != "None"})  # "None" = no event
        assert f["events"] == ("; ".join(names) if names else "none")


@needs_cards
def test_50_random_month_cards_match_direct_queries(real):
    for c in _sample(real["cards"], "item_store_month"):
        k, f = c.key, c.fields
        sim = real["sim_by"][(k["store_id"], k["item_id"])]
        S = sim[sim.date.dt.strftime("%Y-%m") == k["month"]]
        assert len(S) == pd.Period(k["month"]).days_in_month
        assert f["total_units"] == S.demand.sum()
        assert f["mean_daily_units"] == pytest.approx(S.demand.mean(), abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass
        assert f["sd_daily_units"] == pytest.approx(S.demand.std(ddof=1), abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass
        assert f["shortfall_days"] == (S.shortfall > 0).sum()


@needs_cards
def test_50_random_order_cards_match_direct_queries(real):
    orders = real["orders"].set_index("order_id")
    for c in _sample(real["cards"], "order"):
        r, f = orders.loc[int(c.key["order_id"])], c.fields
        assert f["scheduled_days"] == r["days_for_shipment_scheduled"] and f["delivery_status"] == r["delivery_status"]
        assert f["shipping_mode"] == r["shipping_mode"] and f["market"] == r["market"] and f["region"] == r["order_region"]
        assert f["order_date"] == r["order_date"].strftime("%Y-%m-%d")
        assert f["is_cancelled"] == bool(r["is_cancelled"])
        if r["is_cancelled"]:
            assert f["delay_days"] is None and f["late"] is None
            assert f["recorded_real_days"] == r["days_for_shipping_real"] and f["recorded_delay_days"] == r["raw_delay_days"]
        else:
            assert f["real_days"] == r["days_for_shipping_real"] and f["delay_days"] == r["days_for_shipping_real"] - r["days_for_shipment_scheduled"]
            assert f["late"] == (f["delay_days"] > 0)


@needs_cards
def test_all_cancelled_orders_have_cards_with_raw_values(real):
    ords = real["orders"]
    by_id = {c.key["order_id"]: c for c in real["cards"] if c.card_type == "order"}
    assert len(by_id) == len(ords)
    can = ords[ords.is_cancelled]
    assert len(can) > 0 and all(by_id[str(i)].fields["delay_days"] is None for i in can["order_id"])


@needs_cards
def test_50_random_region_month_cards_match_direct_queries(real):
    o = real["all_orders"]  # the region x month cards aggregate every order, not just the sampled ones
    month = o["order_date"].dt.strftime("%Y-%m")
    for c in _sample(real["cards"], "region_month"):
        k, f = c.key, c.fields
        g = o[(o["order_region"] == k["region"]) & (month == k["month"])]
        ok = g[~g.is_cancelled]
        assert f["orders"] == len(g) and f["cancelled_orders"] == g.is_cancelled.sum()
        assert f["late_orders"] == int((ok.raw_delay_days > 0).sum())
        assert f["late_rate"] == pytest.approx((ok.raw_delay_days > 0).mean(), abs=1e-4)
        for mode in SHIPPING_MODES:
            m = ok[ok["shipping_mode"] == mode].raw_delay_days
            expect = None if m.empty else float(m.mean())
            got = f[_mode_field(mode)]
            assert (got is None) if expect is None else got == pytest.approx(expect, abs=0.0051)  # card values are rounded to 2 decimals; an exact half-step must pass


@needs_cards
def test_region_month_cards_are_built_from_all_orders_not_just_the_sample(real):
    from src.config import load_config
    min_orders = load_config()["cards"]["region_month_min_orders"]
    allo, sample = real["all_orders"], real["orders"]
    cells = allo.groupby(["order_region", allo["order_date"].dt.strftime("%Y-%m")]).size()
    cards = [c for c in real["cards"] if c.card_type == "region_month"]
    assert len(cards) == int((cells >= min_orders).sum())                     # one card per region-month cell of ALL orders
    assert sum(c.fields["orders"] for c in cards) > len(sample)                 # more orders than the sample holds
    assert sum(c.fields["orders"] for c in cards) <= len(allo)
    assert len([c for c in real["cards"] if c.card_type == "order"]) == len(sample)  # order cards stay one per sampled order
