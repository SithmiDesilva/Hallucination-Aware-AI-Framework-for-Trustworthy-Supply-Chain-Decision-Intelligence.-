"""Unit tests for the inventory simulator (src/data/simulate_inventory.py).

Check invariants (stock never negative, units sold + shortfall = demand), order arrival after the
lead time, no safety stock/shortfall under constant demand, short series shorter than warmup
returning empty, and a hand-computed day-by-day example.
"""
import math

import numpy as np
import pytest

from src.data.simulate_inventory import SimParams, simulate_series

P = SimParams(lead_time=2, review_cover=1, z=1.0, window=3, warmup=3)
RNG = np.random.default_rng(0)


def random_demand(n=300):
    return RNG.poisson(5, n).astype(float)


def test_stock_never_negative():
    sim = simulate_series(random_demand(), SimParams())
    assert (sim["on_hand_end"] >= 0).all()


def test_units_sold_plus_shortfall_equals_demand():
    sim = simulate_series(random_demand(), SimParams())
    assert np.allclose(sim["sim_units_sold"] + sim["shortfall"], sim["demand"])


def test_order_arrives_lead_time_later():
    p = SimParams()
    sim = simulate_series(random_demand(), p)
    placed = sim[sim["order_placed"]]
    assert len(placed) > 0
    sim = sim.set_index("day_idx")
    for t, row in placed.set_index("day_idx").iterrows():
        arrive = t + p.lead_time
        if arrive in sim.index:
            before = sim.loc[arrive - 1, "on_order"] if arrive - 1 in sim.index else None
            # on-order drops by the ordered quantity (at least) when the order lands
            assert sim.loc[arrive, "on_order"] <= before + sim.loc[arrive, "order_qty"] - row["order_qty"]


def test_constant_demand_has_no_safety_stock_or_shortfall():
    sim = simulate_series(np.full(200, 4.0), SimParams())
    assert (sim["safety_stock"] == 0).all()
    assert (sim["shortfall"] == 0).all()


def test_series_shorter_than_warmup_is_empty():
    assert simulate_series(np.ones(10), SimParams()).empty


def test_hand_computed_days():
    # L=2, R=1, z=1, window=3, warmup=3, demand 2,4,6 | 5,3,1
    sim = simulate_series(np.array([2, 4, 6, 5, 3, 1.0]), P).set_index("day_idx")
    # day 3: window [2,4,6] -> mu=4, sd=2; SS=2*sqrt(2); ROP=8+SS; S=ROP+4
    ss3 = 2 * math.sqrt(2)
    rop3 = 8 + ss3
    s3 = rop3 + 4
    d3 = sim.loc[3]
    assert d3["safety_stock"] == pytest.approx(ss3)
    assert d3["rop"] == pytest.approx(rop3)
    assert d3["on_hand_end"] == math.ceil(s3) - 5  # 15 - 5 = 10
    assert d3["order_placed"] and d3["order_qty"] == math.ceil(s3 - 10)  # position 10 <= 10.83 -> 5
    # day 4: window [4,6,5] -> mu=5, sd=1; ROP=10+sqrt2=11.41; position 7+5=12 > ROP -> no order
    d4 = sim.loc[4]
    assert d4["rop"] == pytest.approx(10 + math.sqrt(2))
    assert d4["on_hand_end"] == 7 and d4["on_order"] == 5 and not d4["order_placed"]
    # day 5: the day-3 order lands (3 + L): 7 + 5 - 1 = 11 on hand. Window [6,5,3] -> mu=14/3,
    # sd=sqrt(7/3); ROP = 28/3 + sqrt(7/3)*sqrt(2) ~ 11.49 >= position 11 -> reorder ceil(S - 11) = 6
    d5 = sim.loc[5]
    assert d5["on_hand_end"] == 11
    assert d5["order_placed"] and d5["order_qty"] == 6 and d5["on_order"] == 6
