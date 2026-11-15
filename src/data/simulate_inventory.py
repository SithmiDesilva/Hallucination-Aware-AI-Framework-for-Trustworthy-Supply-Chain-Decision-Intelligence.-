"""Inventory simulation: observed daily sales are treated as demand under a fixed (s, S)-style policy.

    SS_t  = z * sigma_t * sqrt(L)
    ROP_t = mu_t * L + SS_t
    S_t   = ROP_t + mu_t * R

mu_t / sigma_t use the 28 days BEFORE day t (no look-ahead; sample std, ddof=1). The simulation
starts after the warm-up with on-hand = ceil(S_t). Quantities are whole units. Daily order:
  1. receive orders due today   2. sell min(on-hand, demand) -> sim_units_sold (differs from observed sales on shortfall days), record shortfall
  3. if inventory position <= ROP, order ceil(S - position), arriving L days later
  4. save the state
Parameters are fixed before any question is written (configs/base.yaml).
"""
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config import PROCESSED, load_config

STATE_COLS = [
    "demand", "sim_units_sold", "shortfall", "on_hand_end", "on_order",
    "rop", "safety_stock", "order_placed", "order_qty",
]


@dataclass(frozen=True)
class SimParams:
    lead_time: int = 7
    review_cover: int = 7
    z: float = 1.645
    window: int = 28
    warmup: int = 28


def params(cfg: dict | None = None) -> SimParams:
    """Simulation parameters from the config; z is derived from the service level (0.95 -> 1.645)."""
    s = (cfg or load_config())["simulation"]
    z = round(NormalDist().inv_cdf(s["service_level"]), 3)
    return SimParams(s["lead_time"], s["review"], z, s["window"], s.get("warmup", s["window"]))


def rolling_stats(demand: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample std of the `window` days BEFORE each day t (NaN while t < window)."""
    demand = np.asarray(demand, dtype=float)
    mu = np.full(len(demand), np.nan)
    sigma = np.full(len(demand), np.nan)
    for t in range(window, len(demand)):
        hist = demand[t - window:t]
        mu[t] = hist.mean()
        sigma[t] = hist.std(ddof=1) if window > 1 else 0.0
    return mu, sigma


def simulate_series(demand: np.ndarray, p: SimParams) -> pd.DataFrame:
    """Simulate one series. Returns one row per day from index p.warmup onward, with `day_idx`."""
    demand = np.asarray(demand, dtype=float)
    start = max(p.warmup, p.window)
    n = len(demand)
    if n <= start:
        return pd.DataFrame(columns=["day_idx", *STATE_COLS])

    mu_all, sigma_all = rolling_stats(demand, p.window)
    rows = []
    pending: list[tuple[int, int]] = []  # (arrival day index, quantity)
    on_hand = None
    for t in range(start, n):
        mu, sigma = mu_all[t], sigma_all[t]
        ss = p.z * sigma * math.sqrt(p.lead_time)
        rop = mu * p.lead_time + ss
        s_level = rop + mu * p.review_cover
        if on_hand is None:
            on_hand = math.ceil(s_level)

        # 1. receive
        arrived = sum(q for a, q in pending if a == t)
        pending = [(a, q) for a, q in pending if a != t]
        on_hand += arrived
        # 2. sell
        sold = min(on_hand, demand[t])
        shortfall = demand[t] - sold
        on_hand -= sold
        # 3. reorder
        on_order = sum(q for _, q in pending)
        position = on_hand + on_order
        placed, qty = False, 0
        if position <= rop:
            qty = math.ceil(s_level - position)
            if qty > 0:
                placed = True
                pending.append((t + p.lead_time, qty))
                on_order += qty
            else:
                qty = 0
        # 4. save
        rows.append((t, demand[t], sold, shortfall, on_hand, on_order, rop, ss, placed, qty))
    return pd.DataFrame(rows, columns=["day_idx", *STATE_COLS])


def simulate_all(long: pd.DataFrame, p: SimParams) -> pd.DataFrame:
    """Run every (store, item) series of the processed M5 long table (already sorted by date)."""
    parts = []
    for (store, item), g in long.groupby(["store_id", "item_id"], sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        sim = simulate_series(g["units"].to_numpy(), p)
        if sim.empty:
            continue
        sim["store_id"], sim["item_id"] = store, item
        sim["date"] = g["date"].to_numpy()[sim["day_idx"].to_numpy()]
        sim["wm_yr_wk"] = g["wm_yr_wk"].to_numpy()[sim["day_idx"].to_numpy()]
        parts.append(sim)
    return pd.concat(parts, ignore_index=True)


def build_simulation(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    long = pd.read_parquet(PROCESSED / "m5_long.parquet")
    out = simulate_all(long, params(cfg))
    out.to_parquet(PROCESSED / "m5_sim.parquet", index=False)
    return out


if __name__ == "__main__":
    df = build_simulation()
    print(f"m5_sim.parquet: {len(df):,} rows, total shortfall {df['shortfall'].sum():,.0f} units")
