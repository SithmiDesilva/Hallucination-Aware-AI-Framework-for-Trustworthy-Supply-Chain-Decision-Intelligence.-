"""Fix the pool of DataCo orders reserved for the conflicting-evidence slice.

Every cancelled order is held out of the main question set. Cancelled orders still carry a raw
shipping-delay value (see results/data_validation), so a system that reads it will answer "shipped N
days late" about an order that never shipped. Strata by the sign of the raw delay:

    delay_gt0  the headline case: the raw field says 'late'  (474 of 854 in the 20k sample)
    delay_eq0  the raw field says 'on time'
    delay_lt0  the raw field says 'early'

    python -m src.bench.conflict_slice     ->  benchmark/conflict_slice/dataco_cancelled_pool.csv
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, ROOT

OUT = ROOT / "benchmark" / "conflict_slice"
KEEP = ["Order Id", "Order Status", "Shipping Mode", "Market", "Order Region",
        "Days for shipping (real)", "Days for shipment (scheduled)", "raw_delay_days"]


def build_pool(orders: pd.DataFrame) -> pd.DataFrame:
    pool = orders.loc[orders["is_cancelled"], KEEP].copy()
    pool["stratum"] = np.select(
        [pool["raw_delay_days"] > 0, pool["raw_delay_days"] == 0], ["delay_gt0", "delay_eq0"], default="delay_lt0")
    return pool.sort_values("Order Id").reset_index(drop=True)


if __name__ == "__main__":
    pool = build_pool(pd.read_parquet(PROCESSED / "dataco_orders.parquet"))
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT / "dataco_cancelled_pool.csv", index=False)
    print(f"{len(pool)} cancelled orders reserved for the conflict slice")
    print(pool["stratum"].value_counts().to_string())
