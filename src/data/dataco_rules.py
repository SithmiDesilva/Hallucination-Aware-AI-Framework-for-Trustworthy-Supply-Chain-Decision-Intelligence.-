"""Ground-truth semantics for DataCo delivery questions. Question generation, the gold labeller and
the verifier's lookups all read lateness through this module so the rule lives in one place.

Rule: a cancelled order has no delivery, so it has no delay and cannot be late.
"""
import pandas as pd

CANCELLED_LATE_ANSWER = "The order was cancelled, so there is no delivery to be late."


def lateness_gold(order: pd.Series | dict) -> dict:
    """Gold answer to 'was order X late?' for one processed order row."""
    if bool(order["is_cancelled"]):
        return {"outcome": "cancelled", "late": None, "delay_days": None, "answer": CANCELLED_LATE_ANSWER}
    late = bool(order["late"])
    delay = int(order["shipping_delay_days"])
    return {
        "outcome": "late" if late else "not_late",
        "late": late,
        "delay_days": delay,
        "answer": f"The order was {'late' if late else 'not late'}; actual minus scheduled shipping days = {delay}.",
    }
