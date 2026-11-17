"""Prove the cancelled-order rule on the full DataCo file and on the 20,000-order sample.

    python -m src.data.dataco_checks

Writes everything to results/data_validation/ :
  dataco_cancelled_rule_checks.csv / .md   the one summary table for the thesis
  dataco_cancelled_crosstab.csv            Order Status x Delivery Status (all orders), supporting evidence
  dataco_cancelled_rule_checks.json        the same results, machine-readable
"""
import json

import pandas as pd

from src.config import ROOT, load_config
from src.data.load_dataco import CANCELLED_STATUS, STATUS_SIGN, clean, load_raw, sample_orders, to_orders

OUT = ROOT / "results" / "data_validation"
CANCELLED_ORDER_STATUSES = {"CANCELED", "SUSPECTED_FRAUD"}


def _row(check, claim, scope, n, value, note=""):
    return {"check": check, "claim": claim, "scope": scope, "n": int(n), "value": value, "note": note}


def check_delay_sign(orders: pd.DataFrame, scope: str) -> list[dict]:
    """Check 1: among cancelled orders, how many have a recorded delay > 0, = 0, < 0."""
    c = orders[orders["is_cancelled"]]
    d = c["raw_delay_days"]
    return [
        _row("1a", "cancelled orders (total)", scope, len(c), len(c)),
        _row("1b", "cancelled with recorded delay > 0", scope, (d > 0).sum(), int((d > 0).sum())),
        _row("1c", "cancelled with recorded delay = 0", scope, (d == 0).sum(), int((d == 0).sum())),
        _row("1d", "cancelled with recorded delay < 0", scope, (d < 0).sum(), int((d < 0).sum())),
    ]


def check_late_flag(orders: pd.DataFrame, items: pd.DataFrame | None, scope: str) -> list[dict]:
    """Check 2: for non-cancelled orders, late (delay > 0) == Late_delivery_risk, and Delivery Status is the sign of the delay."""
    nc = orders[~orders["is_cancelled"]]
    agree = (nc["late_delivery_risk"] == 1) == (nc["raw_delay_days"] > 0)
    rows = [_row("2a", "non-cancelled orders: Late_delivery_risk == (real > scheduled)", scope, len(nc),
                 f"{agree.mean() * 100:.4f}%", f"{int(agree.sum())}/{len(nc)} agree")]
    status_ok = nc["delivery_status"].map(STATUS_SIGN) == nc["raw_delay_days"].apply(lambda d: (d > 0) - (d < 0))
    rows.append(_row("2c", "non-cancelled orders: Delivery Status is the sign of the delay (Late > 0, Advance < 0, On time = 0)", scope, len(nc),
                     f"{status_ok.mean() * 100:.4f}%", f"{int(status_ok.sum())}/{len(nc)} agree"))
    if items is not None:
        inc = items[items["delivery_status"] != CANCELLED_STATUS]
        ia = (inc["late_delivery_risk"] == 1) == (inc["days_for_shipping_real"] > inc["days_for_shipment_scheduled"])
        rows.append(_row("2b", "same check at item-row level", scope, len(inc),
                         f"{ia.mean() * 100:.4f}%", f"{int(ia.sum())}/{len(inc)} agree"))
    return rows


def check_cancelled_definition(orders: pd.DataFrame, scope: str) -> tuple[list[dict], pd.DataFrame]:
    """Check 3: Delivery Status 'Shipping canceled' vs Order Status in {CANCELED, SUSPECTED_FRAUD}."""
    by_delivery = orders["delivery_status"] == CANCELLED_STATUS
    by_order = orders["order_status"].isin(CANCELLED_ORDER_STATUSES)
    both, only_d, only_o = (by_delivery & by_order).sum(), (by_delivery & ~by_order).sum(), (~by_delivery & by_order).sum()
    rows = [
        _row("3a", "cancelled by both Delivery Status and Order Status", scope, both, int(both)),
        _row("3b", "cancelled by Delivery Status only", scope, only_d, int(only_d)),
        _row("3c", "cancelled by Order Status only", scope, only_o, int(only_o)),
        _row("3d", "the two definitions agree on every order", scope, len(orders),
             bool(only_d == 0 and only_o == 0)),
    ]
    crosstab = pd.crosstab(orders["order_status"], orders["delivery_status"])
    return rows, crosstab


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def run() -> pd.DataFrame:
    cfg = load_config()
    items = clean(load_raw())
    orders = to_orders(items)
    sample = sample_orders(orders, cfg["data"]["dataco_orders"], cfg["seed"])

    rows: list[dict] = []
    crosstab = None
    for scope, o, item_rows in (
        (f"all orders ({len(orders):,})", orders, items),
        (f"20k sample ({len(sample):,})", sample, None),
    ):
        rows += check_delay_sign(o, scope)
        rows += check_late_flag(o, item_rows, scope)
        r3, ct = check_cancelled_definition(o, scope)
        rows += r3
        if crosstab is None:
            crosstab = ct
    table = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "dataco_cancelled_rule_checks.csv", index=False)
    (OUT / "dataco_cancelled_rule_checks.md").write_text(to_markdown(table), encoding="utf-8")
    crosstab.to_csv(OUT / "dataco_cancelled_crosstab.csv")
    (OUT / "dataco_cancelled_rule_checks.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return table


if __name__ == "__main__":
    print(to_markdown(run()))
