# Data validation findings

Evidence behind data-definition decisions, kept in one place for the thesis. Regenerate with the command shown.

| Finding | Command | Files |
|---|---|---|
| DataCo cancelled-order rule (delay sign, late-flag identity, definition of "cancelled") | `python -m src.data.dataco_checks` | `dataco_cancelled_rule_checks.{csv,md,json}`, `dataco_cancelled_crosstab.csv` |
| M5 ambiguities (zero sales vs stockout, observed vs simulated sales, not-on-sale vs zero, week vs month) | `python -m src.data.m5_checks` | `m5_ambiguity_checks.{csv,md,json}` |

Both feed the conflicting-evidence slice. DataCo orders held out for it are listed in
`benchmark/conflict_slice/dataco_cancelled_pool.csv` (`python -m src.bench.conflict_slice`).

## What each file is

These are generated outputs (CSV/JSON cannot hold comments and the scripts overwrite them), so each file is described here.

| File | Contents |
|---|---|
| `dataco_cancelled_rule_checks.csv` / `.json` / `.md` | Same table in three formats: counts for DataCo cancelled orders (recorded delay >0 / =0 / <0), whether `Late_delivery_risk` equals (real > scheduled) for non-cancelled orders, and whether "cancelled" is defined by Delivery Status, Order Status, or both. |
| `dataco_cancelled_crosstab.csv` | Order Status x Delivery Status counts; shows cancelled and suspected-fraud orders only ever carry "Shipping canceled". |
| `m5_ambiguity_checks.csv` / `.json` / `.md` | Same table in three formats: how often M5 zero-sales days are ambiguous (no demand vs stockout), observed vs simulated sales disagreements, and related week/month ambiguities. |
