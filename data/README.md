# Data definitions

## DataCo (orders)
- **is_cancelled:** `Delivery Status == "Shipping canceled"`. This is identical to `Order Status` in
  {`CANCELED`, `SUSPECTED_FRAUD`} on every order (checked on all 65,752 orders).
- **raw_delay_days:** `Days for shipping (real)` minus `Days for shipment (scheduled)`, kept for every order.
- **shipping_delay_days:** `raw_delay_days`, **null for cancelled orders** (nothing was delivered).
- **late:** `Late_delivery_risk == 1`, **null for cancelled orders**. For every non-cancelled order this equals
  `real > scheduled` (100%), so lateness and delay never disagree.
- **status:** `Delivery Status` as given. The raw `Days for shipping (real)` and `Late_delivery_risk` columns are untouched.
- **Gold answer for "was a cancelled order late?":** "The order was cancelled, so there is no delivery to be late."
  The rule lives in `src/data/dataco_rules.py` and is used by question generation, the gold labeller and the
  verifier's lookups. Late rates and mean delays are computed over non-cancelled orders only.
- Evidence for the rule: `python -m src.data.dataco_checks` -> `results/data_validation/`.
- Source rows are order *items*; they are aggregated to one row per `Order Id`, and the build fails
  if order-level fields disagree within an order. Customer name, email, password and street are dropped.
- 20,000 orders sampled with seed 42, stratified by `Market` x `Shipping Mode` (19,999 after rounding).

## M5 (inventory)
- 3 stores (one each from CA, TX, WI) and 300 items stratified over FOODS / HOBBIES / HOUSEHOLD,
  seed 42 -> 900 item-store series. Sampling happens before the wide-to-long reshape.
- Only priced days are kept (a missing `sell_price` means the item was not on sale).
- Weeks are Walmart weeks (`wm_yr_wk`, Saturday to Friday).
- Inventory is simulated by `src/data/simulate_inventory.py` with the parameters in `configs/base.yaml`
  (L = 7, R = 7, z = 1.645, trailing 28-day window with no look-ahead, 28-day warm-up, lost sales).

Rebuild everything with `python -m src.data.load_all` then `python -m src.data.simulate_inventory`.
