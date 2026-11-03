# Raw data (git-ignored)

Place downloads here, unmodified:

    data/raw/m5/sales_train_evaluation.csv
    data/raw/m5/calendar.csv
    data/raw/m5/sell_prices.csv
    data/raw/dataco/DataCoSupplyChainDataset.csv
    data/raw/dataco/DescriptionDataCoSupplyChain.csv

Sources: M5 Forecasting - Accuracy (Kaggle); DataCo Smart Supply Chain (Mendeley Data, V5).
Note: M5 uses `sales_train_evaluation.csv` (1941 days), not `sales_train_validation.csv`.

Record hashes with `python -m src.data.hash_raw` and paste the output below.

## File hashes (SHA-256)

Recorded 2026-10-30 (`DescriptionDataCoSupplyChain.csv` not downloaded; not used by the pipeline).

    fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6  dataco/DataCoSupplyChainDataset.csv
    d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5  m5/calendar.csv
    4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0  m5/sales_train_evaluation.csv
    9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef  m5/sell_prices.csv
    f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724  m5/sales_train_validation.csv  (not used)
