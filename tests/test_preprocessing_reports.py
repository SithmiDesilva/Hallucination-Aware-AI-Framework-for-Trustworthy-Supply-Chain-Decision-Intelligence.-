"""The two preprocessing checklists (M5 and DataCo) must pass in full: they are the thesis evidence that every
Do and Don't rule holds on the real data. Skipped until the raw and processed data exist."""
import pytest

from src.config import CARDS, PROCESSED, RAW

READY = (
    (RAW / "m5" / "sales_train_evaluation.csv").exists() and (RAW / "dataco" / "DataCoSupplyChainDataset.csv").exists()
    and (PROCESSED / "m5_long.parquet").exists() and (PROCESSED / "m5_sim.parquet").exists()
    and (PROCESSED / "dataco_orders_all.parquet").exists() and (CARDS / "cards.jsonl").exists()
)
needs_data = pytest.mark.skipif(not READY, reason="raw or processed data not available")


@needs_data
def test_the_m5_checklist_passes_in_full():
    from src.data.m5_preprocessing_report import run
    table = run()
    assert len(table) == 15 and set(table["rule_type"]) == {"Do", "Don't"}
    assert (table["result"] == "PASS").all(), table[table["result"] != "PASS"][["rule", "evidence"]].to_string()


@needs_data
def test_the_dataco_checklist_passes_in_full():
    from src.data.dataco_preprocessing_report import run
    table = run()
    assert len(table) == 16 and set(table["rule_type"]) == {"Do", "Don't"}
    assert (table["result"] == "PASS").all(), table[table["result"] != "PASS"][["rule", "evidence"]].to_string()
