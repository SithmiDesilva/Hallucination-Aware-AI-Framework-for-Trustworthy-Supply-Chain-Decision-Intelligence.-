"""Config and path helpers shared by all packages."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CARDS = ROOT / "data" / "cards"


def load_config(path: str | Path = ROOT / "configs" / "base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
