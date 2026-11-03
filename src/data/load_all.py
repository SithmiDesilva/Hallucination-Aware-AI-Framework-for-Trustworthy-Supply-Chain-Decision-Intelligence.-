"""One command to build both processed datasets:  python -m src.data.load_all"""
from src.data.load_dataco import build_dataco
from src.data.load_m5 import build_m5

if __name__ == "__main__":
    m5 = build_m5()
    print(f"M5: {len(m5):,} long rows")
    dc = build_dataco()
    print(f"DataCo: {len(dc):,} orders")
