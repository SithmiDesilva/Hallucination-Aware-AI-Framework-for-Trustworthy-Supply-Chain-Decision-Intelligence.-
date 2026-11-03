"""Print SHA-256 hashes of everything in data/raw so they can be recorded in data/raw/README.md."""
import hashlib

from src.config import RAW


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    for p in sorted(RAW.rglob("*.csv")):
        print(f"{sha256(p)}  {p.relative_to(RAW).as_posix()}")
