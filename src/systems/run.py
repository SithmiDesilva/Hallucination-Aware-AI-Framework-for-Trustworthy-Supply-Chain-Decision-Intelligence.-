"""Run one system over one split and write runs/<date>_<system>_<split>_<git-hash>/ :

    config.yaml   the exact config used, plus run metadata      outputs.jsonl   one Answer per line
    env.txt       pip freeze, GPU, CUDA, Ollama version and model digest

    python -m src.systems.run --system b0 --split dev [--questions PATH] [--limit N] [--allow-dirty] [--resume RUN_DIR]

Rules from the plan: a run refuses to start on a dirty git tree (use --allow-dirty for development runs; the
folder name then ends in -dirty), the test split needs frozen benchmark checksums, and a run folder is never
overwritten (--resume continues an interrupted run in place, skipping questions already answered).
Every answer is written and flushed as soon as it exists, so a crash or sleep loses nothing.
"""
import argparse
import datetime
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from src.cards.schemas import Answer, Question, read_jsonl
from src.config import ROOT, RUNS, load_config
from src.systems import b0
from src.systems.llm import make_llm

SYSTEMS = {"b0": b0}  # b1, b2, b3 and the V variants are added as they are built


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_hash() -> str:
    try:
        return _git("rev-parse", "--short", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def git_dirty() -> bool:
    try:
        return bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def new_run_dir(runs_dir: Path, system: str, split: str, dirty: bool) -> Path:
    base = f"{datetime.date.today():%Y-%m-%d}_{system}_{split}_{git_hash()}{'-dirty' if dirty else ''}"
    path, n = runs_dir / base, 1
    while path.exists():  # never overwrite a run folder
        n += 1
        path = runs_dir / f"{base}_{n}"
    return path


def write_env(path: Path, cfg: dict) -> None:
    parts = []

    def section(title: str, fn) -> None:
        try:
            parts.append(f"## {title}\n{fn()}\n")
        except Exception as e:  # an unavailable tool must not stop a run
            parts.append(f"## {title}\nunavailable: {e}\n")

    section("python", lambda: sys.version)
    section("pip freeze", lambda: subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.strip())
    section("gpu", lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip())
    section("cuda", lambda: next(l for l in subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.splitlines() if "CUDA Version" in l).strip())

    def ollama() -> str:
        base = "http://localhost:11434"
        version = json.load(urllib.request.urlopen(base + "/api/version", timeout=5))["version"]
        tags = json.load(urllib.request.urlopen(base + "/api/tags", timeout=5))["models"]
        digest = next((m["digest"][:12] for m in tags if m["name"] == cfg["generator"]["model"]), "model not found")
        return f"ollama {version}; {cfg['generator']['model']} digest {digest}; server_env (configured) {cfg['generator'].get('server_env')}"

    section("ollama", ollama)
    path.write_text("\n".join(parts), encoding="utf-8")


def resolve_questions_path(path: Path | str | None, split: str) -> Path:
    """The questions file: as given (relative to the current folder), else relative to the repo root.
    Without --questions it is benchmark/<split>.jsonl, which exists only after the benchmark is built."""
    wanted = Path(path) if path else ROOT / "benchmark" / f"{split}.jsonl"
    for candidate in (wanted, ROOT / wanted):
        if candidate.is_file():
            return candidate
    hint = ("" if path else f"; the {split!r} benchmark is not built yet, so pass --questions PATH")
    raise SystemExit(f"questions file not found: {wanted} (looked in {Path.cwd()} and in {ROOT}){hint}")


def record_gpu_share(run_dir: Path, cfg: dict, share: float | None) -> None:
    """env.txt lists the configured server settings; this adds what the server actually did."""
    if share is None:
        return
    with open(run_dir / "env.txt", "a", encoding="utf-8") as f:
        f.write(f"\n## measured\n{cfg['generator']['model']} is {share:.0%} on the GPU\n")
    if share < 0.99:
        print(f"  WARNING: the model is only {share:.0%} on the GPU, so this run is slower than planned; "
              "start the server with scripts/start_ollama.ps1")


def main(system: str, split: str, *, config_path: Path | None = None, questions_path: Path | None = None,
         limit: int | None = None, allow_dirty: bool = False, resume: Path | None = None,
         llm=None, runs_dir: Path | None = None) -> Path:
    if system not in SYSTEMS:
        raise SystemExit(f"unknown system {system!r}; available: {sorted(SYSTEMS)}")
    cfg = load_config(config_path) if config_path else load_config()
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise SystemExit("the git working tree has uncommitted changes; commit them, or use --allow-dirty for a development run")
    if split == "test" and not (ROOT / cfg["benchmark"]["checksums"]).exists():
        raise SystemExit("the test split needs the frozen benchmark checksums; run `make validate` first")

    qpath = resolve_questions_path(questions_path, split)
    questions = list(read_jsonl(qpath, Question))
    if limit:
        questions = questions[:limit]

    if resume:
        run_dir = Path(resume)
        outputs = run_dir / "outputs.jsonl"
        done = {a.qid for a in read_jsonl(outputs, Answer)} if outputs.exists() else set()
    else:
        run_dir, done = new_run_dir(Path(runs_dir) if runs_dir else RUNS, system, split, dirty), set()
        run_dir.mkdir(parents=True)
        meta = {"system": system, "split": split, "git_hash": git_hash(), "dirty": dirty,
                "questions_file": str(qpath), "n_questions": len(questions)}
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "run": meta}, sort_keys=False), encoding="utf-8")
        write_env(run_dir / "env.txt", cfg)

    todo = [q for q in questions if q.qid not in done]
    print(f"{run_dir.name}: {len(done)} done, {len(todo)} to run")
    module = SYSTEMS[system]
    with (llm if llm is not None else make_llm(cfg)) as model, open(run_dir / "outputs.jsonl", "a", encoding="utf-8", newline="\n") as out:
        for i, q in enumerate(todo, 1):
            # the prompt version comes from the config (so it is recorded with the run); b0 takes no cards,
            # retrieval-based systems get theirs from runs/retrieval_<split>.jsonl
            ans = module.answer(q, None, model, prompt=cfg["prompts"][system])
            out.write(ans.model_dump_json() + "\n")
            out.flush()
            print(f"  [{i}/{len(todo)}] {q.qid}  {ans.seconds:.1f}s{'  (truncated)' if ans.truncated else ''}")
            if i == 1 and hasattr(model, "gpu_share"):
                record_gpu_share(run_dir, cfg, model.gpu_share())
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--questions", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--resume", type=Path)
    a = ap.parse_args()
    print(main(a.system, a.split, questions_path=a.questions, limit=a.limit, allow_dirty=a.allow_dirty, resume=a.resume))
