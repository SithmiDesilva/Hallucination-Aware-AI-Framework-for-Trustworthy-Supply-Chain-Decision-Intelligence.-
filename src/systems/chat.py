"""Chat with a system in the terminal: type a question, read the answer.

    python -m src.systems.chat              (B0, the LLM with no data; the other systems are added as they are built)

Every question is answered on its own: the system keeps no memory of earlier turns, exactly as in the
benchmark runs. Each exchange is appended to runs/chat/<date>_<system>.jsonl so manual tests leave a record.
Commands: /help  /quit  (Ctrl+C or Ctrl+D also quit).
"""
import argparse
import datetime
import json
import textwrap
from pathlib import Path

from src.cards.schemas import Question
from src.config import RUNS, load_config
from src.systems import b0
from src.systems.llm import make_llm

SYSTEMS = {"b0": (b0, "LLM only: no evidence cards, so anything specific it says is from memory")}
WIDTH = 100
HELP = "Type a question and press Enter. Commands: /help  /quit"


def manual_question(text: str, n: int) -> Question:
    """Wrap typed text in a Question record (the gold fields are placeholders; B0 never reads them)."""
    return Question(qid=f"CHAT-{n:03d}", split="dev", domain="inventory", template_id="manual", question_type="lookup",
                    answerable=False, unanswerable_type="absent_attribute", question=text, reference_answer="n/a")


def _wrap(text: str) -> str:
    return "\n".join(textwrap.fill(line, WIDTH, subsequent_indent="    ") if line.strip() else "" for line in text.splitlines())


def chat(system: str = "b0", *, config_path: Path | None = None, llm=None, input_fn=input, output_fn=print,
         log_dir: Path | None = None) -> int:
    """Run the chat loop; returns the number of questions answered. `llm`, `input_fn` and `output_fn` are
    injectable so the loop can be tested without a model or a keyboard."""
    module, description = SYSTEMS[system]
    cfg = load_config(config_path) if config_path else load_config()
    log_path = Path(log_dir or RUNS / "chat") / f"{datetime.date.today():%Y-%m-%d}_{system}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = system.upper()

    output_fn(f"{label} chat ({description}).\nEach question is answered on its own, with no memory of earlier ones. "
              f"{HELP}\nThe first answer loads the model and can take a minute.\nLog: {log_path}\n")
    n = 0
    with (llm if llm is not None else make_llm(cfg)) as model:
        while True:
            try:
                text = input_fn("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                output_fn("")
                break
            if not text:
                continue
            if text.lower() in ("/quit", "/exit", "/q"):
                break
            if text.lower() == "/help":
                output_fn(HELP + "\n")
                continue
            if text.startswith("/"):
                output_fn(f"Unknown command {text.split()[0]!r}. {HELP}\n")
                continue

            n += 1
            output_fn("  thinking...")
            try:
                ans = module.answer(manual_question(text, n), None, model, prompt=cfg["prompts"][system])
            except (ConnectionError, RuntimeError) as e:
                output_fn(f"\n{e}\n")
                n -= 1
                break
            output_fn(f"\n{label}: {_wrap(ans.final_answer)}\n    [{ans.seconds:.1f}s | abstained: {'yes' if ans.abstained else 'no'}"
                      f"{' | CUT OFF at the token limit' if ans.truncated else ''}]\n")
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"), "question": text,
                                    **ans.model_dump()}) + "\n")
            if n == 1 and hasattr(model, "gpu_share"):
                share = model.gpu_share()
                if share is not None and share < 0.99:
                    output_fn(f"  Note: the model is only {share:.0%} on the GPU, so answers are slower; "
                              "start the server with scripts\\start_ollama.ps1\n")
    output_fn("Bye.")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="b0", choices=sorted(SYSTEMS))
    ap.add_argument("--config", type=Path)
    a = ap.parse_args()
    chat(a.system, config_path=a.config)
