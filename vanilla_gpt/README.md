# vanilla_gpt (side experiments, not part of the main pipeline)

Early API-based probes kept for reference. Nothing in `src/` imports from here.

- `main.py` - one GPT-4o call on a hand-written unanswerable prompt (needs `OPENAI_API_KEY` in `.env`).
- `run_vanilla.py` - batch-runs a prompts CSV (default `../initial_tests/numerical.csv`) through DeepSeek
  (needs `DEEPSEEK_API_KEY`). Run from the repo root: `python vanilla_gpt/run_vanilla.py`.

The main study uses local open-source models via Ollama (see `configs/base.yaml`). The reproducible
LLM-only baseline there is **B0**.
