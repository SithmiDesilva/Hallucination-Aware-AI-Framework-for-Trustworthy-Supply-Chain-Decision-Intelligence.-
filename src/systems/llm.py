"""One LLM wrapper with two back ends, chosen by `generator.backend` in the config:
Ollama on the laptop, vLLM on Kaggle (not implemented yet: written when the Kaggle runs are set up).

Temperature 0 and a fixed seed, context capped at `num_ctx`. Use it as a context manager: the model stays
loaded for the whole pipeline step (one reload, not one per question) and is unloaded on exit when the
config says keep_alive: 0, so the next step (embedder, NLI verifier) has the GPU to itself.
`batch` appends every result to a JSONL log the moment it arrives and skips keys already in that log, so an
interrupted step resumes where it stopped.
"""
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

OLLAMA_URL = "http://localhost:11434"
SESSION_KEEP_ALIVE = "10m"


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool = False  # stopped at max tokens, so the text may be cut off
    model: str = ""


class LLMBackend:
    """The interface every back end provides; the systems and the run script rely on exactly this.

    Used as a context manager: the model stays loaded for the whole step and is unloaded on exit when the
    config says keep_alive: 0.
    """
    cfg: dict

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        raise NotImplementedError

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> LLMResult:
        raise NotImplementedError

    def gpu_share(self) -> float | None:
        return None

    def unload(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.cfg.get("keep_alive", 0) == 0:
            try:
                self.unload()
            except ConnectionError:
                pass


class OllamaLLM(LLMBackend):
    def __init__(self, generator_cfg: dict, seed: int = 42, url: str = OLLAMA_URL):
        self.cfg, self.seed, self.url = generator_cfg, seed, url

    # -- transport
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # the server answered with an error (e.g. the model failed to load)
            raise RuntimeError(f"Ollama returned HTTP {e.code} for {path}: {e.read().decode('utf-8', 'replace')[:500]}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama is not reachable at {self.url} ({e.reason}); start it with scripts/start_ollama.ps1") from e

    def _chat(self, system: str, user: str, max_tokens: int | None, schema: dict | None) -> LLMResult:
        c = self.cfg
        body = {
            "model": c["model"], "stream": False, "keep_alive": SESSION_KEEP_ALIVE,
            "options": {"num_ctx": c["num_ctx"], "temperature": c.get("temperature", 0), "top_p": c.get("top_p", 1),
                        "num_predict": max_tokens or c["max_new_tokens"], "seed": self.seed},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema is not None:
            body["format"] = schema
        t0 = time.perf_counter()
        out = self._post("/api/chat", body)
        seconds = time.perf_counter() - t0
        res = LLMResult(text=out["message"]["content"], prompt_tokens=out.get("prompt_eval_count", 0),
                        completion_tokens=out.get("eval_count", 0), seconds=seconds,
                        truncated=out.get("done_reason") == "length", model=c["model"])
        if res.prompt_tokens >= c["num_ctx"] - 1:  # Ollama would silently cut the prompt
            raise RuntimeError(f"prompt filled the whole {c['num_ctx']}-token context window; shorten it")
        return res

    # -- public API
    def chat(self, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        return self._chat(system, user, max_tokens, None)

    def chat_json(self, system: str, user: str, schema: dict, max_tokens: int | None = None) -> LLMResult:
        """Output constrained to a JSON schema; `text` is the JSON string."""
        return self._chat(system, user, max_tokens, schema)

    def gpu_share(self) -> float | None:
        """Fraction of the loaded model that sits on the GPU (1.0 = fully on GPU); None if it cannot be read."""
        try:
            with urllib.request.urlopen(self.url + "/api/ps", timeout=10) as r:
                for m in json.load(r)["models"]:
                    if m["name"] == self.cfg["model"] and m["size"]:
                        return m["size_vram"] / m["size"]
        except Exception:
            pass
        return None

    def unload(self) -> None:
        self._post("/api/generate", {"model": self.cfg["model"], "keep_alive": 0})


class VllmLLM(LLMBackend):
    def __init__(self, generator_cfg: dict, seed: int = 42):
        raise NotImplementedError("the vLLM (Kaggle) back end is written when the Kaggle runs are set up")


def make_llm(cfg: dict) -> LLMBackend:
    """The back end named by generator.backend."""
    backend = cfg["generator"]["backend"]
    if backend == "ollama":
        return OllamaLLM(cfg["generator"], seed=cfg["seed"])
    if backend == "vllm":
        return VllmLLM(cfg["generator"], seed=cfg["seed"])
    raise ValueError(f"unknown backend: {backend!r}")


def batch(llm, requests: Iterable[dict], log_path: str | Path) -> dict[str, LLMResult]:
    """Run {key, system, user, [max_tokens], [schema]} requests, logging each result as it arrives.

    Keys already present in the log are not re-run. Returns {key: LLMResult} for every request.
    """
    path = Path(log_path)
    done: dict[str, LLMResult] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = LLMResult(**{k: rec[k] for k in LLMResult.__dataclass_fields__})
    results = {}
    with open(path, "a", encoding="utf-8", newline="\n") as log:
        for req in requests:
            key = req["key"]
            if key not in done:
                if req.get("schema") is not None:
                    done[key] = llm.chat_json(req["system"], req["user"], req["schema"], req.get("max_tokens"))
                else:
                    done[key] = llm.chat(req["system"], req["user"], req.get("max_tokens"))
                log.write(json.dumps({"key": key, "system": req["system"], "user": req["user"], **asdict(done[key])}) + "\n")
                log.flush()
            results[key] = done[key]
    return results
