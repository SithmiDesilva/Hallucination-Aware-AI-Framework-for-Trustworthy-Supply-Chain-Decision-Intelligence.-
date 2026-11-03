"""Phase 0 check: the generator answers a prompt, returns schema-valid JSON, and sits fully on the GPU.

    python -m src.systems.check_ollama
"""
import json
import urllib.request

from src.config import load_config

URL = "http://localhost:11434"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "units": {"type": "integer"}},
    "required": ["answer", "units"],
}


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    m = load_config()["models"]
    out = _post("/api/chat", {
        "model": m["generator"], "stream": False, "format": SCHEMA, "keep_alive": "5m",
        "options": {"num_ctx": m["generator_num_ctx"], "temperature": 0},
        "messages": [{"role": "user", "content": "Demand last month was 1200 units. "
                      "Forecast next month with 15% growth. Reply as JSON."}],
    })
    parsed = json.loads(out["message"]["content"])
    assert set(SCHEMA["required"]) <= set(parsed), parsed
    print("schema-valid JSON:", parsed)

    with urllib.request.urlopen(URL + "/api/ps", timeout=30) as r:
        for mod in json.load(r)["models"]:
            gpu = mod["size_vram"] / mod["size"]
            print(f"{mod['name']}: {gpu:.0%} on GPU")
            assert gpu > 0.99, "layers offloaded to CPU: lower num_ctx or try the other model"
    _post("/api/generate", {"model": m["generator"], "keep_alive": 0})  # unload


if __name__ == "__main__":
    main()
