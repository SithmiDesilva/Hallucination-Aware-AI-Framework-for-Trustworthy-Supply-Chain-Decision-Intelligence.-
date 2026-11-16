"""Tests for the B0 pipeline: prompt formatting, abstention rule, LLM wrapper, B0 itself and the run script.

All use fakes (no GPU, no server) except the last test, which talks to the real Ollama server and runs only
with RUN_LLM_TESTS=1: it checks the smoke questions end to end and that the same input gives the same output.
"""
import json
import os
import urllib.error
from email.message import Message
from pathlib import Path

import pytest
import yaml

from src.cards.schemas import Answer, Card, Question, read_jsonl
from src.systems import b0, run
from src.systems.format import cards_block, fill, fill_prompt, load_prompt
from src.systems.llm import LLMResult, OllamaLLM, batch, make_llm

SMOKE = Path(__file__).parent / "fixtures" / "smoke_questions.jsonl"
GEN_CFG = {"backend": "ollama", "model": "llama3.1:8b-instruct-q4_K_M", "num_ctx": 4096, "max_new_tokens": 300,
           "temperature": 0, "keep_alive": 0}


def questions():
    return list(read_jsonl(SMOKE, Question))


# ----------------------------------------------------------------------------- format
def test_fill_keeps_literal_json_braces_and_fills_placeholders():
    out = fill('Reply as {"answer": "...", "units": 0}. Question: {{ q }}', q="how many?")
    assert out == 'Reply as {"answer": "...", "units": 0}. Question: how many?'


def test_fill_raises_on_a_missing_placeholder():
    with pytest.raises(Exception):
        fill("Question: {{ question }}")


def test_b0_prompt_is_the_plans_text():
    system, user = fill_prompt("gen_b0_v1", question="How many units?")
    assert system == (
        "You are an assistant for supply-chain operations. Answer the user's question about inventory or deliveries.\n\n"
        "Rules:\n"
        "1. Give specific numbers, dates and entities in your answer.\n"
        "2. When the question needs a calculation (a total, an average, a difference, a rate), show the numbers you used and the result.\n"
        "3. Answer in at most 5 sentences.")
    assert user == "Question: How many units?\nAnswer:"
    assert load_prompt("gen_b0_v1")[1].count("{{") == 1


def test_cards_block_format():
    c = Card(card_id="DC-ORD-1", source="dataco", card_type="order", key={"order_id": "1"}, fields={"x": 1}, text="Order 1 ...")
    assert cards_block([c, c]) == "[DC-ORD-1] Order 1 ...\n[DC-ORD-1] Order 1 ..."


# ----------------------------------------------------------------------------- LLM wrapper (HTTP faked)
class FakeOllama(OllamaLLM):
    def __init__(self, replies=None, **kw):
        super().__init__(GEN_CFG, **kw)
        self.calls, self.replies = [], list(replies or [])

    def _post(self, path, body):
        self.calls.append((path, body))
        if path == "/api/chat":
            r = self.replies.pop(0) if self.replies else {"text": "ok"}
            return {"message": {"content": r["text"]}, "prompt_eval_count": r.get("prompt", 50), "eval_count": 7,
                    "done_reason": r.get("done", "stop")}
        return {}


def test_chat_sends_deterministic_options_and_returns_a_result():
    llm = FakeOllama(seed=42)
    r = llm.chat("sys", "hello")
    path, body = llm.calls[0]
    assert path == "/api/chat" and body["stream"] is False and body["model"] == GEN_CFG["model"]
    assert body["options"] == {"num_ctx": 4096, "temperature": 0, "top_p": 1, "num_predict": 300, "seed": 42}
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert "format" not in body
    assert (r.text, r.completion_tokens, r.truncated) == ("ok", 7, False) and r.seconds >= 0


def test_chat_json_passes_the_schema_and_truncation_is_flagged():
    llm = FakeOllama(replies=[{"text": '{"a": 1}', "done": "length"}])
    r = llm.chat_json("s", "u", {"type": "object"}, max_tokens=50)
    body = llm.calls[0][1]
    assert body["format"] == {"type": "object"} and body["options"]["num_predict"] == 50
    assert r.truncated is True


def test_a_prompt_that_fills_the_context_window_is_an_error():
    with pytest.raises(RuntimeError):
        FakeOllama(replies=[{"text": "x", "prompt": 4096}]).chat("s", "u")


def test_context_manager_unloads_when_keep_alive_is_zero():
    llm = FakeOllama()
    with llm:
        llm.chat("s", "u")
    assert llm.calls[-1][0] == "/api/generate" and llm.calls[-1][1]["keep_alive"] == 0


def test_unreachable_server_gives_a_helpful_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ConnectionError, match="start_ollama"):
        OllamaLLM(GEN_CFG).chat("s", "u")


def test_a_server_side_error_is_not_reported_as_unreachable(monkeypatch):
    import io
    def err(*a, **k):
        raise urllib.error.HTTPError("http://x", 500, "boom", Message(), io.BytesIO(b'{"error":"model runner stopped"}'))
    monkeypatch.setattr("urllib.request.urlopen", err)
    with pytest.raises(RuntimeError, match="HTTP 500.*model runner stopped"):
        OllamaLLM(GEN_CFG).chat("s", "u")


def test_make_llm_picks_the_backend_and_vllm_is_not_ready():
    assert isinstance(make_llm({"generator": GEN_CFG, "seed": 42}), OllamaLLM)
    with pytest.raises(NotImplementedError):
        make_llm({"generator": {**GEN_CFG, "backend": "vllm"}, "seed": 42})
    with pytest.raises(ValueError):
        make_llm({"generator": {**GEN_CFG, "backend": "x"}, "seed": 42})


def test_batch_logs_each_result_and_resumes(tmp_path):
    log = tmp_path / "log.jsonl"
    reqs = [{"key": f"k{i}", "system": "s", "user": f"u{i}"} for i in range(3)]
    first = FakeOllama(replies=[{"text": "a"}, {"text": "b"}])
    with pytest.raises(IndexError):  # a crash on the 3rd request: the replies list is empty and the fake is strict
        class Strict(FakeOllama):
            def _post(self, path, body):
                if not self.replies:
                    raise IndexError
                return super()._post(path, body)
        batch(Strict(replies=[{"text": "a"}, {"text": "b"}]), reqs, log)
    assert len(log.read_text().splitlines()) == 2
    second = FakeOllama(replies=[{"text": "c"}])
    out = batch(second, reqs, log)
    assert [r.text for r in out.values()] == ["a", "b", "c"] and len(second.calls) == 1  # only the missing one ran
    assert len(log.read_text().splitlines()) == 3


# ----------------------------------------------------------------------------- B0
class FakeLLM:
    """Records prompts; replies with a canned text. Has the context-manager protocol like the real wrapper."""
    def __init__(self, texts=None, fail_on=None):
        self.calls, self.texts, self.fail_on = [], list(texts or []), fail_on

    def chat(self, system, user, max_tokens=None):
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise KeyboardInterrupt
        self.calls.append((system, user))
        text = self.texts.pop(0) if self.texts else f"echo: {user[-40:]}"
        return LLMResult(text=text, prompt_tokens=10, completion_tokens=5, seconds=0.25)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_b0_asks_only_the_question_and_never_shows_cards():
    q = questions()[0]
    secret = Card(card_id="M5-X", source="m5", card_type="item_store_week", key={"a": "b"}, fields={"u": 1}, text="SECRET CARD TEXT")
    llm = FakeLLM(["About 40 units."])
    ans = b0.answer(q, [secret], llm)
    system, user = llm.calls[0]
    assert q.question in user and "SECRET" not in system + user and "M5-X" not in system + user
    assert (ans.system, ans.qid, ans.card_ids, ans.prompt_version, ans.n_calls) == ("b0", q.qid, [], "gen_b0_v1", 1)
    assert ans.final_answer == "About 40 units." and ans.abstained is False and ans.seconds == 0.25


def test_b0_records_abstention_and_truncation():
    q = questions()[2]
    refusal = b0.answer(q, None, FakeLLM(["The cards do not contain that item."]))
    assert refusal.abstained
    assert not b0.answer(q, None, FakeLLM(["FOODS_3_999 sold 40 units."])).abstained
    class Cut(FakeLLM):
        def chat(self, *a, **k):
            r = super().chat(*a, **k)
            r.truncated = True
            return r
    assert b0.answer(q, None, Cut(["The answer is"])).truncated


# ----------------------------------------------------------------------------- run script
@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "git_dirty", lambda: False)
    monkeypatch.setattr(run, "write_env", lambda path, cfg: path.write_text("env", encoding="utf-8"))
    return tmp_path


def test_run_writes_config_env_and_one_answer_per_question(runs):
    d = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)
    assert sorted(p.name for p in d.iterdir()) == ["config.yaml", "env.txt", "outputs.jsonl"]
    answers = list(read_jsonl(d / "outputs.jsonl", Answer))
    assert [a.qid for a in answers] == [q.qid for q in questions()]
    meta = yaml.safe_load((d / "config.yaml").read_text())["run"]
    assert meta["system"] == "b0" and meta["split"] == "dev" and meta["n_questions"] == 3 and meta["dirty"] is False
    assert {a.prompt_version for a in answers} == {"gen_b0_v1"}  # from config prompts.b0
    assert d.name.startswith("20") and "_b0_dev_" in d.name
    assert yaml.safe_load((d / "config.yaml").read_text())["generator"]["model"] == GEN_CFG["model"]


def test_a_run_folder_is_never_overwritten(runs):
    a = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)
    b = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)
    assert a != b and a.exists() and b.exists()


def test_an_interrupted_run_resumes_without_repeating_work(runs):
    with pytest.raises(KeyboardInterrupt):
        run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(fail_on=2), runs_dir=runs)
    d = next(runs.iterdir())
    assert len(list(read_jsonl(d / "outputs.jsonl", Answer))) == 2  # the first two were saved before the crash
    llm = FakeLLM()
    assert run.main("b0", "dev", questions_path=SMOKE, llm=llm, resume=d, runs_dir=runs) == d
    assert len(llm.calls) == 1  # only the third question was run
    assert [a.qid for a in read_jsonl(d / "outputs.jsonl", Answer)] == [q.qid for q in questions()]


def test_same_config_gives_identical_outputs(runs):
    texts = ["x", "y", "z"]
    a = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(list(texts)), runs_dir=runs)
    b = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(list(texts)), runs_dir=runs)
    strip = lambda d: [(x.qid, x.raw_output, x.final_answer, x.abstained) for x in read_jsonl(d / "outputs.jsonl", Answer)]
    assert strip(a) == strip(b)


def test_gpu_share_is_read_from_the_server(monkeypatch):
    import io
    body = json.dumps({"models": [{"name": GEN_CFG["model"], "size": 5_000, "size_vram": 4_000}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(body))
    assert OllamaLLM(GEN_CFG).gpu_share() == pytest.approx(0.8)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b'{"models": []}'))
    assert OllamaLLM(GEN_CFG).gpu_share() is None


def test_run_records_the_measured_gpu_share_and_warns_when_partial(runs, capsys):
    class Partial(FakeLLM):
        def gpu_share(self):
            return 0.8
    d = run.main("b0", "dev", questions_path=SMOKE, llm=Partial(), runs_dir=runs, limit=2)
    assert "is 80% on the GPU" in (d / "env.txt").read_text()
    assert "WARNING" in capsys.readouterr().out
    class Full(FakeLLM):
        def gpu_share(self):
            return 1.0
    d = run.main("b0", "dev", questions_path=SMOKE, llm=Full(), runs_dir=runs, limit=1)
    assert "is 100% on the GPU" in (d / "env.txt").read_text() and "WARNING" not in capsys.readouterr().out


def test_a_dirty_tree_is_refused_unless_allowed(runs, monkeypatch):
    monkeypatch.setattr(run, "git_dirty", lambda: True)
    with pytest.raises(SystemExit, match="uncommitted"):
        run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)
    d = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs, allow_dirty=True)
    assert d.name.endswith("-dirty")


def test_questions_path_is_found_from_the_repo_root_or_reports_a_clear_error(runs, monkeypatch, tmp_path):
    from src.config import ROOT
    monkeypatch.chdir(tmp_path)  # run from an unrelated folder, with a path relative to the repo root
    rel = SMOKE.relative_to(ROOT)
    d = run.main("b0", "dev", questions_path=rel, llm=FakeLLM(), runs_dir=runs, limit=1)
    assert len(list(read_jsonl(d / "outputs.jsonl", Answer))) == 1
    with pytest.raises(SystemExit, match="questions file not found: nope.jsonl"):
        run.main("b0", "dev", questions_path=Path("nope.jsonl"), llm=FakeLLM(), runs_dir=runs)
    with pytest.raises(SystemExit, match="benchmark is not built yet"):
        run.main("b0", "no_such_split", llm=FakeLLM(), runs_dir=runs)


def test_the_test_split_needs_frozen_checksums(runs):
    with pytest.raises(SystemExit, match="checksums"):
        run.main("b0", "test", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)


def test_limit_and_unknown_system(runs):
    d = run.main("b0", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs, limit=2)
    assert len(list(read_jsonl(d / "outputs.jsonl", Answer))) == 2
    with pytest.raises(SystemExit):
        run.main("nope", "dev", questions_path=SMOKE, llm=FakeLLM(), runs_dir=runs)


# ----------------------------------------------------------------------------- real model (opt-in)
@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to run against the Ollama server")
def test_real_b0_on_smoke_questions_is_deterministic(tmp_path):
    from src.config import load_config
    cfg = load_config()
    with make_llm(cfg) as llm:
        first = [b0.answer(q, None, llm) for q in questions()]
        again = b0.answer(questions()[0], None, llm)
    assert all(a.final_answer for a in first)
    assert again.raw_output == first[0].raw_output  # temperature 0 + fixed seed: same input, same output
