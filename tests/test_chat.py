"""Tests for the terminal chat (src/systems/chat.py), with a fake model and scripted keyboard input.

Covers: each question gets an answer and is logged; commands and blank lines; end of input and Ctrl+C quit
cleanly; questions are independent (no earlier turn reaches the prompt); a server error ends the chat with a
message instead of a traceback; the GPU warning; the log is appended to, never overwritten.
"""
import json

import pytest

from src.systems.chat import chat, manual_question
from src.systems.llm import LLMResult


class FakeLLM:
    def __init__(self, texts=None, error=None, share=1.0):
        self.calls, self.texts, self.error, self.share, self.exited = [], list(texts or []), error, share, False

    def chat(self, system, user, max_tokens=None):
        if self.error:
            raise self.error
        self.calls.append((system, user))
        return LLMResult(text=self.texts.pop(0) if self.texts else "ok", prompt_tokens=5, completion_tokens=5, seconds=1.5)

    def gpu_share(self):
        return self.share

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


def run_chat(inputs, tmp_path, llm=None):
    lines, out = iter(inputs), []

    def fake_input(prompt):
        try:
            nxt = next(lines)
        except StopIteration:
            raise EOFError
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    llm = llm or FakeLLM()
    n = chat("b0", llm=llm, input_fn=fake_input, output_fn=out.append, log_dir=tmp_path)
    return n, "\n".join(out), llm


def test_answers_each_question_and_logs_it(tmp_path):
    n, out, llm = run_chat(["How many units?", "Was order 5 late?", "/quit"], tmp_path, FakeLLM(["About 40.", "Yes, 3 days."]))
    assert n == 2 and "B0: About 40." in out and "B0: Yes, 3 days." in out and "abstained: no" in out
    (log,) = list(tmp_path.glob("*_b0.jsonl"))
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in rows] == ["How many units?", "Was order 5 late?"]
    assert rows[0]["final_answer"] == "About 40." and rows[0]["system"] == "b0" and rows[0]["card_ids"] == []
    assert llm.exited and "Bye." in out


def test_questions_are_independent_no_memory_of_earlier_turns(tmp_path):
    _, _, llm = run_chat(["first question here", "second question here", "/quit"], tmp_path)
    (_, user1), (_, user2) = llm.calls
    assert "first question here" in user1 and "second question here" not in user1
    assert "first question here" not in user2 and "second question here" in user2


def test_commands_blank_lines_and_unknown_commands(tmp_path):
    n, out, llm = run_chat(["", "   ", "/help", "/wat", "/QUIT"], tmp_path)
    assert n == 0 and llm.calls == []
    assert "Commands: /help  /quit" in out and "Unknown command '/wat'" in out


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
def test_end_of_input_and_ctrl_c_quit_cleanly(tmp_path, ending):
    n, out, llm = run_chat(["a question", ending], tmp_path)
    assert n == 1 and "Bye." in out and llm.exited


def test_a_server_problem_ends_the_chat_with_a_message(tmp_path):
    n, out, llm = run_chat(["a question", "another"], tmp_path, FakeLLM(error=ConnectionError("Ollama is not reachable ...")))
    assert n == 0 and "Ollama is not reachable" in out and "Bye." in out and llm.exited
    assert not any(p.read_text() for p in tmp_path.glob("*.jsonl"))  # nothing is logged for a failed question


def test_refusals_are_flagged_and_a_partial_gpu_warns_once(tmp_path):
    _, out, _ = run_chat(["q1", "q2", "/quit"], tmp_path, FakeLLM(["INSUFFICIENT EVIDENCE. Not in the data.", "fine"], share=0.8))
    assert "abstained: yes" in out and out.count("only 80% on the GPU") == 1
    _, out, _ = run_chat(["q1", "/quit"], tmp_path, FakeLLM(share=1.0))
    assert "GPU" not in out.replace("Note", "")


def test_the_log_is_appended_across_sessions(tmp_path):
    run_chat(["one", "/quit"], tmp_path)
    run_chat(["two", "/quit"], tmp_path)
    (log,) = list(tmp_path.glob("*_b0.jsonl"))
    assert [json.loads(l)["question"] for l in log.read_text(encoding="utf-8").splitlines()] == ["one", "two"]


def test_manual_question_is_a_valid_record():
    q = manual_question("How many units?", 7)
    assert q.qid == "CHAT-007" and q.question == "How many units?"
