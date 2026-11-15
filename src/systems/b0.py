"""B0, LLM only: the question with no evidence cards.

It shows what the model says from memory alone, so it doubles as the memorisation check: a number it gets
right with no cards cannot be credited to retrieval. It is told nothing about abstaining. Whether the
reply happens to be a refusal is recorded by the frozen abstention rule (abstain.py).
"""
from typing import Sequence

from src.cards.schemas import Answer, Card, Question
from src.systems.abstain import is_abstention
from src.systems.format import fill_prompt

SYSTEM = "b0"
PROMPT = "gen_b0_v1"


def answer(question: Question, cards: Sequence[Card] | None, llm, prompt: str = PROMPT) -> Answer:
    """`cards` is accepted so every system has the same signature, and is deliberately never used."""
    system, user = fill_prompt(prompt, question=question.question)
    r = llm.chat(system, user)
    text = r.text.strip()
    return Answer(qid=question.qid, system=SYSTEM, card_ids=[], prompt_version=prompt, raw_output=r.text,
                  final_answer=text, abstained=is_abstention(text), seconds=r.seconds, n_calls=1,
                  truncated=r.truncated)
