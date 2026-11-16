"""Prompt files and placeholder filling.

A prompt file in prompts/ has two sections, `### SYSTEM` and `### USER`, and Jinja2 placeholders such as
{{ question }}. Prompts contain literal JSON braces, so str.format must not be used.
"""
from typing import Sequence

from jinja2 import Environment, StrictUndefined

from src.cards.schemas import Card
from src.config import ROOT

_env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)
PROMPTS = ROOT / "prompts"


def fill(template: str, **slots) -> str:
    """Fill {{ placeholders }}. A placeholder with no value raises, so a prompt is never sent half-filled."""
    return _env.from_string(template).render(**slots)


def load_prompt(name: str) -> tuple[str, str]:
    """(system, user template) from prompts/<name>.txt."""
    text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    head, sep, rest = text.partition("### USER")
    if not sep or not head.lstrip().startswith("### SYSTEM"):
        raise ValueError(f"prompt {name!r} needs a '### SYSTEM' section followed by a '### USER' section")
    return head.replace("### SYSTEM", "", 1).strip(), rest.strip()


def fill_prompt(name: str, **slots) -> tuple[str, str]:
    system, user = load_prompt(name)
    return fill(system, **slots), fill(user, **slots)


def cards_block(cards: Sequence[Card]) -> str:
    """The {cards} block of the RAG prompts: one `[CARD_ID] text` line per card."""
    return "\n".join(f"[{c.card_id}] {c.text}" for c in cards)
