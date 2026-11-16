"""Tests for src/systems/abstain.py: the plan's version-1 abstention pattern.

The plan: refine the pattern on the development set by reading every answer it misses or wrongly catches,
then freeze it (and report its precision and recall on the 100 human-annotated answers). Until that
refinement, these tests pin the pattern to the plan's text, and the "known misses" list records what version 1
does not catch so the refinement has a starting point. Change the pattern and these lists together, once.
"""
import re

import pytest

from src.systems.abstain import ABSTAIN_PHRASE, PATTERN, is_abstention

PLAN_PATTERN = (
    r"^insufficient evidence\b"
    r"|\b(the (evidence|cards?) (do|does) not (contain|include|show|state))\b"
    r"|\b(cannot|can't|unable to) (be )?(determine|answer|find)\b.*\b(evidence|cards?|data)\b"
    r"|\bnot enough (information|evidence|data)\b"
)

ABSTENTIONS = [
    "INSUFFICIENT EVIDENCE",
    "insufficient evidence",
    "  INSUFFICIENT EVIDENCE. The cards do not record carriers.  ",
    "INSUFFICIENT EVIDENCE. The answer could not be verified against the evidence.",   # B3's withdrawal
    "INSUFFICIENT EVIDENCE. The evidence does not confirm: CA_1 sold 412 units.",       # V's abstention
    "INSUFFICIENT EVIDENCE. The evidence does not support an answer to this question.",
    "The evidence does not contain the shipping date.",
    "The cards do not include that item.",
    "The card does not show a reorder point.",
    "The evidence does not state the supplier.",
    "I cannot determine the units sold from the cards.",
    "We are unable to find that order in the data.",
    "There is not enough information to answer.",
    "Not enough evidence to say.",
]
ANSWERS = [
    "412 units.", "Yes, the order was 2 days late [DC-ORD-1042].", "The cards show 412 units sold [M5-CA_1-FOODS_3_090-W11512].",
    "The evidence shows CA_1 sold more: 412 against 380 units.", "Shortfall was 0 units that week.",
    "The order was cancelled, so there is no delivery to be late.", "Orders cannot be negative; the total was 30 units.",
    "A reorder of 180 units is recommended because stock (58) is below the reorder point (96.4).",
    "The answer is INSUFFICIENT EVIDENCE.",  # the plan anchors the exact phrase at the start of the answer
    "FOODS_3_999 sold 40 units at CA_1.",
]
# What version 1 does NOT catch. Typical of B0, which has no cards to mention. Review these on the development set.
KNOWN_MISSES = [
    "I do not have access to your store's sales data.",
    "To answer this question, I would need to access the sales data for store CA_1.",
    "I don't have specific information about FOODS_3_999.",
    "It is not possible to tell from the information given.",
    "The answer can't be determined from the evidence provided.",  # passive: "determined" has no word boundary after "determine"
]


def test_pattern_is_exactly_the_plans_version_1():
    assert PATTERN.pattern == PLAN_PATTERN and PATTERN.flags & re.IGNORECASE
    assert ABSTAIN_PHRASE == "INSUFFICIENT EVIDENCE"


@pytest.mark.parametrize("text", ABSTENTIONS)
def test_abstentions_are_detected(text):
    assert is_abstention(text)


@pytest.mark.parametrize("text", ANSWERS)
def test_answers_are_not_abstentions(text):
    assert not is_abstention(text)


@pytest.mark.parametrize("text", KNOWN_MISSES)
def test_known_misses_of_version_1(text):
    assert not is_abstention(text)  # if you refine the pattern, move these into ABSTENTIONS


def test_whitespace_is_trimmed_before_matching():
    assert is_abstention("\n\n  INSUFFICIENT EVIDENCE\n")
