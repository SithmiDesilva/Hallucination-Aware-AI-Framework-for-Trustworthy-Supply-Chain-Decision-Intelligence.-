"""What counts as an abstention: the plan's pattern (Prompts, "Abstention detection"), applied
case-insensitively after trimming whitespace.

The plan says to refine this pattern on the development set by reading every answer it misses or wrongly
catches, then freeze it, and to record its precision and recall on the 100 human-annotated answers. Until
then it is exactly the version-1 pattern below; any edit after the freeze invalidates earlier results.
V's own abstention ("INSUFFICIENT EVIDENCE. The evidence does not confirm: ...") and B3's withdrawal both
start with the exact phrase, so they are always detected.
"""
import re

ABSTAIN_PHRASE = "INSUFFICIENT EVIDENCE"

PATTERN = re.compile(
    r"^insufficient evidence\b"
    r"|\b(the (evidence|cards?) (do|does) not (contain|include|show|state))\b"
    r"|\b(cannot|can't|unable to) (be )?(determine|answer|find)\b.*\b(evidence|cards?|data)\b"
    r"|\bnot enough (information|evidence|data)\b",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    return bool(PATTERN.search(text.strip()))
