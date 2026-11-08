"""One definition of every record that moves through the pipeline (pydantic v2).

    Card        one retrievable evidence unit; `text` is what retrieval and the generator see,
                `fields` is what the rule checks and the gold labeller use
    GoldFact    one verified fact behind a benchmark question
    Question    a benchmark item (frozen in benchmark/*.jsonl)
    Claim       one atomic statement extracted from an answer; the verifier adds `label` and `score`
    GoldLabel   the independent g_E / g_T labels for one claim (written by eval/, never by systems/)
    Answer      one system's output for one question (a line of runs/<run>/outputs.jsonl)

Unknown fields are rejected (extra="forbid") so a typo in a JSONL file fails loudly instead of being dropped.
`read_jsonl` / `write_jsonl` are the only way records should be (de)serialised.
"""
from pathlib import Path
from typing import Iterable, Iterator, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

# Strict members keep 412 an int, 1.25 a float and True a bool through a JSON round trip.
Scalar = Union[StrictBool, StrictInt, StrictFloat, StrictStr, None]

Source = Literal["m5", "dataco"]
CardType = Literal["item_store_week", "item_store_month", "order", "region_month"]
Domain = Literal["inventory", "delivery"]
Split = Literal["dev", "test", "conflict"]
QuestionType = Literal["lookup", "aggregation", "comparison", "policy_outcome", "recommendation"]
UnanswerableType = Literal["nonexistent_entity", "out_of_range_date", "absent_attribute", "partial_evidence"]
FactType = Literal["numerical", "temporal", "entity", "categorical", "boolean"]
ClaimType = Literal["numerical", "temporal", "entity", "derived", "recommendation", "other"]
ClaimLabel = Literal["supported", "contradicted", "misattributed", "unsupported"]
System = Literal["b0", "b1", "b2", "b3", "v_flag", "v_remove", "v_full"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Card(_Model):
    card_id: str = Field(min_length=1)
    source: Source
    card_type: CardType
    key: dict[str, str]
    fields: dict[str, Scalar] = Field(min_length=1)
    text: str = Field(min_length=1)


class Tolerance(_Model):
    abs: float | None = None
    rel: float | None = None


class GoldFact(_Model):
    key: dict[str, str]
    attribute: str
    value: Scalar
    type: FactType
    tolerance: Tolerance | None = None


class Tags(_Model):
    """Difficulty tags (src/bench/tags.py)."""
    n_gold_cards: int = 0
    confusable: bool = False  # a near-duplicate card (other store, neighbouring item or week) exists
    near_miss: bool = False   # comparison whose two sides differ by 1-5%
    conflict: bool = False    # the evidence is internally conflicting (e.g. a cancelled order with a raw delay)


class Question(_Model):
    qid: str = Field(min_length=1)
    split: Split
    domain: Domain
    template_id: str
    question_type: QuestionType
    answerable: bool
    unanswerable_type: UnanswerableType | None = None
    question: str = Field(min_length=1)
    assumptions: str | None = None  # the stated policy, for recommendation questions
    reference_answer: str
    gold_facts: list[GoldFact] = []
    gold_card_ids: list[str] = []
    hard_negative_card_ids: list[str] = []
    withheld_card_ids: list[str] = []  # removed from retrieval for this question only (partial evidence)
    tags: Tags = Tags()

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable:
            if self.unanswerable_type is not None:
                raise ValueError("an answerable question cannot have an unanswerable_type")
            if not self.gold_facts or not self.gold_card_ids:
                raise ValueError("an answerable question needs gold_facts and gold_card_ids")
        elif self.unanswerable_type is None:
            raise ValueError("an unanswerable question needs an unanswerable_type")
        if (self.unanswerable_type == "partial_evidence") != bool(self.withheld_card_ids):
            raise ValueError("withheld_card_ids must be set exactly for partial_evidence questions")
        return self


class Claim(_Model):
    qid: str
    system: System
    claim_id: str
    text: str
    claim_type: ClaimType
    key: dict[str, str] | None = None
    attribute: str | None = None
    value: Scalar = None
    cited_card_ids: list[str] = []
    is_core: bool = False       # directly answers the question, as opposed to supporting detail
    label: ClaimLabel | None = None   # set by the verifier (V only)
    score: float | None = Field(default=None, ge=0, le=1)  # 0/1 on rule paths, a probability on the NLI path


class GoldLabel(_Model):
    """gold.jsonl: g_E checks the claim against the cards retrieved for the question, g_T against the source tables.
    Free-text claims (claim_type 'other') get no automatic label; they are covered by the human sample."""
    qid: str
    system: System
    claim_id: str
    g_E: ClaimLabel | None = None
    g_T: ClaimLabel | None = None


class Answer(_Model):
    qid: str
    system: System
    card_ids: list[str] = []
    prompt_version: str
    raw_output: str
    final_answer: str
    abstained: bool
    seconds: float = Field(ge=0)
    n_calls: int = Field(default=1, ge=0)


M = TypeVar("M", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[M]) -> Iterator[M]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n
