"""Tests for src/cards/schemas.py.

Round trip (model -> JSON -> model is unchanged, types preserved) using the plan's own example card,
question and claim; the Question consistency rules; rejection of unknown fields; the JSONL helpers.
"""
import pytest
from pydantic import ValidationError

from src.cards.schemas import Answer, Card, Claim, GoldFact, GoldLabel, Question, read_jsonl, write_jsonl

CARD = {
    "card_id": "M5-CA_1-FOODS_3_090-W2015-12",
    "source": "m5",
    "card_type": "item_store_week",
    "key": {"store_id": "CA_1", "item_id": "FOODS_3_090", "week": "2015-W12"},
    "fields": {"units_sold": 412, "mean_price": 1.25, "snap_days": 3, "on_hand_end": 58, "rop": 96.4,
               "safety_stock": 31.2, "order_placed": True, "order_qty": 180, "shortfall": 0},
    "text": "Item FOODS_3_090 at store CA_1, week 2015-W12: 412 units sold at a mean price of $1.25.",
}
QUESTION = {
    "qid": "M5-AGG-0137", "split": "test", "domain": "inventory", "template_id": "m5_weekly_units",
    "question_type": "aggregation", "answerable": True, "unanswerable_type": None,
    "question": "How many units of FOODS_3_090 did store CA_1 sell in the week of 16 March 2015?",
    "assumptions": None, "reference_answer": "412 units.",
    "gold_facts": [{"key": {"store_id": "CA_1", "item_id": "FOODS_3_090", "week": "2015-W12"},
                    "attribute": "units_sold", "value": 412, "type": "numerical", "tolerance": {"abs": 1}}],
    "gold_card_ids": ["M5-CA_1-FOODS_3_090-W2015-12"],
    "hard_negative_card_ids": ["M5-CA_1-FOODS_3_091-W2015-12", "M5-CA_2-FOODS_3_090-W2015-12"],
}
CLAIM = {
    "qid": "M5-AGG-0137", "system": "v_full", "claim_id": "c2", "text": "CA_1 sold 412 units of FOODS_3_090 that week.",
    "claim_type": "numerical", "key": {"store_id": "CA_1", "item_id": "FOODS_3_090", "week": "2015-W12"},
    "attribute": "units_sold", "value": 412, "cited_card_ids": ["M5-CA_1-FOODS_3_090-W2015-12"],
}


@pytest.mark.parametrize("model, data", [(Card, CARD), (Question, QUESTION), (Claim, CLAIM)])
def test_round_trip_is_unchanged(model, data):
    obj = model.model_validate(data)
    again = model.model_validate_json(obj.model_dump_json())
    assert again == obj


def test_value_types_survive_the_round_trip():
    card = Card.model_validate_json(Card.model_validate(CARD).model_dump_json())
    f = card.fields
    assert type(f["units_sold"]) is int and type(f["mean_price"]) is float and type(f["order_placed"]) is bool
    assert type(card.fields["rop"]) is float


def test_remaining_models_round_trip():
    for obj in (
        GoldFact(key={"order_id": "1"}, attribute="late", value=None, type="boolean"),
        GoldLabel(qid="q", system="b1", claim_id="c1", g_E="supported", g_T="contradicted"),
        Answer(qid="q", system="b2", card_ids=["a"], prompt_version="gen_b2_v1", raw_output="x",
               final_answer="INSUFFICIENT EVIDENCE", abstained=True, seconds=1.5),
        Claim.model_validate({**CLAIM, "label": "supported", "score": 1.0, "is_core": True}),
    ):
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Card.model_validate({**CARD, "extra_field": 1})
    with pytest.raises(ValidationError):
        Question.model_validate({**QUESTION, "answerble": True})


def test_invalid_enums_and_ranges_are_rejected():
    with pytest.raises(ValidationError):
        Claim.model_validate({**CLAIM, "label": "maybe"})
    with pytest.raises(ValidationError):
        Claim.model_validate({**CLAIM, "score": 1.5})
    with pytest.raises(ValidationError):
        Card.model_validate({**CARD, "fields": {}})


def test_answerable_question_needs_gold_facts_and_cards():
    with pytest.raises(ValidationError):
        Question.model_validate({**QUESTION, "gold_facts": []})
    with pytest.raises(ValidationError):
        Question.model_validate({**QUESTION, "unanswerable_type": "nonexistent_entity"})


def test_dataset_wide_questions_need_facts_but_not_cards():
    base = {**QUESTION, "scope": "dataset", "gold_card_ids": [], "hard_negative_card_ids": []}
    q = Question.model_validate(base)
    assert q.scope == "dataset" and q.gold_card_ids == [] and Question.model_validate_json(q.model_dump_json()) == q
    with pytest.raises(ValidationError):
        Question.model_validate({**base, "gold_facts": []})                   # still needs the real values
    with pytest.raises(ValidationError):
        Question.model_validate({**base, "answerable": False, "unanswerable_type": "nonexistent_entity", "gold_facts": []})
    assert Question.model_validate(QUESTION).scope == "cards"                  # the default is unchanged
    with pytest.raises(ValidationError):
        Question.model_validate({**QUESTION, "gold_card_ids": []})            # card questions still need cards


def test_unanswerable_question_rules():
    base = {**QUESTION, "answerable": False, "gold_facts": [], "gold_card_ids": [], "reference_answer": "INSUFFICIENT EVIDENCE"}
    with pytest.raises(ValidationError):
        Question.model_validate(base)  # no unanswerable_type
    assert Question.model_validate({**base, "unanswerable_type": "nonexistent_entity"})
    with pytest.raises(ValidationError):  # partial evidence must say which card is withheld
        Question.model_validate({**base, "unanswerable_type": "partial_evidence"})
    ok = Question.model_validate({**base, "unanswerable_type": "partial_evidence",
                                  "withheld_card_ids": ["M5-CA_1-FOODS_3_090-W2015-12"]})
    assert ok.withheld_card_ids
    with pytest.raises(ValidationError):  # a mask on any other type is a bug
        Question.model_validate({**base, "unanswerable_type": "nonexistent_entity",
                                 "withheld_card_ids": ["M5-CA_1-FOODS_3_090-W2015-12"]})


def test_jsonl_helpers_round_trip(tmp_path):
    cards = [Card.model_validate(CARD), Card.model_validate({**CARD, "card_id": "M5-other"})]
    path = tmp_path / "cards.jsonl"
    assert write_jsonl(path, cards) == 2
    assert list(read_jsonl(path, Card)) == cards
    assert path.read_text(encoding="utf-8").count("\n") == 2
