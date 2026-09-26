import copy
import json
import random
from collections import Counter

import pytest

import app as application
from rag.answers import (
    ANSWER_CONTRACT_VERSION,
    answer_explanation_consistency_errors,
    answer_letter_from_value,
    remap_explanation_options,
)
from rag.generator import parse_mcq_response
from rag.quality import is_too_similar, validate_generated_mcq


def _question(**changes):
    return {
        "question": "What is 7 + 5?",
        "options": ["10", "12", "14", "16"],
        "answer": "B",
        "correct_answer": "12",
        "explanation": "Correct result: 12. Adding seven and five gives twelve.",
        **changes,
    }


def test_model_letter_is_derived_from_explicit_value():
    parsed = parse_mcq_response(json.dumps({"questions": [_question(answer="C")]}), require_correct_answer=True)
    assert parsed[0]["answer"] == "B"
    assert parsed[0]["correct_answer"] == "12"


@pytest.mark.parametrize("value", ["99", "B", "", None])
def test_missing_or_nonmatching_value_cannot_be_graded(value):
    assert parse_mcq_response(json.dumps({"questions": [_question(correct_answer=value)]}), require_correct_answer=True) == []


def test_new_generation_requires_value_even_when_a_letter_is_valid():
    question = _question()
    question.pop("correct_answer")
    assert parse_mcq_response(json.dumps({"questions": [question]}), require_correct_answer=True) == []
    assert parse_mcq_response(json.dumps({"questions": [question]}))[0]["answer"] == "B"


@pytest.mark.parametrize("values", [["1/2", "0.5", "3/4", "1"], ["50%", "0.5", "25%", "0.1"]])
def test_equivalent_correct_options_are_rejected(values):
    with pytest.raises(ValueError, match="exactly one"):
        answer_letter_from_value("1/2", values)


def test_negative_distractor_mentions_do_not_become_the_answer():
    question = _question(
        question="Which value represents the given total?",
        explanation="10 omits one of the addends and is incorrect. Add both given quantities.",
    )
    assert validate_generated_mcq(question) == []


@pytest.mark.parametrize("explanation", [
    "Correct result: 14. This is the calculated total.",
    "Adding seven and five gives 14.",
    "Option C is correct.",
    "The correct answer is C.",
    "Option B is correct, and option C is correct.",
])
def test_contradictory_explanations_are_rejected(explanation):
    assert answer_explanation_consistency_errors(_question(explanation=explanation))


def test_decimal_prefix_is_not_confused_with_a_smaller_integer():
    question = _question(
        question="Which is the stated measured value?",
        options=["12", "12.5", "15", "20"],
        answer="A", correct_answer="12", explanation="Correct result: 12.5. This is the measured value.",
    )
    assert answer_explanation_consistency_errors(question)


@pytest.mark.parametrize(("stem", "options", "value"), [
    ("What is 7 + 5?", ["10", "12", "14", "16"], "14"),
    ("Calculate 0.1 + 0.2?", ["0.1", "0.2", "0.3", "0.4"], "0.4"),
    ("Evaluate (2 + 3)^2?", ["10", "20", "25", "30"], "20"),
])
def test_wrong_arithmetic_is_rejected_even_if_value_and_explanation_agree(stem, options, value):
    question = _question(question=stem, options=options, correct_answer=value,
                         answer=answer_letter_from_value(value, options), explanation=f"Correct result: {value}.")
    assert any("independently calculated" in error for error in validate_generated_mcq(question))


def test_shuffling_balances_positions_and_preserves_the_explanation_and_value():
    original = [_question(question=f"Total for item {index}?", explanation="Option B is correct. Option A omits an addend.") for index in range(56)]
    before = copy.deepcopy(original)
    for seed in range(12):
        paper = application._personalize_quiz(original, randomizer=random.Random(seed))
        assert Counter(q["answer"] for q in paper) == {letter: 14 for letter in "ABCD"}
        for question in paper:
            assert question["options"]["ABCD".index(question["answer"])] == "12"
            assert f"Option {question['answer']} is correct" in question["explanation"]
            assert answer_explanation_consistency_errors(question) == []
            assert question["answer_contract_version"] == ANSWER_CONTRACT_VERSION
    assert original == before


@pytest.mark.parametrize("count", [1, 2, 3, 4, 10, 11, 56])
def test_correct_positions_differ_by_at_most_one(count):
    paper = application._personalize_quiz([_question() for _ in range(count)], randomizer=random.Random(4))
    positions = Counter(question["answer"] for question in paper)
    assert max(positions.get(letter, 0) for letter in "ABCD") - min(positions.get(letter, 0) for letter in "ABCD") <= 1


def test_geometry_labels_are_preserved_when_option_labels_move():
    text = "In quadrilateral ABCD, AB = CD and AD = BC. Option C correctly applies the property; option A is incorrect."
    remapped = remap_explanation_options(text, {"A": "D", "B": "C", "C": "B", "D": "A"})
    assert "ABCD, AB = CD and AD = BC" in remapped
    assert "Option B correctly" in remapped
    assert "option D is incorrect" in remapped


def test_option_lists_and_ordinals_are_remapped_together():
    remapped = remap_explanation_options(
        "Options A and B are incorrect. The third option is correct. Options A, B, and D are distractors.",
        {"A": "D", "B": "C", "C": "B", "D": "A"},
    )
    assert "Options D and C" in remapped
    assert "option B is correct" in remapped
    assert "Options D, C, and A" in remapped


def test_indefinite_article_is_not_mistaken_for_option_a():
    explanation = "The correct answer is a rational number because it is a ratio of two integers."
    question = _question(
        question="Classify the given number.", options=["Irrational", "Rational", "Imaginary", "Undefined"],
        correct_answer="Rational", explanation=explanation,
    )
    assert validate_generated_mcq(question) == []
    assert remap_explanation_options(explanation, {"A": "D", "B": "C", "C": "B", "D": "A"}) == explanation


def test_reordered_finite_sets_are_equivalent_correct_options():
    with pytest.raises(ValueError, match="exactly one"):
        answer_letter_from_value("{1,2}", ["{1,2}", "{2,1}", "{3}", "{}"]) 


def test_wrong_simple_set_operation_is_detected_independently():
    question = _question(
        question="For A = {1, 2} and B = {2, 3}, find A ∩ B.",
        options=["{1}", "{2}", "{1,2,3}", "{3}"],
        answer="C", correct_answer="{1,2,3}", explanation="Correct result: {1,2,3}.",
    )
    assert any("independently calculated set result" in error for error in validate_generated_mcq(question))


def test_new_quantities_and_set_operations_are_not_false_duplicates():
    assert not is_too_similar(
        "A bag contains 3 red and 7 blue counters. Find the probability of drawing red.",
        ["A bag contains 4 red and 6 blue counters. Find the probability of drawing red."], threshold=0.84,
    )
    assert not is_too_similar("For A = {1, 2}, B = {2, 3}, find A ∩ B.",
                              ["For A = {1, 2}, B = {2, 3}, find A ∪ B."], threshold=0.84)


def test_cosmetic_rewording_and_equivalent_numeric_data_still_count_as_repeats():
    assert is_too_similar("Find the perimeter of a rectangle measuring 5 cm by 3 cm.",
                          ["Find the perimeter of a rectangle measuring 5.0 cm by 3.0 cm."], threshold=0.84)
    assert is_too_similar("Question 1: Define the empty set.", ["Question 2: Define the empty set."], threshold=0.84)


def test_internal_correct_value_is_not_exposed_before_submission():
    paper = application._personalize_quiz([_question()])
    public = application._public_paper(paper)[0]
    assert not {"answer", "correct_answer", "explanation", "answer_contract_version"}.intersection(public)
