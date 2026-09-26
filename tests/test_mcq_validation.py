import copy
import json
import random

import pytest
import requests

import app as application
import config
import rag.generator as generator
import rag.runpod_inference as inference
import rag.validation as validation


CONTEXT = "The perimeter of a rectangle is twice the sum of its length and breadth."


def _question():
    return {
        "question": "A rectangular garden is 12 m long and 7 m wide. A 2 m opening is left in the boundary. How much fencing is needed?",
        "options": ["34 m", "36 m", "38 m", "40 m"], "answer": "B", "correct_answer": "36 m",
        "explanation": "Correct result: 36 m. Author-only explanation marker.",
        "topic_id": "perimeter", "grade": 10, "difficulty_level": "Apply",
    }


def _verdict(**changes):
    result = {
        "answer_value": "36 m", "valid_option_values": ["36 m"],
        "explanation": "Twice the sum of 12 and 7, less the 2 m opening, is 36 m.",
        "difficulty_level": "Apply", "unambiguous": True, "concept_relevant": True,
        "textbook_supported": True, "distractors_plausible": True, "application_required": True,
        "reason": "",
    }
    result.update(changes)
    return result


def _review(question):
    return validation.review_mcq(question, topic_name="Perimeter", difficulty_level="Apply",
                                 source_context=CONTEXT, require_application=True)


def test_blinded_solver_checks_answer_and_provides_the_final_explanation(monkeypatch):
    calls = []

    def reviewer(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return _verdict()

    monkeypatch.setattr(validation, "runpod_mcq_review", reviewer)
    question = _question()
    assert _review(question) == []
    assert validation.has_current_review(question)
    payload = json.loads(calls[0][0].split("\n", 1)[1])
    assert not {"answer", "correct_answer", "explanation"}.intersection(payload)
    assert "Author-only explanation marker" not in calls[0][0]
    assert payload["textbook_context"] == CONTEXT
    assert "Twice the sum" in question["explanation"]
    shuffled = application._personalize_quiz([question] * 25, randomizer=random.Random(19))
    assert all(validation.has_current_review(item) for item in shuffled)
    assert all(item["options"]["ABCD".index(item["answer"])] == "36 m" for item in shuffled)


def test_consistent_but_wrong_author_answer_is_rejected(monkeypatch):
    question = _question()
    question.update(answer="A", correct_answer="34 m", explanation="Correct result: 34 m.")
    monkeypatch.setattr(validation, "runpod_mcq_review", lambda *_args, **_kwargs: _verdict())
    assert "disagrees" in "; ".join(_review(question))
    assert "quality_review" not in question


@pytest.mark.parametrize("changes, expected", [
    ({"concept_relevant": False}, "requested concept"),
    ({"textbook_supported": False}, "textbook context"),
    ({"distractors_plausible": False}, "plausible"),
    ({"unambiguous": False}, "ambiguous"),
    ({"application_required": False}, "too direct"),
    ({"difficulty_level": "Remember"}, "Bloom level"),
    ({"valid_option_values": ["34 m", "36 m"]}, "exactly one"),
    ({"valid_option_values": []}, "exactly one"),
    ({"valid_option_values": "36 m"}, "exactly one"),
    ({"answer_value": "37 m"}, "uniquely match"),
    ({"concept_relevant": "true"}, "requested concept"),
    ({"explanation": ""}, "justification"),
    ({"explanation": "The correct answer is option C."}, "option positions"),
    ({"explanation": "The correct answer is B."}, "option positions"),
])
def test_semantic_review_fails_closed(monkeypatch, changes, expected):
    monkeypatch.setattr(validation, "runpod_mcq_review", lambda *_args, **_kwargs: _verdict(**changes))
    question = _question()
    assert expected in "; ".join(_review(question))
    assert "quality_review" not in question


@pytest.mark.parametrize("field, value", [
    ("question", "Define the perimeter of a rectangle."),
    ("options", ["34 m", "36 m", "38 m", "1000 m"]),
    ("options", ["34 m", "36 m", "38 cm", "40 m"]),
])
def test_obvious_recall_and_distractor_defects_do_not_spend_a_review_call(monkeypatch, field, value):
    monkeypatch.setattr(validation, "runpod_mcq_review", lambda *_args, **_kwargs: pytest.fail("Unexpected inference"))
    question = _question()
    question[field] = value
    assert _review(question)


@pytest.mark.parametrize("options", [
    ["-1", "0", "1", "2"], ["1/4", "1/3", "1/2", "2/3"], ["34 m", "36 m", "38 m", "40 m"],
])
def test_comparable_numeric_distractors_pass_conservative_local_gate(options):
    question = _question()
    question["options"] = options
    assert validation.challenge_errors(question, require_application=True) == []


@pytest.mark.parametrize("field, value", [("answer", "C"), ("question", "Changed stem"),
                                         ("explanation", "Changed explanation"), ("grade", 11),
                                         ("difficulty_level", "Evaluate"), ("topic_id", "area")])
def test_review_record_is_bound_to_the_actual_served_content(monkeypatch, field, value):
    monkeypatch.setattr(validation, "runpod_mcq_review", lambda *_args, **_kwargs: _verdict())
    question = _question()
    assert _review(question) == []
    question[field] = value
    assert not validation.has_current_review(question)


def test_rejected_independent_answer_is_regenerated_with_feedback(monkeypatch):
    prompts = []
    rows = [_question(), _question()]
    rows[0].update(answer="A", correct_answer="34 m", explanation="Correct result: 34 m.")
    for row in rows:
        row["evidence_id"] = "E1"

    def generate(prompt, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"questions": [rows.pop(0)]})

    monkeypatch.setattr(generator, "_invoke_llm", generate)
    monkeypatch.setattr(validation, "runpod_mcq_review", lambda *_args, **_kwargs: _verdict())
    monkeypatch.setattr(config, "GENERATION_RETRY_BASE_SECONDS", 0)
    reference = {"source": "grade-10.pdf", "page": 3, "grade": 10, "chunk_id": "chunk-1", "corpus_version": "v1"}
    result = generator._generate_mcqs_block((
        "perimeter", "Perimeter", "Apply", [CONTEXT], 1,
        {"grade": 10, "strict_textbook_grounding": True, "require_application": True},
        [reference], [{"text": CONTEXT, "source_ref": reference}], [], [],
    ))
    assert len(result) == 1 and result[0]["answer"] == "B"
    assert validation.has_current_review(result[0])
    assert len(prompts) == 2
    assert "independently solved answer disagrees" in prompts[1]


def test_unavailable_reviewer_never_returns_unreviewed_candidates(monkeypatch):
    row = _question()
    monkeypatch.setattr(generator, "_invoke_llm", lambda *_args, **_kwargs: json.dumps({"questions": [row]}))

    def unavailable(*_args, **_kwargs):
        raise requests.Timeout("review unavailable")

    monkeypatch.setattr(validation, "runpod_mcq_review", unavailable)
    monkeypatch.setattr(config, "GENERATION_RETRY_BASE_SECONDS", 0)
    result = generator._generate_mcqs_block(("perimeter", "Perimeter", "Apply", [CONTEXT], 1,
                                            {"grade": 10, "require_application": True}, [], [], [], []))
    assert result == []


def test_review_uses_existing_provider_and_expands_truncated_output_once(monkeypatch):
    calls = []

    class Session:
        def post(self, _url, **kwargs):
            calls.append(kwargs["json"])
            count = len(calls)

            class Response:
                status_code = 200

                def json(self):
                    return {"choices": [{"finish_reason": "length" if count == 1 else "stop",
                                         "message": {"content": json.dumps(_verdict())}}]}
            return Response()

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    assert inference.runpod_mcq_review("review", response_format=validation.review_response_format()) == _verdict()
    assert [call["max_tokens"] for call in calls] == [1024, 2048]
    assert all(call["temperature"] == 0.0 for call in calls)
    assert all(call["model"].startswith(config.RUNPOD_MCQ_GENERATION_MODEL) for call in calls)
    assert calls[0]["response_format"]["json_schema"]["name"] == "neuromath_mcq_review"


def test_permanent_review_provider_error_is_not_retried(monkeypatch):
    calls = []

    class Session:
        def post(self, *_args, **_kwargs):
            calls.append(1)

            class Response:
                status_code = 403
                text = "Forbidden"

                def json(self):
                    return {"error": "Forbidden"}
            return Response()

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    with pytest.raises(RuntimeError, match="not accepted"):
        inference.runpod_mcq_review("review", response_format=validation.review_response_format())
    assert len(calls) == 1


def test_deterministic_set_difference_repairs_only_the_answer_key_before_review(monkeypatch):
    question = {
        "question": "For A = {1, 2}, B = {2, 3} and C = {2}, find (A ∪ B) \\ C.",
        "options": ["{1, 2, 3}", "{2}", "{1, 3}", "{2, 3}"],
        "answer": "A",
        "correct_answer": "{1, 2, 3}",
        "explanation": "Correct result: {1, 2, 3}.",
        "topic_id": "sets",
        "grade": 7,
        "difficulty_level": "Apply",
    }
    assert application is not None
    from rag.answers import repair_deterministic_answer

    assert repair_deterministic_answer(question) is True
    assert question["answer"] == "C"
    assert question["correct_answer"] == "{1, 3}"
    assert question["question"].endswith("find (A ∪ B) \\ C.")
    assert question["options"] == ["{1, 2, 3}", "{2}", "{1, 3}", "{2, 3}"]
    assert "quality_review" not in question


def test_candidate_surplus_stops_reviewing_once_required_count_is_reached(monkeypatch):
    calls = []
    rows = []
    for number in range(1, 7):
        row = _question()
        row["question"] = (
            f"A rectangular garden is {10 + number} m long and {5 + number} m wide. "
            "A 2 m opening is left in the boundary. How much fencing is needed?"
        )
        correct = f"{2 * ((10 + number) + (5 + number)) - 2} m"
        row["options"] = [f"{int(correct[:-2]) - 2} m", correct, f"{int(correct[:-2]) + 2} m", f"{int(correct[:-2]) + 4} m"]
        row["answer"] = "B"
        row["correct_answer"] = correct
        rows.append(row)

    monkeypatch.setattr(
        generator,
        "_invoke_llm",
        lambda *_args, **kwargs: json.dumps({"questions": rows[: kwargs["question_count"]]}),
    )

    def reviewer(_prompt, **_kwargs):
        calls.append(1)
        row = rows[len(calls) - 1]
        return _verdict(
            answer_value=row["correct_answer"],
            valid_option_values=[row["correct_answer"]],
            explanation="The perimeter is twice the sum of length and width, less the opening.",
        )

    monkeypatch.setattr(validation, "runpod_mcq_review", reviewer)
    result = generator.generate_mcqs(
        "Perimeter", "Apply", [CONTEXT], num_questions=6, grade=10,
        concept_id="perimeter", require_application=True, accept_limit=4,
    )
    assert len(result) == 4
    assert len(calls) == 4
