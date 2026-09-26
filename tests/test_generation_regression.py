import json
import logging
import re

import pytest

import config
import rag.generator as generator
import rag.runpod_inference as inference
from rag.diagnostic import build_diagnostic_syllabus, validate_diagnostic_paper
from rag.prompts import LEVEL_INSTRUCTIONS, get_mcq_prompt
from rag.quality import validate_generated_mcq


EVIDENCE = "The perimeter of a rectangle is twice the sum of its length and breadth."
pytestmark = pytest.mark.usefixtures("stub_quality_review")


def _reference(grade, index=1):
    return {
        "source": f"grade-{grade}.pdf",
        "page": index,
        "page_start": index,
        "page_end": index,
        "grade": grade,
        "chunk_id": f"grade-{grade}-chunk-{index}",
        "corpus_version": "fixture-corpus",
    }




def _fixture_diagnostic_syllabus():
    mappings = []
    for grade, count in ((10, 31), (11, 25)):
        for number in range(1, count + 1):
            name = "Perimeter" if grade == 10 and number == 1 else f"Topic {grade}-{number}"
            mappings.append({
                "topic_id": f"g{grade}_t1_{number:02d}_{name.lower().replace(' ', '_')}",
                "topic_name": name,
                "grade": grade,
                "term": 1,
                "topic_number": number,
                "required_levels": [3],
                "weightage_percent": 0,
                "min_questions": 1,
                "max_questions": 1,
                "strict_textbook_grounding": True,
            })
    return {
        "syllabus_name": "Fixture Grade 10 and 11 Diagnostic",
        "assessment_type": "diagnostic",
        "grades": [10, 11],
        "total_questions": len(mappings),
        "time_limit_minutes": 120,
        "topic_mappings": mappings,
    }

def _retrieved(syllabus):
    return {
        mapping["topic_id"]: {
            "chunks": [EVIDENCE],
            "source_refs": [_reference(mapping["grade"], index)],
            "grounding_units": [{
                "text": EVIDENCE,
                "source_ref": _reference(mapping["grade"], index),
            }],
        }
        for index, mapping in enumerate(syllabus["topic_mappings"], start=1)
    }


@pytest.mark.parametrize("level", list(LEVEL_INSTRUCTIONS))
@pytest.mark.parametrize("strict_grounding", [False, True])
@pytest.mark.parametrize("use_past_questions", [False, True])
def test_prompt_formats_literal_json_and_preserves_source_braces(
    level, strict_grounding, use_past_questions
):
    context = r"Let S = {1, 2, 3} and x = \frac{1}{2}."
    past = '[{"question_id":"source-01","question":"Find {x}."}]'
    prompt = get_mcq_prompt(
        level,
        strict_grounding=strict_grounding,
        use_past_questions=use_past_questions,
    ).format(
        topic="Sets",
        grade=10,
        concept_id="g10_sets",
        context=context,
        past_questions=past,
        avoid_question_stems="[]",
        num_questions=1,
    )
    assert context in prompt
    assert past in prompt
    assert 'Output exactly one JSON object in this form:\n{"questions": [' in prompt
    assert "Number of questions: 1" in prompt
    if strict_grounding:
        assert '{"questions": []}' in prompt
        assert "evidence_id" in prompt
        assert "Copy source and page" not in prompt
    if use_past_questions:
        assert "source_question_id" in prompt


@pytest.mark.parametrize("workers", [1, 3])
def test_full_diagnostic_uses_real_prompt_transport_parser_and_grounding(
    monkeypatch, workers
):
    syllabus = _fixture_diagnostic_syllabus()
    calls = []

    class Session:
        def post(self, url, **kwargs):
            payload = kwargs["json"]
            calls.append(payload)
            prompt = payload["messages"][-1]["content"]

            def field(name):
                return re.search(rf"^{name}: (.+)$", prompt, re.MULTILINE).group(1)

            concept = field("Concept ID")
            row = {
                "question": f"For fixture {concept}, which of these numbered results is specified?",
                "options": ["1", "2", "3", "4"],
                "correct_answer": "2",
                "explanation": "This is a simulated provider answer for a pipeline test.",
                "evidence_id": "E1",
            }

            class Response:
                status_code = 200

                def json(self):
                    return {
                        "choices": [{
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"questions": [row]})},
                        }]
                    }

            return Response()

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", workers)
    paper = generator.generate_paper(syllabus, _retrieved(syllabus))

    validate_diagnostic_paper(paper, syllabus)
    assert len(calls) == len(paper) == 56
    assert [q["topic_id"] for q in paper] == [
        m["topic_id"] for m in syllabus["topic_mappings"]
    ]
    assert sum(q["grade"] == 10 for q in paper) == 31
    assert sum(q["grade"] == 11 for q in paper) == 25
    assert all(q["answer"] == "B" for q in paper)
    assert all(q["source_refs"][0]["corpus_version"] == "fixture-corpus" for q in paper)
    assert all(q["grounding_evidence"] == EVIDENCE for q in paper)
    assert all("evidence_id" not in q for q in paper)
    assert all(payload["model"] == config.RUNPOD_MCQ_GENERATION_MODEL for payload in calls)
    assert all(payload["response_format"]["type"] == "json_schema" for payload in calls)


def test_prompt_error_stops_at_first_concept_without_http_or_retries(monkeypatch):
    syllabus = _fixture_diagnostic_syllabus()
    calls = []
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 1)

    def broken_prompt(*args, **kwargs):
        calls.append(args)
        return '{"questions": []}'

    monkeypatch.setattr(generator, "get_mcq_prompt", broken_prompt)
    monkeypatch.setattr(
        generator,
        "_invoke_llm",
        lambda _prompt, **_kwargs: pytest.fail("Unexpected HTTP call"),
    )
    monkeypatch.setattr(generator.time, "sleep", lambda _: pytest.fail("Unexpected retry"))

    with pytest.raises(RuntimeError, match="prompt formatting failed"):
        generator.generate_paper(syllabus, _retrieved(syllabus))
    assert len(calls) == 1


def test_permanent_provider_error_stops_the_paper_and_redacts_token(monkeypatch, caplog):
    syllabus = _fixture_diagnostic_syllabus()
    calls = []
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 1)
    monkeypatch.setattr(config, "RUNPOD_API_KEY", "rpa_test_secret_token")

    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs)

            class Response:
                status_code = 401

                def json(self):
                    return {"error": "Invalid token rpa_test_secret_token"}

            return Response()

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(generator.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as error:
        generator.generate_paper(syllabus, _retrieved(syllabus))
    assert len(calls) == 1
    assert "Perimeter" in str(error.value)
    assert "rpa_test_secret_token" not in str(error.value)
    assert "rpa_test_secret_token" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_past_paper_transform_reaches_generation_with_source_answer(monkeypatch):
    source = {
        "question_id": "fixture-source-01",
        "question": "A rectangle has sides 5 cm and 3 cm. Calculate the perimeter.",
        "canonical_answer": "16 cm",
    }
    row = {
        "question": "A 5 cm by 3 cm rectangular label needs edging around all sides. How much edging is needed?",
        "options": ["8 cm", "15 cm", "16 cm", "30 cm"],
        "correct_answer": "16 cm",
        "explanation": "The edging length is 2 × (5 + 3) = 16 cm.",
        "source_question_id": source["question_id"],
    }
    prompts = []

    def invoke(prompt, **_kwargs):
        prompts.append(prompt)
        return json.dumps({"questions": [row]})

    monkeypatch.setattr(generator, "_invoke_llm", invoke)
    result = generator.generate_mcqs(
        "Perimeter", "Apply", [EVIDENCE], num_questions=1, past_questions=[source]
    )
    assert len(result) == 1
    assert result[0]["source_question_id"] == source["question_id"]
    assert result[0]["options"]["ABCD".index(result[0]["answer"])] == "16 cm"
    assert source["question_id"] in prompts[0]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("Circle", "B"), ("B", "B"), ("option B", "B"), ("B.", "B"), ("Because it is round", None)],
)
def test_answer_text_is_not_misread_as_its_first_letter(answer, expected):
    row = {
        "question": "Which of these shapes has a curved boundary?",
        "options": ["Triangle", "Circle", "Square", "Rectangle"],
        "answer": answer,
        "explanation": "A circle has a curved boundary.",
    }
    parsed = generator.parse_mcq_response(json.dumps({"questions": [row]}))
    assert (parsed[0]["answer"] if parsed else None) == expected


def test_malformed_option_count_is_rejected_without_index_error():
    row = {
        "question": "What is the result?",
        "options": ["1", "2", "3", "4", "5"],
        "answer": "5",
        "explanation": "The result is five.",
    }
    assert generator.parse_mcq_response(json.dumps({"questions": [row]})) == []


def test_overlong_explanation_reaches_quality_gate_without_truncation():
    row = {
        "question": "Which result is specified?",
        "options": ["1", "2", "3", "4"],
        "answer": "A",
        "explanation": "x" * 601,
    }
    parsed = generator.parse_mcq_response(json.dumps({"questions": [row]}))
    assert len(parsed[0]["explanation"]) == 601
    assert "a concise explanation is required" in validate_generated_mcq(parsed[0])
