import copy

import pytest

import app as app_module
from rag.diagnostic import DiagnosticPaperError, build_diagnostic_syllabus, validate_diagnostic_paper
from werkzeug.security import generate_password_hash


def _valid_paper(syllabus):
    paper = [
        {
            "question": (
                f"{mapping['topic_id']} focuses on {mapping['topic_name']}. "
                f"Which listed result correctly applies this specific curriculum concept?"
            ),
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "answer": "A",
            "explanation": "The Grade textbook context supports Option 1.",
            "grounding_evidence": (
                "This exact textbook passage provides the mathematical method used for this question."
            ),
            "topic_id": mapping["topic_id"],
            "grade": mapping["grade"],
            "term": mapping["term"],
            "difficulty_level": "Apply",
            "source_refs": [
                {
                    "source": f"maths-g-{mapping['grade']}.pdf",
                    "page": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "grade": mapping["grade"],
                    "chunk_id": f"chunk-{mapping['topic_id']}",
                    "corpus_version": "test-corpus-v1",
                }
            ],
        }
        for mapping in syllabus["topic_mappings"]
    ]
    from rag.adaptive import LEVELS, POLICY_VERSION
    from rag.validation import REVIEW_VERSION, _content_digest

    for question, mapping in zip(paper, syllabus["topic_mappings"]):
        question["difficulty_level"] = LEVELS[mapping["required_levels"][0]]
        question["adaptive_policy_version"] = POLICY_VERSION
        question["quality_review"] = {
            "version": REVIEW_VERSION,
            "model_id": app_module.config.RUNPOD_MCQ_GENERATION_MODEL,
            "provider": "runpod",
            "content_sha256": _content_digest(question),
        }
    return paper


def test_grade_10_11_catalogue_has_exactly_one_mapping_per_concept():
    syllabus = build_diagnostic_syllabus()
    mappings = syllabus["topic_mappings"]
    assert syllabus["assessment_type"] == "diagnostic"
    assert syllabus["grades"] == [10, 11]
    assert len(mappings) == syllabus["total_questions"] == 56
    assert sum(mapping["grade"] == 10 for mapping in mappings) == 31
    assert sum(mapping["grade"] == 11 for mapping in mappings) == 25
    assert len({mapping["topic_id"] for mapping in mappings}) == 56
    assert all(mapping["min_questions"] == mapping["max_questions"] == 1 for mapping in mappings)
    assert all(mapping["strict_textbook_grounding"] for mapping in mappings)


def test_complete_grounded_diagnostic_is_accepted():
    syllabus = build_diagnostic_syllabus()
    validate_diagnostic_paper(_valid_paper(syllabus), syllabus)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "wrong_grade", "no_source", "no_evidence"],
)
def test_partial_duplicate_or_ungrounded_diagnostic_fails_closed(mutation):
    syllabus = build_diagnostic_syllabus()
    paper = _valid_paper(syllabus)
    if mutation == "missing":
        paper.pop()
    elif mutation == "duplicate":
        paper[-1]["topic_id"] = paper[0]["topic_id"]
    elif mutation == "wrong_grade":
        paper[0]["grade"] = 11 if paper[0]["grade"] == 10 else 10
    elif mutation == "no_source":
        paper[0]["source_refs"] = []
    elif mutation == "no_evidence":
        paper[0]["grounding_evidence"] = ""
    with pytest.raises(DiagnosticPaperError):
        validate_diagnostic_paper(copy.deepcopy(paper), syllabus)


def test_grounding_evidence_must_be_an_exact_excerpt():
    from rag.generator import _grounding_unit_for_evidence

    units = [
        {
            "text": "The perimeter of a rectangle is twice the sum of its length and breadth.",
            "source_ref": {"source": "grade-10.pdf", "page": 12, "grade": 10},
        }
    ]
    match = _grounding_unit_for_evidence(
        "The perimeter of a rectangle is twice the sum of its length and breadth",
        units,
    )
    assert match == units[0]
    assert _grounding_unit_for_evidence(
        "This unsupported statement was invented outside of the supplied textbook material",
        units,
    ) is None


def test_generation_block_ties_evidence_to_its_exact_textbook_page(monkeypatch):
    import rag.generator as generator

    evidence = "The perimeter of a rectangle is twice the sum of its length and breadth"
    monkeypatch.setattr(generator.config, "GENERATION_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(
        generator,
        "generate_mcqs",
        lambda **_kwargs: [
            {
                "question": "A rectangle is 5 cm by 3 cm. What is its perimeter?",
                "options": ["8 cm", "15 cm", "16 cm", "30 cm"],
                "answer": "C",
                "explanation": "Twice the sum of 5 and 3 is 16.",
                "grounding_evidence": evidence,
                "grade": 10,
                "concept_id": "g10_perimeter",
                "source": "grade-10.pdf",
                "page": 12,
            }
        ],
    )
    source_ref = {
        "source": "grade-10.pdf",
        "page": 12,
        "page_start": 12,
        "page_end": 12,
        "grade": 10,
        "chunk_id": "chunk-perimeter",
        "corpus_version": "test-corpus-v1",
    }
    result = generator._generate_mcqs_block(
        (
            "g10_perimeter",
            "Perimeter",
            "Apply",
            [evidence + "."],
            1,
            {"grade": 10, "term": 1, "strict_textbook_grounding": True},
            [source_ref],
            [{"text": evidence + ".", "source_ref": source_ref}],
            [],
            [],
        )
    )
    assert len(result) == 1
    assert result[0]["source_refs"] == [source_ref]
    assert result[0]["grade"] == 10


def test_generation_block_rejects_invented_grounding_evidence(monkeypatch):
    import rag.generator as generator

    monkeypatch.setattr(generator.config, "GENERATION_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(
        generator,
        "generate_mcqs",
        lambda **_kwargs: [
            {
                "question": "A rectangle is 5 cm by 3 cm. What is its perimeter?",
                "options": ["8 cm", "15 cm", "16 cm", "30 cm"],
                "answer": "C",
                "explanation": "Twice the sum of 5 and 3 is 16.",
                "grounding_evidence": (
                    "This invented sentence is not present anywhere in the supplied textbook chunk"
                ),
                "grade": 10,
                "concept_id": "g10_perimeter",
                "source": "grade-10.pdf",
                "page": 12,
            }
        ],
    )
    source_ref = {
        "source": "grade-10.pdf",
        "page": 12,
        "page_start": 12,
        "page_end": 12,
        "grade": 10,
        "chunk_id": "chunk-perimeter",
        "corpus_version": "test-corpus-v1",
    }
    result = generator._generate_mcqs_block(
        (
            "g10_perimeter",
            "Perimeter",
            "Apply",
            ["The perimeter of a rectangle is twice the sum of its length and breadth."],
            1,
            {"grade": 10, "term": 1, "strict_textbook_grounding": True},
            [source_ref],
            [
                {
                    "text": "The perimeter of a rectangle is twice the sum of its length and breadth.",
                    "source_ref": source_ref,
                }
            ],
            [],
            [],
        )
    )
    assert result == []


def test_validated_template_is_reused_as_private_answer_hidden_quiz(monkeypatch):
    syllabus = build_diagnostic_syllabus()
    paper = _valid_paper(syllabus)
    monkeypatch.setattr(app_module, "_has_llm_config", lambda: True)
    with app_module.app.app_context():
        user = app_module.User(
            email="diagnostic@example.com",
            name="Diagnostic Student",
            password_hash=generate_password_hash("Correct-Horse-42!"),
            email_verified_at=app_module._utcnow(),
        )
        app_module.db.session.add(user)
        app_module.db.session.commit()
        syllabus = app_module._student_diagnostic_syllabus(user)
        paper = _valid_paper(syllabus)
        fingerprint = app_module._diagnostic_fingerprint(syllabus)
        app_module.db.session.add(
            app_module.AssessmentTemplate(
                fingerprint=fingerprint,
                paper_json=app_module.json.dumps(paper),
                **app_module._artifact_versions(),
            )
        )
        app_module.db.session.commit()

        result = app_module._generate_and_save_paper(user)
        active, stored_paper = app_module._load_active_quiz(user)
        assert result["cached"] is True
        assert result["count"] == 25
        assert result["quiz_id"] == active.quiz_id
        assert "answer" not in result["paper"][0]
        assert active is not None
        assert stored_paper[0]["answer"] in {"A", "B", "C", "D"}
        assert stored_paper[0]["options"]["ABCD".index(stored_paper[0]["answer"])] == "Option 1"
