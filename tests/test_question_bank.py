import json

import app as app_module
import config
from rag.diagnostic import build_diagnostic_syllabus
from rag.question_bank import (
    load_question_bank,
    question_bank_version,
    questions_for_concept,
    validate_question_bank,
)


def _record(mapping, *, question_id="2022-p1-q01", status="verified"):
    competency_levels = mapping.get("competency_levels", [])
    return {
        "question_id": question_id,
        "year": 2022,
        "paper": "G.C.E. O/L Mathematics Paper I",
        "question_number": "1",
        "grade": mapping["grade"],
        "concept_id": mapping["topic_id"],
        "concept_name": mapping["topic_name"],
        "competency_level": competency_levels[0] if competency_levels else "",
        "question": "A length of 7 cm is increased by 5 cm. Find the new length.",
        "options": ["10 cm", "11 cm", "12 cm", "13 cm"],
        "answer": "C",
        "solution": "Add 7 cm and 5 cm to obtain 12 cm.",
        "source_file": "ol-mathematics-2022.pdf",
        "source_page": 2,
        "mapping_status": status,
        "reviewed_by": "curriculum-reviewer@example.com",
        "reviewed_at": "2026-09-08T09:00:00+00:00",
        "mapping_evidence": "The method and terminology align with this reviewed concept.",
    }


def _write_bank(directory, records):
    path = directory / "questions.json"
    path.write_text(
        json.dumps({"schema_version": "1", "questions": records}),
        encoding="utf-8",
    )
    return path


def test_validated_bank_loads_exact_reviewed_concept_mapping(tmp_path, monkeypatch):
    mapping = build_diagnostic_syllabus()["topic_mappings"][0]
    _write_bank(tmp_path, [_record(mapping)])
    monkeypatch.setattr(config, "QUESTION_BANK_REQUIRED", True)

    status = validate_question_bank(tmp_path)
    questions = load_question_bank(tmp_path)

    assert status["ready"] is True
    assert status["question_count"] == 1
    assert status["mapped_concepts"] == [mapping["topic_id"]]
    assert questions[0].canonical_answer == "12 cm"
    assert question_bank_version(questions) != "none"


def test_unreviewed_or_wrong_concept_mapping_fails_closed(tmp_path):
    mapping = build_diagnostic_syllabus()["topic_mappings"][0]
    unreviewed = _record(mapping, status="pending")
    wrong_concept = _record(mapping, question_id="2022-p1-q02")
    wrong_concept["concept_id"] = "not-a-curriculum-concept"
    wrong_concept["question"] = "A distinct source question with a different stem."
    _write_bank(tmp_path, [unreviewed, wrong_concept])

    status = validate_question_bank(tmp_path)

    assert status["ready"] is False
    assert any("mapping must be verified" in error for error in status["errors"])
    assert any("unknown concept_id" in error for error in status["errors"])


def test_question_selection_excludes_sources_already_seen_by_student(tmp_path):
    mapping = build_diagnostic_syllabus()["topic_mappings"][0]
    first = _record(mapping)
    second = _record(mapping, question_id="2023-p1-q01")
    second["year"] = 2023
    second["question"] = "A ribbon is 9 cm long and gains 4 cm. Find its new length."
    _write_bank(tmp_path, [first, second])
    questions = load_question_bank(tmp_path)

    selected = questions_for_concept(
        mapping["topic_id"],
        grade=mapping["grade"],
        exclude_question_ids={first["question_id"]},
        seed="student-quiz-seed",
        questions=questions,
    )

    assert [question.question_id for question in selected] == [second["question_id"]]


def test_question_bank_status_is_admin_only_and_reports_missing_data(client, monkeypatch):
    with app_module.app.app_context():
        user = app_module.User(
            email="reviewer@example.com",
            name="Reviewer",
            password_hash="not-used-in-this-test",
            email_verified_at=app_module._utcnow(),
        )
        app_module.db.session.add(user)
        app_module.db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["session_version"] = 0

    denied = client.get("/api/question-bank/status")
    assert denied.status_code == 403

    monkeypatch.setattr(config, "ADMIN_EMAILS", {"reviewer@example.com"})
    response = client.get("/api/question-bank/status")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["ready"] is False
    assert payload["question_count"] == 0
