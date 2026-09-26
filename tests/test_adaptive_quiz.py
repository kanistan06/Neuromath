import copy
from collections import Counter

import pytest
from sqlalchemy import text

import app as application
from rag.adaptive import POLICY_VERSION, select_diagnostic_syllabus, topic_progress
from rag.diagnostic import build_diagnostic_syllabus


def _history(syllabus, attempt=1, correct=True, level="Apply"):
    return {row["topic_id"]: [{
        "attempt_id": attempt, "attempt_number": attempt, "is_correct": correct,
        "difficulty_level": level, "question": f"Previous calculation for {row['topic_id']}",
        "source_question_id": f"source-{row['topic_id']}-{attempt}",
    }] for row in syllabus["topic_mappings"]}


def test_25_selected_concepts_rotate_and_cover_catalogue_after_success():
    catalogue = build_diagnostic_syllabus()
    original = copy.deepcopy(catalogue)
    history, served = {}, set()
    selections = []
    for attempt in range(3):
        syllabus = select_diagnostic_syllabus(
            catalogue, student_key="17", history_by_topic=history,
            attempt_count=attempt, diagnostic_count=attempt, latest_attempt_id=attempt,
        )
        ids = {row["topic_id"] for row in syllabus["topic_mappings"]}
        assert len(ids) == syllabus["total_questions"] == 25
        assert sorted(Counter(row["grade"] for row in syllabus["topic_mappings"]).values()) == [12, 13]
        assert all(row["required_levels"][0] >= 3 and row["require_application"] for row in syllabus["topic_mappings"])
        selections.append(ids)
        served.update(ids)
        for key, rows in _history(syllabus, attempt + 1).items():
            history.setdefault(key, []).extend(rows)
    assert selections[0].isdisjoint(selections[1])
    assert len(served) == 56
    assert catalogue == original


def test_same_snapshot_is_reproducible_and_other_students_get_different_selection():
    catalogue = build_diagnostic_syllabus()
    first = select_diagnostic_syllabus(catalogue, student_key="alice")
    assert first == select_diagnostic_syllabus(catalogue, student_key="alice")
    second = select_diagnostic_syllabus(catalogue, student_key="bob")
    assert {row["topic_id"] for row in first["topic_mappings"]} != {row["topic_id"] for row in second["topic_mappings"]}
    assert first["selection_seed"] != second["selection_seed"]


def test_weak_concepts_get_review_without_displacing_new_coverage():
    catalogue = build_diagnostic_syllabus()
    first = select_diagnostic_syllabus(catalogue, student_key="student")
    history = _history(first, correct=False)
    second = select_diagnostic_syllabus(catalogue, student_key="student", history_by_topic=history,
                                         attempt_count=1, diagnostic_count=1, latest_attempt_id=1)
    reviews = [row for row in second["topic_mappings"] if row["topic_id"] in history]
    assert len(reviews) == 8
    assert all(row["required_levels"] == [3] for row in reviews)
    assert all(row["avoid_question_stems"] and row["excluded_source_question_ids"] for row in reviews)


def test_teaching_periods_increase_selection_priority():
    catalogue = build_diagnostic_syllabus()
    for row in catalogue["topic_mappings"]:
        row["teaching_periods"] = 4
        row["competency_levels"] = []
    high, low = catalogue["topic_mappings"][:2]
    high["teaching_periods"] = 12
    counts = Counter()
    for seed in range(120):
        syllabus = select_diagnostic_syllabus(catalogue, student_key=str(seed))
        counts.update(row["topic_id"] for row in syllabus["topic_mappings"])
    assert counts[high["topic_id"]] > counts[low["topic_id"]] * 1.5


@pytest.mark.parametrize("level, expected", [("Apply", 4), ("Analyse", 5), ("unknown", 3), ("Remember", 3)])
def test_difficulty_requires_success_on_two_distinct_attempts(level, expected):
    rows = [{"attempt_id": attempt, "attempt_number": attempt, "is_correct": True,
             "difficulty_level": level} for attempt in (1, 2)]
    assert topic_progress(rows)["next_level"] == expected
    assert topic_progress([rows[0]] * 10)["next_level"] == 3
    rows[-1]["is_correct"] = False
    assert topic_progress(rows)["next_level"] == 3


def _user(email):
    user = application.User(email=email, name="Student", password_hash="test-only",
                            email_verified_at=application._utcnow())
    application.db.session.add(user)
    application.db.session.flush()
    return user


def _attempt(user, topic, level="Apply", correct=True, kind="diagnostic"):
    attempt = application.Attempt(user_id=user.id, quiz_kind=kind, total_questions=1,
                                  correct=int(correct), incorrect=int(not correct), score_percent=100 * int(correct))
    application.db.session.add(attempt)
    application.db.session.flush()
    application.db.session.add(application.AttemptQuestion(
        attempt_id=attempt.id, question_index=0, topic_id=topic, question=f"Question from attempt {attempt.id}",
        is_correct=correct, difficulty_level=level, source_question_id=f"source-{attempt.id}",
    ))
    application.db.session.commit()
    return attempt.id


def test_memory_survives_session_reload_and_is_scoped_to_student():
    with application.app.app_context():
        alice, bob = _user("alice@example.com"), _user("bob@example.com")
        application.db.session.commit()
        alice_id, bob_id = alice.id, bob.id
        topic = build_diagnostic_syllabus()["topic_mappings"][0]["topic_id"]
        _attempt(alice, topic)
        _attempt(alice, topic, kind="practice")
        before = application._student_diagnostic_syllabus(alice)
        _attempt(bob, topic, correct=False)
        application.db.session.remove()
        alice = application.db.session.get(application.User, alice_id)
        bob = application.db.session.get(application.User, bob_id)
        memory = application._student_memory(alice)
        assert memory["attempt_count"] == 2 and memory["diagnostic_count"] == 1
        assert topic_progress(memory["history_by_topic"][topic])["next_level"] == 4
        assert topic_progress(application._student_memory(bob)["history_by_topic"][topic])["accuracy"] == 0
        assert application._student_diagnostic_syllabus(alice) == before
        assert application._diagnostic_fingerprint(before) != application._diagnostic_fingerprint(application._student_diagnostic_syllabus(bob))


def test_legacy_difficulty_column_is_added_idempotently_without_losing_answers():
    with application.app.app_context():
        user = _user("legacy-memory@example.com")
        _attempt(user, "legacy-topic")
        application.db.session.execute(text("ALTER TABLE attempt_questions DROP COLUMN difficulty_level"))
        application.db.session.commit()
        application._ensure_attempt_question_columns()
        application._ensure_attempt_question_columns()
        row = application.db.session.query(application.AttemptQuestion).one()
        assert row.is_correct is True and row.difficulty_level == "unknown"


def test_full_student_quiz_lifecycle_uses_submitted_history_and_keeps_unfinished_quiz(monkeypatch, client, csrf_post):
    import rag.generator as generator
    import rag.ingest as ingest
    import rag.retriever as retriever
    from test_diagnostic import _valid_paper

    generated = []
    retrievals = []

    def retrieve(syllabus, **kwargs):
        retrievals.append((copy.deepcopy(syllabus), kwargs))
        return {}

    def generate(syllabus, _content, **_kwargs):
        generated.append(copy.deepcopy(syllabus))
        return _valid_paper(syllabus)

    monkeypatch.setattr(application, "_has_llm_config", lambda: True)
    monkeypatch.setattr(ingest, "ensure_vector_index_ready", lambda **_kwargs: {"count": 100})
    monkeypatch.setattr(retriever, "retrieve_for_syllabus", retrieve)
    monkeypatch.setattr(generator, "generate_paper", generate)
    with application.app.app_context():
        user = _user("lifecycle@example.com")
        application.db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["session_version"] = 0

    first_response = csrf_post("/api/quiz/load", {})
    assert first_response.status_code == 200, first_response.get_json()
    first = first_response.get_json()
    assert first["count"] == 25
    assert all(not {"answer", "correct_answer", "explanation", "quality_review"}.intersection(q) for q in first["paper"])
    again = csrf_post("/api/quiz/load", {}).get_json()
    assert again["quiz_id"] == first["quiz_id"] and again["paper"] == first["paper"]
    assert len(generated) == 1
    with application.app.app_context():
        user = application.db.session.get(application.User, user_id)
        assert application._student_memory(user)["attempt_count"] == 0

    answers = {str(index): "ABCD"[question["options"].index("Option 1")] for index, question in enumerate(first["paper"])}
    assert sorted(Counter(answers.values()).values()) == [6, 6, 6, 7]
    result = csrf_post("/api/submit", {"quiz_id": first["quiz_id"], "answers": answers})
    assert result.status_code == 200 and result.get_json()["correct"] == 25
    duplicate = csrf_post("/api/submit", {"quiz_id": first["quiz_id"], "answers": answers})
    assert duplicate.status_code == 409
    with application.app.app_context():
        application.db.session.remove()
        user = application.db.session.get(application.User, user_id)
        memory = application._student_memory(user)
        assert memory["attempt_count"] == 1 and len(memory["history_by_topic"]) == 25
        assert all(row.difficulty_level == "Apply" for row in application.AttemptQuestion.query.all())
    second = csrf_post("/api/quiz/load", {}).get_json()
    assert second["count"] == 25 and second["quiz_id"] != first["quiz_id"]
    assert {q["topic_id"] for q in first["paper"]}.isdisjoint(q["topic_id"] for q in second["paper"])
    assert len(generated) == 2
    assert retrievals[0][1]["question_seed"] != retrievals[1][1]["question_seed"]
    assert generated[0]["selection_version"] == POLICY_VERSION


def test_failed_validation_does_not_publish_or_cache_a_partial_diagnostic(monkeypatch):
    import rag.generator as generator
    import rag.retriever as retriever

    monkeypatch.setattr(application, "_has_llm_config", lambda: True)
    monkeypatch.setattr(retriever, "retrieve_for_syllabus", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generator, "generate_paper", lambda *_args, **_kwargs: [])
    with application.app.app_context():
        user = _user("incomplete@example.com")
        application.db.session.commit()
        with pytest.raises(RuntimeError, match="expected 25"):
            application._generate_and_save_paper(user)
        assert application.ActiveQuiz.query.count() == 0
        assert application.AssessmentTemplate.query.count() == 0


def test_template_cache_is_not_shared_between_students(monkeypatch):
    import rag.generator as generator
    import rag.retriever as retriever
    from test_diagnostic import _valid_paper

    calls = []

    def generate(syllabus, _content, **_kwargs):
        calls.append(syllabus["selection_seed"])
        return _valid_paper(syllabus)

    monkeypatch.setattr(application, "_has_llm_config", lambda: True)
    monkeypatch.setattr(retriever, "retrieve_for_syllabus", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generator, "generate_paper", generate)
    with application.app.app_context():
        alice, bob = _user("private-a@example.com"), _user("private-b@example.com")
        application.db.session.commit()
        first = application._generate_and_save_paper(alice)
        second = application._generate_and_save_paper(bob)
        assert first["quiz_id"] != second["quiz_id"]
        assert len(set(calls)) == 2
        assert application.ActiveQuiz.query.count() == application.AssessmentTemplate.query.count() == 2
