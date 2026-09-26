import json

import app as application


def _verified_user(email: str):
    user = application.User(
        email=email,
        name=email.split("@", 1)[0],
        password_hash="test-only",
        email_verified_at=application._utcnow(),
    )
    application.db.session.add(user)
    application.db.session.commit()
    return user


def _paper(label: str):
    return [{
        "question": f"Private question for {label}",
        "options": ["1", "2", "3", "4"],
        "answer": "B",
        "correct_answer": "2",
        "explanation": "Correct result: 2.",
        "topic_id": f"topic-{label}",
        "difficulty_level": "Apply",
        "grade": 10,
    }]


def _login(client, user_id: int):
    with client.session_transaction() as session:
        session["user_id"] = int(user_id)
        session["session_version"] = 0


def _csrf(client):
    return client.get("/api/csrf-token").get_json()["csrf_token"]


def test_foreign_quiz_identifier_cannot_cross_user_boundary():
    first_client = application.app.test_client()
    second_client = application.app.test_client()

    with application.app.app_context():
        first = _verified_user("isolation-a@example.com")
        second = _verified_user("isolation-b@example.com")
        first_id, second_id = first.id, second.id
        first_active, first_private = application._store_active_quiz(first, _paper("a"), "practice")
        second_active, _second_private = application._store_active_quiz(second, _paper("b"), "practice")
        first_quiz_id = first_active.quiz_id
        second_quiz_id = second_active.quiz_id
        first_answer = first_private[0]["answer"]

    _login(first_client, first_id)
    _login(second_client, second_id)

    # User B cannot use User A's opaque quiz identifier to submit or inspect state.
    second_token = _csrf(second_client)
    submit = second_client.post(
        "/api/submit",
        json={"quiz_id": first_quiz_id, "answers": {"0": first_answer}},
        headers={"X-CSRFToken": second_token},
    )
    assert submit.status_code == 409

    batch_status = second_client.get(f"/api/quiz/batch-status?quiz_id={first_quiz_id}")
    assert batch_status.status_code == 409

    second_paper = second_client.get("/api/paper").get_json()
    assert second_paper["quiz_id"] == second_quiz_id
    assert second_paper["paper"][0]["question"] == "Private question for b"

    with application.app.app_context():
        assert application.Attempt.query.count() == 0
        assert application.ActiveQuiz.query.filter_by(user_id=first_id, quiz_id=first_quiz_id).count() == 1
        assert application.ActiveQuiz.query.filter_by(user_id=second_id, quiz_id=second_quiz_id).count() == 1


def test_production_error_response_does_not_expose_internal_exception(monkeypatch):
    client = application.app.test_client()
    monkeypatch.setattr(application, "_is_production", True)
    monkeypatch.setattr(
        application,
        "_load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("postgresql://secret-user:secret-password@private-host/internal")
        ),
    )

    response = client.get("/api/topics")
    assert response.status_code == 500
    payload = response.get_json()
    serialized = json.dumps(payload)
    assert "secret-password" not in serialized
    assert "private-host" not in serialized
    assert payload["message"] == "Topics are temporarily unavailable."


def test_non_object_json_body_is_rejected_without_server_error(client, csrf_post):
    token = _csrf(client)
    response = client.post(
        "/api/signin",
        json=["not", "an", "object"],
        headers={"X-CSRFToken": token},
    )
    assert response.status_code == 400
    assert response.get_json()["message"] == "Password is required."


def test_attempt_numbers_are_scoped_per_user_not_global_primary_keys():
    first_client = application.app.test_client()
    second_client = application.app.test_client()

    with application.app.app_context():
        first = _verified_user("attempt-scope-a@example.com")
        second = _verified_user("attempt-scope-b@example.com")
        first_id, second_id = first.id, second.id

        first_old = application.Attempt(
            user_id=first_id,
            quiz_id="attempt-scope-a-old",
            quiz_kind="practice",
            total_questions=10,
            correct=7,
            incorrect=3,
            score_percent=70.0,
        )
        application.db.session.add(first_old)
        application.db.session.commit()

        second_only = application.Attempt(
            user_id=second_id,
            quiz_id="attempt-scope-b-only",
            quiz_kind="diagnostic",
            total_questions=25,
            correct=20,
            incorrect=5,
            score_percent=80.0,
        )
        application.db.session.add(second_only)
        application.db.session.commit()

        first_new = application.Attempt(
            user_id=first_id,
            quiz_id="attempt-scope-a-new",
            quiz_kind="practice",
            total_questions=10,
            correct=9,
            incorrect=1,
            score_percent=90.0,
        )
        application.db.session.add(first_new)
        application.db.session.commit()

        first_old_id = first_old.id
        second_only_id = second_only.id
        first_new_id = first_new.id

    _login(first_client, first_id)
    _login(second_client, second_id)

    first_attempts = first_client.get("/api/attempts").get_json()["attempts"]
    second_attempts = second_client.get("/api/attempts").get_json()["attempts"]

    assert [item["attempt_id"] for item in first_attempts] == [first_new_id, first_old_id]
    assert [item["attempt_number"] for item in first_attempts] == [2, 1]
    assert [item["attempt_id"] for item in second_attempts] == [second_only_id]
    assert [item["attempt_number"] for item in second_attempts] == [1]


def test_public_error_messages_never_expose_internal_exception_text():
    secret = RuntimeError("postgresql://secret-user:secret-password@private-host/internal")
    assert application._public_error_message(secret, "Please try again shortly.") == "Please try again shortly."


def test_inference_failure_response_is_student_safe():
    from rag.inference_errors import GenerationBudgetExceeded

    with application.app.app_context():
        response = application._inference_failure_response(GenerationBudgetExceeded())
        assert response is not None
        payload = response.get_json()
        assert payload["code"] == "generation_budget_exhausted"
        assert payload["message"] == "Question generation is taking longer than expected. Please try again shortly."
        assert "inference-call limit" not in payload["message"]
