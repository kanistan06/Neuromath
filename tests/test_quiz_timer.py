import json
from datetime import timedelta

import app as application
from rag.answers import ANSWER_CONTRACT_VERSION


def _login_user(client, email="timer@example.com"):
    with application.app.app_context():
        user = application.User(
            email=email,
            name="Timer Student",
            password_hash="test-only",
            email_verified_at=application._utcnow(),
        )
        application.db.session.add(user)
        application.db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["session_version"] = 0
    return user_id


def test_practice_timer_fallback_tracks_topic_difficulty(monkeypatch):
    monkeypatch.setattr(application, "_load_json", lambda *_args, **_kwargs: {})
    assert application._practice_time_limit_minutes(7, "Sets", 1) == 12
    assert application._practice_time_limit_minutes(7, "Sets", 2) == 15
    assert application._practice_time_limit_minutes(7, "Sets", 3) == 20


def test_timer_setting_defaults_enabled_and_can_be_disabled(client, csrf_post):
    user_id = _login_user(client)

    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.get_json()["quiz_timer_enabled"] is True

    response = csrf_post(
        "/api/settings",
        {"theme": "dark", "difficulty": "medium", "quiz_timer_enabled": False},
    )
    assert response.status_code == 200
    assert response.get_json()["quiz_timer_enabled"] is False

    with application.app.app_context():
        settings = application.UserSettings.query.filter_by(user_id=user_id).one()
        assert settings.quiz_timer_enabled is False


def test_timed_quiz_requires_answers_before_expiry_and_allows_timeout_submit(client, csrf_post):
    user_id = _login_user(client, "timer-submit@example.com")
    quiz_id = "timer-quiz-identifier-123456"
    paper = [
        {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "answer": "B",
            "correct_answer": "4",
            "explanation": "2 + 2 = 4.",
            "topic_id": "test_topic",
            "difficulty_level": "Remember",
            "answer_contract_version": ANSWER_CONTRACT_VERSION,
        }
    ]

    with application.app.app_context():
        application.db.session.add(
            application.ActiveQuiz(
                user_id=user_id,
                quiz_id=quiz_id,
                quiz_kind="diagnostic",
                status="active",
                paper_json=json.dumps(paper),
                timer_enabled=True,
                timer_duration_seconds=600,
                expires_at=application._utcnow() + timedelta(minutes=10),
            )
        )
        application.db.session.commit()

    response = csrf_post("/api/submit", {"quiz_id": quiz_id, "answers": {}})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["code"] == "incomplete_quiz"
    assert payload["first_unanswered"] == 0

    with application.app.app_context():
        active = application.ActiveQuiz.query.filter_by(quiz_id=quiz_id).one()
        active.expires_at = application._utcnow() - timedelta(seconds=1)
        application.db.session.commit()

    response = csrf_post("/api/submit", {"quiz_id": quiz_id, "answers": {}})
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    assert payload["timed_out"] is True
    assert payload["correct"] == 0
    assert payload["incorrect"] == 1
