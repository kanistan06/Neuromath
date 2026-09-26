from datetime import timedelta

import app as app_module
from email_validator import EmailNotValidError
from auth_security import (
    hash_otp,
    hash_token,
    new_numeric_otp,
    new_one_time_token,
    validate_password,
)
from werkzeug.security import generate_password_hash


STRONG_PASSWORD = "Correct-Horse-42!"
NEW_PASSWORD = "New-Correct-84!"


def test_password_policy_rejects_each_missing_requirement():
    assert not validate_password("Short1!").valid
    assert not validate_password("NOLOWERCASE42!").valid
    assert not validate_password("nouppercase42!").valid
    assert not validate_password("NoNumbersHere!").valid
    assert not validate_password("NoSymbolsHere42").valid
    assert validate_password(STRONG_PASSWORD).valid


def test_one_time_tokens_are_random_and_only_hashes_need_persisting():
    first_raw, first_digest = new_one_time_token()
    second_raw, second_digest = new_one_time_token()
    assert first_raw != second_raw
    assert first_digest == hash_token(first_raw)
    assert second_digest == hash_token(second_raw)
    assert first_raw not in first_digest
    assert len(first_digest) == 64


def test_numeric_otp_is_six_digits_and_uses_a_keyed_context_bound_digest():
    otp = new_numeric_otp()
    assert len(otp) == 6 and otp.isdigit()
    first = hash_otp(otp, user_id=1, token_id=10)
    assert len(first) == 64
    assert otp not in first
    assert first != hash_otp(otp, user_id=1, token_id=11)


def test_csrf_rejects_state_change_without_token(client):
    response = client.post(
        "/api/signup",
        json={"name": "Alice", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    assert response.status_code == 400
    assert response.get_json()["message"].startswith("Security token")


def test_signup_verify_and_signin_lifecycle(client, csrf_post, monkeypatch):
    delivered = {}
    monkeypatch.setattr(
        app_module,
        "send_verification_email",
        lambda email, name, token: delivered.update(email=email, token=token),
    )
    signup = csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    assert signup.status_code == 202
    assert signup.get_json()["status"] == "ok"

    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email="alice@example.com").one()
        stored = app_module.AuthToken.query.filter_by(user_id=user.id, purpose="verify_email").one()
        assert user.email_verified_at is None
        assert stored.token_hash == hash_token(delivered["token"])
        assert delivered["token"] not in stored.token_hash

    blocked = csrf_post(
        "/api/signin", {"email": "alice@example.com", "password": STRONG_PASSWORD}
    )
    assert blocked.status_code == 403
    assert blocked.get_json()["needs_verification"] is True

    verified = csrf_post("/api/verify-email", {"token": delivered["token"]})
    assert verified.status_code == 200
    reused = csrf_post("/api/verify-email", {"token": delivered["token"]})
    assert reused.status_code == 400

    signed_in = csrf_post(
        "/api/signin", {"email": "alice@example.com", "password": STRONG_PASSWORD}
    )
    assert signed_in.status_code == 200
    assert client.get("/api/me").get_json()["email"] == "alice@example.com"


def test_signup_rejects_an_email_domain_that_cannot_receive_mail(client, csrf_post, monkeypatch):
    monkeypatch.setattr(app_module.config, "EMAIL_CHECK_DELIVERABILITY", True)

    def reject_undeliverable(_email, *, check_deliverability):
        assert check_deliverability is True
        raise EmailNotValidError("The domain does not accept email.")

    monkeypatch.setattr(app_module, "validate_email", reject_undeliverable)
    response = csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@invalid.test", "password": STRONG_PASSWORD},
    )
    assert response.status_code == 400
    assert "receive verification messages" in response.get_json()["message"]
    with app_module.app.app_context():
        assert app_module.User.query.count() == 0


def test_signup_and_signin_are_enumeration_safe(client, csrf_post, monkeypatch):
    monkeypatch.setattr(app_module, "send_verification_email", lambda *args: None)
    csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email="alice@example.com").one()
        user.email_verified_at = app_module._utcnow()
        app_module.db.session.commit()

    existing = csrf_post(
        "/api/signup",
        {"name": "Someone", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    fresh = csrf_post(
        "/api/signup",
        {"name": "Someone", "email": "other@example.com", "password": STRONG_PASSWORD},
    )
    assert existing.status_code == fresh.status_code == 202
    assert existing.get_json()["message"] == fresh.get_json()["message"]

    wrong_existing = csrf_post(
        "/api/signin", {"email": "alice@example.com", "password": "Wrong-Password-42!"}
    )
    wrong_missing = csrf_post(
        "/api/signin", {"email": "missing@example.com", "password": "Wrong-Password-42!"}
    )
    assert wrong_existing.status_code == wrong_missing.status_code == 401
    assert wrong_existing.get_json()["message"] == wrong_missing.get_json()["message"]


def test_forgot_reset_is_generic_one_time_and_invalidates_sessions(client, csrf_post, monkeypatch):
    reset_delivery = {}
    monkeypatch.setattr(app_module, "send_verification_email", lambda *args: None)
    monkeypatch.setattr(
        app_module,
        "send_password_reset_otp",
        lambda email, name, otp: reset_delivery.update(email=email, otp=otp),
    )
    monkeypatch.setattr(app_module, "send_password_changed_notice", lambda *args: None)
    csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email="alice@example.com").one()
        user.email_verified_at = app_module._utcnow()
        app_module.db.session.commit()

    csrf_post("/api/signin", {"email": "alice@example.com", "password": STRONG_PASSWORD})
    assert client.get("/api/me").get_json()["email"] == "alice@example.com"

    existing = csrf_post("/api/forgot-password", {"email": "alice@example.com"})
    missing = csrf_post("/api/forgot-password", {"email": "missing@example.com"})
    assert existing.status_code == missing.status_code == 200
    assert existing.get_json()["message"] == missing.get_json()["message"]
    with app_module.app.app_context():
        stored = app_module.AuthToken.query.filter_by(purpose="reset_password").one()
        assert reset_delivery["otp"] not in stored.token_hash
        assert stored.failed_attempts == 0

    reset = csrf_post(
        "/api/reset-password",
        {"email": "alice@example.com", "otp": reset_delivery["otp"], "password": NEW_PASSWORD},
    )
    assert reset.status_code == 200
    assert client.get("/api/me").get_json()["email"] is None
    reused = csrf_post(
        "/api/reset-password",
        {"email": "alice@example.com", "otp": reset_delivery["otp"], "password": STRONG_PASSWORD},
    )
    assert reused.status_code == 400
    old_login = csrf_post(
        "/api/signin", {"email": "alice@example.com", "password": STRONG_PASSWORD}
    )
    assert old_login.status_code == 401
    new_login = csrf_post(
        "/api/signin", {"email": "alice@example.com", "password": NEW_PASSWORD}
    )
    assert new_login.status_code == 200


def test_password_reset_otp_locks_after_five_failed_attempts():
    with app_module.app.app_context():
        user = app_module.User(
            email="locked@example.com",
            name="Locked Student",
            password_hash=generate_password_hash(STRONG_PASSWORD),
            email_verified_at=app_module._utcnow(),
        )
        app_module.db.session.add(user)
        app_module.db.session.flush()
        otp = app_module._issue_password_reset_otp(user)
        app_module.db.session.commit()

        wrong_otp = "000000" if otp != "000000" else "000001"
        for _ in range(app_module.config.PASSWORD_RESET_MAX_ATTEMPTS):
            assert app_module._check_password_reset_otp(user, wrong_otp) is None
        assert app_module._check_password_reset_otp(user, otp) is None
        stored = app_module.AuthToken.query.filter_by(
            user_id=user.id, purpose="reset_password"
        ).one()
        assert stored.used_at is not None


def test_password_reset_otp_also_verifies_mailbox_ownership(client, csrf_post, monkeypatch):
    reset_delivery = {}
    monkeypatch.setattr(app_module, "send_verification_email", lambda *args: None)
    monkeypatch.setattr(
        app_module,
        "send_password_reset_otp",
        lambda email, name, otp: reset_delivery.update(email=email, otp=otp),
    )
    monkeypatch.setattr(app_module, "send_password_changed_notice", lambda *args: None)
    csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )

    requested = csrf_post("/api/forgot-password", {"email": "alice@example.com"})
    assert requested.status_code == 200
    reset = csrf_post(
        "/api/reset-password",
        {"email": "alice@example.com", "otp": reset_delivery["otp"], "password": NEW_PASSWORD},
    )
    assert reset.status_code == 200
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email="alice@example.com").one()
        assert user.email_verified_at is not None


def test_expired_token_is_rejected(client, csrf_post, monkeypatch):
    delivery = {}
    monkeypatch.setattr(
        app_module,
        "send_verification_email",
        lambda email, name, token: delivery.update(token=token),
    )
    csrf_post(
        "/api/signup",
        {"name": "Alice Example", "email": "alice@example.com", "password": STRONG_PASSWORD},
    )
    with app_module.app.app_context():
        token = app_module.AuthToken.query.filter_by(purpose="verify_email").one()
        token.expires_at = app_module._utcnow() - timedelta(seconds=1)
        app_module.db.session.commit()
    response = csrf_post("/api/verify-email", {"token": delivery["token"]})
    assert response.status_code == 400


def test_signin_rate_limit(client, csrf_post):
    responses = [
        csrf_post(
            "/api/signin",
            {"email": "target@example.com", "password": "Wrong-Password-42!"},
        )
        for _ in range(6)
    ]
    assert [response.status_code for response in responses[:5]] == [401] * 5
    assert responses[5].status_code == 429


def test_active_papers_are_private_per_user_and_hide_answers(client):
    second_client = app_module.app.test_client()
    first_paper = [
        {
            "question": "First private question?",
            "options": ["1", "2", "3", "4"],
            "answer": "B",
            "explanation": "Two is correct.",
            "topic_id": "first",
            "difficulty_level": "Apply",
        }
    ]
    second_paper = [
        {
            "question": "Second private question?",
            "options": ["5", "6", "7", "8"],
            "answer": "D",
            "explanation": "Eight is correct.",
            "topic_id": "second",
            "difficulty_level": "Apply",
        }
    ]
    with app_module.app.app_context():
        first = app_module.User(
            email="first@example.com",
            name="First",
            password_hash=generate_password_hash(STRONG_PASSWORD),
            email_verified_at=app_module._utcnow(),
        )
        second = app_module.User(
            email="second@example.com",
            name="Second",
            password_hash=generate_password_hash(STRONG_PASSWORD),
            email_verified_at=app_module._utcnow(),
        )
        app_module.db.session.add_all([first, second])
        app_module.db.session.commit()
        first_id, second_id = first.id, second.id
        first_active, first_private = app_module._store_active_quiz(
            first, first_paper, "diagnostic"
        )
        second_active, _second_private = app_module._store_active_quiz(
            second, second_paper, "diagnostic"
        )
        first_quiz_id = first_active.quiz_id
        second_quiz_id = second_active.quiz_id
        first_answer = first_private[0]["answer"]

    with client.session_transaction() as session:
        session["user_id"] = first_id
        session["session_version"] = 0
    with second_client.session_transaction() as session:
        session["user_id"] = second_id
        session["session_version"] = 0

    first_response = client.get("/api/paper").get_json()["paper"]
    second_response = second_client.get("/api/paper").get_json()["paper"]
    assert first_response[0]["question"] == "First private question?"
    assert second_response[0]["question"] == "Second private question?"
    assert "answer" not in first_response[0]
    assert "explanation" not in first_response[0]

    csrf = client.get("/api/csrf-token").get_json()["csrf_token"]
    submitted = client.post(
        "/api/submit",
        json={"quiz_id": first_quiz_id, "answers": {"0": first_answer}},
        headers={"X-CSRFToken": csrf},
    )
    assert submitted.status_code == 200
    assert submitted.get_json()["correct"] == 1
    assert client.get("/api/paper").status_code == 404
    second_payload = second_client.get("/api/paper").get_json()
    assert second_payload["quiz_id"] == second_quiz_id
    first_attempts = client.get("/api/attempts").get_json()["attempts"]
    second_attempts = second_client.get("/api/attempts").get_json()["attempts"]
    assert len(first_attempts) == 1
    assert first_attempts[0]["quiz_id"] == first_quiz_id
    assert first_attempts[0]["quiz_kind"] == "diagnostic"
    assert second_attempts == []

    replay = client.post(
        "/api/submit",
        json={"quiz_id": first_quiz_id, "answers": {"0": first_answer}},
        headers={"X-CSRFToken": csrf},
    )
    assert replay.status_code == 409
