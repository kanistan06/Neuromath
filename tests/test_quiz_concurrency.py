import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_concurrent_students_and_duplicate_submissions_are_isolated(tmp_path):
    database_path = (tmp_path / "concurrent-quiz.sqlite3").as_posix()
    environment = {
        **os.environ,
        "APP_ENV": "development",
        "DATABASE_URL": f"sqlite:///{database_path}",
        "EMAIL_DELIVERY_MODE": "console",
        "OBSERVABILITY_ENABLED": "false",
        "RUNPOD_API_KEY": "",
        "QUESTION_BANK_REQUIRED": "false",
        "QUESTION_BANK_DIR": str(tmp_path / "past_papers"),
        "RATELIMIT_ENABLED": "false",
        "CACHE_REDIS_URL": "memory://",
        "AUTO_INGEST_ON_STARTUP": "false",
    }
    script = textwrap.dedent(
        """
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import app as module

        module.app.config.update(TESTING=True)
        students = []
        with module.app.app_context():
            for index in range(8):
                user = module.User(
                    email=f"student-{index}@example.com",
                    name=f"Student {index}",
                    password_hash="test-only",
                    email_verified_at=module._utcnow(),
                )
                module.db.session.add(user)
                module.db.session.commit()
                active, paper = module._store_active_quiz(user, [{
                    "question": f"Private arithmetic task for student {index}",
                    "options": ["1", "2", "3", "4"],
                    "answer": "C",
                    "explanation": "The specified result is three.",
                    "topic_id": f"fixture-{index}",
                }], "diagnostic")
                students.append((user.id, active.quiz_id, paper[0]["answer"]))

        barrier = threading.Barrier(8)

        def take_quiz(student):
            user_id, quiz_id, answer = student
            client = module.app.test_client()
            with client.session_transaction() as session:
                session["user_id"] = user_id
                session["session_version"] = 0
            public = client.get("/api/paper").get_json()
            assert public["quiz_id"] == quiz_id
            assert "answer" not in public["paper"][0]
            csrf = client.get("/api/csrf-token").get_json()["csrf_token"]
            barrier.wait(timeout=10)
            response = client.post(
                "/api/submit",
                json={"quiz_id": quiz_id, "answers": {"0": answer}},
                headers={"X-CSRFToken": csrf},
            )
            assert response.status_code == 200, response.get_json()
            assert response.get_json()["correct"] == 1
            attempts = client.get("/api/attempts").get_json()["attempts"]
            assert len(attempts) == 1
            assert attempts[0]["quiz_id"] == quiz_id
            return response.status_code

        with ThreadPoolExecutor(max_workers=8) as executor:
            statuses = list(executor.map(take_quiz, students))

        with module.app.app_context():
            assert module.Attempt.query.count() == 8
            assert module.AttemptQuestion.query.count() == 8
            assert module.ActiveQuiz.query.count() == 0
            user = module.db.session.get(module.User, students[0][0])
            active, paper = module._store_active_quiz(user, [{
                "question": "Duplicate submission race",
                "options": ["5", "6", "7", "8"],
                "answer": "D",
                "explanation": "The specified result is eight.",
                "topic_id": "fixture-race",
            }], "practice")
            race_quiz_id = active.quiz_id
            race_answer = paper[0]["answer"]

        race_barrier = threading.Barrier(2)

        def submit_same_quiz(_):
            client = module.app.test_client()
            with client.session_transaction() as session:
                session["user_id"] = students[0][0]
                session["session_version"] = 0
            csrf = client.get("/api/csrf-token").get_json()["csrf_token"]
            race_barrier.wait(timeout=10)
            return client.post(
                "/api/submit",
                json={"quiz_id": race_quiz_id, "answers": {"0": race_answer}},
                headers={"X-CSRFToken": csrf},
            ).status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            race_statuses = sorted(executor.map(submit_same_quiz, range(2)))
        assert race_statuses == [200, 409], race_statuses
        with module.app.app_context():
            assert module.Attempt.query.filter_by(quiz_id=race_quiz_id).count() == 1
            assert module.Attempt.query.count() == 9
        print(json.dumps({"students": len(statuses), "race_statuses": race_statuses}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary == {"students": 8, "race_statuses": [200, 409]}
