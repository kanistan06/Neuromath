import json
import re
from collections import Counter
from fractions import Fraction

import pytest

import app as application
import config
import rag.generator as generator
import rag.runpod_inference as inference
from rag.quality import validate_generated_mcq


TOPICS = ["Sets", "Probability", "Real Numbers", "Constructions", "Data Representation & Prediction"]
pytestmark = pytest.mark.usefixtures("stub_quality_review")


def _row(topic, number):
    if topic == "Sets":
        a, b, c = number, number + 1, number + 2
        stem = f"For A = {{{a}, {b}}} and B = {{{b}, {c}}}, find A ∪ B."
        options = [f"{{{a}, {b}}}", f"{{{b}, {c}}}", f"{{{a}, {b}, {c}}}", f"{{{b}}}"]
        reason = "The union contains each distinct element from either set."
    elif topic == "Probability":
        probability = Fraction(number, number + 10)
        step = Fraction(1, 2 * (number + 10))
        stem = f"A bag contains {number} red counters and 10 blue counters. What is the probability of drawing red at random?"
        options = [str(probability - step), str(probability + step), str(probability), "1"]
        reason = "Divide the number of red counters by the total number of counters."
    elif topic == "Real Numbers":
        stem = f"Evaluate {number}^2?"
        value = number**2
        options = [str(value - 1), str(value + 1), str(value), str(value + 2)]
        reason = "Squaring multiplies the number by itself."
    elif topic == "Constructions":
        stem = f"A perpendicular bisector divides a segment of length {2 * number} cm equally. What is each half's length?"
        options = [f"{number + 1} cm", f"{number + 2} cm", f"{number} cm", f"{number + 3} cm"]
        reason = "A bisector divides the segment into two equal lengths."
    else:
        stem = f"A data sample contains {number}, {number + 2}, and {number + 4}. What is its arithmetic mean?"
        options = [str(number), str(number + 1), str(number + 2), str(number + 3)]
        reason = "Divide the sum of the three observations by three."
    return {
        "question": stem, "options": options, "correct_answer": options[2],
        "explanation": f"Correct result: {options[2]}. {reason}",
    }


def _syllabus(topic, grade=9):
    return {
        "total_questions": 10,
        "topic_mappings": [{
            "topic_id": "practice-topic", "topic_name": topic, "grade": grade,
            "required_levels": [1, 2, 3], "min_questions": 10, "max_questions": 10,
        }],
    }


def _install_provider(monkeypatch, topic, *, reject_first=False):
    calls = []
    next_number = 1

    class Session:
        def post(self, _url, **kwargs):
            nonlocal next_number
            request = kwargs["json"]
            calls.append(request)
            schema = request["response_format"]["json_schema"]["schema"]["properties"]["questions"]
            assert "correct_answer" in schema["items"]["required"]
            assert "answer" not in schema["items"]["properties"]
            count = schema["maxItems"]
            rows = [_row(topic, next_number + index) for index in range(count)]
            next_number += count
            if reject_first and len(calls) == 1:
                rows[0]["correct_answer"] = rows[0]["options"][0]

            class Response:
                status_code = 200

                def json(self):
                    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"questions": rows})}}]}

            return Response()

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "GENERATION_RETRY_BASE_SECONDS", 0)
    return calls


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("workers", [1, 4])
def test_ten_fresh_numerical_practice_items_survive_all_generation_gates(monkeypatch, topic, workers):
    calls = _install_provider(monkeypatch, topic)
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", workers)
    paper = generator.generate_paper(_syllabus(topic), {"practice-topic": {"chunks": ["The textbook explains the selected topic and its mathematical methods."]}})
    assert len(paper) == 10
    assert len(calls) == 3
    assert Counter(question["difficulty_level"] for question in paper) == {"Remember": 4, "Understand": 3, "Apply": 3}
    assert all(question["answer"] == "C" for question in paper)
    assert all(validate_generated_mcq(question) == [] for question in paper)
    for request, expected_history in zip(calls, [0, 4, 7]):
        prompt = request["messages"][-1]["content"]
        history = re.search(r"Previously Served Question Stems[^\n]*\n([^\n]*)", prompt).group(1)
        assert len(json.loads(history)) == expected_history


def test_bad_answer_candidate_is_regenerated_with_feedback(monkeypatch):
    # This test specifically exercises the retry-feedback path. Production uses
    # candidate surplus, which can replace one rejected item from the same batch
    # without an extra author call. Disable surplus only for this test so a retry
    # is required and its feedback propagation remains covered.
    monkeypatch.setattr(config, "GENERATION_CANDIDATE_MULTIPLIER", 1)
    calls = _install_provider(monkeypatch, "Real Numbers", reject_first=True)
    paper = generator.generate_paper(_syllabus("Real Numbers"), {"practice-topic": {"chunks": ["Squaring multiplies a number by itself."]}})
    assert len(paper) == 10
    assert len(calls) == 4
    assert calls[1]["response_format"]["json_schema"]["schema"]["properties"]["questions"]["maxItems"] == 1
    assert "Correct these issues from the previous attempt" in calls[1]["messages"][-1]["content"]
    assert not any(question["question"] == "Evaluate 1^2?" for question in paper)


def test_topic_quiz_is_generated_stored_and_scored_using_displayed_values(monkeypatch, client, csrf_post, tmp_path):
    import rag.ingest as ingest
    import rag.retriever as retriever

    _install_provider(monkeypatch, "Sets")
    monkeypatch.setattr(application, "_has_llm_config", lambda: True)
    monkeypatch.setattr(application, "_find_topic_id_for_grade_topic", lambda *_args: "practice-topic")
    monkeypatch.setattr(application, "PAPER_FILE", tmp_path / "paper.json")
    monkeypatch.setattr(ingest, "ensure_vector_index_ready", lambda **_kwargs: {"ready": True})
    monkeypatch.setattr(retriever, "retrieve_for_syllabus", lambda *_args, **_kwargs: {"practice-topic": {"chunks": ["The union of two sets contains all the distinct elements of both sets."]}})
    with application.app.app_context():
        user = application.User(email="practice@example.com", name="Practice", password_hash="test-only", email_verified_at=application._utcnow())
        application.db.session.add(user)
        application.db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["session_version"] = 0

    response = csrf_post("/api/practice/quiz", {"grade": 9, "topic": "Sets"})
    assert response.status_code == 200, response.get_json()
    quiz = response.get_json()
    assert quiz["count"] == 10
    assert all(question["difficulty_level"] == "Apply" for question in quiz["paper"])
    assert all("correct_answer" not in question and "answer" not in question for question in quiz["paper"])
    answers = {}
    for index, question in enumerate(quiz["paper"]):
        sets = re.findall(r"\{(\d+), (\d+)\}", question["question"])
        union = sorted({int(value) for pair in sets for value in pair})
        expected = "{" + ", ".join(map(str, union)) + "}"
        answers[str(index)] = "ABCD"[question["options"].index(expected)]
    result = csrf_post("/api/submit", {"quiz_id": quiz["quiz_id"], "answers": answers})
    assert result.status_code == 200, result.get_json()
    assert result.get_json()["correct"] == 10
    assert max(Counter(answers.values()).values()) <= 3


def test_old_active_quiz_cannot_resume_an_obsolete_answer_contract(client):
    with application.app.app_context():
        user = application.User(email="legacy@example.com", name="Legacy", password_hash="test-only", email_verified_at=application._utcnow())
        application.db.session.add(user)
        application.db.session.commit()
        application.db.session.add(application.ActiveQuiz(user_id=user.id, quiz_id="legacy-quiz-identifier-123", quiz_kind="practice", status="active", paper_json=json.dumps([{"question": "Legacy question", "options": ["1", "2", "3", "4"], "answer": "B"}])))
        application.db.session.commit()
        _, paper = application._load_active_quiz(user)
        assert paper == []
