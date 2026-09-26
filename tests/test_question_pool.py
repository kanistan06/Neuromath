import json

import app as application
import rag.validation as validation
from rag.question_pool import add_pool_question, select_unseen_questions


def _reviewed_question(number: int, *, topic_id: str = "g7_t1_01_sets"):
    answer_value = str(number + 10)
    question = {
        "question": f"A learner combines two finite sets in example {number}. What is the required result?",
        "options": [str(number + 7), str(number + 8), answer_value, str(number + 11)],
        "answer": "C",
        "correct_answer": answer_value,
        "explanation": f"Correct result: {answer_value}. The stated set operation gives {answer_value}.",
        "topic_id": topic_id,
        "difficulty_level": "Apply",
        "grade": 7,
        "term": 1,
        "grounding_evidence": "This is a sufficiently long verified textbook evidence excerpt for the tested set operation and method.",
        "source_refs": [{
            "source": "grade7.pdf", "page": 1, "page_start": 1, "page_end": 1,
            "grade": 7, "chunk_id": f"chunk-{number}", "corpus_version": "test-v1",
        }],
    }
    question["quality_review"] = {
        "version": validation.REVIEW_VERSION,
        "model_id": application.config.RUNPOD_MCQ_GENERATION_MODEL,
        "provider": "runpod",
        "content_sha256": validation._content_digest(question),
    }
    return question


def test_question_pool_does_not_repeat_questions_for_same_user():
    with application.app.app_context():
        user = application.User(
            email="pool@example.com",
            name="Pool User",
            password_hash="test-only",
            email_verified_at=application._utcnow(),
        )
        application.db.session.add(user)
        application.db.session.commit()

        for number in range(20):
            assert add_pool_question(
                _reviewed_question(number),
                topic_name="Sets",
                source_type="llm_generated",
            )

        first = select_unseen_questions(
            user_id=user.id,
            topic_id="g7_t1_01_sets",
            grade=7,
            difficulty_level="Apply",
            count=10,
            quiz_kind="practice",
        )
        assert len(first) == 10
        active, _paper = application._store_active_quiz(user, first, "practice", time_limit_minutes=15)
        first_ids = {q["_pool_question_id"] for q in first}
        assert application.UserQuestionExposure.query.filter_by(user_id=user.id).count() == 10

        second = select_unseen_questions(
            user_id=user.id,
            topic_id="g7_t1_01_sets",
            grade=7,
            difficulty_level="Apply",
            count=10,
            quiz_kind="practice",
        )
        second_ids = {q["_pool_question_id"] for q in second}
        assert len(second) == 10
        assert first_ids.isdisjoint(second_ids)
        assert active.quiz_id


def test_pool_question_metadata_survives_private_quiz_storage():
    with application.app.app_context():
        user = application.User(
            email="pool-meta@example.com",
            name="Pool Meta",
            password_hash="test-only",
            email_verified_at=application._utcnow(),
        )
        application.db.session.add(user)
        application.db.session.commit()
        add_pool_question(_reviewed_question(99), topic_name="Sets", source_type="llm_generated")
        selected = select_unseen_questions(
            user_id=user.id,
            topic_id="g7_t1_01_sets",
            grade=7,
            difficulty_level="Apply",
            count=1,
            quiz_kind="practice",
        )
        active, private_paper = application._store_active_quiz(user, selected, "practice", time_limit_minutes=15)
        stored = json.loads(active.paper_json)
        assert stored[0]["_pool_question_id"] == private_paper[0]["_pool_question_id"]
        assert "_pool_question_id" not in application._public_paper(private_paper)[0]
