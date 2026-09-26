import copy
import json
import re
import threading

import pytest

import app as application
import config
import rag.generator as generator
import rag.runpod_inference as inference
import rag.inference_errors as errors
import rag.validation as validation
from rag.quality import is_too_similar


CONTEXT = (
    "A union contains every distinct element in either set. An intersection contains "
    "only shared elements. A set difference removes the elements of the second set "
    "from the first. Combine these operations to solve problems involving finite sets."
)


def _set_text(values):
    return "{" + ", ".join(map(str, sorted(values))) + "}"


def _row(number):
    a, b, c = number, number + 1, number + 2
    options = [_set_text({a, b, c}), _set_text({b}), _set_text({a, c}), _set_text({b, c})]
    return {
        "question": f"For A = {{{a}, {b}}}, B = {{{b}, {c}}} and C = {{{b}}}, find (A ∪ B) \\ C.",
        "options": options, "correct_answer": options[2],
        "explanation": f"Correct result: {options[2]}. Form the union, then remove every element in C.",
    }


def _solve(stem):
    sets = {name: {int(value.strip()) for value in elements.split(",")}
            for name, elements in re.findall(r"\b([ABC]) = \{([^}]*)\}", stem)}
    return _set_text((sets["A"] | sets["B"]) - sets["C"])


def _verdict(payload):
    value = _solve(payload["question"])
    assert value in payload["options"]
    assert "set difference" in payload["textbook_context"]
    assert not {"answer", "correct_answer", "explanation"}.intersection(payload)
    return {
        "answer_value": value, "valid_option_values": [value],
        "explanation": f"Form the union of A and B, then remove all elements of C to obtain {value}.",
        "difficulty_level": "Apply", "unambiguous": True, "concept_relevant": True,
        "textbook_supported": True, "distractors_plausible": True,
        "application_required": True, "reason": "",
    }


class _Provider:
    def __init__(self, monkeypatch, fail_review=None):
        self.fail_review = fail_review
        self.review_calls = 0
        self.author_counts = []
        self.calls = []
        self.next_number = 1
        monkeypatch.setattr(inference, "_http_session", lambda: self)
        monkeypatch.setattr(config, "GENERATION_RETRY_BASE_SECONDS", 0)

    def post(self, _url, **kwargs):
        request = kwargs["json"]
        self.calls.append(request)
        schema = request["response_format"]["json_schema"]
        status = 200
        if schema["name"] == "neuromath_mcq_review":
            self.review_calls += 1
            if self.review_calls == self.fail_review:
                status = 402
                result = {"error": {"message": "You have depleted your monthly included credits."}}
            else:
                payload = json.loads(request["messages"][-1]["content"].split("\n", 1)[1])
                result = _verdict(payload)
        else:
            count = schema["schema"]["properties"]["questions"]["maxItems"]
            self.author_counts.append(count)
            result = {"questions": [_row(self.next_number + i * 3) for i in range(count)]}
            if "evidence_id" in schema["schema"]["properties"]["questions"]["items"]["properties"]:
                for row in result["questions"]:
                    row["evidence_id"] = "E1"
            self.next_number += count * 3

        class Response:
            status_code = status

            def json(self):
                if status >= 400:
                    return result
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}

        return Response()


def _syllabus(count=10):
    return {
        "total_questions": count,
        "topic_mappings": [{"topic_id": "sets-test", "topic_name": "Sets", "grade": 7,
                            "required_levels": [3], "require_application": True,
                            "min_questions": count, "max_questions": count}],
    }


def _content():
    return {"sets-test": {"chunks": [CONTEXT]}}


def _login(client, email="recovery@example.com"):
    with application.app.app_context():
        user = application.User(email=email, name="Recovery", password_hash="test-only",
                                email_verified_at=application._utcnow())
        application.db.session.add(user)
        application.db.session.commit()
        user_id = user.id
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["session_version"] = 0
    return user_id


def _install_app(monkeypatch, tmp_path):
    import rag.ingest as ingest
    import rag.retriever as retriever

    retrievals = []

    def retrieve(*args, **kwargs):
        retrievals.append((args, kwargs))
        return _content()

    monkeypatch.setattr(application, "_has_llm_config", lambda: True)
    monkeypatch.setattr(application, "_find_topic_id_for_grade_topic", lambda *_args: "sets-test")
    monkeypatch.setattr(application, "PAPER_FILE", tmp_path / "paper.json")
    monkeypatch.setattr(ingest, "ensure_vector_index_ready", lambda **_kwargs: {"ready": True})
    monkeypatch.setattr(retriever, "retrieve_for_syllabus", retrieve)
    return retrievals


def test_credit_failure_mid_review_saves_private_item_and_resumes_only_missing_questions(
    monkeypatch, client, csrf_post, tmp_path,
):
    retrievals = _install_app(monkeypatch, tmp_path)
    provider = _Provider(monkeypatch, fail_review=2)
    user_id = _login(client)
    clock = [errors.time.time()]
    monkeypatch.setattr(errors.time, "time", lambda: clock[0])

    response = csrf_post("/api/practice/quiz", {"grade": 7, "topic": "Sets"})
    assert response.status_code == 503
    assert response.get_json()["code"] == "inference_credits_exhausted"
    assert response.headers["Retry-After"] == "60"
    assert "billing" in response.get_json()["message"].lower()
    assert len(provider.calls) == 3
    with application.app.app_context():
        application.db.session.remove()
        draft = application.GenerationDraft.query.filter_by(user_id=user_id).one()
        saved = json.loads(draft.state_json)["paper"]
        assert len(saved) == 1 and validation.has_current_review(saved[0])
        assert application.ActiveQuiz.query.count() == 0

    response = csrf_post("/api/practice/quiz", {"grade": 7, "topic": "Sets"})
    assert response.status_code == 503
    assert len(provider.calls) == 3 and len(retrievals) == 1

    clock[0] += 61
    response = csrf_post("/api/practice/quiz", {"grade": 7, "topic": "Sets"})
    assert response.status_code == 200, response.get_json()
    quiz = response.get_json()
    assert quiz["count"] == 10
    assert provider.author_counts == [6, 6, 6, 2]
    assert len(retrievals) == 1
    assert sum(q["question"] == saved[0]["question"] for q in quiz["paper"]) == 1
    assert all(not {"answer", "correct_answer", "explanation", "quality_review"}.intersection(q)
               for q in quiz["paper"])
    answers = {str(i): "ABCD"[q["options"].index(_solve(q["question"]))]
               for i, q in enumerate(quiz["paper"])}
    with application.app.app_context():
        assert application.GenerationDraft.query.count() == 0
    response = csrf_post("/api/submit", {"quiz_id": quiz["quiz_id"], "answers": answers})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["correct"] == 10


def test_eight_validated_questions_resume_with_two_new_questions(monkeypatch):
    provider = _Provider(monkeypatch)
    saved = generator.generate_paper(_syllabus(8), _content())
    provider.calls.clear()
    provider.author_counts.clear()
    result = generator.generate_paper(_syllabus(10), _content(), initial_paper=saved)
    assert len(result) == 10
    assert provider.author_counts == [4]
    assert len(provider.calls) == 3
    assert result[:8] == saved


def test_diagnostic_draft_resumes_missing_concept_with_original_grade_and_citation(monkeypatch, client):
    import rag.retriever as retriever
    from rag.diagnostic import validate_diagnostic_paper

    provider = _Provider(monkeypatch, fail_review=2)
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 1)
    user_id = _login(client)
    content = {}
    syllabus = {"assessment_type": "diagnostic", "total_questions": 2, "topic_mappings": []}
    for grade in [10, 11]:
        mapping = dict(_syllabus(1)["topic_mappings"][0], topic_id=f"g{grade}_sets",
                       grade=grade, strict_textbook_grounding=True)
        syllabus["topic_mappings"].append(mapping)
        reference = {"source": f"grade-{grade}.pdf", "page": 8, "grade": grade,
                     "chunk_id": f"g{grade}-sets-8", "corpus_version": "test-corpus-v1"}
        content[mapping["topic_id"]] = {"chunks": [CONTEXT], "source_refs": [reference],
                                       "grounding_units": [{"text": CONTEXT, "source_ref": reference}]}
    retrievals = []
    monkeypatch.setattr(retriever, "retrieve_for_syllabus",
                        lambda *_args, **_kwargs: retrievals.append(1) or content)
    clock = [errors.time.time()]
    monkeypatch.setattr(errors.time, "time", lambda: clock[0])
    with application.app.app_context():
        user = application.db.session.get(application.User, user_id)
        with pytest.raises(RuntimeError) as caught:
            application._generate_with_draft(user, "diagnostic", "diagnostic", syllabus, {})
        assert isinstance(errors.access_failure(caught.value), errors.InferenceCreditsError)
        application.db.session.remove()
        draft = application.GenerationDraft.query.filter_by(user_id=user_id).one()
        saved = json.loads(draft.state_json)["paper"]
        assert len(saved) == 1 and saved[0]["grade"] == 10
        clock[0] += 61
        user = application.db.session.get(application.User, user_id)
        paper, _draft_id = application._generate_with_draft(user, "diagnostic", "diagnostic", syllabus, {})
        validate_diagnostic_paper(paper, syllabus)
        assert len(paper) == 2 and provider.author_counts == [2, 2, 2]
        assert paper[0] == saved[0]
        assert len(retrievals) == 1
        assert {q["grade"] for q in paper} == {10, 11}
        assert all(q["source_refs"][0]["grade"] == q["grade"] for q in paper)


def test_invalid_or_obsolete_saved_questions_are_replaced(monkeypatch):
    provider = _Provider(monkeypatch)
    saved = generator.generate_paper(_syllabus(2), _content())
    tampered, obsolete = copy.deepcopy(saved)
    tampered["correct_answer"] = tampered["options"][0]
    obsolete["quality_review"]["version"] = "independent-mcq-v1"
    provider.author_counts.clear()
    result = generator.generate_paper(_syllabus(2), _content(), initial_paper=[None, tampered, obsolete])
    assert len(result) == 2 and provider.author_counts == [4]
    assert all(validation.has_current_review(q) for q in result)


def test_account_cooldown_blocks_other_student_before_retrieval(monkeypatch, client, csrf_post, tmp_path):
    retrievals = _install_app(monkeypatch, tmp_path)
    provider = _Provider(monkeypatch, fail_review=1)
    first = _login(client)
    assert csrf_post("/api/practice/quiz", {"grade": 7, "topic": "Sets"}).status_code == 503
    second = _login(client, "second@example.com")
    assert csrf_post("/api/practice/quiz", {"grade": 7, "topic": "Sets"}).status_code == 503
    assert len(retrievals) == 1 and len(provider.calls) == 2
    with application.app.app_context():
        assert application.GenerationDraft.query.filter_by(user_id=first).count() == 1
        assert application.GenerationDraft.query.filter_by(user_id=second).count() == 0


def test_inference_call_budget_is_shared_by_parallel_concepts(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 3)
    monkeypatch.setattr(config, "GENERATION_INFERENCE_CALLS_PER_QUESTION", 4)
    monkeypatch.setattr(config, "GENERATION_INFERENCE_CALL_BUDGET_MAX", 12)
    calls = []
    lock = threading.Lock()

    def generate(_tasks):
        for _ in range(5):
            errors.consume_inference_call()
            with lock:
                calls.append(1)
        return []

    monkeypatch.setattr(generator, "_generate_topic_tasks", generate)
    syllabus = {"total_questions": 3, "topic_mappings": [
        {"topic_id": str(i), "topic_name": "Sets", "required_levels": [3], "min_questions": 1}
        for i in range(3)
    ]}
    with pytest.raises(errors.GenerationBudgetExceeded):
        generator.generate_paper(syllabus, {str(i): {"chunks": [CONTEXT]} for i in range(3)})
    assert len(calls) == 12


def test_25_question_diagnostic_gets_full_bounded_call_budget(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_INFERENCE_CALLS_PER_QUESTION", 12)
    monkeypatch.setattr(config, "GENERATION_INFERENCE_CALL_BUDGET_MAX", 300)
    assert generator._generation_inference_call_budget({"total_questions": 25}) == 300


def test_generation_and_review_see_the_same_complete_source_window(monkeypatch):
    contexts = ["First passage. " * 130, "Later set difference passage. " * 70]
    seen = []
    monkeypatch.setattr(generator, "_invoke_llm", lambda prompt, **_kwargs:
                        seen.append(prompt) or json.dumps({"questions": [_row(3)]}))

    def review(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        seen.append(payload["textbook_context"])
        return _verdict(payload)

    monkeypatch.setattr(validation, "runpod_mcq_review", review)
    result = generator.generate_mcqs("Sets", "Apply", contexts, 1, grade=7,
                                     concept_id="sets-test", grounding_attempt=1, require_application=True)
    assert len(result) == 1
    assert seen[1] == contexts[1].strip()
    assert seen[1] in seen[0] and len(seen[1]) <= 3600
    assert contexts[0].strip() not in seen[0]


def test_source_window_accepts_a_complete_passage_exactly_at_the_limit():
    passage = "a" * generator._MCQ_CONTEXT_CHAR_CAP
    assert generator._source_window([passage], 0) == passage


def test_correct_key_with_discarded_author_self_correction_requires_clean_independent_review(monkeypatch):
    row = _row(3)
    row["explanation"] += " Wait, let me reconsider."
    monkeypatch.setattr(generator, "_invoke_llm", lambda *_args, **_kwargs: json.dumps({"questions": [row]}))
    calls = []

    def review(prompt, **_kwargs):
        calls.append(prompt)
        return _verdict(json.loads(prompt.split("\n", 1)[1]))

    monkeypatch.setattr(validation, "runpod_mcq_review", review)
    result = generator.generate_mcqs("Sets", "Apply", [CONTEXT], 1, grade=7, require_application=True)
    assert len(result) == 1 and len(calls) == 1
    assert "reconsider" not in result[0]["explanation"]
    assert validation.has_current_review(result[0])

    row["correct_answer"] = row["options"][0]
    assert generator.generate_mcqs("Sets", "Apply", [CONTEXT], 1, grade=7, require_application=True) == []
    assert len(calls) == 1


def test_set_membership_changes_are_distinct_but_reordering_is_a_repeat():
    first = "For A = {1, 2}, B = {2, 3} and C = {3, 4}, find (A ∪ B) \\ C."
    redistributed = "For A = {1, 3}, B = {2, 3} and C = {2, 4}, find (A ∪ B) \\ C."
    reordered = "For A = {2, 1}, B = {3, 2} and C = {4, 3}, find (A ∪ B) \\ C."
    renamed = "For B = {1, 2}, A = {2, 3} and C = {3, 4}, find (A ∪ B) \\ C."
    assert not is_too_similar(redistributed, [first], threshold=0.84)
    assert is_too_similar(reordered, [first], threshold=0.84)
    assert is_too_similar(renamed, [first], threshold=0.84)


@pytest.mark.parametrize("status", [402, 403])
def test_credit_failure_is_actionable_sanitized_and_billing_account_scoped(monkeypatch, status):
    monkeypatch.setattr(config, "RUNPOD_API_KEY", "rpa_private_test_token")
    with pytest.raises(errors.InferenceCreditsError) as caught:
        errors.record_credit_failure(status, "Your credits are exhausted. rpa_private_test_token")
    assert "rpa_private_test_token" not in str(caught.value)
    with pytest.raises(errors.InferenceCreditsError):
        errors.ensure_inference_available()
    monkeypatch.setattr(config, "RUNPOD_API_KEY", "rpa_different_test_key")
    errors.ensure_inference_available()
