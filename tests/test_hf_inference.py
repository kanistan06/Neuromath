import json

import pytest

import config
import rag.runpod_inference as inference


@pytest.fixture(autouse=True)
def _runpod_endpoints(monkeypatch):
    monkeypatch.setattr(config, "RUNPOD_LLM_ENDPOINT_ID", "test-endpoint")
    monkeypatch.setattr(config, "RUNPOD_MCQ_ENDPOINT_ID", "test-endpoint")

class Response:
    def __init__(self, content="", *, status=200, finish_reason="stop"):
        self.status_code = status
        self.content = content
        self.finish_reason = finish_reason

    def json(self):
        if self.status_code >= 400:
            return {"error": {"message": self.content}}
        return {
            "choices": [{
                "message": {"content": self.content},
                "finish_reason": self.finish_reason,
            }]
        }


def test_qwen_requests_current_http_router_contract(monkeypatch):
    captured = {}

    class Session:
        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response(json.dumps({"questions": []}))

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 1)
    response = inference.runpod_json_generation(
        "Textbook-grounded prompt", max_new_tokens=900, temperature=0.2
    )

    assert json.loads(response) == {"questions": []}
    assert captured["url"] == "https://api.runpod.ai/v2/test-endpoint/openai/v1/chat/completions"
    assert captured["json"]["model"] == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "Textbook-grounded prompt"}
    ]
    assert captured["json"]["max_tokens"] == 900
    assert captured["json"]["temperature"] == 0.2
    assert captured["json"]["stream"] is False
    assert captured["timeout"] == config.RUNPOD_TIMEOUT_SECONDS


def test_qwen_uses_configured_model(monkeypatch):
    assert inference._routed_model_id() == config.RUNPOD_LLM_MODEL


def test_http_session_authenticates_with_runpod_key(monkeypatch):
    monkeypatch.setattr(config, "RUNPOD_API_KEY", "rpa_test_only")
    inference._http_session.cache_clear()
    session = None
    try:
        session = inference._http_session()
        assert session.headers["Authorization"] == "Bearer rpa_test_only"
        assert "X-HF-Bill-To" not in session.headers
    finally:
        if session is not None:
            session.close()
        inference._http_session.cache_clear()


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_permanent_router_error_is_not_retried(monkeypatch, status):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            return Response("Provider rejected the request.", status=status)

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 6)
    monkeypatch.setattr(inference.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    message = "RunPod inference balance" if status == 402 else "Provider rejected"
    with pytest.raises(RuntimeError, match=message):
        inference.runpod_json_generation("prompt")
    assert len(calls) == 1


def test_transient_router_error_retries_within_configured_budget(monkeypatch):
    responses = iter([
        Response("Temporarily unavailable", status=503),
        Response('{"questions": []}'),
    ])
    calls = []
    sleeps = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            return next(responses)

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 2)
    monkeypatch.setattr(inference.time, "sleep", sleeps.append)
    assert json.loads(inference.runpod_json_generation("prompt")) == {"questions": []}
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_truncated_output_is_not_accepted_and_budget_expands(monkeypatch):
    budgets = []

    class Session:
        def post(self, *args, **kwargs):
            budgets.append(kwargs["json"]["max_tokens"])
            if len(budgets) == 1:
                return Response('{"questions": []}', finish_reason="length")
            return Response('{"questions": []}')

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_LLM_MAX_NEW_TOKENS", 900)
    assert json.loads(inference.runpod_json_generation("prompt")) == {"questions": []}
    assert budgets == [900, 1800]


def test_explicit_output_limit_is_respected(monkeypatch):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs["json"]["max_tokens"])
            return Response('{"questions": []}', finish_reason="length")

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    with pytest.raises(RuntimeError, match="output limit"):
        inference.runpod_json_generation("prompt", max_new_tokens=900)
    assert calls == [900]


def test_mcq_authoring_uses_non_thinking_structured_route(monkeypatch):
    captured = {}

    class Session:
        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response(json.dumps({
                "questions": [{
                    "question": "What is 2 + 2?",
                    "options": ["4", "5", "3", "6"],
                    "correct_answer": "4",
                    "explanation": "2 + 2 equals 4.",
                    "evidence_id": "E1",
                }]
            }))

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MCQ_GENERATION_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setattr(config, "RUNPOD_MCQ_MAX_NEW_TOKENS", 900)
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 1)

    response = inference.runpod_mcq_generation(
        "Create one grounded MCQ.",
        strict_grounding=True,
        question_count=1,
    )

    assert json.loads(response)["questions"][0]["evidence_id"] == "E1"
    assert captured["url"] == "https://api.runpod.ai/v2/test-endpoint/openai/v1/chat/completions"
    assert captured["json"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["json"]["messages"][-1] == {
        "role": "user",
        "content": "Create one grounded MCQ.",
    }
    assert captured["json"]["max_tokens"] == 900
    schema = captured["json"]["response_format"]["json_schema"]["schema"]
    item = schema["properties"]["questions"]["items"]
    assert "evidence_id" in item["properties"]
    assert "grounding_evidence" not in item["properties"]
    assert "grade" not in item["properties"]
    assert "source" not in item["properties"]
    assert "correct_answer" in item["required"]
    assert "answer" not in item["properties"]
    assert schema["properties"]["questions"]["maxItems"] == 1


def test_mcq_past_paper_schema_preserves_source_question_id(monkeypatch):
    captured = {}

    class Session:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            return Response(json.dumps({
                "questions": [{
                    "question": "A transformed question?",
                    "options": ["1", "2", "3", "4"],
                    "correct_answer": "2",
                    "explanation": "The verified value is 2.",
                    "source_question_id": "paper-01",
                }]
            }))

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 1)
    inference.runpod_mcq_generation(
        "prompt",
        strict_grounding=False,
        use_past_questions=True,
        question_count=1,
    )
    item = captured["json"]["response_format"]["json_schema"]["schema"]["properties"]["questions"]["items"]
    assert "source_question_id" in item["properties"]
    assert "source_question_id" in item["required"]


def test_structured_mcq_length_retries_once_at_2048(monkeypatch):
    budgets = []

    class Session:
        def post(self, *args, **kwargs):
            budgets.append(kwargs["json"]["max_tokens"])
            if len(budgets) == 1:
                return Response('{"questions":[', finish_reason="length")
            return Response('{"questions": []}', finish_reason="stop")

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MCQ_MAX_NEW_TOKENS", 900)
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 1)

    assert json.loads(
        inference.runpod_mcq_generation(
            "prompt",
            strict_grounding=True,
            question_count=1,
        )
    ) == {"questions": []}
    assert budgets == [900, 2048]


def test_explicit_structured_mcq_limit_never_expands(monkeypatch):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs["json"]["max_tokens"])
            return Response('{"questions":[', finish_reason="length")

    monkeypatch.setattr(inference, "_http_session", lambda: Session())
    monkeypatch.setattr(config, "RUNPOD_MAX_RETRIES", 1)
    with pytest.raises(RuntimeError, match="token limit"):
        inference.runpod_mcq_generation(
            "prompt",
            strict_grounding=True,
            question_count=1,
            max_new_tokens=900,
        )
    assert calls == [900]
