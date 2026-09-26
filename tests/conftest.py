import os
import sys
from pathlib import Path

import pytest


os.environ.update(
    {
        "APP_ENV": "development",
        "AUTO_INGEST_ON_STARTUP": "false",
        "CACHE_REDIS_URL": "memory://",
        "DATABASE_URL": "sqlite:///:memory:",
        "EMAIL_DELIVERY_MODE": "console",
        "EMAIL_CHECK_DELIVERABILITY": "false",
        "FLASK_SECRET_KEY": "test-secret-only-not-for-production",
        "RUNPOD_API_KEY": "",
        "RUNPOD_LLM_ENDPOINT_ID": "math-endpoint",
        "RUNPOD_MCQ_ENDPOINT_ID": "mcq-endpoint",
        "HF_API_KEY": "",
        "VECTOR_STORE_PROVIDER": "chroma",
        "OBSERVABILITY_ENABLED": "false",
        "QUESTION_BANK_REQUIRED": "false",
        "QUESTION_POOL_RUNTIME_ENABLED": "false",
        "RATELIMIT_ENABLED": "true",
        "RATELIMIT_STORAGE_URI": "memory://",
    }
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def clean_database():
    from rag.inference_errors import _blocked_until

    _blocked_until.clear()
    app_module.app.config.update(TESTING=True)
    app_module.limiter.reset()
    with app_module.app.app_context():
        app_module.db.drop_all()
        app_module.db.create_all()
    yield
    with app_module.app.app_context():
        app_module.db.session.remove()
        app_module.db.drop_all()


@pytest.fixture
def client():
    return app_module.app.test_client()


@pytest.fixture(autouse=True)
def isolated_question_bank(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.config, "QUESTION_BANK_DIR", tmp_path / "past_papers")


@pytest.fixture
def csrf_post(client):
    def post(path, payload):
        token_response = client.get("/api/csrf-token")
        token = token_response.get_json()["csrf_token"]
        return client.post(path, json=payload, headers={"X-CSRFToken": token})

    return post


@pytest.fixture
def stub_quality_review(monkeypatch):
    import rag.validation as validation

    def approve(question, **_kwargs):
        question["quality_review"] = {
            "version": validation.REVIEW_VERSION,
            "model_id": app_module.config.RUNPOD_MCQ_GENERATION_MODEL,
            "provider": "runpod",
            "content_sha256": validation._content_digest(question),
        }
        return []

    monkeypatch.setattr(validation, "review_mcq", approve)
    return approve
