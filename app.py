"""Flask web application for the RAG MCQ Assessment system."""

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from email_validator import EmailNotValidError, validate_email
from flask import Flask, g, has_request_context, jsonify, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFError, CSRFProtect, generate_csrf
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

import config
from rag.answers import (
    ANSWER_CONTRACT_VERSION,
    answer_explanation_consistency_errors,
    answer_letter_from_value,
    remap_explanation_options,
)
from auth_security import (
    hash_otp,
    hash_token,
    new_numeric_otp,
    new_one_time_token,
    rate_limit_identity,
    validate_password,
)
from infrastructure.cache import (
    cache_key,
    delete as cache_delete,
    generation_lock,
    get_json as cache_get_json,
    ping as cache_ping,
    set_json as cache_set_json,
)
from infrastructure.observability import configure_observability, pseudonymous_ref
from mailer import (
    EmailDeliveryError,
    send_password_changed_notice,
    send_password_reset_otp,
    send_verification_email,
)
from rag.image_gen import is_safe_quiz_image_url
from rag.inference_errors import access_failure, ensure_inference_available
from rag.quality import question_fingerprint


app = Flask(__name__)

_is_production = config.IS_PRODUCTION

_secret = config.FLASK_SECRET_KEY
if _is_production and (
    not _secret
    or _secret == "dev-secret-change-me"
    or len(_secret) < 32
):
    raise RuntimeError(
        "FLASK_SECRET_KEY must be set to a strong random value of at least 32 characters in production"
    )
if _is_production:
    production_errors: list[str] = []
    database = urlparse(config.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://"))
    if database.scheme not in {"postgresql", "postgres"} or not str(database.hostname or "").endswith(".neon.tech"):
        production_errors.append("DATABASE_URL must use Neon PostgreSQL")
    redis_url = urlparse(config.RATELIMIT_STORAGE_URI)
    if redis_url.scheme != "rediss" or not str(redis_url.hostname or "").endswith(".upstash.io"):
        production_errors.append("RATELIMIT_STORAGE_URI must use Upstash Redis over TLS")
    if config.CACHE_REDIS_URL != config.RATELIMIT_STORAGE_URI:
        production_errors.append("CACHE_REDIS_URL must use the approved Upstash Redis instance")
    email_configured = all(
        (
            config.EMAIL_DELIVERY_MODE == "smtp",
            config.SMTP_HOST.lower() == "smtp.resend.com",
            config.SMTP_PORT == 587,
            config.SMTP_USERNAME == "resend",
            bool(config.SMTP_PASSWORD),
            config.SMTP_USE_TLS,
            not config.SMTP_USE_SSL,
            config.SMTP_FROM_EMAIL.lower() == "learn@neuromath.io",
        )
    )
    if not email_configured:
        production_errors.append("transactional email must use Resend SMTP from learn@neuromath.io")
    if not config.PUBLIC_BASE_URL.startswith("https://"):
        production_errors.append("PUBLIC_BASE_URL must use HTTPS")
    qdrant = urlparse(config.QDRANT_URL)
    if config.VECTOR_STORE_PROVIDER != "qdrant" or not str(qdrant.hostname or "").endswith(".cloud.qdrant.io"):
        production_errors.append("the vector store must use Qdrant Cloud")
    if not config.QDRANT_API_KEY:
        production_errors.append("QDRANT_API_KEY is required")
    if config.QDRANT_WRITE_API_KEY:
        production_errors.append("QDRANT_WRITE_API_KEY must not be present in the web service")
    if not config.RUNPOD_API_KEY:
        production_errors.append("RUNPOD_API_KEY is required for RunPod inference")
    if not config.RUNPOD_MCQ_ENDPOINT_ID:
        production_errors.append("RUNPOD_MCQ_ENDPOINT_ID is required for MCQ generation")
    if config.RUNPOD_MCQ_GENERATION_MODEL != config.SELECTED_MCQ_GENERATION_MODEL:
        production_errors.append(
            "RUNPOD_MCQ_GENERATION_MODEL must match the approved MCQ generation model"
        )
    if not config.HF_API_KEY:
        production_errors.append("HF_API_KEY is required for Hugging Face embeddings")
    if (
        config.HF_EMBEDDING_MODEL != config.SELECTED_EMBEDDING_MODEL
        or config.HF_EMBEDDING_DIMENSIONS != config.SELECTED_EMBEDDING_DIMENSIONS
    ):
        production_errors.append("embeddings must use 1024-dimensional BAAI/bge-m3")
    if config.CHUNK_TOKENIZER_MODEL != config.SELECTED_LLM_MODEL:
        production_errors.append("chunking must use the selected Qwen tokenizer")
    if not (
        300 <= config.CHUNK_TARGET_TOKENS <= 450
        and 300 <= config.CHUNK_MIN_TOKENS <= config.CHUNK_TARGET_TOKENS
        and config.CHUNK_TARGET_TOKENS <= config.CHUNK_MAX_TOKENS <= 450
        and 50 <= config.CHUNK_OVERLAP_TOKENS <= 75
    ):
        production_errors.append("chunking must use the approved 300-450 token range and 50-75 overlap")
    if config.DIAGNOSTIC_GRADES != (10, 11):
        production_errors.append("diagnostic generation must cover Grades 10 and 11")
    if not config.QUESTION_BANK_REQUIRED or not config.QUESTION_BANK_REQUIRE_REVIEW:
        production_errors.append(
            "the structured past-paper bank and verified mapping reviews are required"
        )
    if config.IMAGE_GEN_ENABLED:
        production_errors.append("image generation must remain disabled for the selected stack")
    if not config.CORPUS_VERSION:
        production_errors.append("CORPUS_VERSION must match the completed Qdrant ingestion")
    if not config.OBSERVABILITY_ENABLED or not config.BETTER_STACK_SOURCE_TOKEN:
        production_errors.append("Better Stack observability is required")
    if production_errors:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(production_errors))
app.secret_key = _secret or "dev-secret-change-me"
app.config.update(
    SESSION_COOKIE_NAME="neuromath_session",
    SESSION_COOKIE_SECURE=bool(_is_production),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=config.SESSION_LIFETIME_HOURS),
    # Flask-WTF forwards this value to itsdangerous as integer seconds.
    WTF_CSRF_TIME_LIMIT=config.SESSION_LIFETIME_HOURS * 60 * 60,
    MAX_CONTENT_LENGTH=config.MAX_REQUEST_BYTES,
)

csrf = CSRFProtect(app)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per hour"],
    storage_uri=config.RATELIMIT_STORAGE_URI,
    enabled=config.RATELIMIT_ENABLED,
)

from werkzeug.middleware.proxy_fix import ProxyFix
if config.TRUST_PROXY_HEADERS:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
configure_observability(app)


def _normalized_database_url() -> str:
    raw = config.DATABASE_URL
    if raw.startswith("postgres://"):
        return "postgresql://" + raw[len("postgres://") :]
    if raw:
        return raw

    return f"sqlite:///{(config.PROJECT_ROOT / 'app.db').as_posix()}"


app.config["SQLALCHEMY_DATABASE_URI"] = _normalized_database_url()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgresql"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_use_lifo": True,
        "pool_size": config.DATABASE_POOL_SIZE,
        "max_overflow": config.DATABASE_MAX_OVERFLOW,
        "pool_recycle": config.DATABASE_POOL_RECYCLE_SECONDS,
        "connect_args": {
            "connect_timeout": config.DATABASE_CONNECT_TIMEOUT_SECONDS,
            "sslmode": "require",
        },
    }

db = SQLAlchemy(app)


def _utcnow() -> datetime:
    """Naive UTC for database compatibility without deprecated utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _quiz_log(
    event: str,
    *,
    level: str = "info",
    user_id: int | None = None,
    quiz_id: str | None = None,
    quiz_kind: str | None = None,
    topic_id: str | None = None,
    grade: int | None = None,
    batch_offset: int | None = None,
    question_count: int | None = None,
    outcome: str | None = None,
    phase: str | None = None,
) -> None:
    """Emit correlation-friendly quiz logs without question/answer or raw user data."""
    extra: dict[str, Any] = {"event": str(event)}
    if has_request_context():
        request_id = getattr(g, "request_id", None)
        if request_id:
            extra["request_id"] = request_id
    if user_id is not None:
        extra["user_ref"] = pseudonymous_ref("user", int(user_id))
    if quiz_id:
        extra["quiz_ref"] = pseudonymous_ref("quiz", str(quiz_id))
    if quiz_kind:
        extra["quiz_kind"] = str(quiz_kind)
    if topic_id:
        extra["topic_id"] = str(topic_id)
    if grade is not None:
        extra["grade"] = int(grade)
    if batch_offset is not None:
        extra["batch_offset"] = int(batch_offset)
    if question_count is not None:
        extra["question_count"] = int(question_count)
    if outcome:
        extra["outcome"] = str(outcome)
    if phase:
        extra["phase"] = str(phase)
    log_method = getattr(app.logger, level, app.logger.info)
    log_method(str(event), extra=extra)



class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    joined_at = db.Column(db.DateTime, nullable=False, default=_utcnow)
    email_verified_at = db.Column(db.DateTime, nullable=True)
    password_changed_at = db.Column(db.DateTime, nullable=True)
    session_version = db.Column(db.Integer, nullable=False, default=0)


class AuthToken(db.Model):
    __tablename__ = "auth_tokens"
    __table_args__ = (
        db.CheckConstraint(
            "purpose IN ('verify_email', 'reset_password')",
            name="ck_auth_tokens_purpose",
        ),
        db.Index("ix_auth_tokens_user_purpose", "user_id", "purpose"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    purpose = db.Column(db.String(32), nullable=False)
    token_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    used_at = db.Column(db.DateTime, nullable=True)
    failed_attempts = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


class UserSettings(db.Model):
    __tablename__ = "user_settings"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    theme = db.Column(db.String(16), nullable=False, default="dark")
    difficulty = db.Column(db.String(16), nullable=False, default="medium")
    quiz_timer_enabled = db.Column(db.Boolean, nullable=False, default=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


class Attempt(db.Model):
    __tablename__ = "attempts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    quiz_id = db.Column(db.String(64), nullable=True, unique=True, index=True)
    quiz_kind = db.Column(db.String(32), nullable=False, default="diagnostic")
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)
    total_questions = db.Column(db.Integer, nullable=False)
    correct = db.Column(db.Integer, nullable=False)
    incorrect = db.Column(db.Integer, nullable=False)
    score_percent = db.Column(db.Float, nullable=False)


class Mistake(db.Model):
    __tablename__ = "mistakes"

    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("attempts.id"), nullable=False, index=True)
    question_index = db.Column(db.Integer, nullable=False)
    question = db.Column(db.Text, nullable=False, default="")
    topic_id = db.Column(db.String(255), nullable=False, default="unknown")
    difficulty_level = db.Column(db.String(64), nullable=False, default="unknown")
    correct_answer = db.Column(db.String(16), nullable=False, default="")
    student_answer = db.Column(db.String(16), nullable=True)


class AttemptQuestion(db.Model):
    __tablename__ = "attempt_questions"

    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("attempts.id"), nullable=False, index=True)
    question_index = db.Column(db.Integer, nullable=False)
    question = db.Column(db.Text, nullable=False, default="")
    options_json = db.Column(db.Text, nullable=False, default="[]")
    correct_answer = db.Column(db.String(16), nullable=False, default="")
    student_answer = db.Column(db.String(16), nullable=True)
    correct_option_text = db.Column(db.Text, nullable=True)
    student_option_text = db.Column(db.Text, nullable=True)
    explanation = db.Column(db.Text, nullable=False, default="")
    is_correct = db.Column(db.Boolean, nullable=False, default=False)
    image_url = db.Column(db.String(512), nullable=True)
    topic_id = db.Column(db.String(255), nullable=False, default="unknown", index=True)
    source_question_id = db.Column(db.String(128), nullable=True, index=True)
    question_hash = db.Column(db.String(64), nullable=False, default="", index=True)
    difficulty_level = db.Column(db.String(64), nullable=False, default="unknown")
    pool_question_id = db.Column(db.Integer, db.ForeignKey("question_pool_items.id"), nullable=True, index=True)


class QuestionPoolItem(db.Model):
    """A globally reusable, pre-generated and independently reviewed MCQ."""

    __tablename__ = "question_pool_items"
    __table_args__ = (
        db.UniqueConstraint("question_hash", name="uq_question_pool_question_hash"),
        db.Index("ix_question_pool_lookup", "grade", "topic_id", "difficulty_level", "is_active"),
    )

    id = db.Column(db.Integer, primary_key=True)
    question_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    source_type = db.Column(db.String(32), nullable=False, index=True)
    source_question_id = db.Column(db.String(128), nullable=True, index=True)
    topic_id = db.Column(db.String(255), nullable=False, index=True)
    topic_name = db.Column(db.String(255), nullable=False)
    grade = db.Column(db.Integer, nullable=False, index=True)
    difficulty_level = db.Column(db.String(64), nullable=False, index=True)
    eligible_diagnostic = db.Column(db.Boolean, nullable=False, default=True, index=True)
    eligible_practice = db.Column(db.Boolean, nullable=False, default=True, index=True)
    question_json = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)


class UserQuestionExposure(db.Model):
    """Tracks pool questions already shown to a student so they are never repeated."""

    __tablename__ = "user_question_exposures"
    __table_args__ = (
        db.UniqueConstraint("user_id", "pool_question_id", name="uq_user_pool_question_exposure"),
        db.Index("ix_user_question_exposures_quiz", "user_id", "quiz_id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    pool_question_id = db.Column(db.Integer, db.ForeignKey("question_pool_items.id"), nullable=False, index=True)
    quiz_id = db.Column(db.String(64), nullable=False, index=True)
    quiz_kind = db.Column(db.String(32), nullable=False, index=True)
    served_at = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)
    answered_at = db.Column(db.DateTime, nullable=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("attempts.id"), nullable=True, index=True)


class ActiveQuiz(db.Model):
    """Private server-side answer key for the one quiz currently open per user."""

    __tablename__ = "active_quizzes"

    __table_args__ = (
        db.CheckConstraint(
            "status IN ('active', 'submitting')",
            name="ck_active_quizzes_status",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True, index=True)
    quiz_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    quiz_kind = db.Column(db.String(32), nullable=False, default="diagnostic")
    status = db.Column(db.String(16), nullable=False, default="active", index=True)
    paper_json = db.Column(db.Text, nullable=False)
    timer_enabled = db.Column(db.Boolean, nullable=False, default=False)
    timer_duration_seconds = db.Column(db.Integer, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    target_question_count = db.Column(db.Integer, nullable=True)
    batch_state_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


class AssessmentTemplate(db.Model):
    """A validated diagnostic version reused through private per-user copies."""

    __tablename__ = "assessment_templates"

    id = db.Column(db.Integer, primary_key=True)
    fingerprint = db.Column(db.String(64), nullable=False, unique=True, index=True)
    paper_json = db.Column(db.Text, nullable=False)
    model_id = db.Column(db.String(255), nullable=False)
    generation_provider = db.Column(db.String(64), nullable=False, default="unknown")
    embedding_model = db.Column(db.String(255), nullable=False, default=config.SELECTED_EMBEDDING_MODEL)
    embedding_provider = db.Column(db.String(64), nullable=False, default="unknown")
    question_bank_version = db.Column(db.String(64), nullable=False, default="none")
    corpus_version = db.Column(db.String(128), nullable=False, default="unknown")
    prompt_version = db.Column(db.String(32), nullable=False, default="unknown")
    validator_version = db.Column(db.String(32), nullable=False, default="unknown")
    chunker_version = db.Column(db.String(64), nullable=False, default=config.CHUNKER_VERSION)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)


class GenerationDraft(db.Model):
    __tablename__ = "generation_drafts"
    __table_args__ = (db.UniqueConstraint("user_id", "quiz_kind", "scope", name="uq_generation_draft_scope"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    quiz_kind = db.Column(db.String(32), nullable=False)
    scope = db.Column(db.String(255), nullable=False)
    fingerprint = db.Column(db.String(64), nullable=False)
    state_json = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False, default=_utcnow)


def _ensure_attempt_question_columns() -> None:
    try:
        from sqlalchemy import inspect, text

        insp = inspect(db.engine)
        if "attempt_questions" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("attempt_questions")}
        definitions = {
            "image_url": "VARCHAR(512)",
            "topic_id": "VARCHAR(255) NOT NULL DEFAULT 'unknown'",
            "source_question_id": "VARCHAR(128)",
            "question_hash": "VARCHAR(64) NOT NULL DEFAULT ''",
            "difficulty_level": "VARCHAR(64) NOT NULL DEFAULT 'unknown'",
            "pool_question_id": "INTEGER",
        }
        with db.engine.begin() as conn:
            for name, definition in definitions.items():
                if name not in cols:
                    conn.execute(
                        text(
                            f"ALTER TABLE attempt_questions ADD COLUMN {name} {definition}"
                        )
                    )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_attempt_questions_topic_id "
                    "ON attempt_questions (topic_id)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_attempt_questions_source_question_id "
                    "ON attempt_questions (source_question_id)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_attempt_questions_question_hash "
                    "ON attempt_questions (question_hash)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_attempt_questions_pool_question_id "
                    "ON attempt_questions (pool_question_id)"
                )
            )
    except Exception as exc:
        print(f"Note: could not migrate attempt_questions: {exc}")


def _ensure_active_quiz_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "active_quizzes" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("active_quizzes")}
    with db.engine.begin() as connection:
        if "quiz_id" not in columns:
            connection.execute(
                text("ALTER TABLE active_quizzes ADD COLUMN quiz_id VARCHAR(64)")
            )
        if "status" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE active_quizzes ADD COLUMN status "
                    "VARCHAR(16) NOT NULL DEFAULT 'active'"
                )
            )
        if "timer_enabled" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE active_quizzes ADD COLUMN timer_enabled "
                    "BOOLEAN NOT NULL DEFAULT FALSE"
                )
            )
        if "timer_duration_seconds" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE active_quizzes ADD COLUMN timer_duration_seconds INTEGER"
                )
            )
        if "expires_at" not in columns:
            connection.execute(
                text("ALTER TABLE active_quizzes ADD COLUMN expires_at TIMESTAMP")
            )
        if "target_question_count" not in columns:
            connection.execute(
                text("ALTER TABLE active_quizzes ADD COLUMN target_question_count INTEGER")
            )
        if "batch_state_json" not in columns:
            connection.execute(
                text("ALTER TABLE active_quizzes ADD COLUMN batch_state_json TEXT")
            )
        rows = connection.execute(
            text("SELECT id FROM active_quizzes WHERE quiz_id IS NULL OR quiz_id = ''")
        ).fetchall()
        for row in rows:
            connection.execute(
                text("UPDATE active_quizzes SET quiz_id = :quiz_id WHERE id = :id"),
                {"quiz_id": secrets.token_urlsafe(24), "id": row[0]},
            )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ix_active_quizzes_quiz_id ON active_quizzes (quiz_id)"
            )
        )


def _ensure_user_settings_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "user_settings" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("user_settings")}
    if "quiz_timer_enabled" in columns:
        return
    with db.engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE user_settings ADD COLUMN quiz_timer_enabled "
                "BOOLEAN NOT NULL DEFAULT TRUE"
            )
        )


def _ensure_attempt_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "attempts" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("attempts")}
    with db.engine.begin() as connection:
        if "quiz_id" not in columns:
            connection.execute(
                text("ALTER TABLE attempts ADD COLUMN quiz_id VARCHAR(64)")
            )
        if "quiz_kind" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE attempts ADD COLUMN quiz_kind "
                    "VARCHAR(32) NOT NULL DEFAULT 'diagnostic'"
                )
            )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ix_attempts_quiz_id ON attempts (quiz_id)"
            )
        )


def _ensure_auth_security_columns() -> None:
    """Forward-compatible migration for databases created by the earlier prototype."""
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    dialect = db.engine.dialect.name
    added_verification_column = "email_verified_at" not in columns
    statements: list[str] = []
    if added_verification_column:
        statements.append("ALTER TABLE users ADD COLUMN email_verified_at TIMESTAMP")
    if "password_changed_at" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN password_changed_at TIMESTAMP")
    if "session_version" not in columns:
        statements.append(
            "ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 0"
        )
    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        if added_verification_column:
            # Preserve access for accounts created before email verification existed.
            connection.execute(
                text(
                    "UPDATE users SET email_verified_at = "
                    "COALESCE(joined_at, CURRENT_TIMESTAMP) WHERE email_verified_at IS NULL"
                )
            )
        if "auth_tokens" in inspector.get_table_names():
            auth_columns = {
                column["name"] for column in inspector.get_columns("auth_tokens")
            }
            if "failed_attempts" not in auth_columns:
                connection.execute(
                    text(
                        "ALTER TABLE auth_tokens ADD COLUMN failed_attempts "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                )


def _ensure_assessment_template_metadata_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "assessment_templates" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("assessment_templates")}
    definitions = {
        "generation_provider": "VARCHAR(64) NOT NULL DEFAULT 'unknown'",
        "embedding_model": "VARCHAR(255) NOT NULL DEFAULT 'BAAI/bge-m3'",
        "embedding_provider": "VARCHAR(64) NOT NULL DEFAULT 'unknown'",
        "question_bank_version": "VARCHAR(64) NOT NULL DEFAULT 'none'",
        "corpus_version": "VARCHAR(128) NOT NULL DEFAULT 'unknown'",
        "prompt_version": "VARCHAR(32) NOT NULL DEFAULT 'unknown'",
        "validator_version": "VARCHAR(32) NOT NULL DEFAULT 'unknown'",
        "chunker_version": "VARCHAR(64) NOT NULL DEFAULT 'structure-token-v1'",
    }
    with db.engine.begin() as connection:
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(
                    text(f"ALTER TABLE assessment_templates ADD COLUMN {name} {definition}")
                )


with app.app_context():
    db.create_all()
    _ensure_attempt_question_columns()
    _ensure_active_quiz_columns()
    _ensure_user_settings_columns()
    _ensure_attempt_columns()
    _ensure_auth_security_columns()
    _ensure_assessment_template_metadata_columns()

    if config.AUTO_INGEST_ON_STARTUP:
        try:
            from rag.ingest import ensure_vector_index_ready

            ensure_vector_index_ready(repair=True)
        except Exception as exc:
            print(f"Automatic vector-index preparation failed: {exc}")


PAPER_FILE = config.PROJECT_ROOT / "generated_paper.json"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _option_text(options: list, answer_letter: str | None) -> str | None:
    if not isinstance(options, list):
        return None
    letter = str(answer_letter or "").strip().upper()[:1]
    index_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    idx = index_map.get(letter)
    if idx is None or idx >= len(options):
        return None
    return str(options[idx])


def _has_llm_config() -> bool:
    return bool(
        config.RUNPOD_API_KEY
        and config.RUNPOD_MCQ_ENDPOINT_ID
        and config.RUNPOD_MCQ_GENERATION_MODEL == config.SELECTED_MCQ_GENERATION_MODEL
    )


def _has_embedding_config() -> bool:
    return bool(
        config.HF_API_KEY
        and config.HF_EMBEDDING_MODEL == config.SELECTED_EMBEDDING_MODEL
        and config.HF_EMBEDDING_DIMENSIONS == config.SELECTED_EMBEDDING_DIMENSIONS
    )


def _infer_syllabus_from_reference_docs() -> dict:
    supported_ext = {".pdf", ".md", ".txt"}
    files = []
    for configured_dir in config.QUIZ_REFERENCE_DIRS:
        base_dir = config.resolve_reference_dir(configured_dir)
        if not base_dir.exists():
            continue
        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in supported_ext:
                continue
            if not config.include_quiz_reference_file(p):
                continue
            files.append(p)
    files = sorted(files, key=lambda p: p.name.lower())

    mappings = []
    if files:
        for idx, path in enumerate(files[:8], start=1):
            label = path.stem.replace("_", " ").replace("-", " ").strip()
            topic_id = f"doc_{idx:03d}"
            mappings.append(
                {
                    "topic_id": topic_id,
                    "topic_name": label or topic_id,
                    "required_levels": [1, 2, 3],
                    "weightage_percent": 0,
                    "min_questions": 6,
                    "max_questions": 10,
                    "notes": f"Auto-inferred from {path.name}",
                }
            )
    else:
        mappings.append(
            {
                "topic_id": "doc_001",
                "topic_name": "Mathematics",
                "required_levels": [1, 2, 3],
                "weightage_percent": 100,
                "min_questions": 12,
                "max_questions": 20,
                "notes": "Auto-generated fallback when no reference files are found.",
            }
        )

    n = len(mappings)
    for i, m in enumerate(mappings):
        m["weightage_percent"] = 100 // n + (1 if i < (100 % n) else 0)

    total_questions = min(40, max(12, n * 6))
    return {
        "syllabus_name": "Auto-Inferred from reference docs (G10/G11)",
        "total_questions": total_questions,
        "time_limit_minutes": max(30, total_questions * 2),
        "topic_mappings": mappings,
    }


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return s or "topic"


def _infer_syllabus_from_topics_file() -> dict | None:
    try:
        raw = _load_json(config.TOPICS_FILE)
    except Exception:
        return None

    flat_topics: list[dict] = []
    if isinstance(raw, list):
        # Legacy format: [{"id": "...", "name": "..."}]
        for i, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            topic_id = str(item.get("id", "")).strip() or f"topic_{i:03d}"
            flat_topics.append({"topic_id": topic_id, "topic_name": name, "periods": 4})
    elif isinstance(raw, dict):
        # Grade/term format.
        for grade in raw.get("grades", []):
            grade_no = grade.get("grade")
            for term in grade.get("terms", []):
                term_no = term.get("term")
                for idx, topic in enumerate(term.get("topics", []), start=1):
                    if not isinstance(topic, dict):
                        continue
                    name = str(topic.get("topic", "")).strip()
                    if not name:
                        continue
                    no = topic.get("no", idx)
                    topic_id = f"g{grade_no}_t{term_no}_{int(no):02d}_{_slug(name)}"
                    try:
                        periods = int(topic.get("periods", 4))
                    except Exception:
                        periods = 4
                    flat_topics.append(
                        {
                            "topic_id": topic_id,
                            "topic_name": name,
                            "periods": max(1, periods),
                        }
                    )

    if not flat_topics:
        return None

    # Keep quiz size practical; choose top weighted topics.
    flat_topics.sort(key=lambda t: t.get("periods", 1), reverse=True)
    chosen = flat_topics[:8]
    total_periods = max(1, sum(t.get("periods", 1) for t in chosen))

    mappings = []
    for t in chosen:
        periods = t.get("periods", 1)
        weight = max(1, round(periods * 100 / total_periods))
        min_q = 2 if periods <= 4 else 3
        mappings.append(
            {
                "topic_id": t["topic_id"],
                "topic_name": t["topic_name"],
                "required_levels": [1, 2, 3],
                "weightage_percent": weight,
                "min_questions": min_q,
                "max_questions": max(min_q + 1, min_q + 3),
                "notes": "Auto-inferred from topics.json",
            }
        )

    # Normalize weightage to exactly 100.
    diff = 100 - sum(m["weightage_percent"] for m in mappings)
    if mappings:
        mappings[0]["weightage_percent"] += diff

    total_questions = max(12, min(40, sum(m["min_questions"] for m in mappings)))
    return {
        "syllabus_name": "Auto-Inferred from topics.json",
        "total_questions": total_questions,
        "time_limit_minutes": max(30, total_questions * 2),
        "topic_mappings": mappings,
    }


def _load_or_infer_syllabus() -> dict:
    from rag.diagnostic import build_diagnostic_syllabus

    # topics.json is the reviewed source of truth for Grade 10/11 concept coverage.
    return build_diagnostic_syllabus()


def _topic_name_lookup_from_topics_file() -> dict[str, str]:
    """
    Build topic_id -> topic_name from data/topics.json using the same id format
    used by _infer_syllabus_from_topics_file.
    """
    lookup: dict[str, str] = {}
    try:
        raw = _load_json(config.TOPICS_FILE)
    except Exception:
        return lookup

    if not isinstance(raw, dict):
        return lookup

    for grade in raw.get("grades", []):
        grade_no = grade.get("grade")
        for term in grade.get("terms", []):
            term_no = term.get("term")
            for idx, topic in enumerate(term.get("topics", []), start=1):
                if not isinstance(topic, dict):
                    continue
                name = str(topic.get("topic", "")).strip()
                if not name:
                    continue
                no = topic.get("no", idx)
                topic_id = f"g{grade_no}_t{term_no}_{int(no):02d}_{_slug(name)}"
                lookup[topic_id] = name
    return lookup


def _textbook_index_by_grade() -> dict[int, list[str]]:
    """Index textbook filenames by detected grade number (6-11)."""
    books_dir = config.DATA_DIR / "TextBooks"
    if not books_dir.exists():
        return {}

    by_grade: dict[int, list[str]] = {}
    for p in books_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".pdf":
            continue
        name = p.name
        m = re.search(r"\bg[\s\-_]?(\d{1,2})\b", name, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            grade = int(m.group(1))
        except Exception:
            continue
        if grade < 6 or grade > 11:
            continue
        by_grade.setdefault(grade, []).append(name)

    for grade in by_grade:
        by_grade[grade] = sorted(by_grade[grade], key=lambda x: x.lower())
    return by_grade


def _find_topic_id_for_grade_topic(grade: int, topic_name: str) -> str:
    """Resolve to existing topics.json id where possible, otherwise fallback."""
    target = re.sub(r"[^a-z0-9]+", " ", str(topic_name or "").lower()).strip()
    if not target:
        return f"practice_g{grade}_topic"

    try:
        raw = _load_json(config.TOPICS_FILE)
    except Exception:
        raw = {}

    if isinstance(raw, dict):
        for g in raw.get("grades", []):
            if int(g.get("grade", -1)) != int(grade):
                continue
            for term in g.get("terms", []):
                term_no = term.get("term")
                for idx, topic in enumerate(term.get("topics", []), start=1):
                    if not isinstance(topic, dict):
                        continue
                    name = str(topic.get("topic", "")).strip()
                    norm = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
                    if not norm:
                        continue
                    if target == norm or target in norm or norm in target:
                        no = topic.get("no", idx)
                        return f"g{grade}_t{term_no}_{int(no):02d}_{_slug(name)}"

    return f"practice_g{grade}_{_slug(topic_name)}"


def _build_practice_catalog() -> list[dict]:
    """
    Build practice tabs from concept_mapping.json:
    - one tab per main_topic
    - subtopics grouped by grade (11 -> 6)
    """
    try:
        concept_mapping = _load_json(config.DATA_DIR / "concept_mapping.json")
    except Exception:
        concept_mapping = []

    textbooks = _textbook_index_by_grade()
    tabs: list[dict] = []
    if not isinstance(concept_mapping, list):
        return tabs

    for main in concept_mapping:
        main_topic = str(main.get("main_topic", "")).strip()
        if not main_topic:
            continue

        grade_groups: dict[int, dict] = {}
        for strand in main.get("strands", []):
            strand_name = str(strand.get("strand_name", "")).strip()
            for step in strand.get("progression", []):
                grade = step.get("grade")
                topic = str(step.get("topic", "")).strip()
                if not isinstance(grade, int) or not topic:
                    continue
                if grade < 6 or grade > 11:
                    continue

                grp = grade_groups.setdefault(
                    grade,
                    {"grade": grade, "subtopics": []},
                )
                if any(s["topic"] == topic for s in grp["subtopics"]):
                    continue

                grp["subtopics"].append(
                    {
                        "topic": topic,
                        "strand_name": strand_name,
                        "textbooks": textbooks.get(grade, []),
                    }
                )

        grades_payload = [
            {
                "grade": grade,
                "subtopics": sorted(items["subtopics"], key=lambda s: s["topic"].lower()),
            }
            for grade, items in sorted(grade_groups.items(), key=lambda x: x[0], reverse=True)
        ]
        tabs.append(
            {
                "tab_id": _slug(main_topic),
                "label": main_topic,
                "grades": grades_payload,
            }
        )

    return tabs


def _public_paper(paper: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only fields needed to take a quiz; never expose answer keys early."""
    allowed = {
        "question",
        "options",
        "topic_id",
        "difficulty_level",
        "grade",
        "term",
        "image_url",
    }
    return [{key: value for key, value in question.items() if key in allowed} for question in paper]


def _personalize_quiz(
    paper: list[dict[str, Any]],
    *,
    randomizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Create an answer-equivalent private ordering for one student."""
    rng = randomizer or secrets.SystemRandom()
    personalized = json.loads(json.dumps(paper, ensure_ascii=False))
    positions = list("ABCD")
    rng.shuffle(positions)
    targets = positions * (len(personalized) // 4) + positions[:len(personalized) % 4]
    rng.shuffle(targets)
    for question, target in zip(personalized, targets):
        options = question.get("options")
        answer = str(question.get("answer", "")).upper()
        if (
            not isinstance(options, list)
            or len(options) != 4
            or len({str(option).strip() for option in options}) != 4
            or answer not in {"A", "B", "C", "D"}
        ):
            raise ValueError("Cannot personalize an invalid MCQ.")
        if answer_explanation_consistency_errors(question):
            raise ValueError("Cannot serve a question with an inconsistent answer and explanation.")
        correct_option = str(options["ABCD".index(answer)])
        if "correct_answer" in question:
            answer = answer_letter_from_value(question["correct_answer"], options)
            correct_option = str(options["ABCD".index(answer)])
        old_options = list(options)
        distractors = [
            option for index, option in enumerate(options)
            if index != "ABCD".index(answer)
        ]
        rng.shuffle(distractors)
        distractors.insert("ABCD".index(target), correct_option)
        question["options"] = distractors
        question["answer"] = answer_letter_from_value(correct_option, distractors)
        question["correct_answer"] = correct_option
        letter_map = {
            "ABCD"[index]: "ABCD"[distractors.index(option)]
            for index, option in enumerate(old_options)
        }
        question["explanation"] = remap_explanation_options(
            str(question.get("explanation", "")), letter_map
        )
        question["answer_contract_version"] = ANSWER_CONTRACT_VERSION
        if answer_explanation_consistency_errors(question):
            raise ValueError("Option shuffling produced an inconsistent answer and explanation.")
    rng.shuffle(personalized)
    return personalized


def _orm_identity_id(instance: Any) -> int:
    """Return a model primary key without triggering a refresh on a detached instance."""
    state = sqlalchemy_inspect(instance)
    if state.identity and len(state.identity) == 1:
        return int(state.identity[0])
    value = getattr(instance, "id", None)
    if value is None:
        raise RuntimeError("Cannot persist quiz state for an unsaved user.")
    return int(value)


def _get_or_create_user_settings(user_id: int) -> UserSettings:
    """Return one user's settings without failing a concurrent first-read/create race."""
    settings = UserSettings.query.filter_by(user_id=int(user_id)).first()
    if settings is not None:
        return settings
    settings = UserSettings(
        user_id=int(user_id),
        theme="dark",
        difficulty="medium",
        quiz_timer_enabled=True,
    )
    db.session.add(settings)
    try:
        db.session.commit()
        return settings
    except IntegrityError:
        db.session.rollback()
        settings = UserSettings.query.filter_by(user_id=int(user_id)).first()
        if settings is None:
            raise
        _quiz_log(
            "settings.concurrent_create",
            user_id=int(user_id),
            outcome="reused_existing",
        )
        return settings


def _user_quiz_timer_enabled(user_id: int) -> bool:
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if settings is None:
        return True
    return bool(settings.quiz_timer_enabled)


def _practice_time_limit_minutes(
    grade: int,
    topic_name: str,
    difficulty_level: int | None = None,
) -> int:
    """Return a stable topic-based practice limit using curriculum metadata when available."""
    target = re.sub(r"[^a-z0-9]+", " ", str(topic_name or "").lower()).strip()
    periods = 0
    try:
        raw = _load_json(config.TOPICS_FILE)
    except Exception:
        raw = {}

    if isinstance(raw, dict):
        for grade_row in raw.get("grades", []):
            if int(grade_row.get("grade", -1)) != int(grade):
                continue
            for term in grade_row.get("terms", []):
                for topic in term.get("topics", []):
                    if not isinstance(topic, dict):
                        continue
                    name = str(topic.get("topic", "")).strip()
                    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
                    if normalized and (target == normalized or target in normalized or normalized in target):
                        try:
                            periods = max(1, int(topic.get("periods", 0)))
                        except Exception:
                            periods = 0
                        break
                if periods:
                    break
            if periods:
                break

    if periods <= 0:
        return {1: 12, 2: 15, 3: 20}.get(int(difficulty_level or 0), 20)
    if periods <= 4:
        return 12
    if periods <= 7:
        return 15
    if periods <= 10:
        return 18
    if periods <= 14:
        return 20
    return 25


def _quiz_timer_payload(active: ActiveQuiz | None) -> dict[str, Any]:
    if (
        active is None
        or not active.timer_enabled
        or not active.expires_at
        or not active.timer_duration_seconds
    ):
        return {"enabled": False}

    expires_at = active.expires_at.replace(tzinfo=timezone.utc)
    remaining_seconds = max(0, int((active.expires_at - _utcnow()).total_seconds()))
    return {
        "enabled": True,
        "duration_seconds": int(active.timer_duration_seconds),
        "remaining_seconds": remaining_seconds,
        "expires_at": expires_at.isoformat(),
    }


def _store_active_quiz(
    user: User,
    paper: list[dict[str, Any]],
    quiz_kind: str,
    *,
    time_limit_minutes: int | None = None,
    target_question_count: int | None = None,
    batch_state: dict[str, Any] | None = None,
) -> tuple[ActiveQuiz, list[dict[str, Any]]]:
    personalized = _personalize_quiz(paper)
    # _generate_with_draft intentionally releases the request-scoped SQLAlchemy
    # session while remote inference runs. The original User object is therefore
    # detached afterwards. SQLAlchemy keeps the primary-key identity on detached
    # objects, so use that identity instead of touching user.id (which may be
    # expired and would otherwise trigger DetachedInstanceError).
    user_id = _orm_identity_id(user)
    active = ActiveQuiz.query.filter_by(user_id=user_id).first()
    if not active:
        active = ActiveQuiz(user_id=user_id, quiz_id=secrets.token_urlsafe(24))
        db.session.add(active)
    else:
        active.quiz_id = secrets.token_urlsafe(24)
    active.quiz_kind = quiz_kind
    active.status = "active"
    active.paper_json = json.dumps(personalized, ensure_ascii=False)
    active.created_at = _utcnow()
    active.target_question_count = (
        max(len(personalized), int(target_question_count))
        if target_question_count is not None
        else len(personalized)
    )
    active.batch_state_json = (
        json.dumps(batch_state, ensure_ascii=False) if batch_state is not None else None
    )

    timer_enabled = (
        _user_quiz_timer_enabled(user_id)
        and time_limit_minutes is not None
        and int(time_limit_minutes) > 0
    )
    active.timer_enabled = bool(timer_enabled)
    if timer_enabled:
        duration_seconds = int(time_limit_minutes) * 60
        active.timer_duration_seconds = duration_seconds
        active.expires_at = active.created_at + timedelta(seconds=duration_seconds)
    else:
        active.timer_duration_seconds = None
        active.expires_at = None

    for question in personalized:
        pool_question_id = question.get("_pool_question_id")
        if not isinstance(pool_question_id, int):
            continue
        exposure = UserQuestionExposure.query.filter_by(
            user_id=user_id, pool_question_id=pool_question_id
        ).first()
        if exposure is None:
            db.session.add(
                UserQuestionExposure(
                    user_id=user_id,
                    pool_question_id=pool_question_id,
                    quiz_id=active.quiz_id,
                    quiz_kind=quiz_kind,
                )
            )

    db.session.commit()
    return active, personalized


def _load_active_quiz(user: User) -> tuple[ActiveQuiz | None, list[dict[str, Any]]]:
    from rag.validation import has_current_review

    active = ActiveQuiz.query.filter_by(user_id=user.id, status="active").first()
    if not active:
        return None, []
    try:
        paper = json.loads(active.paper_json)
    except Exception:
        return active, []
    if not isinstance(paper, list) or any(
        not isinstance(question, dict)
        or question.get("answer_contract_version") != ANSWER_CONTRACT_VERSION
        or ("quality_review" in question and not has_current_review(question))
        for question in paper
    ):
        return active, []
    return active, paper


def _active_quiz_for_update(
    user_id: int,
    quiz_id: str,
) -> tuple[ActiveQuiz | None, list[dict[str, Any]]]:
    """Lock one active quiz row for a short state transition."""
    from rag.validation import has_current_review

    active = (
        ActiveQuiz.query.filter(
            ActiveQuiz.user_id == int(user_id),
            ActiveQuiz.quiz_id == str(quiz_id),
            ActiveQuiz.status == "active",
        )
        .with_for_update()
        .first()
    )
    if active is None:
        return None, []
    try:
        paper = json.loads(active.paper_json or "[]")
    except (TypeError, ValueError):
        return active, []
    if not isinstance(paper, list) or any(
        not isinstance(question, dict)
        or question.get("answer_contract_version") != ANSWER_CONTRACT_VERSION
        or ("quality_review" in question and not has_current_review(question))
        for question in paper
    ):
        return active, []
    return active, paper


def _claim_active_quiz(user: User, quiz_id: str) -> bool:
    claimed = ActiveQuiz.query.filter(
        ActiveQuiz.user_id == user.id,
        ActiveQuiz.quiz_id == quiz_id,
        ActiveQuiz.status == "active",
    ).update({"status": "submitting"}, synchronize_session=False)
    db.session.flush()
    return claimed == 1


def _recent_question_stems(user: User, topic_id: str = "") -> list[str]:
    query = (
        AttemptQuestion.query.join(
            Attempt,
            AttemptQuestion.attempt_id == Attempt.id,
        )
        .filter(Attempt.user_id == user.id)
        .order_by(AttemptQuestion.id.desc())
    )
    if topic_id:
        query = query.filter(AttemptQuestion.topic_id == topic_id)
    return [
        row.question
        for row in query.limit(config.QUESTION_HISTORY_LIMIT).all()
        if str(row.question or "").strip()
    ]


def _previous_source_question_ids(user: User, topic_id: str = "") -> set[str]:
    query = (
        AttemptQuestion.query.join(
            Attempt,
            AttemptQuestion.attempt_id == Attempt.id,
        )
        .filter(
            Attempt.user_id == user.id,
            AttemptQuestion.source_question_id.is_not(None),
        )
    )
    if topic_id:
        query = query.filter(AttemptQuestion.topic_id == topic_id)
    return {
        str(row.source_question_id)
        for row in query.limit(config.QUESTION_HISTORY_LIMIT).all()
        if row.source_question_id
    }


def _configured_model_id() -> str:
    # AssessmentTemplate.model_id records the model that actually authors MCQs.
    return config.RUNPOD_MCQ_GENERATION_MODEL


def _student_memory(user: User) -> dict[str, Any]:
    from sqlalchemy import func

    ordered = (
        db.session.query(
            AttemptQuestion.topic_id, AttemptQuestion.question,
            AttemptQuestion.source_question_id, AttemptQuestion.is_correct,
            AttemptQuestion.difficulty_level, AttemptQuestion.attempt_id,
            func.dense_rank().over(order_by=Attempt.id).label("attempt_number"),
            func.row_number().over(
                partition_by=AttemptQuestion.topic_id,
                order_by=AttemptQuestion.id.desc(),
            ).label("topic_position"),
        )
        .join(Attempt, AttemptQuestion.attempt_id == Attempt.id)
        .filter(Attempt.user_id == user.id)
        .subquery()
    )
    rows = (db.session.query(ordered).filter(ordered.c.topic_position <= 50)
            .order_by(ordered.c.topic_id, ordered.c.topic_position).all())
    history: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row._mapping)
        history.setdefault(str(item["topic_id"]), []).append(item)
    attempts = Attempt.query.filter_by(user_id=user.id)
    return {
        "history_by_topic": history,
        "attempt_count": attempts.count(),
        "diagnostic_count": attempts.filter_by(quiz_kind="diagnostic").count(),
        "latest_attempt_id": db.session.query(func.max(Attempt.id)).filter(Attempt.user_id == user.id).scalar() or 0,
    }


def _student_diagnostic_syllabus(user: User | None, *, variant: str = "") -> dict[str, Any]:
    from rag.adaptive import select_diagnostic_syllabus

    return select_diagnostic_syllabus(
        _load_or_infer_syllabus(),
        student_key=str(user.id) if user else "preview",
        variant=variant,
        **(_student_memory(user) if user else {}),
    )


def _pool_shortage_error(
    *,
    topic_name: str,
    difficulty_level: str,
    required: int,
    available: int,
) -> RuntimeError:
    return RuntimeError(
        "Question pool and on-demand generation could not provide enough unseen questions for "
        f"'{topic_name}' at {difficulty_level}: required {required}, available {available}."
    )


QUIZ_BATCH_SIZE = 5


def _pool_source_type_for_question(question: dict[str, Any]) -> str:
    from rag.question_pool import POOL_SOURCE_LLM, POOL_SOURCE_PAST_PAPER_TRANSFORMED

    return (
        POOL_SOURCE_PAST_PAPER_TRANSFORMED
        if str(question.get("source_question_id", "")).strip()
        else POOL_SOURCE_LLM
    )


def _persist_generated_questions_to_pool(
    paper: list[dict[str, Any]],
    *,
    topic_names: dict[str, str],
    diagnostic_topic_ids: set[str] | None = None,
) -> int:
    """Store newly validated runtime questions for reuse by other students."""
    from rag.question_pool import add_pool_question

    diagnostic_ids = diagnostic_topic_ids or set()
    added = 0
    for question in paper:
        topic_key = str(question.get("topic_id", "")).strip()
        topic_name = topic_names.get(topic_key, topic_key or "Mathematics")
        grade = int(question.get("grade", 0) or 0)
        diagnostic_eligible = topic_key in diagnostic_ids or (
            grade in set(config.DIAGNOSTIC_GRADES)
            and topic_key.startswith(f"g{grade}_t")
        )
        if add_pool_question(
            question,
            topic_name=topic_name,
            source_type=_pool_source_type_for_question(question),
            eligible_diagnostic=diagnostic_eligible,
            eligible_practice=True,
        ):
            added += 1
    return added


def _generate_runtime_pool_batch(
    *,
    user_id: int,
    syllabus: dict[str, Any],
    active_paper: list[dict[str, Any]] | None = None,
    diagnostic_topic_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the existing RAG generation path for one small batch and cache the result globally."""
    from rag.generator import generate_paper
    from rag.ingest import ensure_vector_index_ready
    from rag.retriever import retrieve_for_syllabus

    user = db.session.get(User, int(user_id))
    if user is None:
        raise RuntimeError("Student account no longer exists.")

    active_questions = list(active_paper or [])
    runtime_syllabus = json.loads(json.dumps(syllabus, ensure_ascii=False))
    excluded_by_topic: dict[str, set[str]] = {}
    for mapping in runtime_syllabus.get("topic_mappings", []):
        topic_key = str(mapping.get("topic_id", ""))
        blocked_stems = list(mapping.get("avoid_question_stems", []))
        blocked_stems.extend(_recent_question_stems(user, topic_key))
        blocked_stems.extend(
            str(question.get("question", ""))
            for question in active_questions
            if str(question.get("topic_id", "")) == topic_key
            and str(question.get("question", "")).strip()
        )
        mapping["avoid_question_stems"] = list(dict.fromkeys(blocked_stems))

        excluded = set(_previous_source_question_ids(user, topic_key))
        excluded.update(
            str(question.get("source_question_id", "")).strip()
            for question in active_questions
            if str(question.get("topic_id", "")) == topic_key
            and str(question.get("source_question_id", "")).strip()
        )
        excluded_by_topic[topic_key] = excluded
        mapping["excluded_source_question_ids"] = sorted(excluded)
        mapping["generation_offset"] = int(mapping.get("generation_offset", 0) or 0) + secrets.randbelow(1_000_000)

    required_grades = tuple(sorted({
        int(mapping.get("grade", 0) or 0)
        for mapping in runtime_syllabus.get("topic_mappings", [])
        if int(mapping.get("grade", 0) or 0) > 0
    }))
    ensure_vector_index_ready(
        repair=config.INGEST_ON_QUIZ_LOAD,
        required_grades=required_grades or None,
    )
    retrieved = retrieve_for_syllabus(
        runtime_syllabus,
        exclude_question_ids_by_topic=excluded_by_topic,
        question_seed=secrets.token_hex(16),
    )

    # Do not hold a Neon connection while RunPod performs the long inference work.
    db.session.remove()
    paper = generate_paper(runtime_syllabus, retrieved)
    topic_names = {
        str(mapping.get("topic_id", "")): str(mapping.get("topic_name", ""))
        for mapping in runtime_syllabus.get("topic_mappings", [])
    }
    _persist_generated_questions_to_pool(
        paper,
        topic_names=topic_names,
        diagnostic_topic_ids=diagnostic_topic_ids,
    )
    return paper


def _practice_batch_syllabus(
    *,
    grade: int,
    topic_id: str,
    topic_name: str,
    level_id: int,
    count: int,
) -> dict[str, Any]:
    return {
        "syllabus_name": f"Practice: Grade {grade} - {topic_name}",
        "assessment_type": "practice",
        "grades": [int(grade)],
        "total_questions": int(count),
        "topic_mappings": [
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "grade": int(grade),
                "required_levels": [int(level_id)],
                "require_application": True,
                "weightage_percent": 100,
                "min_questions": int(count),
                "max_questions": int(count),
                "strict_textbook_grounding": True,
                "avoid_question_stems": [],
                "notes": f"Topic practice for Grade {grade}",
            }
        ],
    }


def _pool_private_exclusions(
    paper: list[dict[str, Any]] | None,
) -> tuple[set[int], set[str], set[str]]:
    pool_ids: set[int] = set()
    question_hashes: set[str] = set()
    source_ids: set[str] = set()
    for question in paper or []:
        pool_id = question.get("_pool_question_id")
        if isinstance(pool_id, int):
            pool_ids.add(pool_id)
        pool_hash = str(question.get("_pool_question_hash", "")).strip()
        if pool_hash:
            question_hashes.add(pool_hash)
        source_id = str(question.get("source_question_id", "")).strip()
        if source_id:
            source_ids.add(source_id)
    return pool_ids, question_hashes, source_ids


def _practice_batch_from_pool_or_generate(
    *,
    user_id: int,
    topic_id: str,
    topic_name: str,
    grade: int,
    level_id: int,
    difficulty_level: str,
    count: int,
    active_paper: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    from rag.question_pool import select_unseen_questions

    blocked_pool_ids, blocked_hashes, blocked_source_ids = _pool_private_exclusions(active_paper)
    generated = False
    for _ in range(2):
        selected = select_unseen_questions(
            user_id=int(user_id),
            topic_id=topic_id,
            grade=int(grade),
            difficulty_level=difficulty_level,
            count=int(count),
            quiz_kind="practice",
            exclude_pool_question_ids=blocked_pool_ids,
            exclude_question_hashes=blocked_hashes,
            exclude_source_question_ids=blocked_source_ids,
        )
        if len(selected) >= count:
            return selected[:count], generated

        missing = count - len(selected)
        syllabus = _practice_batch_syllabus(
            grade=grade,
            topic_id=topic_id,
            topic_name=topic_name,
            level_id=level_id,
            count=missing,
        )
        _generate_runtime_pool_batch(
            user_id=int(user_id),
            syllabus=syllabus,
            active_paper=active_paper,
        )
        generated = True

    selected = select_unseen_questions(
        user_id=int(user_id),
        topic_id=topic_id,
        grade=int(grade),
        difficulty_level=difficulty_level,
        count=int(count),
        quiz_kind="practice",
        exclude_pool_question_ids=blocked_pool_ids,
        exclude_question_hashes=blocked_hashes,
        exclude_source_question_ids=blocked_source_ids,
    )
    if len(selected) != count:
        raise _pool_shortage_error(
            topic_name=topic_name,
            difficulty_level=difficulty_level,
            required=count,
            available=len(selected),
        )
    return selected, generated


def _diagnostic_partial_syllabus(
    full_syllabus: dict[str, Any],
    mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    partial = json.loads(json.dumps(full_syllabus, ensure_ascii=False))
    partial["topic_mappings"] = json.loads(json.dumps(mappings, ensure_ascii=False))
    partial["total_questions"] = len(mappings)
    # Full adaptive diagnostics must contain all 25 mappings, but a runtime batch
    # is intentionally validated as a strict partial diagnostic.
    partial.pop("selection_version", None)
    partial.pop("selection_seed", None)
    return partial


def _diagnostic_batch_from_pool_or_generate(
    *,
    user_id: int,
    full_syllabus: dict[str, Any],
    mappings: list[dict[str, Any]],
    active_paper: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    from rag.adaptive import LEVELS, POLICY_VERSION
    from rag.diagnostic import validate_diagnostic_paper
    from rag.question_pool import select_unseen_questions

    blocked_pool_ids, blocked_hashes, blocked_source_ids = _pool_private_exclusions(active_paper)
    generated = False
    missing_mappings: list[dict[str, Any]] = []
    selected_by_topic: dict[str, dict[str, Any]] = {}

    for mapping in mappings:
        level_id = max(int(value) for value in mapping.get("required_levels", [3]))
        difficulty = LEVELS.get(level_id, "Apply")
        topic_key = str(mapping["topic_id"])
        rows = select_unseen_questions(
            user_id=int(user_id),
            topic_id=topic_key,
            grade=int(mapping["grade"]),
            difficulty_level=difficulty,
            count=1,
            quiz_kind="diagnostic",
            exclude_pool_question_ids=blocked_pool_ids,
            exclude_question_hashes=blocked_hashes,
            exclude_source_question_ids=blocked_source_ids,
        )
        if rows:
            question = rows[0]
            question["adaptive_policy_version"] = POLICY_VERSION
            selected_by_topic[topic_key] = question
        else:
            missing_mappings.append(mapping)

    if missing_mappings:
        partial = _diagnostic_partial_syllabus(full_syllabus, missing_mappings)
        _generate_runtime_pool_batch(
            user_id=int(user_id),
            syllabus=partial,
            active_paper=active_paper,
            diagnostic_topic_ids={str(mapping["topic_id"]) for mapping in missing_mappings},
        )
        generated = True
        for mapping in missing_mappings:
            level_id = max(int(value) for value in mapping.get("required_levels", [3]))
            difficulty = LEVELS.get(level_id, "Apply")
            topic_key = str(mapping["topic_id"])
            rows = select_unseen_questions(
                user_id=int(user_id),
                topic_id=topic_key,
                grade=int(mapping["grade"]),
                difficulty_level=difficulty,
                count=1,
                quiz_kind="diagnostic",
                exclude_pool_question_ids=blocked_pool_ids,
                exclude_question_hashes=blocked_hashes,
                exclude_source_question_ids=blocked_source_ids,
            )
            if rows:
                question = rows[0]
                question["adaptive_policy_version"] = POLICY_VERSION
                selected_by_topic[topic_key] = question

    batch: list[dict[str, Any]] = []
    for mapping in mappings:
        topic_key = str(mapping["topic_id"])
        question = selected_by_topic.get(topic_key)
        if question is None:
            level_id = max(int(value) for value in mapping.get("required_levels", [3]))
            raise _pool_shortage_error(
                topic_name=str(mapping.get("topic_name") or topic_key),
                difficulty_level=LEVELS.get(level_id, "Apply"),
                required=1,
                available=0,
            )
        batch.append(question)

    validate_diagnostic_paper(batch, _diagnostic_partial_syllabus(full_syllabus, mappings))
    return batch, generated


def _active_batch_state(active: ActiveQuiz) -> dict[str, Any]:
    try:
        state = json.loads(active.batch_state_json or "{}")
    except (TypeError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _append_active_quiz_batch(
    *,
    active: ActiveQuiz,
    user_id: int,
    paper: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        current = json.loads(active.paper_json or "[]")
    except (TypeError, ValueError):
        current = []
    if not isinstance(current, list):
        current = []

    personalized = _personalize_quiz(paper)
    combined = current + personalized
    active.paper_json = json.dumps(combined, ensure_ascii=False)
    for question in personalized:
        pool_question_id = question.get("_pool_question_id")
        if not isinstance(pool_question_id, int):
            continue
        exposure = UserQuestionExposure.query.filter_by(
            user_id=int(user_id), pool_question_id=pool_question_id
        ).first()
        if exposure is None:
            db.session.add(
                UserQuestionExposure(
                    user_id=int(user_id),
                    pool_question_id=pool_question_id,
                    quiz_id=active.quiz_id,
                    quiz_kind=active.quiz_kind,
                )
            )
    db.session.flush()
    return personalized, combined


def _batch_response_metadata(active: ActiveQuiz, loaded_count: int) -> dict[str, Any]:
    target = int(active.target_question_count or loaded_count)
    return {
        "loaded_count": int(loaded_count),
        "total_count": target,
        "batch_size": QUIZ_BATCH_SIZE,
        "has_more": int(loaded_count) < target,
    }


_batch_prefetch_inflight: set[tuple[int, str]] = set()
_batch_prefetch_guard = threading.Lock()
_batch_prefetch_executor = ThreadPoolExecutor(
    max_workers=config.QUIZ_PREFETCH_MAX_WORKERS,
    thread_name_prefix="quiz-prefetch",
)


def _prefetch_batches(state: dict[str, Any]) -> list[dict[str, Any]]:
    prefetch = state.get("prefetch")
    if not isinstance(prefetch, dict):
        return []
    batches = prefetch.get("batches")
    if not isinstance(batches, list):
        return []
    return [item for item in batches if isinstance(item, dict)]


def _prefetch_is_stale(prefetch: dict[str, Any]) -> bool:
    updated_at = str(prefetch.get("updated_at", "")).strip()
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return True
    if updated.tzinfo is not None:
        updated = updated.astimezone(timezone.utc).replace(tzinfo=None)
    return (_utcnow() - updated).total_seconds() > config.QUIZ_PREFETCH_STALE_SECONDS


def _batch_prefetch_for_offset(state: dict[str, Any], offset: int) -> dict[str, Any]:
    prefetch = state.get("prefetch")
    if not isinstance(prefetch, dict):
        return {}

    for item in _prefetch_batches(state):
        try:
            item_offset = int(item.get("offset", -1))
        except (TypeError, ValueError):
            continue
        if item_offset == int(offset):
            return item

    # Backward compatibility with the previous single-batch state shape.
    try:
        legacy_offset = int(prefetch.get("offset", -1))
    except (TypeError, ValueError):
        legacy_offset = -1
    if legacy_offset == int(offset) and "batch" in prefetch:
        return prefetch

    status = str(prefetch.get("status", "idle"))
    if status in {"queued", "preparing"} and _prefetch_is_stale(prefetch):
        return {"offset": int(offset), "status": "stale"}
    try:
        start_offset = int(prefetch.get("start_offset", -1))
    except (TypeError, ValueError):
        start_offset = -1
    if start_offset >= 0 and int(offset) >= start_offset and status in {"queued", "preparing"}:
        return {"offset": int(offset), "status": "preparing"}
    if start_offset >= 0 and int(offset) >= start_offset and status == "failed":
        return {
            "offset": int(offset),
            "status": "failed",
            "error": str(prefetch.get("error", "")),
        }
    return {}


def _set_prefetch_status(
    active: ActiveQuiz,
    *,
    status: str,
    start_offset: int | None = None,
    error: str | None = None,
) -> None:
    state = _active_batch_state(active)
    prefetch = state.get("prefetch")
    if not isinstance(prefetch, dict) or "batches" not in prefetch:
        prefetch = {"batches": []}
    if start_offset is not None:
        previous_start = prefetch.get("start_offset")
        try:
            previous_start_int = int(previous_start)
        except (TypeError, ValueError):
            previous_start_int = int(start_offset)
        prefetch["start_offset"] = min(previous_start_int, int(start_offset))
    prefetch["status"] = str(status)
    prefetch["updated_at"] = _utcnow().isoformat()
    if error:
        prefetch["error"] = str(error)[:500]
    else:
        prefetch.pop("error", None)
    state["prefetch"] = prefetch
    active.batch_state_json = json.dumps(state, ensure_ascii=False)


def _store_prefetched_batch(
    active: ActiveQuiz,
    *,
    offset: int,
    batch: list[dict[str, Any]],
    generated: bool,
) -> None:
    state = _active_batch_state(active)
    prefetch = state.get("prefetch")
    if not isinstance(prefetch, dict) or "batches" not in prefetch:
        prefetch = {
            "start_offset": int(offset),
            "status": "preparing",
            "batches": [],
        }
    batches = [
        item
        for item in prefetch.get("batches", [])
        if isinstance(item, dict) and int(item.get("offset", -1)) != int(offset)
    ]
    batches.append(
        {
            "offset": int(offset),
            "status": "ready",
            "generated": bool(generated),
            "batch": batch,
        }
    )
    batches.sort(key=lambda item: int(item.get("offset", -1)))
    prefetch["batches"] = batches
    prefetch["generated"] = bool(prefetch.get("generated")) or bool(generated)
    prefetch["status"] = "preparing"
    prefetch["updated_at"] = _utcnow().isoformat()
    prefetch.pop("error", None)
    state["prefetch"] = prefetch
    active.batch_state_json = json.dumps(state, ensure_ascii=False)


def _consume_prefetched_batch(
    active: ActiveQuiz,
    *,
    offset: int,
) -> tuple[list[dict[str, Any]], bool]:
    state = _active_batch_state(active)
    prefetch = state.get("prefetch")
    if not isinstance(prefetch, dict):
        return [], False

    selected: dict[str, Any] | None = None
    remaining: list[dict[str, Any]] = []
    for item in _prefetch_batches(state):
        try:
            item_offset = int(item.get("offset", -1))
        except (TypeError, ValueError):
            item_offset = -1
        if item_offset == int(offset) and selected is None:
            selected = item
        else:
            remaining.append(item)

    # Consume the legacy single-batch shape if this quiz was started before an app reload.
    if selected is None:
        try:
            legacy_offset = int(prefetch.get("offset", -1))
        except (TypeError, ValueError):
            legacy_offset = -1
        if legacy_offset == int(offset) and isinstance(prefetch.get("batch"), list):
            selected = prefetch
            prefetch.pop("offset", None)
            prefetch.pop("batch", None)

    if selected is None:
        return [], False

    batch = selected.get("batch")
    if not isinstance(batch, list) or not all(isinstance(question, dict) for question in batch):
        return [], False

    prefetch["batches"] = remaining
    prefetch["updated_at"] = _utcnow().isoformat()
    state["prefetch"] = prefetch
    active.batch_state_json = json.dumps(state, ensure_ascii=False)
    return batch, bool(selected.get("generated"))


def _prefetched_private_questions(state: dict[str, Any]) -> list[dict[str, Any]]:
    private_questions: list[dict[str, Any]] = []
    for item in sorted(_prefetch_batches(state), key=lambda row: int(row.get("offset", -1))):
        batch = item.get("batch")
        if item.get("status") == "ready" and isinstance(batch, list):
            private_questions.extend(question for question in batch if isinstance(question, dict))
    return private_questions


def _schedule_next_batch_prefetch(*, user_id: int, quiz_id: str) -> None:
    """Start preparing every remaining quiz question after the first five are returned."""
    active = ActiveQuiz.query.filter_by(
        user_id=int(user_id),
        quiz_id=str(quiz_id),
        status="active",
    ).first()
    if active is None:
        return
    try:
        current_paper = json.loads(active.paper_json or "[]")
    except (TypeError, ValueError):
        return
    if not isinstance(current_paper, list):
        return

    target = int(active.target_question_count or len(current_paper))
    offset = len(current_paper)
    if offset >= target:
        locked, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
        if locked is not None:
            _set_prefetch_status(locked, status="complete", start_offset=offset)
            db.session.commit()
        return

    state = _active_batch_state(active)
    current = _batch_prefetch_for_offset(state, offset)
    prefetch = state.get("prefetch") if isinstance(state.get("prefetch"), dict) else {}
    if current.get("status") == "ready" and str(prefetch.get("status", "")) == "complete":
        return

    if not prefetch or str(prefetch.get("status", "")) in {"failed", "idle", "stale"} or _prefetch_is_stale(prefetch):
        locked, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
        if locked is None:
            return
        _set_prefetch_status(locked, status="queued", start_offset=offset)
        db.session.commit()
        active = locked

    key = (int(user_id), str(quiz_id))
    with _batch_prefetch_guard:
        if key in _batch_prefetch_inflight:
            return
        _batch_prefetch_inflight.add(key)

    _quiz_log(
        "quiz.prefetch.scheduled",
        user_id=int(user_id),
        quiz_id=str(quiz_id),
        quiz_kind=active.quiz_kind,
        batch_offset=offset,
        outcome="queued",
    )
    try:
        _batch_prefetch_executor.submit(
            _run_next_batch_prefetch,
            int(user_id),
            str(quiz_id),
            offset,
        )
    except Exception:
        with _batch_prefetch_guard:
            _batch_prefetch_inflight.discard(key)
        raise


def _run_next_batch_prefetch(user_id: int, quiz_id: str, offset: int) -> None:
    """Prepare all remaining questions privately while the student answers the first five."""
    with app.app_context():
        try:
            active, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
            if active is None:
                return
            _set_prefetch_status(active, status="preparing", start_offset=int(offset))
            quiz_kind = active.quiz_kind
            db.session.commit()
            _quiz_log(
                "quiz.prefetch.started",
                user_id=int(user_id),
                quiz_id=str(quiz_id),
                quiz_kind=quiz_kind,
                batch_offset=int(offset),
                outcome="running",
            )

            while True:
                active = ActiveQuiz.query.filter_by(
                    user_id=int(user_id),
                    quiz_id=str(quiz_id),
                    status="active",
                ).first()
                if active is None:
                    return
                try:
                    current_paper = json.loads(active.paper_json or "[]")
                except (TypeError, ValueError):
                    current_paper = []
                if not isinstance(current_paper, list):
                    current_paper = []

                target = int(active.target_question_count or len(current_paper))
                if len(current_paper) >= target:
                    locked, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
                    if locked is not None:
                        _set_prefetch_status(locked, status="complete", start_offset=int(offset))
                        db.session.commit()
                    _quiz_log(
                        "quiz.prefetch.completed",
                        user_id=int(user_id),
                        quiz_id=str(quiz_id),
                        quiz_kind=active.quiz_kind,
                        batch_offset=int(offset),
                        question_count=target,
                        outcome="complete",
                    )
                    return

                state = _active_batch_state(active)
                prepared_offsets = {
                    int(item.get("offset", -1))
                    for item in _prefetch_batches(state)
                    if item.get("status") == "ready"
                }
                next_offset: int | None = None
                candidate = len(current_paper)
                while candidate < target:
                    if candidate not in prepared_offsets:
                        next_offset = candidate
                        break
                    candidate += QUIZ_BATCH_SIZE

                if next_offset is None:
                    locked, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
                    if locked is not None:
                        _set_prefetch_status(locked, status="complete", start_offset=int(offset))
                        db.session.commit()
                    _quiz_log(
                        "quiz.prefetch.completed",
                        user_id=int(user_id),
                        quiz_id=str(quiz_id),
                        quiz_kind=active.quiz_kind,
                        batch_offset=int(offset),
                        question_count=target,
                        outcome="complete",
                    )
                    return

                batch_count = min(QUIZ_BATCH_SIZE, target - next_offset)
                private_context = list(current_paper) + _prefetched_private_questions(state)
                lock_name = _quiz_prefetch_lock_name(int(user_id), str(quiz_id))
                _quiz_log(
                    "quiz.prefetch.batch_started",
                    user_id=int(user_id),
                    quiz_id=str(quiz_id),
                    quiz_kind=active.quiz_kind,
                    batch_offset=next_offset,
                    question_count=batch_count,
                    outcome="preparing",
                )

                with generation_lock(lock_name) as acquired:
                    if not acquired:
                        # Another process is preparing this student's remaining quiz.
                        return

                    active = ActiveQuiz.query.filter_by(
                        user_id=int(user_id),
                        quiz_id=str(quiz_id),
                        status="active",
                    ).first()
                    if active is None:
                        return
                    state = _active_batch_state(active)
                    if _batch_prefetch_for_offset(state, next_offset).get("status") == "ready":
                        continue

                    generated = False
                    if active.quiz_kind == "practice":
                        batch, generated = _practice_batch_from_pool_or_generate(
                            user_id=int(user_id),
                            topic_id=str(state.get("topic_id", "")),
                            topic_name=str(state.get("topic_name", "")),
                            grade=int(state.get("grade", 0) or 0),
                            level_id=int(state.get("level_id", 3) or 3),
                            difficulty_level=str(state.get("difficulty_level", "Apply")),
                            count=batch_count,
                            active_paper=private_context,
                        )
                    elif active.quiz_kind == "diagnostic":
                        syllabus = state.get("syllabus")
                        if not isinstance(syllabus, dict):
                            raise RuntimeError("Diagnostic batch plan is unavailable.")
                        mappings = list(syllabus.get("topic_mappings", []))[
                            next_offset : next_offset + batch_count
                        ]
                        if len(mappings) != batch_count:
                            raise RuntimeError("Diagnostic batch plan is incomplete.")
                        batch, generated = _diagnostic_batch_from_pool_or_generate(
                            user_id=int(user_id),
                            full_syllabus=syllabus,
                            mappings=mappings,
                            active_paper=private_context,
                        )
                    else:
                        return

                # Runtime generation may release the scoped SQLAlchemy session. Lock the
                # ActiveQuiz row only for the short JSON state mutation so a concurrent
                # next-batch request cannot overwrite a just-prepared private batch.
                active, shown_paper = _active_quiz_for_update(int(user_id), str(quiz_id))
                if active is None:
                    return
                if next_offset < len(shown_paper):
                    db.session.rollback()
                    continue

                state = _active_batch_state(active)
                if _batch_prefetch_for_offset(state, next_offset).get("status") != "ready":
                    _store_prefetched_batch(
                        active,
                        offset=next_offset,
                        batch=batch,
                        generated=generated,
                    )
                    db.session.commit()
                    _quiz_log(
                        "quiz.prefetch.batch_ready",
                        user_id=int(user_id),
                        quiz_id=str(quiz_id),
                        quiz_kind=active.quiz_kind,
                        batch_offset=next_offset,
                        question_count=len(batch),
                        outcome="generated" if generated else "pool",
                    )
                else:
                    db.session.rollback()

        except Exception as exc:
            db.session.rollback()
            try:
                active, _ = _active_quiz_for_update(int(user_id), str(quiz_id))
                if active is not None:
                    _set_prefetch_status(
                        active,
                        status="failed",
                        start_offset=int(offset),
                        error="Background preparation failed; it will be retried.",
                    )
                    db.session.commit()
            except Exception:
                db.session.rollback()
            _quiz_log(
                "quiz.prefetch.failed",
                level="exception",
                user_id=int(user_id),
                quiz_id=str(quiz_id),
                batch_offset=int(offset),
                outcome="failed",
            )
        finally:
            with _batch_prefetch_guard:
                _batch_prefetch_inflight.discard((int(user_id), str(quiz_id)))
            db.session.remove()


def _diagnostic_payload_from_pool(user: User) -> dict[str, Any]:
    from rag.adaptive import QUIZ_SIZE

    user_id = _orm_identity_id(user)
    user = db.session.get(User, user_id)
    if user is None:
        raise RuntimeError("Student account no longer exists.")

    active, active_paper = _load_active_quiz(user)
    state = _active_batch_state(active) if active is not None else {}
    if (
        active is not None
        and active.quiz_kind == "diagnostic"
        and int(active.target_question_count or 0) == QUIZ_SIZE
        and state.get("batch_mode") == "hybrid_pool"
        and active_paper
    ):
        return {
            "status": "ok",
            "paper": _public_paper(active_paper),
            "count": len(active_paper),
            "quiz_id": active.quiz_id,
            "mode": "hybrid_pool",
            "cached": True,
            "timer": _quiz_timer_payload(active),
            **_batch_response_metadata(active, len(active_paper)),
        }

    syllabus = _student_diagnostic_syllabus(user)
    first_mappings = list(syllabus.get("topic_mappings", []))[:QUIZ_BATCH_SIZE]
    batch, generated = _diagnostic_batch_from_pool_or_generate(
        user_id=user_id,
        full_syllabus=syllabus,
        mappings=first_mappings,
    )
    active, private_paper = _store_active_quiz(
        user,
        batch,
        "diagnostic",
        time_limit_minutes=config.DIAGNOSTIC_TIME_LIMIT_MINUTES,
        target_question_count=QUIZ_SIZE,
        batch_state={
            "batch_mode": "hybrid_pool",
            "quiz_kind": "diagnostic",
            "syllabus": syllabus,
        },
    )
    return {
        "status": "ok",
        "paper": _public_paper(private_paper),
        "count": len(private_paper),
        "quiz_id": active.quiz_id,
        "mode": "hybrid_generated" if generated else "pool",
        "cached": False,
        "timer": _quiz_timer_payload(active),
        **_batch_response_metadata(active, len(private_paper)),
    }


def _artifact_versions() -> dict[str, str]:
    from rag.diagnostic import VALIDATOR_VERSION
    from rag.ingest import read_ingest_manifest
    from rag.prompts import PROMPT_VERSION
    from rag.question_bank import question_bank_version

    ingest_manifest = read_ingest_manifest()
    return {
        "model_id": _configured_model_id(),
        "generation_provider": "runpod",
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "embedding_provider": config.HF_EMBEDDING_PROVIDER,
        "question_bank_version": question_bank_version(),
        "corpus_version": config.CORPUS_VERSION or str(
            ingest_manifest.get("corpus_version", "unindexed")
        ),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "chunker_version": config.CHUNKER_VERSION,
    }


def _diagnostic_fingerprint(syllabus: dict[str, Any]) -> str:
    source_files = []
    for configured_dir in config.QUIZ_REFERENCE_DIRS:
        root = config.resolve_reference_dir(configured_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and config.include_quiz_reference_file(path):
                stat = path.stat()
                try:
                    relative_path = str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
                except ValueError:
                    relative_path = path.name
                source_files.append((relative_path, stat.st_size, stat.st_mtime_ns))
    payload = {
        "syllabus": syllabus,
        "source_files": source_files,
        "versions": _artifact_versions(),
        "embedding_dimensions": config.HF_EMBEDDING_DIMENSIONS,
        "generation_temperature": config.RUNPOD_GENERATION_TEMPERATURE,
        "generation_max_tokens": config.RUNPOD_MCQ_MAX_NEW_TOKENS,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generate_and_save_paper(user: User | None = None, *, force: bool = False) -> dict:
    if not _has_llm_config():
        raise RuntimeError(
            "RunPod MCQ generation access is not fully configured."
        )
    from rag.adaptive import POLICY_VERSION, QUIZ_SIZE
    from rag.validation import has_current_review

    # Long-running inference deliberately clears the scoped SQLAlchemy session.
    # A User object retained by the caller can therefore be detached when this
    # function is called again. Recover its persistent primary-key identity
    # without refreshing the detached object, then load an attached instance
    # in the current session before any user attributes are accessed.
    if user is not None:
        user_id = _orm_identity_id(user)
        user = db.session.get(User, user_id)
        if user is None:
            raise RuntimeError("Student account no longer exists.")

    identity = _quiz_start_lock_name(_orm_identity_id(user)) if user else "diagnostic-preview"
    with generation_lock(identity) as acquired:
        if not acquired:
            raise QuizPreparationInProgress("diagnostic")
        if user is not None and not force:
            active, paper = _load_active_quiz(user)
            if (active is not None and active.quiz_kind == "diagnostic" and len(paper) == QUIZ_SIZE
                    and all(q.get("adaptive_policy_version") == POLICY_VERSION and has_current_review(q) for q in paper)):
                return {
                    "status": "ok",
                    "paper": _public_paper(paper),
                    "count": len(paper),
                    "quiz_id": active.quiz_id,
                    "mode": "rag",
                    "cached": True,
                    "timer": _quiz_timer_payload(active),
                }
        syllabus = _student_diagnostic_syllabus(user, variant=secrets.token_hex(16) if force and user else "")
        fingerprint = _diagnostic_fingerprint(syllabus)
        return _generate_and_save_paper_locked(
            user,
            force=force,
            syllabus=syllabus,
            fingerprint=fingerprint,
        )


def _quiz_payload(
    paper: list[dict[str, Any]],
    *,
    user: User | None,
    cached: bool,
) -> dict[str, Any]:
    quiz_id = ""
    timer = {"enabled": False}
    if user is not None:
        active, paper = _store_active_quiz(
            user,
            paper,
            "diagnostic",
            time_limit_minutes=config.DIAGNOSTIC_TIME_LIMIT_MINUTES,
        )
        quiz_id = active.quiz_id
        timer = _quiz_timer_payload(active)
    return {
        "status": "ok",
        "paper": _public_paper(paper) if user is not None else paper,
        "count": len(paper),
        "quiz_id": quiz_id,
        "mode": "rag",
        "cached": cached,
        "timer": timer,
    }


def _template_cache_key(fingerprint: str) -> str:
    return cache_key("assessment-template", fingerprint)


def _load_cached_template(
    fingerprint: str,
    syllabus: dict[str, Any],
) -> list[dict[str, Any]] | None:
    from rag.diagnostic import validate_diagnostic_paper

    versions = _artifact_versions()
    redis_key = _template_cache_key(fingerprint)
    cached_payload = cache_get_json(redis_key)
    if isinstance(cached_payload, dict) and cached_payload.get("versions") == versions:
        paper = cached_payload.get("paper")
        if isinstance(paper, list):
            try:
                validate_diagnostic_paper(paper, syllabus)
                return paper
            except Exception:
                cache_delete(redis_key)

    cutoff = _utcnow() - timedelta(hours=config.DIAGNOSTIC_CACHE_HOURS)
    cached = AssessmentTemplate.query.filter_by(fingerprint=fingerprint).first()
    if not cached or cached.created_at < cutoff:
        return None
    stored_versions = {
        "model_id": cached.model_id,
        "generation_provider": cached.generation_provider,
        "embedding_model": cached.embedding_model,
        "embedding_provider": cached.embedding_provider,
        "question_bank_version": cached.question_bank_version,
        "corpus_version": cached.corpus_version,
        "prompt_version": cached.prompt_version,
        "validator_version": cached.validator_version,
        "chunker_version": cached.chunker_version,
    }
    try:
        paper = json.loads(cached.paper_json)
        if stored_versions != versions or not isinstance(paper, list):
            raise ValueError("Cached template version mismatch.")
        validate_diagnostic_paper(paper, syllabus)
    except Exception:
        db.session.delete(cached)
        db.session.commit()
        cache_delete(redis_key)
        return None
    cache_set_json(
        redis_key,
        {"paper": paper, "versions": versions},
        config.DIAGNOSTIC_CACHE_HOURS * 3600,
    )
    return paper


def _generate_and_save_paper_locked(
    user: User | None = None,
    *,
    force: bool = False,
    syllabus: dict[str, Any],
    fingerprint: str,
) -> dict:
    if not force:
        paper = _load_cached_template(fingerprint, syllabus)
        if paper is not None:
            return _quiz_payload(paper, user=user, cached=True)
    else:
        cache_delete(_template_cache_key(fingerprint))

    try:
        from rag.diagnostic import validate_diagnostic_paper
        paper, draft_id = _generate_with_draft(user, "diagnostic", "diagnostic", syllabus, {
            "exclude_question_ids_by_topic": {
                mapping["topic_id"]: set(mapping.get("excluded_source_question_ids", []))
                for mapping in syllabus["topic_mappings"]
            },
            "question_seed": syllabus.get("selection_seed", ""),
        })
        validate_diagnostic_paper(paper, syllabus)
        versions = _artifact_versions()
        cached = AssessmentTemplate.query.filter_by(fingerprint=fingerprint).first()
        if not cached:
            cached = AssessmentTemplate(fingerprint=fingerprint)
            db.session.add(cached)
        cached.paper_json = json.dumps(paper, ensure_ascii=False)
        cached.model_id = versions["model_id"]
        cached.generation_provider = versions["generation_provider"]
        cached.embedding_model = versions["embedding_model"]
        cached.embedding_provider = versions["embedding_provider"]
        cached.question_bank_version = versions["question_bank_version"]
        cached.corpus_version = versions["corpus_version"]
        cached.prompt_version = versions["prompt_version"]
        cached.validator_version = versions["validator_version"]
        cached.chunker_version = versions["chunker_version"]
        cached.created_at = _utcnow()
        db.session.commit()
        cache_set_json(
            _template_cache_key(fingerprint),
            {"paper": paper, "versions": versions},
            config.DIAGNOSTIC_CACHE_HOURS * 3600,
        )
        payload = _quiz_payload(paper, user=user, cached=False)
        _clear_generation_draft(draft_id)
        return payload
    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Textbook-grounded diagnostic generation failed")
        raise RuntimeError(f"Textbook-grounded diagnostic generation failed: {exc}") from exc


def _generate_with_draft(user, quiz_kind, scope, syllabus, retrieval_options):
    from rag.generator import generate_paper
    from rag.retriever import retrieve_for_syllabus
    from rag.quality import deduplicate_questions
    from rag.validation import has_current_review

    if user is None:
        ensure_inference_available()
        return generate_paper(syllabus, retrieve_for_syllabus(syllabus, **retrieval_options)), None
    fingerprint = _diagnostic_fingerprint({**syllabus, "draft_user_id": user.id})
    draft = GenerationDraft.query.filter_by(user_id=user.id, quiz_kind=quiz_kind, scope=scope).first()
    state = {}
    if draft and draft.fingerprint == fingerprint and draft.updated_at >= _utcnow() - timedelta(hours=config.DIAGNOSTIC_CACHE_HOURS):
        try:
            state = json.loads(draft.state_json)
            if not isinstance(state, dict) or not isinstance(state.get("content"), dict) or not isinstance(state.get("paper"), list):
                state = {}
        except (TypeError, ValueError):
            state = {}
    if not state:
        ensure_inference_available()
        state = {"content": retrieve_for_syllabus(syllabus, **retrieval_options), "paper": [], "round": 0}
    state["round"] = int(state.get("round", 0)) + 1
    if draft is None:
        draft = GenerationDraft(user_id=user.id, quiz_kind=quiz_kind, scope=scope)
        db.session.add(draft)
    draft.fingerprint = fingerprint
    draft.state_json = json.dumps(state, ensure_ascii=False)
    draft.updated_at = _utcnow()
    db.session.commit()
    draft_id = draft.id
    # Generation can take several minutes. Do not keep a request-scoped Neon
    # SQLAlchemy session alive while waiting on remote inference; a fresh session
    # is acquired only when progress or the final quiz is persisted.
    db.session.remove()
    progress_lock = threading.Lock()

    def save_progress(questions):
        with progress_lock:
            accepted = [question for question in questions if has_current_review(question)]
            updated = deduplicate_questions(state["paper"] + accepted, similarity_threshold=config.MCQ_SIMILARITY_THRESHOLD)
            if updated == state["paper"]:
                return
            state["paper"] = updated
            with app.app_context():
                GenerationDraft.query.filter_by(id=draft_id, fingerprint=fingerprint).update({
                    "state_json": json.dumps(state, ensure_ascii=False), "updated_at": _utcnow(),
                }, synchronize_session=False)
                db.session.commit()

    try:
        paper = generate_paper(syllabus, state["content"], initial_paper=state["paper"],
                               on_progress=save_progress, retry_offset=(state["round"] - 1) * 7)
        save_progress(paper)
        return paper, draft_id
    except Exception as exc:
        # Ensure teardown never tries to rollback a connection that became stale
        # during a long inference request. Progress has already been committed by
        # save_progress using short-lived application contexts.
        db.session.remove()
        if access_failure(exc) is not None:
            raise
        count = len(state["paper"])
        raise RuntimeError(
            f"{exc} Validated progress saved: {count}/{syllabus.get('total_questions', 0)}. "
            "Retry to generate the missing questions."
        ) from exc


def _clear_generation_draft(draft_id):
    if draft_id is not None:
        GenerationDraft.query.filter_by(id=draft_id).delete(synchronize_session=False)
        db.session.commit()


class QuizPreparationInProgress(RuntimeError):
    code = "quiz_preparing"

    def __init__(self, quiz_kind: str):
        kind = "practice quiz" if str(quiz_kind) == "practice" else "diagnostic quiz"
        super().__init__(f"Your {kind} is already being prepared. Please wait a moment.")


def _quiz_preparation_response(exc):
    current = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, QuizPreparationInProgress):
            return (
                jsonify(
                    {
                        "status": "preparing",
                        "code": current.code,
                        "message": str(current),
                        "retry_after_ms": 2000,
                    }
                ),
                202,
            )
        current = current.__cause__ or current.__context__
    return None


def _inference_failure_response(exc):
    failure = access_failure(exc)
    if failure is None:
        return None
    code = str(getattr(failure, "code", "inference_unavailable"))
    messages = {
        "inference_credits_exhausted": (
            "Question generation is temporarily unavailable. Please try again shortly."
        ),
        "generation_budget_exhausted": (
            "Question generation is taking longer than expected. Please try again shortly."
        ),
        "hf_embeddings_unavailable": (
            "The learning-content service is temporarily unavailable. Please try again shortly."
        ),
    }
    response = jsonify(
        {
            "status": "error",
            "code": code,
            "message": messages.get(
                code,
                "Question generation is temporarily unavailable. Please try again shortly.",
            ),
        }
    )
    response.status_code = 503
    if getattr(failure, "retry_after", None):
        response.headers["Retry-After"] = str(failure.retry_after)
    return response


def _normalize_email(raw_email: str, *, require_deliverable: bool = False) -> str:
    try:
        result = validate_email(
            str(raw_email or "").strip(),
            check_deliverability=(
                require_deliverable and config.EMAIL_CHECK_DELIVERABILITY
            ),
        )
        return result.normalized.lower()
    except EmailNotValidError as exc:
        message = (
            "Enter a real email address that can receive verification messages."
            if require_deliverable and config.EMAIL_CHECK_DELIVERABILITY
            else "A valid email address is required."
        )
        raise ValueError(message) from exc


def _set_authenticated_session(user: User) -> None:
    session.clear()
    session.permanent = True
    session["user_id"] = int(user.id)
    session["session_version"] = int(user.session_version or 0)


def _current_user() -> User | None:
    raw_user_id = session.get("user_id")
    if not isinstance(raw_user_id, int):
        return None
    user = db.session.get(User, raw_user_id)
    if not user:
        session.clear()
        return None
    if int(session.get("session_version", -1)) != int(user.session_version or 0):
        session.clear()
        return None
    if user.email_verified_at is None:
        session.clear()
        return None
    return user


def _is_admin(user: User | None) -> bool:
    return bool(user and user.email.lower() in config.ADMIN_EMAILS)


def _user_or_ip_rate_key() -> str:
    user_id = session.get("user_id")
    if isinstance(user_id, int):
        return f"user:{user_id}"
    return f"ip:{get_remote_address()}"


def _quiz_start_lock_name(user_id: int) -> str:
    """Serialize only foreground quiz creation for one student.

    Background prefetch deliberately uses a quiz-scoped lock so a long-running
    prefetch from an older active quiz cannot block a new foreground quiz
    request for the same student, while different students always use distinct
    lock identities.
    """
    return f"quiz-start-student-{int(user_id)}"


def _quiz_prefetch_lock_name(user_id: int, quiz_id: str) -> str:
    """Lock one background prefetch job without contending with quiz startup."""
    return f"quiz-prefetch-student-{int(user_id)}-quiz-{str(quiz_id)}"


def _issue_auth_token(user: User, purpose: str, lifetime: timedelta) -> str:
    now = _utcnow()
    # Serialize token rotation per account so concurrent resend requests cannot
    # leave more than one usable token for the same purpose.
    db.session.query(User.id).filter(User.id == user.id).with_for_update().one()
    AuthToken.query.filter_by(user_id=user.id, purpose=purpose, used_at=None).update(
        {"used_at": now}, synchronize_session=False
    )
    raw_token, token_digest = new_one_time_token()
    db.session.add(
        AuthToken(
            user_id=user.id,
            purpose=purpose,
            token_hash=token_digest,
            expires_at=now + lifetime,
        )
    )
    return raw_token


def _active_auth_token(raw_token: str, purpose: str) -> AuthToken | None:
    if not raw_token or len(raw_token) > 512:
        return None
    now = _utcnow()
    return AuthToken.query.filter(
        AuthToken.token_hash == hash_token(raw_token),
        AuthToken.purpose == purpose,
        AuthToken.used_at.is_(None),
        AuthToken.expires_at > now,
    ).first()


def _issue_password_reset_otp(user: User) -> str:
    """Invalidate earlier reset codes and store only an HMAC of the new OTP."""
    now = _utcnow()
    # Lock the account row while rotating reset codes. This keeps simultaneous
    # forgot-password requests from creating two valid OTPs in PostgreSQL.
    db.session.query(User.id).filter(User.id == user.id).with_for_update().one()
    AuthToken.query.filter_by(
        user_id=user.id, purpose="reset_password", used_at=None
    ).update({"used_at": now}, synchronize_session=False)

    raw_otp = new_numeric_otp(config.PASSWORD_RESET_OTP_LENGTH)
    _unused_raw, unique_placeholder = new_one_time_token()
    token = AuthToken(
        user_id=user.id,
        purpose="reset_password",
        token_hash=unique_placeholder,
        expires_at=now + timedelta(minutes=config.PASSWORD_RESET_TTL_MINUTES),
        failed_attempts=0,
    )
    db.session.add(token)
    db.session.flush()
    token.token_hash = hash_otp(raw_otp, user_id=user.id, token_id=token.id)
    return raw_otp


def _check_password_reset_otp(user: User | None, raw_otp: str) -> AuthToken | None:
    """Constant-time OTP verification with expiry, one-time use and attempt lockout."""
    code = str(raw_otp or "").strip()
    now = _utcnow()
    if user is None:
        hmac.compare_digest(
            hash_otp(code, user_id=0, token_id=0),
            _DUMMY_OTP_HASH,
        )
        return None

    token = (
        AuthToken.query.filter(
            AuthToken.user_id == user.id,
            AuthToken.purpose == "reset_password",
            AuthToken.used_at.is_(None),
            AuthToken.expires_at > now,
            AuthToken.failed_attempts < config.PASSWORD_RESET_MAX_ATTEMPTS,
        )
        .order_by(AuthToken.created_at.desc(), AuthToken.id.desc())
        .with_for_update()
        .first()
    )
    if token is None:
        hmac.compare_digest(
            hash_otp(code, user_id=user.id, token_id=0),
            _DUMMY_OTP_HASH,
        )
        return None

    expected = hash_otp(code, user_id=user.id, token_id=token.id)
    if not (
        len(code) == config.PASSWORD_RESET_OTP_LENGTH
        and code.isascii()
        and code.isdigit()
        and hmac.compare_digest(expected, token.token_hash)
    ):
        token.failed_attempts = int(token.failed_attempts or 0) + 1
        if token.failed_attempts >= config.PASSWORD_RESET_MAX_ATTEMPTS:
            token.used_at = now
        db.session.commit()
        return None
    return token


def _json_body() -> dict[str, Any]:
    value = request.get_json(silent=True)
    return value if isinstance(value, dict) else {}


def _public_error_message(exc: Exception, fallback: str) -> str:
    """Return a safe user-facing message; detailed exceptions stay in server logs."""
    return fallback


def _auth_rate_key() -> str:
    body = _json_body()
    return rate_limit_identity(get_remote_address(), body.get("email", ""))


_DUMMY_PASSWORD_HASH = generate_password_hash("NotARealPassword!2026")
_DUMMY_OTP_HASH = hash_otp("000000", user_id=0, token_id=0)


class DBTrackerAdapter:
    """Adapter so existing guidance analyzer/recommender can use DB data."""

    def __init__(self, user: User, attempt_id: int | None = None):
        self.user = user
        self.attempt_id = attempt_id
        attempts = Attempt.query.filter_by(user_id=user.id).order_by(Attempt.created_at.asc()).all()
        self.data = {
            "attempts": [
                {
                    "score_percent": a.score_percent,
                    "mistakes": [],
                }
                for a in attempts
            ]
        }

    def get_all_mistakes(self) -> list[dict]:
        query = (
            Mistake.query.join(Attempt, Mistake.attempt_id == Attempt.id)
            .filter(Attempt.user_id == self.user.id)
        )
        if self.attempt_id is not None:
            query = query.filter(Mistake.attempt_id == self.attempt_id)
        rows = query.all()
        return [
            {
                "question_index": m.question_index,
                "question": m.question,
                "topic_id": m.topic_id,
                "difficulty_level": m.difficulty_level,
                "correct_answer": m.correct_answer,
                "student_answer": m.student_answer,
            }
            for m in rows
        ]


class ConceptMapper:
    """Maps a grade-11 weak topic to lower-grade chapters using concept_mapping.json."""

    def __init__(self):
        self.concept_mapping = self._safe_load(config.DATA_DIR / "concept_mapping.json", default=[])
        self.topics_data = self._safe_load(config.TOPICS_FILE, default={})
        self.grade11_competencies = self._build_grade11_competency_index()

    @staticmethod
    def _safe_load(path: Path, default: Any) -> Any:
        try:
            return _load_json(path)
        except Exception:
            return default

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    @staticmethod
    def _normalize_comp(comp: str) -> str:
        # "6.4 (a part)" -> "6.4"
        m = re.search(r"\d+\.\d+", str(comp or ""))
        return m.group(0) if m else str(comp or "").strip()

    def _build_grade11_competency_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        grades = self.topics_data.get("grades", []) if isinstance(self.topics_data, dict) else []
        for grade in grades:
            if grade.get("grade") != 11:
                continue
            for term in grade.get("terms", []):
                for topic in term.get("topics", []):
                    name = self._norm(topic.get("topic", ""))
                    if not name:
                        continue
                    levels = {
                        self._normalize_comp(level)
                        for level in topic.get("competency_levels", [])
                        if str(level or "").strip()
                    }
                    if levels:
                        index.setdefault(name, set()).update(levels)
        return index

    def map_topic(self, topic_name: str) -> list[dict]:
        topic_norm = self._norm(topic_name)
        if not topic_norm or not isinstance(self.concept_mapping, list):
            return []

        source_comp = self.grade11_competencies.get(topic_norm, set())
        candidates = []

        for main in self.concept_mapping:
            for strand in main.get("strands", []):
                progression = strand.get("progression", [])
                grade11_rows = [row for row in progression if row.get("grade") == 11]
                if not grade11_rows:
                    continue

                g11_topics = {self._norm(row.get("topic", "")) for row in grade11_rows}
                g11_comp = {
                    self._normalize_comp(level)
                    for row in grade11_rows
                    for level in row.get("competency_levels", [])
                    if str(level or "").strip()
                }
                matches_topic = topic_norm in g11_topics
                overlap = source_comp.intersection(g11_comp) if (source_comp and g11_comp) else set()
                overlap_count = len(overlap)
                matches_comp = overlap_count > 0
                if not (matches_topic or matches_comp):
                    continue

                focus_chapters = sorted(
                    [
                        {
                            "grade": int(row.get("grade", 0)),
                            "topic": row.get("topic", ""),
                            "competency_levels": row.get("competency_levels", []),
                        }
                        for row in progression
                        if isinstance(row.get("grade"), int) and row.get("grade") <= 11
                    ],
                    key=lambda row: row["grade"],
                    reverse=True,
                )

                candidates.append(
                    {
                        "main_topic": main.get("main_topic", ""),
                        "strand_name": strand.get("strand_name", ""),
                        "match_type": "topic" if matches_topic else "competency",
                        "score": 100 + overlap_count if matches_topic else overlap_count,
                        "focus_chapters": focus_chapters,
                    }
                )

        if not candidates:
            return []

        # If we have exact topic matches, suppress noisy competency-only fallbacks.
        has_topic_match = any(c["match_type"] == "topic" for c in candidates)
        if has_topic_match:
            candidates = [c for c in candidates if c["match_type"] == "topic"]

        # Rank and dedupe (same strand/path can appear as repeated guidance noise).
        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        deduped = []
        seen = set()
        for c in candidates:
            chapter_path = tuple(
                (row.get("grade"), row.get("topic", ""))
                for row in c.get("focus_chapters", [])
            )
            key = (c.get("main_topic", ""), c.get("strand_name", ""), chapter_path)
            if key in seen:
                continue
            seen.add(key)
            c.pop("score", None)
            deduped.append(c)
            if len(deduped) >= 3:
                break

        return deduped


@app.errorhandler(500)
def handle_500(e):
    app.logger.error(
        "Unhandled application error",
        exc_info=(type(e), e, e.__traceback__),
    )
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Internal server error."}), 500
    return render_template("index.html"), 500


@app.errorhandler(CSRFError)
def handle_csrf_error(_error):
    return jsonify({"status": "error", "message": "Security token expired. Please try again."}), 400


@app.errorhandler(429)
def handle_rate_limit(_error):
    return jsonify({"status": "error", "message": "Too many requests. Please try again later."}), 429


@app.errorhandler(413)
def handle_request_too_large(_error):
    return jsonify({"status": "error", "message": "Request payload is too large."}), 413


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; "
        "object-src 'none'; manifest-src 'self'; "
        "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' "
        "https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; connect-src 'self'",
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if _is_production:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    if request.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
@limiter.exempt
def api_health():
    """Check the dependencies required to serve authenticated diagnostics."""
    try:
        from sqlalchemy import text

        db.session.execute(text("SELECT 1"))
        components: dict[str, Any] = {"database": "ok"}
        if _is_production:
            components["redis"] = "ok" if cache_ping() else "error"
            from rag.ingest import vector_index_status
            from rag.question_bank import validate_question_bank

            components["vector_index"] = (
                "ok" if vector_index_status().get("ready") else "error"
            )
            components["question_bank"] = (
                "ok" if validate_question_bank().get("ready") else "error"
            )
        if "error" in components.values():
            return jsonify(
                {"status": "error", "service": "NeuroMath", "components": components}
            ), 503
        return jsonify(
            {"status": "ok", "service": "NeuroMath", "components": components}
        )
    except Exception:
        app.logger.exception("Application health check failed")
        return jsonify({"status": "error", "message": "Service unavailable."}), 503


@app.route("/api/csrf-token")
@limiter.exempt
def api_csrf_token():
    return jsonify({"status": "ok", "csrf_token": generate_csrf()})


@app.route("/api/signup", methods=["POST"])
@limiter.limit("5 per hour", key_func=_auth_rate_key)
def api_signup():
    body = _json_body()
    name = str(body.get("name", "")).strip()
    password = str(body.get("password", ""))

    if not name:
        return jsonify({"status": "error", "message": "Name is required."}), 400
    if len(name) > 100:
        return jsonify({"status": "error", "message": "Name must not exceed 100 characters."}), 400
    try:
        email = _normalize_email(body.get("email", ""), require_deliverable=True)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    policy = validate_password(password, email=email, name=name)
    if not policy.valid:
        return jsonify({"status": "error", "message": policy.message}), 400
    # Perform the same intentionally expensive password-hash work whether the
    # address is new or already registered, reducing timing-based enumeration.
    submitted_password_hash = generate_password_hash(password)

    generic_message = (
        "If this email can be registered, a verification message has been sent."
    )
    user = User.query.filter_by(email=email).first()
    if user and user.email_verified_at is not None:
        # Same response as a new registration prevents account enumeration.
        return jsonify({"status": "ok", "message": generic_message}), 202
    if not user:
        user = User(
            email=email,
            name=name,
            password_hash=submitted_password_hash,
            email_verified_at=None,
            session_version=0,
        )
        db.session.add(user)
        try:
            db.session.flush()
        except IntegrityError:
            # A concurrent signup for the same email may win the unique-key race.
            # Keep the public response enumeration-safe and let that request own
            # verification delivery.
            db.session.rollback()
            app.logger.info(
                "auth.signup.concurrent_duplicate",
                extra={"event": "auth.signup.concurrent_duplicate", "outcome": "duplicate"},
            )
            return jsonify({"status": "ok", "message": generic_message}), 202
        db.session.add(UserSettings(user_id=user.id, theme="dark", difficulty="medium"))

    raw_token = _issue_auth_token(
        user,
        "verify_email",
        timedelta(hours=config.EMAIL_VERIFICATION_TTL_HOURS),
    )
    db.session.commit()
    try:
        send_verification_email(user.email, user.name, raw_token)
    except EmailDeliveryError as exc:
        # Keep the same public response for every address. The account remains
        # recoverable through the rate-limited resend endpoint after delivery
        # configuration is corrected.
        app.logger.error("Signup verification delivery failed: %s", exc)
    return jsonify({"status": "ok", "message": generic_message}), 202


@app.route("/api/signin", methods=["POST"])
@limiter.limit("5 per minute;20 per hour", key_func=_auth_rate_key)
def api_signin():
    body = _json_body()
    password = str(body.get("password", ""))
    if not password:
        return jsonify({"status": "error", "message": "Password is required."}), 400
    try:
        email = _normalize_email(body.get("email", ""))
    except ValueError:
        email = "invalid@example.invalid"

    user = User.query.filter_by(email=email).first()
    candidate_hash = user.password_hash if user else _DUMMY_PASSWORD_HASH
    password_ok = check_password_hash(candidate_hash, password)
    if not user or not password_ok:
        return jsonify({"status": "error", "message": "Invalid email or password."}), 401
    if user.email_verified_at is None:
        return jsonify(
            {
                "status": "error",
                "message": "Verify your email before signing in.",
                "needs_verification": True,
            }
        ), 403

    _set_authenticated_session(user)
    return jsonify({"status": "ok", "email": user.email, "name": user.name})


@app.route("/api/signout", methods=["POST"])
def api_signout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/api/resend-verification", methods=["POST"])
@limiter.limit("3 per hour", key_func=_auth_rate_key)
def api_resend_verification():
    body = _json_body()
    generic_message = "If the account requires verification, a new message has been sent."
    try:
        email = _normalize_email(body.get("email", ""))
    except ValueError:
        return jsonify({"status": "ok", "message": generic_message})
    user = User.query.filter_by(email=email).first()
    if not user or user.email_verified_at is not None:
        return jsonify({"status": "ok", "message": generic_message})
    raw_token = _issue_auth_token(
        user,
        "verify_email",
        timedelta(hours=config.EMAIL_VERIFICATION_TTL_HOURS),
    )
    db.session.commit()
    try:
        send_verification_email(user.email, user.name, raw_token)
    except EmailDeliveryError as exc:
        app.logger.error("Verification email delivery failed: %s", exc)
    return jsonify({"status": "ok", "message": generic_message})


@app.route("/api/verify-email", methods=["POST"])
@limiter.limit("10 per hour")
def api_verify_email():
    body = _json_body()
    token = _active_auth_token(str(body.get("token", "")), "verify_email")
    if not token:
        return jsonify(
            {"status": "error", "message": "Verification link is invalid or expired."}
        ), 400
    user = db.session.get(User, token.user_id)
    if not user:
        return jsonify(
            {"status": "error", "message": "Verification link is invalid or expired."}
        ), 400
    now = _utcnow()
    consumed = AuthToken.query.filter(
        AuthToken.id == token.id,
        AuthToken.used_at.is_(None),
        AuthToken.expires_at > now,
    ).update({"used_at": now}, synchronize_session=False)
    if consumed != 1:
        db.session.rollback()
        return jsonify(
            {"status": "error", "message": "Verification link is invalid or expired."}
        ), 400
    user.email_verified_at = user.email_verified_at or now
    db.session.commit()
    return jsonify({"status": "ok", "message": "Email verified. You can now sign in."})


@app.route("/api/forgot-password", methods=["POST"])
@limiter.limit("3 per hour", key_func=_auth_rate_key)
def api_forgot_password():
    body = _json_body()
    generic_message = (
        "Password-reset OTP has been sent."
    )
    try:
        email = _normalize_email(body.get("email", ""))
    except ValueError:
        return jsonify({"status": "ok", "message": generic_message})
    user = User.query.filter_by(email=email).first()
    if user:
        raw_otp = _issue_password_reset_otp(user)
        db.session.commit()
        try:
            send_password_reset_otp(user.email, user.name, raw_otp)
        except EmailDeliveryError as exc:
            # The public response stays generic to prevent account enumeration;
            # operators still receive an actionable server-side delivery error.
            app.logger.error("Password-reset OTP delivery failed: %s", exc)
    return jsonify({"status": "ok", "message": generic_message})


@app.route("/api/reset-password", methods=["POST"])
@limiter.limit("5 per 15 minutes;20 per day", key_func=_auth_rate_key)
def api_reset_password():
    body = _json_body()
    try:
        email = _normalize_email(body.get("email", ""))
    except ValueError:
        email = "invalid@example.invalid"
    user = User.query.filter_by(email=email).first()
    token = _check_password_reset_otp(user, str(body.get("otp", "")))
    if token is None or user is None:
        return jsonify(
            {
                "status": "error",
                "message": "The reset code is invalid, expired, or has too many failed attempts.",
            }
        ), 400
    password = str(body.get("password", ""))
    policy = validate_password(password, email=user.email, name=user.name)
    if not policy.valid:
        return jsonify({"status": "error", "message": policy.message}), 400

    now = _utcnow()
    consumed = AuthToken.query.filter(
        AuthToken.id == token.id,
        AuthToken.used_at.is_(None),
        AuthToken.expires_at > now,
        AuthToken.failed_attempts < config.PASSWORD_RESET_MAX_ATTEMPTS,
    ).update({"used_at": now}, synchronize_session=False)
    if consumed != 1:
        db.session.rollback()
        return jsonify(
            {
                "status": "error",
                "message": "The reset code is invalid, expired, or has too many failed attempts.",
            }
        ), 400

    user.password_hash = generate_password_hash(password)
    user.password_changed_at = now
    user.email_verified_at = user.email_verified_at or now
    user.session_version = int(user.session_version or 0) + 1
    AuthToken.query.filter_by(user_id=user.id, purpose="reset_password", used_at=None).update(
        {"used_at": now}, synchronize_session=False
    )
    db.session.commit()
    session.clear()
    try:
        send_password_changed_notice(user.email, user.name)
    except EmailDeliveryError as exc:
        app.logger.error("Password-change notice delivery failed: %s", exc)
    return jsonify({"status": "ok", "message": "Password updated. Sign in with your new password."})


@app.route("/api/me")
def api_me():
    user = _current_user()
    if not user:
        return jsonify({"status": "ok", "email": None, "name": None})
    return jsonify(
        {
            "status": "ok",
            "email": user.email,
            "name": user.name,
            "joined": user.joined_at.isoformat() if user.joined_at else None,
        }
    )


@app.route("/api/profile")
def api_profile_get():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Not signed in."}), 401
    return jsonify(
        {
            "status": "ok",
            "email": user.email,
            "name": user.name,
            "joined": user.joined_at.isoformat() if user.joined_at else None,
        }
    )


@app.route("/api/profile", methods=["POST"])
def api_profile_update():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Not signed in."}), 401

    body = _json_body()
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify({"status": "error", "message": "Name is required."}), 400
    if len(name) > 100:
        return jsonify({"status": "error", "message": "Name must not exceed 100 characters."}), 400

    user.name = name
    db.session.commit()
    return jsonify({"status": "ok", "name": name})


@app.route("/api/settings")
def api_settings_get():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Not signed in."}), 401

    settings = _get_or_create_user_settings(user.id)

    return jsonify(
        {
            "status": "ok",
            "theme": settings.theme,
            "difficulty": settings.difficulty,
            "quiz_timer_enabled": bool(settings.quiz_timer_enabled),
        }
    )


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Not signed in."}), 401

    body = _json_body()
    theme = body.get("theme", "dark")
    difficulty = body.get("difficulty", "medium")
    quiz_timer_enabled = body.get("quiz_timer_enabled", True)
    if theme not in {"light", "dark"}:
        theme = "dark"
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    if not isinstance(quiz_timer_enabled, bool):
        quiz_timer_enabled = True

    settings = _get_or_create_user_settings(user.id)

    settings.theme = theme
    settings.difficulty = difficulty
    settings.quiz_timer_enabled = quiz_timer_enabled
    settings.updated_at = _utcnow()
    db.session.commit()

    return jsonify(
        {
            "status": "ok",
            "theme": settings.theme,
            "difficulty": settings.difficulty,
            "quiz_timer_enabled": bool(settings.quiz_timer_enabled),
        }
    )


@app.route("/api/topics")
def api_topics():
    try:
        return jsonify(_load_json(config.TOPICS_FILE))
    except Exception as exc:
        app.logger.exception("Topics API failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Topics are temporarily unavailable."),
        }), 500


@app.route("/api/question-bank/status")
def api_question_bank_status():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Please sign in first."}), 401
    if not _is_admin(user):
        return jsonify(
            {"status": "error", "message": "Administrator access is required."}
        ), 403
    try:
        from rag.question_bank import validate_question_bank

        result = validate_question_bank()
        result["directory"] = Path(result["directory"]).name
        result["files"] = [Path(path).name for path in result.get("files", [])]
        return jsonify({"status": "ok", **result})
    except Exception:
        app.logger.exception("Question-bank validation failed")
        return jsonify(
            {"status": "error", "message": "Question-bank validation failed."}
        ), 500


@app.route("/api/syllabus")
def api_syllabus():
    try:
        syllabus = _student_diagnostic_syllabus(_current_user())
        for mapping in syllabus["topic_mappings"]:
            mapping.pop("avoid_question_stems", None)
            mapping.pop("excluded_source_question_ids", None)
        return jsonify(syllabus)
    except Exception as exc:
        app.logger.exception("Syllabus API failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Syllabus is temporarily unavailable."),
        }), 500


@app.route("/api/ingest", methods=["POST"])
@limiter.limit("1 per hour", key_func=_user_or_ip_rate_key)
def api_ingest():
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Please sign in first."}), 401
    if not _is_admin(user):
        return jsonify({"status": "error", "message": "Administrator access is required."}), 403
    if config.VECTOR_STORE_PROVIDER == "qdrant" and not config.QDRANT_WRITE_API_KEY:
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Remote index rebuilding is disabled in this runtime. Run the ingestion "
                    "command from a secured administrator environment."
                ),
            }
        ), 503
    if not _has_embedding_config():
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Configure BAAI/bge-m3 Inference Provider access before "
                    "building the vector index."
                ),
            }
        ), 503
    try:
        from rag.ingest import ensure_vector_index_ready

        result = ensure_vector_index_ready(repair=True)
        return jsonify({"status": "ok", "message": result.get("message", "Ingestion complete.")})
    except Exception as exc:
        app.logger.exception("Vector ingestion failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Vector ingestion failed."),
        }), 503


@app.route("/api/generate", methods=["POST"])
@limiter.limit("3 per hour", key_func=_user_or_ip_rate_key)
def api_generate():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401
        if not _is_admin(user):
            return jsonify({"status": "error", "message": "Administrator access is required."}), 403
        return jsonify(_generate_and_save_paper(user, force=True))
    except Exception as exc:
        preparing = _quiz_preparation_response(exc)
        if preparing is not None:
            return preparing
        failure = _inference_failure_response(exc)
        if failure is not None:
            return failure
        app.logger.exception("Administrative quiz generation failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Quiz generation failed."),
        }), 503


@app.route("/api/quiz/load", methods=["POST"])
@limiter.limit("3 per hour", key_func=_user_or_ip_rate_key)
def api_quiz_load():
    """
    Ingest (no-op when sources unchanged) + generate paper in one request.
    Faster UX than chaining /api/ingest and /api/generate from the browser.
    """
    user = _current_user()
    if not user:
        return jsonify({"status": "error", "message": "Please sign in first."}), 401
    if config.QUESTION_POOL_RUNTIME_ENABLED:
        try:
            user_id = _orm_identity_id(user)
            _quiz_log(
                "quiz.diagnostic.requested",
                user_id=user_id,
                quiz_kind="diagnostic",
                outcome="started",
            )
            with generation_lock(_quiz_start_lock_name(user_id)) as acquired:
                if not acquired:
                    raise QuizPreparationInProgress("diagnostic")
                payload = _diagnostic_payload_from_pool(user)
            _schedule_next_batch_prefetch(user_id=user_id, quiz_id=str(payload.get("quiz_id", "")))
            _quiz_log(
                "quiz.diagnostic.ready",
                user_id=user_id,
                quiz_id=str(payload.get("quiz_id", "")),
                quiz_kind="diagnostic",
                question_count=int(payload.get("count", 0) or 0),
                outcome=str(payload.get("mode", "pool")),
            )
            payload["ingest"] = {
                "status": payload.get("mode", "pool"),
                "message": (
                    "Loaded from the global question pool."
                    if payload.get("mode") == "pool"
                    else "Pool shortage filled with the existing RAG pipeline; the validated batch was cached globally."
                ),
            }
            return jsonify(payload)
        except Exception as exc:
            preparing = _quiz_preparation_response(exc)
            if preparing is not None:
                return preparing
            app.logger.exception("Diagnostic pool/fallback preparation failed")
            return jsonify(
                {
                    "status": "error",
                    "code": "question_pool_insufficient",
                    "message": _public_error_message(
                        exc, "Diagnostic questions could not be prepared. Please retry shortly."
                    ),
                }
            ), 503
    try:
        from rag.ingest import ensure_vector_index_ready

        index_status = ensure_vector_index_ready(repair=config.INGEST_ON_QUIZ_LOAD)
        ingest_meta = {
            "status": "ready",
            "message": f"Vector index ready ({index_status.get('count', 0)} chunks).",
        }
    except Exception as exc:
        app.logger.exception("Diagnostic vector-index readiness check failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Diagnostic content is temporarily unavailable."),
        }), 503
    try:
        payload = _generate_and_save_paper(user)
        if isinstance(payload, dict):
            payload["ingest"] = ingest_meta
        return jsonify(payload)
    except Exception as exc:
        preparing = _quiz_preparation_response(exc)
        if preparing is not None:
            return preparing
        failure = _inference_failure_response(exc)
        if failure is not None:
            return failure
        app.logger.exception("Diagnostic generation failed")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": _public_error_message(
                        exc, "Diagnostic questions could not be prepared. Please retry shortly."
                    ),
                    "ingest": ingest_meta,
                }
            ),
            503,
        )


@app.route("/api/practice/quiz", methods=["POST"])
@limiter.limit("10 per hour", key_func=_user_or_ip_rate_key)
def api_practice_quiz():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        body = _json_body()
        try:
            grade = int(body.get("grade"))
        except Exception:
            return jsonify({"status": "error", "message": "Valid grade is required."}), 400
        topic_name = str(body.get("topic", "")).strip()
        if not topic_name:
            return jsonify({"status": "error", "message": "Topic is required."}), 400
        if len(topic_name) > 255:
            return jsonify({"status": "error", "message": "Topic is invalid."}), 400

        topic_id = _find_topic_id_for_grade_topic(grade, topic_name)
        user_id = _orm_identity_id(user)
        _quiz_log(
            "quiz.practice.requested",
            user_id=user_id,
            quiz_kind="practice",
            topic_id=topic_id,
            grade=grade,
            outcome="started",
        )
        previous_stems = _recent_question_stems(user, topic_id)
        previous_source_ids = _previous_source_question_ids(user, topic_id)
        from rag.adaptive import LEVELS, topic_progress

        progress = topic_progress(_student_memory(user)["history_by_topic"].get(topic_id, []))
        practice_time_limit_minutes = _practice_time_limit_minutes(
            grade,
            topic_name,
            progress.get("next_level"),
        )
        syllabus = {
            "syllabus_name": f"Practice: Grade {grade} - {topic_name}",
            "total_questions": 10,
            "time_limit_minutes": practice_time_limit_minutes,
            "topic_mappings": [
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "grade": grade,
                    "required_levels": [progress["next_level"]],
                    "require_application": True,
                    "weightage_percent": 100,
                    "min_questions": 10,
                    "max_questions": 10,
                    "avoid_question_stems": previous_stems,
                    "notes": f"Topic practice for Grade {grade}",
                }
            ],
        }

        if config.QUESTION_POOL_RUNTIME_ENABLED:
            try:
                difficulty = LEVELS.get(int(progress["next_level"]), "Apply")
                with generation_lock(_quiz_start_lock_name(_orm_identity_id(user))) as acquired:
                    if not acquired:
                        raise QuizPreparationInProgress("practice")
                    paper, generated = _practice_batch_from_pool_or_generate(
                        user_id=_orm_identity_id(user),
                        topic_id=topic_id,
                        topic_name=topic_name,
                        grade=grade,
                        level_id=int(progress["next_level"]),
                        difficulty_level=difficulty,
                        count=QUIZ_BATCH_SIZE,
                    )
                    # Runtime generation releases the SQLAlchemy session while inference runs.
                    user = db.session.get(User, _orm_identity_id(user))
                    if user is None:
                        raise RuntimeError("Student account no longer exists.")
                    active, private_paper = _store_active_quiz(
                        user,
                        paper,
                        "practice",
                        time_limit_minutes=practice_time_limit_minutes,
                        target_question_count=10,
                        batch_state={
                            "batch_mode": "hybrid_pool",
                            "quiz_kind": "practice",
                            "grade": grade,
                            "topic_id": topic_id,
                            "topic_name": topic_name,
                            "level_id": int(progress["next_level"]),
                            "difficulty_level": difficulty,
                        },
                    )
                payload = {
                    "status": "ok",
                    "mode": "hybrid_generated" if generated else "pool",
                    "paper": _public_paper(private_paper),
                    "count": len(private_paper),
                    "quiz_id": active.quiz_id,
                    "timer": _quiz_timer_payload(active),
                    **_batch_response_metadata(active, len(private_paper)),
                }
                _schedule_next_batch_prefetch(
                    user_id=_orm_identity_id(user),
                    quiz_id=active.quiz_id,
                )
                _quiz_log(
                    "quiz.practice.ready",
                    user_id=_orm_identity_id(user),
                    quiz_id=active.quiz_id,
                    quiz_kind="practice",
                    topic_id=topic_id,
                    grade=grade,
                    question_count=len(private_paper),
                    outcome="hybrid_generated" if generated else "pool",
                )
                return jsonify(payload)
            except Exception as exc:
                preparing = _quiz_preparation_response(exc)
                if preparing is not None:
                    return preparing
                failure = _inference_failure_response(exc)
                if failure is not None:
                    return failure
                app.logger.exception("Practice pool/fallback preparation failed")
                return jsonify(
                    {
                        "status": "error",
                        "code": "question_pool_fallback_failed",
                        "message": _public_error_message(
                            exc, "Practice questions could not be prepared. Please retry shortly."
                        ),
                    }
                ), 503

        if not _has_llm_config():
            if not config.ALLOW_DEMO_MODE:
                return jsonify(
                    {
                        "status": "error",
                        "message": "A configured mathematics LLM is required for practice generation.",
                    }
                ), 503
            from rag.demo import generate_demo_paper

            paper = generate_demo_paper(syllabus)
            active, private_paper = _store_active_quiz(
                user,
                paper,
                "practice",
                time_limit_minutes=practice_time_limit_minutes,
            )
            return jsonify(
                {
                    "status": "ok",
                    "mode": "demo",
                    "paper": _public_paper(private_paper),
                    "count": len(private_paper),
                    "quiz_id": active.quiz_id,
                    "timer": _quiz_timer_payload(active),
                    "message": "Generated practice quiz in demo mode.",
                }
            )

        try:
            from rag.ingest import ensure_vector_index_ready

            # Practice covers Grades 6-11. The diagnostic readiness check only
            # requires Grades 10/11, so verify the requested practice grade here.
            ensure_vector_index_ready(
                repair=config.INGEST_ON_QUIZ_LOAD,
                required_grades=(grade,),
            )
            with generation_lock(_quiz_start_lock_name(_orm_identity_id(user))) as acquired:
                if not acquired:
                    raise QuizPreparationInProgress("practice")
                paper, draft_id = _generate_with_draft(user, "practice", topic_id, syllabus, {
                    "exclude_question_ids_by_topic": {topic_id: previous_source_ids},
                    "question_seed": secrets.token_hex(16),
                })
                active, private_paper = _store_active_quiz(
                    user,
                    paper,
                    "practice",
                    time_limit_minutes=practice_time_limit_minutes,
                )
                _clear_generation_draft(draft_id)
            return jsonify(
                {
                    "status": "ok",
                    "mode": "rag",
                    "paper": _public_paper(private_paper),
                    "count": len(private_paper),
                    "quiz_id": active.quiz_id,
                    "timer": _quiz_timer_payload(active),
                }
            )
        except Exception as exc:
            preparing = _quiz_preparation_response(exc)
            if preparing is not None:
                return preparing
            failure = _inference_failure_response(exc)
            if failure is not None:
                return failure
            app.logger.exception("Practice RAG generation failed")
            if not config.ALLOW_DEMO_MODE:
                return jsonify(
                    {
                        "status": "error",
                        "message": _public_error_message(
                            exc, "Practice questions could not be prepared. Please retry shortly."
                        ),
                    }
                ), 503
            from rag.demo import generate_demo_paper

            paper = generate_demo_paper(syllabus)
            active, private_paper = _store_active_quiz(
                user,
                paper,
                "practice",
                time_limit_minutes=practice_time_limit_minutes,
            )
            return jsonify(
                {
                    "status": "ok",
                    "mode": "demo",
                    "paper": _public_paper(private_paper),
                    "count": len(private_paper),
                    "quiz_id": active.quiz_id,
                    "timer": _quiz_timer_payload(active),
                    "message": "Practice generation failed; using demo questions.",
                }
            )
    except Exception as exc:
        app.logger.exception("Practice quiz request failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Practice request failed."),
        }), 500


@app.route("/api/practice/catalog")
def api_practice_catalog():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401
        return jsonify({"status": "ok", "tabs": _build_practice_catalog()})
    except Exception as exc:
        app.logger.exception("Practice catalog failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Practice catalog is temporarily unavailable."),
        }), 500


@app.route("/api/quiz/next-batch", methods=["POST"])
@limiter.limit("30 per hour", key_func=_user_or_ip_rate_key)
def api_quiz_next_batch():
    """Reveal only the next prepared five questions; generation stays in the background."""
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        body = _json_body()
        quiz_id = str(body.get("quiz_id", "")).strip()
        if not 16 <= len(quiz_id) <= 64:
            return jsonify({"status": "error", "message": "A valid quiz identifier is required."}), 400
        raw_answers = body.get("answers", {})
        if not isinstance(raw_answers, dict):
            return jsonify({"status": "error", "message": "Answers must be an object."}), 400

        user_id = _orm_identity_id(user)
        active, current_paper = _load_active_quiz(user)
        if not active or not current_paper or not hmac.compare_digest(active.quiz_id, quiz_id):
            return jsonify({"status": "error", "message": "This quiz is no longer active."}), 409

        target = int(active.target_question_count or len(current_paper))
        if len(current_paper) >= target:
            return jsonify(
                {
                    "status": "ok",
                    "batch": [],
                    "quiz_id": active.quiz_id,
                    "timer": _quiz_timer_payload(active),
                    **_batch_response_metadata(active, len(current_paper)),
                }
            )

        normalized_answers: dict[str, str] = {}
        for key, value in raw_answers.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": "Invalid question index."}), 400
            answer = str(value or "").strip().upper()
            if index < 0 or index >= len(current_paper) or answer not in {"A", "B", "C", "D"}:
                return jsonify({"status": "error", "message": "Invalid batch answer."}), 400
            normalized_answers[str(index)] = answer

        missing = [index for index in range(len(current_paper)) if str(index) not in normalized_answers]
        if missing:
            return (
                jsonify(
                    {
                        "status": "error",
                        "code": "incomplete_batch",
                        "message": f"Please select an answer for Question {missing[0] + 1} before continuing.",
                        "first_unanswered": missing[0],
                        "unanswered_count": len(missing),
                    }
                ),
                400,
            )

        expected_loaded = len(current_paper)
        locked_active, locked_paper = _active_quiz_for_update(user_id, quiz_id)
        if locked_active is None or not locked_paper:
            db.session.rollback()
            return jsonify({"status": "error", "message": "This quiz is no longer active."}), 409
        if len(locked_paper) != expected_loaded:
            db.session.rollback()
            _quiz_log(
                "quiz.batch.reveal_conflict",
                user_id=user_id,
                quiz_id=quiz_id,
                quiz_kind=locked_active.quiz_kind,
                batch_offset=expected_loaded,
                outcome="state_changed",
            )
            return jsonify({
                "status": "error",
                "code": "quiz_state_changed",
                "message": "This quiz advanced in another request. Refresh the quiz before continuing.",
            }), 409
        active, current_paper = locked_active, locked_paper

        state = _active_batch_state(active)
        if state.get("batch_mode") != "hybrid_pool":
            return jsonify({"status": "error", "message": "This quiz does not support batched loading."}), 409

        offset = len(current_paper)
        batch_count = min(QUIZ_BATCH_SIZE, target - offset)
        prefetch = _batch_prefetch_for_offset(state, offset)
        if prefetch.get("status") != "ready":
            db.session.rollback()
            _schedule_next_batch_prefetch(user_id=user_id, quiz_id=quiz_id)
            return (
                jsonify(
                    {
                        "status": "preparing",
                        "code": "next_batch_preparing",
                        "message": "The remaining quiz questions are being prepared in the background.",
                        "retry_after_ms": 2000,
                        "quiz_id": active.quiz_id,
                        "timer": _quiz_timer_payload(active),
                        **_batch_response_metadata(active, len(current_paper)),
                    }
                ),
                202,
            )

        batch, generated = _consume_prefetched_batch(active, offset=offset)
        if len(batch) != batch_count or not all(isinstance(question, dict) for question in batch):
            db.session.rollback()
            _schedule_next_batch_prefetch(user_id=user_id, quiz_id=quiz_id)
            return (
                jsonify(
                    {
                        "status": "preparing",
                        "code": "next_batch_preparing",
                        "message": "The next five questions are still being prepared.",
                        "retry_after_ms": 2000,
                        "quiz_id": active.quiz_id,
                        "timer": _quiz_timer_payload(active),
                        **_batch_response_metadata(active, len(current_paper)),
                    }
                ),
                202,
            )

        appended, combined = _append_active_quiz_batch(
            active=active,
            user_id=user_id,
            paper=batch,
        )

        if active.quiz_kind == "diagnostic" and len(combined) == target:
            from rag.diagnostic import validate_diagnostic_paper

            syllabus = _active_batch_state(active).get("syllabus")
            if not isinstance(syllabus, dict):
                raise RuntimeError("Diagnostic batch plan is unavailable.")
            validate_diagnostic_paper(combined, syllabus)

        db.session.commit()
        _quiz_log(
            "quiz.batch.revealed",
            user_id=user_id,
            quiz_id=active.quiz_id,
            quiz_kind=active.quiz_kind,
            batch_offset=offset,
            question_count=len(appended),
            outcome="generated" if generated else "pool",
        )

        # Normally every later batch is already being prepared by the original
        # background worker. This call only recovers from a worker/process restart.
        if len(combined) < target:
            next_state = _active_batch_state(active)
            next_prefetch = _batch_prefetch_for_offset(next_state, len(combined))
            if next_prefetch.get("status") not in {"ready", "preparing"}:
                _schedule_next_batch_prefetch(user_id=user_id, quiz_id=active.quiz_id)

        return jsonify(
            {
                "status": "ok",
                "mode": "hybrid_generated" if generated else "pool",
                "batch": _public_paper(appended),
                "count": len(appended),
                "quiz_id": active.quiz_id,
                "timer": _quiz_timer_payload(active),
                **_batch_response_metadata(active, len(combined)),
            }
        )
    except Exception as exc:
        db.session.rollback()
        preparing = _quiz_preparation_response(exc)
        if preparing is not None:
            return preparing
        failure = _inference_failure_response(exc)
        if failure is not None:
            return failure
        app.logger.exception("Next quiz batch reveal failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "The next question batch is temporarily unavailable."),
        }), 503


@app.route("/api/quiz/batch-status")
@limiter.limit("600 per hour", key_func=_user_or_ip_rate_key)
def api_quiz_batch_status():
    """Report whether the next private five-question batch is ready."""
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        quiz_id = str(request.args.get("quiz_id", "")).strip()
        if not 16 <= len(quiz_id) <= 64:
            return jsonify({"status": "error", "message": "A valid quiz identifier is required."}), 400

        user_id = _orm_identity_id(user)
        active, current_paper = _load_active_quiz(user)
        if not active or not current_paper or not hmac.compare_digest(active.quiz_id, quiz_id):
            return jsonify({"status": "error", "message": "This quiz is no longer active."}), 409

        target = int(active.target_question_count or len(current_paper))
        if len(current_paper) >= target:
            return jsonify({"status": "ok", "prefetch_status": "complete"})

        state = _active_batch_state(active)
        prefetch = _batch_prefetch_for_offset(state, len(current_paper))
        status = str(prefetch.get("status", "idle"))
        if status not in {"ready", "preparing"}:
            _schedule_next_batch_prefetch(user_id=user_id, quiz_id=quiz_id)
            active = ActiveQuiz.query.filter_by(
                user_id=user_id,
                quiz_id=quiz_id,
                status="active",
            ).first()
            if active is not None:
                state = _active_batch_state(active)
                status = str(_batch_prefetch_for_offset(state, len(current_paper)).get("status", "preparing"))
            else:
                status = "preparing"

        return jsonify(
            {
                "status": "ok",
                "prefetch_status": status,
                "retry_after_ms": 2000,
            }
        )
    except Exception as exc:
        app.logger.exception("Quiz batch-status check failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Quiz status is temporarily unavailable."),
        }), 500


@app.route("/api/paper")
def api_paper():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401
        active, paper = _load_active_quiz(user)
        if not paper:
            return jsonify({"status": "error", "message": "No active quiz. Load a quiz first."}), 404
        return jsonify(
            {
                "status": "ok",
                "paper": _public_paper(paper),
                "count": len(paper),
                "quiz_id": active.quiz_id,
                "timer": _quiz_timer_payload(active),
                **_batch_response_metadata(active, len(paper)),
            }
        )
    except Exception as exc:
        app.logger.exception("Active quiz retrieval failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "The active quiz is temporarily unavailable."),
        }), 500


@app.route("/api/submit", methods=["POST"])
@limiter.limit("30 per hour", key_func=_user_or_ip_rate_key)
def api_submit():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        body = _json_body()
        submitted_quiz_id = str(body.get("quiz_id", "")).strip()
        if not 16 <= len(submitted_quiz_id) <= 64:
            return jsonify(
                {"status": "error", "message": "A valid quiz identifier is required."}
            ), 400
        raw_answers = body.get("answers", {})
        if not isinstance(raw_answers, dict):
            return jsonify({"status": "error", "message": "Answers must be an object."}), 400
        normalized_answers: dict[str, str] = {}
        for key, value in raw_answers.items():
            answer = str(value or "").strip().upper()
            if answer not in {"A", "B", "C", "D"}:
                return jsonify(
                    {"status": "error", "message": "Each answer must be A, B, C, or D."}
                ), 400
            try:
                index = int(key)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": "Invalid question index."}), 400
            if index < 0:
                return jsonify({"status": "error", "message": "Invalid question index."}), 400
            normalized_answers[str(index)] = answer
        active_quiz, paper = _load_active_quiz(user)
        if not active_quiz or not paper:
            return jsonify({"status": "error", "message": "No active quiz. Load a quiz first."}), 409
        if not hmac.compare_digest(active_quiz.quiz_id, submitted_quiz_id):
            return jsonify(
                {"status": "error", "message": "This quiz is no longer active."}
            ), 409
        if any(int(index) >= len(paper) for index in normalized_answers):
            return jsonify({"status": "error", "message": "Invalid question index."}), 400

        timer_expired = bool(
            active_quiz.timer_enabled
            and active_quiz.expires_at
            and _utcnow() >= active_quiz.expires_at
        )
        target_question_count = int(active_quiz.target_question_count or len(paper))
        if len(paper) < target_question_count and not timer_expired:
            return (
                jsonify(
                    {
                        "status": "error",
                        "code": "quiz_batches_remaining",
                        "message": (
                            f"Complete the remaining question batches before submitting "
                            f"({len(paper)}/{target_question_count} loaded)."
                        ),
                        "loaded_count": len(paper),
                        "total_count": target_question_count,
                    }
                ),
                400,
            )
        missing_indices = [
            index for index in range(len(paper)) if str(index) not in normalized_answers
        ]
        if active_quiz.timer_enabled and missing_indices and not timer_expired:
            first_unanswered = missing_indices[0]
            return (
                jsonify(
                    {
                        "status": "error",
                        "code": "incomplete_quiz",
                        "message": (
                            f"Please select an answer for Question {first_unanswered + 1} "
                            "before submitting."
                        ),
                        "first_unanswered": first_unanswered,
                        "unanswered_count": len(missing_indices),
                    }
                ),
                400,
            )

        if not _claim_active_quiz(user, submitted_quiz_id):
            db.session.rollback()
            _quiz_log(
                "quiz.submit.conflict",
                user_id=_orm_identity_id(user),
                quiz_id=submitted_quiz_id,
                quiz_kind=active_quiz.quiz_kind,
                outcome="already_submitted",
            )
            return jsonify(
                {"status": "error", "message": "This quiz was already submitted."}
            ), 409

        mistakes_payload = []
        correct = 0

        review = []
        for idx, question in enumerate(paper):
            student_answer = normalized_answers.get(str(idx), "")
            correct_answer = question.get("answer", "")
            options = question.get("options", [])
            is_correct = student_answer == correct_answer
            if is_correct:
                correct += 1
            else:
                mistakes_payload.append(
                    {
                        "question_index": idx,
                        "question": question.get("question", ""),
                        "topic_id": question.get("topic_id", "unknown"),
                        "difficulty_level": question.get("difficulty_level", "unknown"),
                        "correct_answer": correct_answer,
                        "student_answer": student_answer,
                    }
                )

            img = question.get("image_url") or ""
            rev = {
                "index": idx,
                "question": question.get("question", ""),
                "options": options,
                "correct_answer": correct_answer,
                "explanation": question.get("explanation", ""),
                "student_answer": student_answer,
                "correct_option_text": _option_text(options, correct_answer),
                "student_option_text": _option_text(options, student_answer),
                "is_correct": is_correct,
            }
            if img and is_safe_quiz_image_url(img):
                rev["image_url"] = img
            review.append(rev)

        total = len(paper)
        incorrect = len(mistakes_payload)
        score_percent = round(correct / max(total, 1) * 100, 1)

        attempt = Attempt(
            user_id=user.id,
            quiz_id=active_quiz.quiz_id,
            quiz_kind=active_quiz.quiz_kind,
            total_questions=total,
            correct=correct,
            incorrect=incorrect,
            score_percent=score_percent,
        )
        db.session.add(attempt)
        db.session.flush()

        for mistake in mistakes_payload:
            db.session.add(
                Mistake(
                    attempt_id=attempt.id,
                    question_index=mistake["question_index"],
                    question=mistake["question"],
                    topic_id=mistake["topic_id"],
                    difficulty_level=mistake["difficulty_level"],
                    correct_answer=mistake["correct_answer"],
                    student_answer=mistake["student_answer"],
                )
            )

        for item in review:
            iu = item.get("image_url") or ""
            original_question = paper[item["index"]]
            db.session.add(
                AttemptQuestion(
                    attempt_id=attempt.id,
                    question_index=item["index"],
                    question=item.get("question", ""),
                    options_json=json.dumps(item.get("options", []), ensure_ascii=False),
                    correct_answer=item.get("correct_answer", ""),
                    student_answer=item.get("student_answer", ""),
                    correct_option_text=item.get("correct_option_text"),
                    student_option_text=item.get("student_option_text"),
                    explanation=item.get("explanation", ""),
                    is_correct=bool(item.get("is_correct", False)),
                    image_url=(iu if iu and is_safe_quiz_image_url(iu) else None),
                    topic_id=str(original_question.get("topic_id", "unknown")),
                    difficulty_level=str(original_question.get("difficulty_level", "unknown")),
                    source_question_id=(
                        str(original_question.get("source_question_id", "")).strip()
                        or None
                    ),
                    question_hash=question_fingerprint(
                        str(original_question.get("question", ""))
                    ),
                    pool_question_id=(
                        int(original_question["_pool_question_id"])
                        if isinstance(original_question.get("_pool_question_id"), int)
                        else None
                    ),
                )
            )

        UserQuestionExposure.query.filter_by(
            user_id=user.id, quiz_id=active_quiz.quiz_id
        ).update(
            {"attempt_id": attempt.id, "answered_at": _utcnow()},
            synchronize_session=False,
        )
        quiz_kind = active_quiz.quiz_kind
        completed_quiz_id = active_quiz.quiz_id
        db.session.delete(active_quiz)
        db.session.commit()
        _quiz_log(
            "quiz.submit.completed",
            user_id=_orm_identity_id(user),
            quiz_id=completed_quiz_id,
            quiz_kind=quiz_kind,
            question_count=total,
            outcome="timed_out" if timer_expired else "submitted",
        )

        return jsonify(
            {
                "status": "ok",
                "email": user.email,
                "correct": correct,
                "incorrect": incorrect,
                "total": total,
                "score_percent": score_percent,
                "review": review,
                "timed_out": timer_expired,
            }
        )
    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Quiz submission failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Quiz submission failed. Please retry."),
        }), 500


@app.route("/api/recommend")
def api_recommend():
    try:
        from guidance.analyzer import WeaknessAnalyzer

        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        attempt_id = request.args.get("attempt_id", type=int)
        attempts_desc = (
            Attempt.query.filter_by(user_id=user.id).order_by(Attempt.created_at.desc(), Attempt.id.desc()).all()
        )
        selected_attempt = None
        if attempts_desc:
            if attempt_id is None:
                selected_attempt = attempts_desc[0]
            else:
                selected_attempt = next((a for a in attempts_desc if a.id == attempt_id), None)
                if selected_attempt is None:
                    return jsonify({"status": "error", "message": "Selected attempt not found."}), 404

        tracker = DBTrackerAdapter(
            user,
            attempt_id=selected_attempt.id if selected_attempt else None,
        )
        analyzer = WeaknessAnalyzer(tracker)
        concept_mapper = ConceptMapper()

        def _level_counts_for_topic(by_tl: dict[str, dict[str, int]], tid: str) -> dict[str, int]:
            if tid in by_tl:
                return by_tl[tid]
            for k, v in by_tl.items():
                if str(k) == tid:
                    return v
            return {}

        weak_topics = analyzer.weakest_topics(top_n=5)
        topic_lookup = _topic_name_lookup_from_topics_file()
        by_topic_level = analyzer.mistakes_by_topic_and_level()

        weak_topics_payload = []
        chapter_guidance = []
        guidance_notes = []
        study_recommendations: list[dict[str, Any]] = []
        textbook_by_grade = _textbook_index_by_grade()
        practice_topics = []
        for topic_id, mistakes in weak_topics:
            topic_id = str(topic_id)
            topic_name = topic_lookup.get(topic_id) or topic_id
            mapped = concept_mapper.map_topic(topic_name)
            level_counts = _level_counts_for_topic(by_topic_level, topic_id)
            top_levels = sorted(level_counts.items(), key=lambda x: (-x[1], x[0]))[:3]
            difficulty_hint = (
                ", ".join(f"{lv}: {c}" for lv, c in top_levels) if top_levels else None
            )
            weak_topics_payload.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "mistakes": int(mistakes),
                    "difficulty_breakdown": difficulty_hint,
                }
            )
            chapter_guidance.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "mistakes": int(mistakes),
                    "mapped_strands": mapped,
                }
            )

            # Build practice plan using mapped chapters and available textbooks.
            chapters_seen = set()
            practice_plan = []
            for strand in mapped:
                for chapter in strand.get("focus_chapters", []):
                    grade = chapter.get("grade")
                    chapter_name = str(chapter.get("topic", "")).strip()
                    if not isinstance(grade, int) or not chapter_name:
                        continue
                    key = (grade, chapter_name)
                    if key in chapters_seen:
                        continue
                    chapters_seen.add(key)
                    practice_plan.append(
                        {
                            "grade": grade,
                            "chapter": chapter_name,
                            "textbooks": textbook_by_grade.get(grade, []),
                        }
                    )
                    if len(practice_plan) >= 6:
                        break
                if len(practice_plan) >= 6:
                    break

            practice_topics.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "mistakes": int(mistakes),
                    "practice_plan": practice_plan,
                }
            )

            actions = [
                f"Grade {p['grade']}: {p['chapter']}"
                for p in practice_plan[:6]
                if p.get("grade") is not None and p.get("chapter")
            ]
            study_recommendations.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "mistakes": int(mistakes),
                    "practice_actions": actions,
                    "tip": (
                        f"Most missed items in this attempt were tagged {top_levels[0][0]} level."
                        if top_levels
                        else None
                    ),
                }
            )

        if not weak_topics_payload:
            guidance_notes.append("No mistakes in the selected attempt. Keep practicing to maintain performance.")
        else:
            overall_levels = analyzer.mistakes_by_level()
            if overall_levels:
                hardest = next(iter(overall_levels.items()))
                guidance_notes.append(
                    f"Across this attempt, the hardest cognitive level for you was “{hardest[0]}” "
                    f"({hardest[1]} mistake(s)). Mix quick recall with a few harder application items."
                )
            guidance_notes.append(
                "Use the Practice tab: open the grades and chapters listed under Recommendations "
                "before re-attempting a full quiz on the same topics."
            )

        attempts_payload = []
        total_attempts = len(attempts_desc)
        for i, attempt in enumerate(attempts_desc):
            attempts_payload.append(
                {
                    "attempt_id": attempt.id,
                    "label": f"Attempt {total_attempts - i} ({attempt.score_percent:.1f}%)",
                    "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
                    "score_percent": attempt.score_percent,
                    "incorrect": attempt.incorrect,
                }
            )

        return jsonify(
            {
                "status": "ok",
                "email": user.email,
                "selected_attempt_id": selected_attempt.id if selected_attempt else None,
                "attempts": attempts_payload,
                "weak_topics": weak_topics_payload,
                "guidance_notes": guidance_notes,
                "study_recommendations": study_recommendations,
                "chapter_guidance": chapter_guidance,
                "practice_topics": practice_topics,
                "score_trend": analyzer.score_trend(),
                "mistakes_by_topic": analyzer.mistakes_by_topic(),
                "mistakes_by_level": analyzer.mistakes_by_level(),
            }
        )
    except Exception as exc:
        app.logger.exception("Recommendation analysis failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Recommendations are temporarily unavailable."),
        }), 500


def _gamification_badge_defs() -> list[dict[str, str]]:
    return [
        {
            "id": "first_spark",
            "title": "First Spark",
            "description": "Complete your first quiz attempt.",
            "icon": "bolt",
        },
        {
            "id": "steady_pulse",
            "title": "Steady Pulse",
            "description": "Complete 5 quiz attempts.",
            "icon": "monitor_heart",
        },
        {
            "id": "deep_network",
            "title": "Deep Network",
            "description": "Complete 15 quiz attempts.",
            "icon": "hub",
        },
        {
            "id": "synaptic_storm",
            "title": "Synaptic Storm",
            "description": "Complete 40 quiz attempts.",
            "icon": "storm",
        },
        {
            "id": "perfect_loop",
            "title": "Perfect Loop",
            "description": "Score 100% on any attempt.",
            "icon": "all_inclusive",
        },
        {
            "id": "sharp_mind",
            "title": "Sharp Mind",
            "description": "Finish at least 5 attempts with 90% or higher.",
            "icon": "psychology",
        },
        {
            "id": "streak_3",
            "title": "Three-Day Chain",
            "description": "Practice on 3 consecutive calendar days.",
            "icon": "link",
        },
        {
            "id": "streak_7",
            "title": "Weekly Rhythm",
            "description": "Practice on 7 consecutive calendar days.",
            "icon": "calendar_month",
        },
        {
            "id": "wide_reach",
            "title": "Wide Reach",
            "description": "Practice on 10 different calendar days.",
            "icon": "scatter_plot",
        },
        {
            "id": "century_mind",
            "title": "Century Mind",
            "description": "Answer 100 questions correctly across all attempts.",
            "icon": "blur_on",
        },
        {
            "id": "double_century",
            "title": "Double Century",
            "description": "Answer 300 questions correctly across all attempts.",
            "icon": "numbers",
        },
        {
            "id": "comeback",
            "title": "Comeback Trail",
            "description": "Improve your score by 20 points or more versus your previous attempt.",
            "icon": "trending_up",
        },
        {
            "id": "marathon_focus",
            "title": "Marathon Focus",
            "description": "Earn a perfect score on a paper with at least 20 questions.",
            "icon": "emoji_events",
        },
    ]


def _compute_gamification(user: User) -> dict[str, Any]:
    attempts = (
        Attempt.query.filter_by(user_id=user.id).order_by(Attempt.created_at.asc(), Attempt.id.asc()).all()
    )
    by_day: dict[str, int] = defaultdict(int)
    scores: list[float] = []
    total_correct = 0
    perfect_count = 0
    high_90_count = 0
    marathon_perfect = False

    for a in attempts:
        if a.created_at:
            day = a.created_at.date().isoformat()
            by_day[day] += 1
        sp = float(a.score_percent or 0)
        scores.append(sp)
        total_correct += int(a.correct or 0)
        if sp >= 99.5:
            perfect_count += 1
            if int(a.total_questions or 0) >= 20:
                marathon_perfect = True
        if sp >= 90:
            high_90_count += 1

    n = len(attempts)
    distinct_days = len(by_day)

    sorted_days = sorted(by_day.keys())
    best_streak = 0
    cur_streak = 0
    prev_d: datetime | None = None
    for ds in sorted_days:
        d = datetime.fromisoformat(ds).date()
        if prev_d is None:
            cur_streak = 1
        else:
            delta = (d - prev_d).days
            if delta == 1:
                cur_streak += 1
            elif delta == 0:
                pass
            else:
                cur_streak = 1
        best_streak = max(best_streak, cur_streak)
        prev_d = d

    current_streak = 0
    if sorted_days:
        today = _utcnow().date()
        walk = today
        day_set = set(sorted_days)
        while walk.isoformat() in day_set:
            current_streak += 1
            walk = walk - timedelta(days=1)

    comeback = False
    for i in range(1, len(scores)):
        if scores[i] - scores[i - 1] >= 20:
            comeback = True
            break

    stats = {
        "total_attempts": n,
        "total_correct_answers": total_correct,
        "distinct_practice_days": distinct_days,
        "best_day_streak": best_streak,
        "current_day_streak": current_streak,
        "perfect_attempts": perfect_count,
        "attempts_90_plus": high_90_count,
        "has_comeback": comeback,
        "marathon_perfect": marathon_perfect,
    }

    earned_ids: set[str] = set()
    if n >= 1:
        earned_ids.add("first_spark")
    if n >= 5:
        earned_ids.add("steady_pulse")
    if n >= 15:
        earned_ids.add("deep_network")
    if n >= 40:
        earned_ids.add("synaptic_storm")
    if perfect_count >= 1:
        earned_ids.add("perfect_loop")
    if high_90_count >= 5:
        earned_ids.add("sharp_mind")
    if best_streak >= 3:
        earned_ids.add("streak_3")
    if best_streak >= 7:
        earned_ids.add("streak_7")
    if distinct_days >= 10:
        earned_ids.add("wide_reach")
    if total_correct >= 100:
        earned_ids.add("century_mind")
    if total_correct >= 300:
        earned_ids.add("double_century")
    if comeback:
        earned_ids.add("comeback")
    if marathon_perfect:
        earned_ids.add("marathon_focus")

    badges_out = []
    for b in _gamification_badge_defs():
        badges_out.append({**b, "earned": b["id"] in earned_ids})

    return {
        "attempts_by_day": dict(sorted(by_day.items())),
        "stats": stats,
        "badges": badges_out,
        "earned_count": len(earned_ids),
        "total_badges": len(badges_out),
    }


@app.route("/api/gamification")
def api_gamification():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Not signed in."}), 401
        payload = _compute_gamification(user)
        return jsonify({"status": "ok", **payload})
    except Exception as exc:
        app.logger.exception("Gamification summary failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Gamification data is temporarily unavailable."),
        }), 500


@app.route("/api/attempts")
def api_attempts():
    try:
        user = _current_user()
        if not user:
            return jsonify({"status": "error", "message": "Please sign in first."}), 401

        attempts = (
            Attempt.query.filter_by(user_id=user.id).order_by(Attempt.created_at.desc(), Attempt.id.desc()).all()
        )

        payload = []
        total_attempts = len(attempts)
        for list_index, attempt in enumerate(attempts):
            rows = (
                AttemptQuestion.query.filter_by(attempt_id=attempt.id)
                .order_by(AttemptQuestion.question_index.asc())
                .all()
            )
            questions = []
            for row in rows:
                try:
                    options = json.loads(row.options_json or "[]")
                    if not isinstance(options, list):
                        options = []
                except Exception:
                    options = []
                iu = getattr(row, "image_url", None) or ""
                qrow = {
                    "index": row.question_index,
                    "question": row.question,
                    "options": options,
                    "student_answer": row.student_answer or "",
                    "correct_answer": row.correct_answer or "",
                    "student_option_text": row.student_option_text or "",
                    "correct_option_text": row.correct_option_text or "",
                    "explanation": row.explanation or "",
                    "is_correct": bool(row.is_correct),
                }
                if iu and is_safe_quiz_image_url(iu):
                    qrow["image_url"] = iu
                questions.append(qrow)

            payload.append(
                {
                    "attempt_id": attempt.id,
                    "attempt_number": total_attempts - list_index,
                    "quiz_id": attempt.quiz_id,
                    "quiz_kind": attempt.quiz_kind,
                    "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
                    "score_percent": attempt.score_percent,
                    "total_questions": attempt.total_questions,
                    "correct": attempt.correct,
                    "incorrect": attempt.incorrect,
                    "questions": questions,
                }
            )

        return jsonify({"status": "ok", "attempts": payload})
    except Exception as exc:
        app.logger.exception("Attempt history retrieval failed")
        return jsonify({
            "status": "error",
            "message": _public_error_message(exc, "Attempt history is temporarily unavailable."),
        }), 500


if __name__ == "__main__":
    app.run(debug=not _is_production, port=int(os.getenv("PORT", 5000)))
