"""Typed configuration for NeuroMath."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    return max(minimum, min(maximum, value))


def env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    return max(minimum, min(maximum, value))


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STUDENT_RECORDS_DIR = PROJECT_ROOT / "guidance" / "student_records"
TOPICS_FILE = DATA_DIR / "topics.json"
DIFFICULTY_LEVELS_FILE = DATA_DIR / "difficulty_levels.json"
SYLLABUS_FILE = DATA_DIR / "syllabus.json"
_question_bank_dir = Path(
    os.getenv("QUESTION_BANK_DIR", str(DATA_DIR / "past_papers"))
).expanduser()
QUESTION_BANK_DIR = (
    _question_bank_dir
    if _question_bank_dir.is_absolute()
    else PROJECT_ROOT / _question_bank_dir
)

APP_ENV = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "development")).strip().lower()
IS_RAILWAY = bool(
    os.getenv("RAILWAY_ENVIRONMENT_NAME") or os.getenv("RAILWAY_PROJECT_ID")
)
IS_PRODUCTION = APP_ENV == "production" or IS_RAILWAY
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:5000").strip().rstrip("/")
PORT = env_int("PORT", 5000, minimum=1, maximum=65535)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
MAX_REQUEST_BYTES = env_int("MAX_REQUEST_BYTES", 65536, minimum=8192, maximum=1048576)
TRUST_PROXY_HEADERS = env_bool("TRUST_PROXY_HEADERS", IS_PRODUCTION)

QUIZ_REFERENCE_DIR = Path(
    os.getenv("QUIZ_REFERENCE_DIR", str(DATA_DIR / "loadQuizRef"))
).expanduser()
QUIZ_REFERENCE_DIRS = [
    Path(item.strip()).expanduser()
    for item in os.getenv(
        "QUIZ_REFERENCE_DIRS",
        f"{DATA_DIR / 'loadQuizRef'},{DATA_DIR / 'TextBooks'}",
    ).split(",")
    if item.strip()
]


def resolve_reference_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _extract_grade_from_name(path: Path) -> int | None:
    match = re.search(
        r"(?:^|[^a-z0-9])(?:e?gr|g|grade)[\s_-]?(\d{1,2})(?=[^0-9]|$)",
        path.name.lower(),
    )
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def include_reference_file(path: Path) -> bool:
    grade = _extract_grade_from_name(path)
    return grade is not None and 6 <= grade <= 11


def include_quiz_reference_file(path: Path) -> bool:
    if path.suffix.lower() not in {".pdf", ".md", ".txt"}:
        return False
    if _is_within(path, resolve_reference_dir(DATA_DIR / "loadQuizRef")):
        return True
    if _is_within(path, resolve_reference_dir(DATA_DIR / "TextBooks")):
        grade = _extract_grade_from_name(path)
        return grade is not None and 6 <= grade <= 11
    return False


SELECTED_LLM_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
SELECTED_MCQ_GENERATION_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SELECTED_EMBEDDING_MODEL = "BAAI/bge-m3"
SELECTED_EMBEDDING_DIMENSIONS = 1024

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_LLM_ENDPOINT_ID = os.getenv("RUNPOD_LLM_ENDPOINT_ID", "").strip()
RUNPOD_LLM_MODEL = os.getenv("RUNPOD_LLM_MODEL", SELECTED_LLM_MODEL).strip()
RUNPOD_MCQ_ENDPOINT_ID = os.getenv("RUNPOD_MCQ_ENDPOINT_ID", "").strip()
RUNPOD_MCQ_GENERATION_MODEL = os.getenv(
    "RUNPOD_MCQ_GENERATION_MODEL", SELECTED_MCQ_GENERATION_MODEL
).strip()
RUNPOD_TIMEOUT_SECONDS = env_int(
    "RUNPOD_TIMEOUT_SECONDS", 300, minimum=10, maximum=900
)
RUNPOD_MAX_RETRIES = env_int("RUNPOD_MAX_RETRIES", 6, minimum=1, maximum=10)
RUNPOD_RETRY_BASE_SECONDS = env_float(
    "RUNPOD_RETRY_BASE_SECONDS", 3.0, minimum=0.1, maximum=30.0
)
RUNPOD_GENERATION_TEMPERATURE = env_float(
    "RUNPOD_GENERATION_TEMPERATURE", 0.2, minimum=0.0, maximum=0.3
)
RUNPOD_LLM_MAX_NEW_TOKENS = env_int(
    "RUNPOD_LLM_MAX_NEW_TOKENS", 900, minimum=256, maximum=2048
)
RUNPOD_MCQ_MAX_NEW_TOKENS = env_int(
    "RUNPOD_MCQ_MAX_NEW_TOKENS", 900, minimum=256, maximum=2048
)

HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_EMBEDDING_MODEL = SELECTED_EMBEDDING_MODEL
HF_EMBEDDING_PROVIDER = "hf-inference"
HF_EMBEDDING_DIMENSIONS = SELECTED_EMBEDDING_DIMENSIONS
HF_EMBEDDING_BATCH_SIZE = env_int("HF_EMBEDDING_BATCH_SIZE", 16, minimum=1, maximum=32)
HF_EMBEDDING_TIMEOUT_SECONDS = env_int("HF_EMBEDDING_TIMEOUT_SECONDS", 120, minimum=10, maximum=300)
HF_EMBEDDING_MAX_RETRIES = env_int("HF_EMBEDDING_MAX_RETRIES", 3, minimum=1, maximum=5)
HF_EMBEDDING_RETRY_BASE_SECONDS = env_float("HF_EMBEDDING_RETRY_BASE_SECONDS", 2.0, minimum=0.1, maximum=10.0)
HF_BILL_TO = os.getenv("HF_BILL_TO", "").strip()

CHUNK_TOKENIZER_MODEL = os.getenv("CHUNK_TOKENIZER_MODEL", SELECTED_LLM_MODEL).strip()
CHUNK_MIN_TOKENS = env_int("CHUNK_MIN_TOKENS", 300, minimum=250, maximum=400)
CHUNK_TARGET_TOKENS = env_int("CHUNK_TARGET_TOKENS", 360, minimum=300, maximum=425)
CHUNK_MAX_TOKENS = env_int("CHUNK_MAX_TOKENS", 450, minimum=350, maximum=450)
CHUNK_OVERLAP_TOKENS = env_int("CHUNK_OVERLAP_TOKENS", 60, minimum=50, maximum=75)
CHUNKER_VERSION = os.getenv("CHUNKER_VERSION", "structure-token-v1").strip()
CORPUS_LABEL = os.getenv("CORPUS_LABEL", "ol-textbooks-v1").strip()
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "").strip()
INGEST_STATE_DIR = Path(
    os.getenv("INGEST_STATE_DIR", str(PROJECT_ROOT / ".state"))
).expanduser()

GENERATION_MAX_WORKERS = env_int("GENERATION_MAX_WORKERS", 1, minimum=1, maximum=8)
GENERATION_RETRY_ATTEMPTS = env_int(
    "GENERATION_RETRY_ATTEMPTS", 3, minimum=1, maximum=6
)
GENERATION_RETRY_BASE_SECONDS = env_float(
    "GENERATION_RETRY_BASE_SECONDS", 2.0, minimum=0.0, maximum=30.0
)
# Ask the author model for a small surplus of candidates, then retain only
# fully validated questions. This reduces whole-quiz failures without relaxing
# any structural, grounding, repetition, or independent-solution checks.
GENERATION_CANDIDATE_MULTIPLIER = env_int(
    "GENERATION_CANDIDATE_MULTIPLIER", 2, minimum=1, maximum=3
)
GENERATION_CANDIDATE_CAP = env_int(
    "GENERATION_CANDIDATE_CAP", 6, minimum=1, maximum=8
)
GENERATION_COVERAGE_REFILL_ROUNDS = env_int(
    "GENERATION_COVERAGE_REFILL_ROUNDS", 2, minimum=1, maximum=3
)
# A full 25-concept diagnostic needs more inference calls than a single-topic
# practice quiz because every accepted MCQ requires an author call plus an
# independent review, and rejected candidates may be retried.  Keep this
# bounded so the request cannot run away in cost while still allowing the
# configured retry/refill policy to complete.
GENERATION_INFERENCE_CALLS_PER_QUESTION = env_int(
    "GENERATION_INFERENCE_CALLS_PER_QUESTION", 12, minimum=3, maximum=20
)
GENERATION_INFERENCE_CALL_BUDGET_MAX = env_int(
    "GENERATION_INFERENCE_CALL_BUDGET_MAX", 300, minimum=12, maximum=1000
)
QUESTION_BANK_REQUIRED = env_bool("QUESTION_BANK_REQUIRED", IS_PRODUCTION)
QUESTION_BANK_REQUIRE_REVIEW = env_bool("QUESTION_BANK_REQUIRE_REVIEW", True)
QUESTION_BANK_MAX_EXAMPLES = env_int(
    "QUESTION_BANK_MAX_EXAMPLES", 10, minimum=1, maximum=10
)
MCQ_SIMILARITY_THRESHOLD = env_float(
    "MCQ_SIMILARITY_THRESHOLD", 0.84, minimum=0.70, maximum=0.99
)
QUESTION_HISTORY_LIMIT = env_int(
    "QUESTION_HISTORY_LIMIT", 500, minimum=50, maximum=5000
)
QUESTION_POOL_RUNTIME_ENABLED = env_bool("QUESTION_POOL_RUNTIME_ENABLED", True)
QUESTION_POOL_TARGET_PER_TOPIC_LEVEL = env_int(
    "QUESTION_POOL_TARGET_PER_TOPIC_LEVEL", 25, minimum=21, maximum=200
)
AUTO_INGEST_ON_STARTUP = env_bool("AUTO_INGEST_ON_STARTUP", False)
INGEST_ON_QUIZ_LOAD = env_bool("INGEST_ON_QUIZ_LOAD", False)
ALLOW_DEMO_MODE = False

DIAGNOSTIC_GRADES = tuple(
    int(item.strip())
    for item in os.getenv("DIAGNOSTIC_GRADES", "10,11").split(",")
    if item.strip()
)
DIAGNOSTIC_TIME_LIMIT_MINUTES = env_int(
    "DIAGNOSTIC_TIME_LIMIT_MINUTES", 120, minimum=30, maximum=300
)
DIAGNOSTIC_CACHE_HOURS = env_int(
    "DIAGNOSTIC_CACHE_HOURS", 168, minimum=1, maximum=720
)
GENERATION_LOCK_SECONDS = env_int(
    "GENERATION_LOCK_SECONDS", 180, minimum=60, maximum=7200
)
GENERATION_LOCK_WAIT_SECONDS = env_int(
    "GENERATION_LOCK_WAIT_SECONDS", 20, minimum=1, maximum=120
)
QUIZ_PREFETCH_MAX_WORKERS = env_int(
    "QUIZ_PREFETCH_MAX_WORKERS", 4, minimum=1, maximum=16
)
QUIZ_PREFETCH_STALE_SECONDS = env_int(
    "QUIZ_PREFETCH_STALE_SECONDS", 180, minimum=30, maximum=1800
)

VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "qdrant").strip().lower()
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db")
).strip()
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME", "neuromath_mcq_content"
).strip()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_WRITE_API_KEY = os.getenv("QDRANT_WRITE_API_KEY", "").strip()
QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME", "neuromath_mcq_content_v1"
).strip()
QDRANT_TIMEOUT_SECONDS = env_int(
    "QDRANT_TIMEOUT_SECONDS", 60, minimum=5, maximum=300
)
if VECTOR_STORE_PROVIDER not in {"chroma", "qdrant"}:
    raise ValueError("VECTOR_STORE_PROVIDER must be 'qdrant' or 'chroma'.")

RETRIEVAL_DENSE_K = env_int("RETRIEVAL_DENSE_K", 12, minimum=5, maximum=50)
RETRIEVAL_LEXICAL_K = env_int("RETRIEVAL_LEXICAL_K", 12, minimum=5, maximum=50)
RETRIEVAL_FINAL_K = env_int("RETRIEVAL_FINAL_K", 5, minimum=3, maximum=12)
LEXICAL_CACHE_SECONDS = env_int(
    "LEXICAL_CACHE_SECONDS", 3600, minimum=60, maximum=86400
)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_POOL_SIZE = env_int("DATABASE_POOL_SIZE", 5, minimum=1, maximum=20)
DATABASE_MAX_OVERFLOW = env_int("DATABASE_MAX_OVERFLOW", 2, minimum=0, maximum=20)
DATABASE_POOL_RECYCLE_SECONDS = env_int(
    "DATABASE_POOL_RECYCLE_SECONDS", 300, minimum=60, maximum=3600
)
DATABASE_CONNECT_TIMEOUT_SECONDS = env_int(
    "DATABASE_CONNECT_TIMEOUT_SECONDS", 10, minimum=3, maximum=60
)

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "") or "dev-secret-change-me"
EMAIL_CHECK_DELIVERABILITY = env_bool("EMAIL_CHECK_DELIVERABILITY", True)
PASSWORD_MIN_LENGTH = env_int("PASSWORD_MIN_LENGTH", 12, minimum=12, maximum=128)
EMAIL_VERIFICATION_TTL_HOURS = env_int(
    "EMAIL_VERIFICATION_TTL_HOURS", 24, minimum=1, maximum=72
)
PASSWORD_RESET_TTL_MINUTES = env_int(
    "PASSWORD_RESET_TTL_MINUTES", 10, minimum=5, maximum=30
)
PASSWORD_RESET_OTP_LENGTH = 6
PASSWORD_RESET_MAX_ATTEMPTS = env_int(
    "PASSWORD_RESET_MAX_ATTEMPTS", 5, minimum=3, maximum=10
)
SESSION_LIFETIME_HOURS = env_int(
    "SESSION_LIFETIME_HOURS", 12, minimum=1, maximum=48
)
RATELIMIT_ENABLED = env_bool("RATELIMIT_ENABLED", True)
RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://").strip()
CACHE_REDIS_URL = os.getenv("CACHE_REDIS_URL", RATELIMIT_STORAGE_URI).strip()
CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "neuromath:v1").strip()
ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("ADMIN_EMAILS", "").split(",")
    if email.strip()
}

EMAIL_DELIVERY_MODE = os.getenv("EMAIL_DELIVERY_MODE", "smtp").strip().lower()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.resend.com").strip()
SMTP_PORT = env_int("SMTP_PORT", 587, minimum=1, maximum=65535)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "resend").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "learn@neuromath.io").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "NeuroMath").strip() or "NeuroMath"
SMTP_USE_TLS = env_bool("SMTP_USE_TLS", True)
SMTP_USE_SSL = env_bool("SMTP_USE_SSL", False)
SMTP_TIMEOUT_SECONDS = env_int(
    "SMTP_TIMEOUT_SECONDS", 15, minimum=5, maximum=60
)
if EMAIL_DELIVERY_MODE not in {"smtp", "console", "disabled"}:
    raise ValueError("EMAIL_DELIVERY_MODE must be 'smtp', 'console', or 'disabled'.")

OBSERVABILITY_ENABLED = env_bool("OBSERVABILITY_ENABLED", IS_PRODUCTION)
BETTER_STACK_SOURCE_TOKEN = os.getenv("BETTER_STACK_SOURCE_TOKEN", "").strip()
BETTER_STACK_INGEST_URL = os.getenv(
    "BETTER_STACK_INGEST_URL", "https://in.logs.betterstack.com"
).strip().rstrip("/")
BETTER_STACK_HEARTBEAT_URL = os.getenv("BETTER_STACK_HEARTBEAT_URL", "").strip()
SERVICE_NAME = os.getenv("SERVICE_NAME", "neuromath-web").strip()

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip("/")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()
R2_PREFIX = os.getenv("R2_PREFIX", "neuromath/backups").strip().strip("/")
BACKUP_ENCRYPTION_KEY = os.getenv("BACKUP_ENCRYPTION_KEY", "").strip()
BACKUP_RETENTION_DAYS = env_int(
    "BACKUP_RETENTION_DAYS", 30, minimum=7, maximum=3650
)
PG_DUMP_PATH = os.getenv("PG_DUMP_PATH", "pg_dump").strip()
PG_RESTORE_PATH = os.getenv("PG_RESTORE_PATH", "pg_restore").strip()

IMAGE_GEN_ENABLED = env_bool("IMAGE_GEN_ENABLED", False)
IMAGE_GEN_MAX_PER_PAPER = env_int(
    "IMAGE_GEN_MAX_PER_PAPER", 8, minimum=0, maximum=20
)
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/sd-turbo").strip()
HF_IMAGE_FALLBACKS = [
    model.strip()
    for model in os.getenv(
        "HF_IMAGE_FALLBACKS",
        "black-forest-labs/FLUX.1-schnell,runwayml/stable-diffusion-v1-5",
    ).split(",")
    if model.strip()
]
HF_IMAGE_TIMEOUT_SECONDS = env_int(
    "HF_IMAGE_TIMEOUT_SECONDS", 120, minimum=10, maximum=600
)
HF_IMAGE_WIDTH = env_int("HF_IMAGE_WIDTH", 512, minimum=256, maximum=1024)
HF_IMAGE_HEIGHT = env_int("HF_IMAGE_HEIGHT", 512, minimum=256, maximum=1024)
_guidance_scale = os.getenv("HF_IMAGE_GUIDANCE_SCALE", "8.0").strip()
HF_IMAGE_GUIDANCE_SCALE = float(_guidance_scale) if _guidance_scale else None
HF_IMAGE_SEED_MODE = os.getenv("HF_IMAGE_SEED_MODE", "stable").strip().lower()
IMAGE_GEN_FALLBACK_QUESTION = env_bool("IMAGE_GEN_FALLBACK_QUESTION", False)
IMAGE_GEN_FALLBACK_MAX_CHARS = env_int(
    "IMAGE_GEN_FALLBACK_MAX_CHARS", 280, minimum=80, maximum=800
)
IMAGE_GEN_FALLBACK_REQUIRE_KEYWORDS = env_bool(
    "IMAGE_GEN_FALLBACK_REQUIRE_KEYWORDS", True
)
