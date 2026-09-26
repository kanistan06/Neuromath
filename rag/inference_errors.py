"""Bounded inference calls and actionable provider access failures."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

import config
from infrastructure.cache import cache_key, get_json, set_json


_blocked_until: dict[str, float] = {}
_lock = threading.Lock()
_budget: ContextVar[dict | None] = ContextVar("inference_budget", default=None)
CREDIT_COOLDOWN_SECONDS = 60


class InferenceCreditsError(RuntimeError):
    status_code = 402
    code = "inference_credits_exhausted"

    def __init__(self, retry_after: int = CREDIT_COOLDOWN_SECONDS):
        self.retry_after = max(1, retry_after)
        super().__init__(
            "RunPod inference balance is unavailable or exhausted. Add funds in the RunPod "
            "billing dashboard, then wait up to 60 seconds and retry. Creating a new API key "
            "does not restore the account balance."
        )


class GenerationBudgetExceeded(RuntimeError):
    code = "generation_budget_exhausted"

    def __init__(self):
        super().__init__(
            "This generation request reached its inference-call limit. Retry to resume any "
            "validated progress saved by the application."
        )


def _account_key() -> str:
    identity = hashlib.sha256(config.RUNPOD_API_KEY.encode()).hexdigest()
    return cache_key("inference-credit-block", identity)


def ensure_inference_available() -> None:
    key = _account_key()
    cached = get_json(key)
    with _lock:
        until = _blocked_until.get(key, 0)
    if isinstance(cached, dict):
        until = max(until, float(cached.get("until", 0)))
    remaining = until - time.time()
    if remaining > 0:
        raise InferenceCreditsError(math.ceil(remaining))


def record_credit_failure(status: int, message: str) -> None:
    lowered = message.lower()
    if status != 402 and not (
        any(marker in lowered for marker in ("credits", "balance", "funds")) and any(word in lowered for word in ("depleted", "exhausted", "exceeded", "insufficient"))
    ):
        return
    key = _account_key()
    until = time.time() + CREDIT_COOLDOWN_SECONDS
    with _lock:
        _blocked_until[key] = until
    set_json(key, {"until": until}, CREDIT_COOLDOWN_SECONDS)
    raise InferenceCreditsError()


def access_failure(exc: BaseException) -> BaseException | None:
    visited = set()
    while exc is not None and id(exc) not in visited:
        visited.add(id(exc))
        if getattr(exc, "code", "") == "hf_embeddings_unavailable":
            return exc
        if isinstance(exc, (InferenceCreditsError, GenerationBudgetExceeded)):
            return exc
        if getattr(exc, "status_code", None) == 402:
            return InferenceCreditsError()
        exc = exc.__cause__ or exc.__context__
    return None


@contextmanager
def inference_budget(max_calls: int):
    if _budget.get() is not None:
        yield
        return
    token = _budget.set({"remaining": max(1, max_calls), "lock": threading.Lock()})
    try:
        yield
    finally:
        _budget.reset(token)


def consume_inference_call() -> None:
    ensure_inference_available()
    budget = _budget.get()
    if budget is not None:
        with budget["lock"]:
            if budget["remaining"] <= 0:
                raise GenerationBudgetExceeded()
            budget["remaining"] -= 1
