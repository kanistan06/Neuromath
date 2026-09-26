"""Structured application logging for Better Stack."""

from __future__ import annotations

import atexit
import hashlib
import hmac
import logging
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from flask import Flask, g, request, session

import config


_SECRET_RE = re.compile(
    r"(?i)(authorization|cookie|password|token|api[_-]?key)([\s:=]+)([^\s,;]+)"
)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")



def pseudonymous_ref(namespace: str, value: object) -> str:
    """Stable HMAC reference for logs without exposing a raw user/quiz identifier."""
    raw = f"{str(namespace)}:{str(value)}".encode("utf-8")
    key = str(config.FLASK_SECRET_KEY or "neuromath-log-ref").encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest()[:16]


def _redact(value: object) -> str:
    text = str(value)
    text = _SECRET_RE.sub(r"\1\2[REDACTED]", text)
    return _EMAIL_RE.sub(r"[REDACTED]@\1", text)


class BetterStackHandler(logging.Handler):
    def __init__(self, token: str, endpoint: str) -> None:
        super().__init__()
        self.token = token
        self.endpoint = endpoint
        self.records: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1000)
        self.session = requests.Session()
        self.worker = threading.Thread(target=self._send_loop, daemon=True)
        self.worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "dt": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "message": _redact(record.getMessage()),
            "service": config.SERVICE_NAME,
            "environment": config.APP_ENV,
            "logger": record.name,
            "thread": record.threadName,
            "process_id": record.process,
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "attempt",
            "delay_seconds",
            "event",
            "user_ref",
            "quiz_ref",
            "quiz_kind",
            "topic_id",
            "grade",
            "batch_offset",
            "question_count",
            "lock_ref",
            "wait_ms",
            "outcome",
            "phase",
        ):
            if hasattr(record, key):
                payload[key] = _redact(getattr(record, key))
        try:
            self.records.put_nowait(payload)
        except queue.Full:
            return

    def _send_loop(self) -> None:
        while True:
            payload = self.records.get()
            if payload is None:
                return
            try:
                self.session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=payload,
                    timeout=5,
                ).raise_for_status()
            except requests.RequestException:
                continue

    def close(self) -> None:
        try:
            self.records.put_nowait(None)
            self.worker.join(timeout=2)
        except Exception:
            pass
        self.session.close()
        super().close()


def configure_observability(app: Flask) -> None:
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(handler, "neuromath_console", False) for handler in root.handlers):
        console = logging.StreamHandler()
        console.neuromath_console = True
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(console)

    if config.OBSERVABILITY_ENABLED and config.BETTER_STACK_SOURCE_TOKEN:
        if not any(isinstance(handler, BetterStackHandler) for handler in root.handlers):
            remote = BetterStackHandler(
                config.BETTER_STACK_SOURCE_TOKEN,
                config.BETTER_STACK_INGEST_URL,
            )
            remote.setLevel(level)
            root.addHandler(remote)
            atexit.register(remote.close)

    @app.before_request
    def begin_request() -> None:
        supplied = request.headers.get("X-Request-ID", "")
        g.request_id = supplied[:64] if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else uuid.uuid4().hex
        g.request_started_at = time.perf_counter()
        raw_user_id = session.get("user_id")
        g.user_ref = (
            pseudonymous_ref("user", raw_user_id)
            if isinstance(raw_user_id, int)
            else None
        )

    @app.after_request
    def complete_request(response):
        request_id = getattr(g, "request_id", uuid.uuid4().hex)
        started_at = getattr(g, "request_started_at", None)
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000, 2)
            if started_at is not None
            else None
        )
        response.headers["X-Request-ID"] = request_id
        app.logger.info(
            "http.request.completed",
            extra={
                "event": "http.request.completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "user_ref": getattr(g, "user_ref", None),
            },
        )
        return response
