"""Upstash Redis cache and distributed-lock adapter."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from redis import Redis
from redis.exceptions import LockError, RedisError

import config
from infrastructure.observability import pseudonymous_ref


logger = logging.getLogger(__name__)
_local_generation_locks: dict[str, threading.Lock] = {}
_local_generation_locks_guard = threading.Lock()


def _local_lock(identity: str) -> threading.Lock:
    """Return a process-local generation lock scoped to one generation identity."""
    with _local_generation_locks_guard:
        lock = _local_generation_locks.get(identity)
        if lock is None:
            lock = threading.Lock()
            _local_generation_locks[identity] = lock
        return lock


@lru_cache(maxsize=1)
def redis_client() -> Redis | None:
    url = config.CACHE_REDIS_URL
    if not url or url == "memory://":
        return None
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )


def cache_key(namespace: str, identity: str) -> str:
    return f"{config.CACHE_KEY_PREFIX}:{namespace}:{identity}"


def get_json(key: str) -> object | None:
    client = redis_client()
    if client is None:
        return None
    try:
        value = client.get(key)
        return json.loads(value) if value else None
    except (RedisError, ValueError):
        logger.exception("Redis cache read failed", extra={"cache_key": key})
        return None


def set_json(key: str, value: object, ttl_seconds: int) -> None:
    client = redis_client()
    if client is None:
        return
    try:
        client.setex(
            key,
            max(1, int(ttl_seconds)),
            json.dumps(value, separators=(",", ":"), ensure_ascii=False),
        )
    except RedisError:
        logger.exception("Redis cache write failed", extra={"cache_key": key})


def delete(key: str) -> None:
    client = redis_client()
    if client is None:
        return
    try:
        client.delete(key)
    except RedisError:
        logger.exception("Redis cache delete failed", extra={"cache_key": key})


def ping() -> bool:
    client = redis_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except RedisError:
        return False


def _lock_log(event: str, identity: str, *, wait_ms: float | None = None, outcome: str = "") -> None:
    lock_ref = pseudonymous_ref("generation-lock", identity)
    extra = {
        "event": event,
        "lock_ref": lock_ref,
        "outcome": outcome,
    }
    rounded_wait = None
    if wait_ms is not None:
        rounded_wait = round(float(wait_ms), 2)
        extra["wait_ms"] = rounded_wait
    # Keep the console useful during local concurrency testing without exposing
    # the raw user id, quiz id, or Redis key. Better Stack receives the same
    # structured fields through ``extra``.
    parts = [event, f"lock_ref={lock_ref}", f"outcome={outcome or 'unknown'}"]
    if rounded_wait is not None:
        parts.append(f"wait_ms={rounded_wait}")
    logger.info(" ".join(parts), extra=extra)


def _renew_distributed_lock(lock, stop: threading.Event, identity: str) -> None:
    """Keep a long generation lock alive while its owner is still working."""
    interval = max(5.0, min(60.0, float(config.GENERATION_LOCK_SECONDS) / 3.0))
    while not stop.wait(interval):
        try:
            lock.extend(config.GENERATION_LOCK_SECONDS, replace_ttl=True)
        except LockError:
            logger.exception(
                "generation.lock.renew_failed",
                extra={
                    "event": "generation.lock.renew_failed",
                    "lock_ref": pseudonymous_ref("generation-lock", identity),
                    "outcome": "ownership_lost",
                },
            )
            return
        except RedisError:
            # A transient Redis outage must not permanently disable renewal. The
            # existing lease remains valid, so retry on the next heartbeat.
            logger.exception(
                "generation.lock.renew_failed",
                extra={
                    "event": "generation.lock.renew_failed",
                    "lock_ref": pseudonymous_ref("generation-lock", identity),
                    "outcome": "redis_retry",
                },
            )


@contextmanager
def generation_lock(identity: str) -> Iterator[bool]:
    """Serialize one generation identity across threads/processes.

    Development ``memory://`` uses a per-identity process lock. A configured
    production Redis lock fails closed on Redis errors instead of silently
    falling back to a process-local lock, which would permit split-brain
    generation across Gunicorn processes.
    """
    started = time.monotonic()
    client = redis_client()
    if client is None:
        local_lock = _local_lock(identity)
        acquired = local_lock.acquire(timeout=config.GENERATION_LOCK_WAIT_SECONDS)
        _lock_log(
            "generation.lock.acquire",
            identity,
            wait_ms=(time.monotonic() - started) * 1000,
            outcome="acquired" if acquired else "timeout",
        )
        try:
            yield acquired
        finally:
            if acquired:
                local_lock.release()
                _lock_log("generation.lock.release", identity, outcome="released")
        return

    lock = client.lock(
        cache_key("generation-lock", identity),
        timeout=config.GENERATION_LOCK_SECONDS,
        blocking_timeout=config.GENERATION_LOCK_WAIT_SECONDS,
        thread_local=False,
    )
    try:
        acquired = bool(lock.acquire(blocking=True))
    except (RedisError, LockError):
        logger.exception(
            "generation.lock.redis_unavailable",
            extra={
                "event": "generation.lock.redis_unavailable",
                "lock_ref": pseudonymous_ref("generation-lock", identity),
                "outcome": "fail_closed" if config.IS_PRODUCTION else "local_fallback",
            },
        )
        if config.IS_PRODUCTION:
            yield False
            return
        local_lock = _local_lock(identity)
        acquired = local_lock.acquire(timeout=config.GENERATION_LOCK_WAIT_SECONDS)
        _lock_log(
            "generation.lock.acquire",
            identity,
            wait_ms=(time.monotonic() - started) * 1000,
            outcome="local_fallback_acquired" if acquired else "local_fallback_timeout",
        )
        try:
            yield acquired
        finally:
            if acquired:
                local_lock.release()
                _lock_log("generation.lock.release", identity, outcome="local_fallback_released")
        return

    _lock_log(
        "generation.lock.acquire",
        identity,
        wait_ms=(time.monotonic() - started) * 1000,
        outcome="acquired" if acquired else "timeout",
    )
    renew_stop = threading.Event()
    renewer: threading.Thread | None = None
    if acquired:
        renewer = threading.Thread(
            target=_renew_distributed_lock,
            args=(lock, renew_stop, identity),
            daemon=True,
            name="generation-lock-renewer",
        )
        renewer.start()
    try:
        yield acquired
    finally:
        renew_stop.set()
        if renewer is not None:
            renewer.join(timeout=1)
        if acquired:
            try:
                lock.release()
                _lock_log("generation.lock.release", identity, outcome="released")
            except (RedisError, LockError):
                logger.exception(
                    "generation.lock.release_failed",
                    extra={
                        "event": "generation.lock.release_failed",
                        "lock_ref": pseudonymous_ref("generation-lock", identity),
                        "outcome": "release_failed",
                    },
                )
