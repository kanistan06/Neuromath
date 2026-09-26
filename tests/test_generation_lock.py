import threading
from concurrent.futures import ThreadPoolExecutor

import config
from infrastructure import cache


def test_local_generation_locks_allow_different_users_to_generate_concurrently(monkeypatch):
    monkeypatch.setattr(cache, "redis_client", lambda: None)
    monkeypatch.setattr(config, "GENERATION_LOCK_WAIT_SECONDS", 0.2)

    entered = threading.Barrier(3)
    release = threading.Event()

    def generate(identity: str) -> bool:
        with cache.generation_lock(identity) as acquired:
            if not acquired:
                return False
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(generate, "practice-student-1")
        second = executor.submit(generate, "practice-student-2")
        entered.wait(timeout=2)
        release.set()
        assert first.result(timeout=2) is True
        assert second.result(timeout=2) is True


def test_local_generation_lock_still_blocks_duplicate_generation_for_same_user(monkeypatch):
    monkeypatch.setattr(cache, "redis_client", lambda: None)
    monkeypatch.setattr(config, "GENERATION_LOCK_WAIT_SECONDS", 0.05)

    started = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with cache.generation_lock("practice-student-1") as acquired:
            assert acquired is True
            started.set()
            release.wait(timeout=2)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert started.wait(timeout=2)

    try:
        with cache.generation_lock("practice-student-1") as acquired:
            assert acquired is False
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()


def test_production_redis_lock_failure_fails_closed(monkeypatch):
    from redis.exceptions import RedisError

    class BrokenLock:
        def acquire(self, *, blocking=True):
            raise RedisError("redis unavailable")

    class BrokenRedis:
        def lock(self, *_args, **_kwargs):
            return BrokenLock()

    monkeypatch.setattr(cache, "redis_client", lambda: BrokenRedis())
    monkeypatch.setattr(config, "IS_PRODUCTION", True)

    with cache.generation_lock("quiz-generation-student-7") as acquired:
        assert acquired is False


def test_quiz_prefetch_lock_does_not_block_foreground_start_for_same_user(monkeypatch):
    """Long background prefetch must never make a new quiz-start request wait."""
    monkeypatch.setattr(cache, "redis_client", lambda: None)
    monkeypatch.setattr(config, "GENERATION_LOCK_WAIT_SECONDS", 0.05)

    started = threading.Event()
    release = threading.Event()

    def hold_prefetch() -> None:
        with cache.generation_lock("quiz-prefetch-student-7-quiz-old") as acquired:
            assert acquired is True
            started.set()
            release.wait(timeout=2)

    worker = threading.Thread(target=hold_prefetch)
    worker.start()
    assert started.wait(timeout=2)

    try:
        with cache.generation_lock("quiz-start-student-7") as acquired:
            assert acquired is True
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()


def test_foreground_quiz_start_locks_are_isolated_between_students(monkeypatch):
    monkeypatch.setattr(cache, "redis_client", lambda: None)
    monkeypatch.setattr(config, "GENERATION_LOCK_WAIT_SECONDS", 0.2)

    entered = threading.Barrier(3)
    release = threading.Event()

    def hold(identity: str) -> bool:
        with cache.generation_lock(identity) as acquired:
            if not acquired:
                return False
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(hold, "quiz-start-student-101")
        second = executor.submit(hold, "quiz-start-student-202")
        entered.wait(timeout=2)
        release.set()
        assert first.result(timeout=2) is True
        assert second.result(timeout=2) is True
