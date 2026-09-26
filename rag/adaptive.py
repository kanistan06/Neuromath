"""Curriculum-weighted selection from a student's completed assessment history."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any


POLICY_VERSION = "adaptive-25-v1"
QUIZ_SIZE = 25
LEVELS = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyse", 5: "Evaluate", 6: "Create"}


def topic_progress(history: list[dict[str, Any]]) -> dict[str, Any]:
    attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        attempts[int(row["attempt_id"])].append(row)
    recent = [attempts[key] for key in sorted(attempts, reverse=True)[:2]]
    accuracy = (
        sum(bool(row["is_correct"]) for row in recent[0]) / len(recent[0])
        if recent else 0.0
    )
    level = 3
    for candidate in (4, 5):
        passed = []
        for rows in recent:
            eligible = [
                row for row in rows
                if next((key for key, name in LEVELS.items()
                         if name == row.get("difficulty_level")), 0) >= candidate - 1
            ]
            passed.append(
                bool(eligible)
                and sum(bool(row["is_correct"]) for row in rows) / len(rows) >= 0.8
                and sum(bool(row["is_correct"]) for row in eligible) / len(eligible) >= 0.8
            )
        if len(passed) == 2 and all(passed):
            level = candidate
    return {
        "seen": bool(history),
        "accuracy": accuracy,
        "next_level": level,
        "last_attempt": max((int(row.get("attempt_number", 0)) for row in history), default=0),
    }


def select_diagnostic_syllabus(
    catalogue: dict[str, Any],
    *,
    student_key: str,
    history_by_topic: dict[str, list[dict[str, Any]]] | None = None,
    attempt_count: int = 0,
    diagnostic_count: int = 0,
    latest_attempt_id: int = 0,
    variant: str = "",
) -> dict[str, Any]:
    history = history_by_topic or {}
    mappings = catalogue.get("topic_mappings", [])
    if len({row["topic_id"] for row in mappings}) != len(mappings) or len(mappings) < QUIZ_SIZE:
        raise ValueError("The diagnostic catalogue must contain at least 25 distinct concepts.")
    grades = sorted({int(row["grade"]) for row in mappings})
    if grades != [10, 11]:
        raise ValueError("The diagnostic catalogue must cover Grades 10 and 11.")
    seed = hashlib.sha256(
        f"{POLICY_VERSION}:{student_key}:{attempt_count}:{latest_attempt_id}:{variant}".encode()
    ).hexdigest()
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    family_counts: Counter = Counter()

    def choose(pool: list[dict[str, Any]], count: int) -> None:
        races = {}
        for mapping in pool:
            progress = topic_progress(history.get(mapping["topic_id"], []))
            age = max(0, attempt_count - progress["last_attempt"])
            # Curriculum teaching periods are a priority proxy, not exam marks.
            periods = max(1, min(30, int(mapping.get("teaching_periods", 5))))
            weight = periods * (1 + min(age, 20) * 0.4)
            races[mapping["topic_id"]] = -math.log(max(rng.random(), 1e-12)) / weight
        for _ in range(min(count, len(pool))):
            def priority(mapping: dict[str, Any]) -> tuple[float, str]:
                families = {str(value).split(".")[0] for value in mapping.get("competency_levels", [])}
                penalty = 1 + 0.6 * sum(family_counts[family] for family in families)
                return races[mapping["topic_id"]] * penalty, mapping["topic_id"]

            chosen = min(pool, key=priority)
            pool.remove(chosen)
            selected.append(chosen)
            family_counts.update({str(value).split(".")[0] for value in chosen.get("competency_levels", [])})

    for index, grade in enumerate(grades):
        quota = QUIZ_SIZE // 2 + int(index == diagnostic_count % 2)
        pool = [row for row in mappings if int(row["grade"]) == grade]
        if len(pool) < quota:
            raise ValueError(f"Grade {grade} requires at least {quota} available concepts.")
        unseen, weak = [], []
        for row in pool:
            progress = topic_progress(history.get(row["topic_id"], []))
            if not progress["seen"]:
                unseen.append(row)
            elif progress["accuracy"] < 0.8:
                weak.append(row)
        before = len(selected)
        choose(weak, min(len(weak), quota // 3))
        choose(unseen, quota - (len(selected) - before))
        used = {row["topic_id"] for row in selected}
        choose([row for row in pool if row["topic_id"] not in used], quota - (len(selected) - before))

    chosen_mappings = []
    for row in selected:
        mapping = copy.deepcopy(row)
        topic_history = history.get(mapping["topic_id"], [])
        progress = topic_progress(topic_history)
        mapping.update({
            "required_levels": [progress["next_level"]],
            "min_questions": 1,
            "max_questions": 1,
            "weightage_percent": 4,
            "require_application": True,
            "adaptive_policy_version": POLICY_VERSION,
            "generation_offset": rng.randrange(1_000_000),
            "avoid_question_stems": list(dict.fromkeys(
                str(item["question"]) for item in topic_history if item.get("question")
            )),
            "excluded_source_question_ids": sorted({
                str(item["source_question_id"]) for item in topic_history if item.get("source_question_id")
            }),
        })
        chosen_mappings.append(mapping)
    result = copy.deepcopy(catalogue)
    result.update({
        "total_questions": QUIZ_SIZE,
        "catalogue_concept_count": len(mappings),
        "topic_mappings": chosen_mappings,
        "selection_version": POLICY_VERSION,
        "selection_seed": seed,
    })
    return result
