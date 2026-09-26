"""Deterministic MCQ structure, answer and repetition checks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable

from rag.answers import (
    answer_explanation_consistency_errors,
    answer_values_equal,
    normalized_answer,
    numeric_answer,
)


def normalize_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9√π%+\-*/=.^∪∩∈∉⊂⊆⊃⊇≤≥≠<>²³{}\[\]]+", " ", str(value or "").lower()),
    ).strip()


def question_fingerprint(stem: str) -> str:
    return hashlib.sha256(normalize_text(stem).encode("utf-8")).hexdigest()


def _token_ngrams(value: str, size: int = 2) -> set[tuple[str, ...]]:
    tokens = normalize_text(value).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def stem_similarity(first: str, second: str) -> float:
    left = normalize_text(first)
    right = normalize_text(second)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    sequence_score = SequenceMatcher(None, left, right).ratio()
    left_grams = _token_ngrams(left)
    right_grams = _token_ngrams(right)
    union = left_grams | right_grams
    ngram_score = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(sequence_score, ngram_score)


def is_too_similar(
    stem: str,
    previous_stems: Iterable[str],
    *,
    threshold: float,
) -> bool:
    return any(
        stem_similarity(stem, previous) >= threshold
        and not _different_mathematical_data(stem, previous)
        for previous in previous_stems
    )


def _different_mathematical_data(first: str, second: str) -> bool:
    def set_bindings(value: str) -> dict[str, tuple[str, ...]]:
        return {
            name.upper(): tuple(sorted({normalized_answer(element) for element in elements.split(",")}))
            for name, elements in re.findall(r"\b([A-Z])\s*=\s*\{([^{}]*)\}", value)
        }

    left_sets, right_sets = set_bindings(first), set_bindings(second)
    if (left_sets and right_sets and left_sets != right_sets
            and sorted(left_sets.values()) != sorted(right_sets.values())):
        return True

    def signature(value: str) -> tuple[Counter, Counter]:
        value = re.sub(r"\b(?:question|item|fixture)\s+[\w-]+", "", value, flags=re.IGNORECASE)
        value = value.replace("−", "-")
        numbers = Counter(
            str(numeric_answer(match)[0])
            for match in re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:/\d+)?%?", value)
            if numeric_answer(match) is not None
        )
        operators = Counter(re.findall(r"[∪∩∈∉⊂⊆⊃⊇≤≥≠<>]|\b(?:union|intersection|complement)\b", value.lower()))
        return numbers, operators

    left_numbers, left_operators = signature(first)
    right_numbers, right_operators = signature(second)
    return bool(
        (left_numbers and right_numbers and left_numbers != right_numbers)
        or (left_operators and right_operators and left_operators != right_operators)
    )


def _answer_text(question: dict[str, Any]) -> str:
    options = question.get("options")
    answer = str(question.get("answer", "")).strip().upper()
    if not isinstance(options, list) or len(options) != 4 or answer not in {"A", "B", "C", "D"}:
        return ""
    return str(options["ABCD".index(answer)]).strip()


def _normalized_answer(value: Any) -> str:
    return normalized_answer(value)


def validate_generated_mcq(
    question: dict[str, Any],
    *,
    past_questions: Iterable[dict[str, Any]] = (),
    blocked_stems: Iterable[str] = (),
    similarity_threshold: float = 0.84,
) -> list[str]:
    """Return deterministic rejection reasons; an empty list means accepted."""
    errors: list[str] = []
    stem = str(question.get("question", "")).strip()
    options = question.get("options")
    answer = str(question.get("answer", "")).strip().upper()
    explanation = str(question.get("explanation", "")).strip()
    normalized_options = (
        [str(option).strip() for option in options]
        if isinstance(options, list)
        else []
    )
    if not stem:
        errors.append("empty question")
    if (
        len(normalized_options) != 4
        or any(not option for option in normalized_options)
        or len({option.casefold() for option in normalized_options}) != 4
    ):
        errors.append("four unique options are required")
    elif any(
        answer_values_equal(option, previous)
        for index, option in enumerate(normalized_options)
        for previous in normalized_options[:index]
    ):
        errors.append("four mathematically distinct options are required")
    if answer not in {"A", "B", "C", "D"}:
        errors.append("answer must be A, B, C or D")
    if any(
        re.search(
            r"\b(?:all|none|both|neither)\b.{0,18}\b(?:above|below)\b|\boptions?\s+[A-D]\b",
            option,
            re.IGNORECASE,
        )
        for option in normalized_options
    ):
        errors.append("option text must not depend on option letters or positions")
    if not explanation or len(explanation) > 600 or "<think" in explanation.lower():
        errors.append("a concise explanation is required")
    errors.extend(answer_explanation_consistency_errors(question))
    if stem and is_too_similar(
        stem,
        blocked_stems,
        threshold=similarity_threshold,
    ):
        errors.append("question repeats or closely paraphrases prior content")

    sources = {
        str(item.get("question_id", "")): item
        for item in past_questions
        if isinstance(item, dict) and str(item.get("question_id", ""))
    }
    source_id = str(question.get("source_question_id", "")).strip()
    if sources:
        if not source_id:
            errors.append("source_question_id is required")
        elif source_id not in sources:
            errors.append("source_question_id is not in the retrieved question set")
        else:
            source = sources[source_id]
            if source.get("concept_id") and source["concept_id"] != question.get("topic_id"):
                errors.append("past-paper source belongs to a different concept")
            if source.get("grade") and source["grade"] != question.get("grade"):
                errors.append("past-paper source belongs to a different grade")
            source_stem = str(source.get("question", "")).strip()
            if source_stem and stem_similarity(stem, source_stem) >= similarity_threshold:
                errors.append("question too closely copies the past-paper source")
            canonical = _normalized_answer(source.get("canonical_answer"))
            generated = _normalized_answer(_answer_text(question))
            if not canonical or not answer_values_equal(generated, canonical):
                errors.append("correct option does not match the verified source answer")
    elif source_id:
        errors.append("source_question_id was returned without a retrieved source")
    return errors


def deduplicate_questions(
    questions: Iterable[dict[str, Any]],
    *,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    exact_stems: set[str] = set()
    stems_by_concept: dict[str, list[str]] = {}
    for question in questions:
        stem = str(question.get("question", "")).strip()
        normalized = normalize_text(stem)
        concept_id = str(question.get("topic_id", ""))
        concept_stems = stems_by_concept.setdefault(concept_id, [])
        if stem:
            if normalized in exact_stems:
                continue
            if is_too_similar(
                stem,
                concept_stems,
                threshold=similarity_threshold,
            ):
                continue
        accepted.append(question)
        if stem:
            exact_stems.add(normalized)
            concept_stems.append(stem)
    return accepted
