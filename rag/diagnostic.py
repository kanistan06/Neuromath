"""Authoritative Grade 10/11 diagnostic assessment definition and validation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import config
from rag.quality import (
    answer_explanation_consistency_errors,
    is_too_similar,
    normalize_text,
)


VALIDATOR_VERSION = "8"


class DiagnosticDefinitionError(ValueError):
    pass


class DiagnosticPaperError(ValueError):
    pass


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug or "topic"


def topic_id(grade: int, term: int, number: int, name: str) -> str:
    return f"g{grade}_t{term}_{number:02d}_{_slug(name)}"


def _load_topics(path: Path | None = None) -> dict[str, Any]:
    source = path or config.TOPICS_FILE
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("grades"), list):
        raise DiagnosticDefinitionError("topics.json must contain a grades array.")
    return data


def build_diagnostic_syllabus(path: Path | None = None) -> dict[str, Any]:
    """Build exactly one mapping for every configured Grade 10/11 curriculum topic."""
    data = _load_topics(path)
    allowed_grades = set(config.DIAGNOSTIC_GRADES)
    mappings: list[dict[str, Any]] = []

    for grade_row in data["grades"]:
        grade = int(grade_row.get("grade", 0))
        if grade not in allowed_grades:
            continue
        for term_row in grade_row.get("terms", []):
            term = int(term_row.get("term", 0))
            for position, raw_topic in enumerate(term_row.get("topics", []), start=1):
                if not isinstance(raw_topic, dict):
                    continue
                name = str(raw_topic.get("topic", "")).strip()
                if not name:
                    raise DiagnosticDefinitionError(
                        f"Grade {grade}, term {term} contains an unnamed topic."
                    )
                number = int(raw_topic.get("no", position))
                mappings.append(
                    {
                        "topic_id": topic_id(grade, term, number, name),
                        "topic_name": name,
                        "grade": grade,
                        "term": term,
                        "topic_number": number,
                        "competency_levels": list(raw_topic.get("competency_levels", [])),
                        "teaching_periods": max(1, int(raw_topic.get("periods", 5) or 5)),
                        # One application-level MCQ is the diagnostic signal for this concept.
                        "required_levels": [3],
                        "weightage_percent": 0,
                        "min_questions": 1,
                        "max_questions": 1,
                        "strict_textbook_grounding": True,
                        "notes": "One textbook-grounded G.C.E. O/L diagnostic MCQ.",
                    }
                )

    if not mappings:
        raise DiagnosticDefinitionError("No configured diagnostic concepts were found.")

    seen = Counter(item["topic_id"] for item in mappings)
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        raise DiagnosticDefinitionError(f"Duplicate diagnostic topic IDs: {duplicates}")

    base_weight, remainder = divmod(100, len(mappings))
    for index, mapping in enumerate(mappings):
        mapping["weightage_percent"] = base_weight + (1 if index < remainder else 0)

    return {
        "syllabus_name": "G.C.E. O/L Grade 10 and 11 Diagnostic Assessment",
        "assessment_type": "diagnostic",
        "grades": sorted(allowed_grades),
        "total_questions": len(mappings),
        "time_limit_minutes": config.DIAGNOSTIC_TIME_LIMIT_MINUTES,
        "topic_mappings": mappings,
    }


def validate_diagnostic_paper(paper: list[dict[str, Any]], syllabus: dict[str, Any]) -> None:
    """Fail closed unless every concept has one grounded, structurally valid MCQ."""
    if syllabus.get("assessment_type") != "diagnostic":
        return
    from rag.adaptive import LEVELS, POLICY_VERSION, QUIZ_SIZE
    from rag.validation import has_current_review
    expected = [str(m["topic_id"]) for m in syllabus.get("topic_mappings", [])]
    actual = [str(q.get("topic_id", "")) for q in paper]
    actual_counts = Counter(actual)
    missing = [topic for topic in expected if actual_counts[topic] == 0]
    repeated = [topic for topic in expected if actual_counts[topic] > 1]
    unexpected = sorted(set(actual).difference(expected))

    errors: list[str] = []
    if syllabus.get("selection_version") and (
        syllabus["selection_version"] != POLICY_VERSION or len(expected) != QUIZ_SIZE
        or syllabus.get("total_questions") != QUIZ_SIZE
    ):
        errors.append("adaptive diagnostics require exactly 25 selected concepts")
    if len(paper) != len(expected):
        errors.append(f"expected {len(expected)} questions, received {len(paper)}")
    if missing:
        errors.append(f"missing concepts: {', '.join(missing[:8])}")
    if repeated:
        errors.append(f"repeated concepts: {', '.join(repeated[:8])}")
    if unexpected:
        errors.append(f"unexpected concepts: {', '.join(unexpected[:8])}")

    exact_stems: set[str] = set()
    stems_by_concept: dict[str, list[str]] = {}
    used_past_questions: set[str] = set()
    expected_grade = {str(m["topic_id"]): int(m["grade"]) for m in syllabus["topic_mappings"]}
    for index, question in enumerate(paper, start=1):
        qid = str(question.get("topic_id", ""))
        if syllabus.get("selection_version"):
            mapping = next((item for item in syllabus["topic_mappings"] if item["topic_id"] == qid), {})
            requested = [LEVELS.get(level) for level in mapping.get("required_levels", [])]
            if question.get("difficulty_level") not in requested:
                errors.append(f"question {index} has incorrect adaptive difficulty")
            if question.get("adaptive_policy_version") != POLICY_VERSION:
                errors.append(f"question {index} has no current adaptive policy")
        if not has_current_review(question):
            errors.append(f"question {index} has no current independent quality review")
        stem = re.sub(r"\s+", " ", str(question.get("question", "")).strip().lower())
        options = question.get("options", [])
        sources = question.get("source_refs", [])
        if not stem:
            errors.append(f"question {index} has an empty stem")
        else:
            normalized_stem = normalize_text(stem)
            concept_stems = stems_by_concept.setdefault(qid, [])
            if normalized_stem in exact_stems or is_too_similar(
                stem,
                concept_stems,
                threshold=config.MCQ_SIMILARITY_THRESHOLD,
            ):
                errors.append(
                    f"question {index} repeats or closely paraphrases another stem"
                )
            exact_stems.add(normalized_stem)
            concept_stems.append(stem)
        normalized_options = [str(option).strip() for option in options] if isinstance(options, list) else []
        if (
            len(normalized_options) != 4
            or any(not option for option in normalized_options)
            or len(set(normalized_options)) != 4
        ):
            errors.append(f"question {index} must have four unique options")
        if question.get("answer") not in {"A", "B", "C", "D"}:
            errors.append(f"question {index} has an invalid answer")
        explanation = str(question.get("explanation", "")).strip()
        if not explanation:
            errors.append(f"question {index} has no explanation")
        elif len(explanation) > 600 or "<think" in explanation.lower():
            errors.append(f"question {index} has an invalid concise explanation")
        for reason in answer_explanation_consistency_errors(question):
            errors.append(f"question {index} {reason}")
        evidence = str(question.get("grounding_evidence", "")).strip()
        if not 8 <= len(re.findall(r"\S+", evidence)) <= 40:
            errors.append(f"question {index} has no verified textbook evidence")
        if int(question.get("grade", 0) or 0) != expected_grade.get(qid):
            errors.append(f"question {index} has incorrect grade metadata")
        if not isinstance(sources, list) or not sources:
            errors.append(f"question {index} has no textbook source reference")
        else:
            for source in sources:
                try:
                    source_grade = int(source.get("grade", 0) or 0)
                    page_start = int(source.get("page_start", source.get("page", 0)) or 0)
                    page_end = int(source.get("page_end", page_start) or page_start)
                except (TypeError, ValueError):
                    source_grade = page_start = page_end = 0
                if source_grade != expected_grade.get(qid):
                    errors.append(f"question {index} uses a source from the wrong grade")
                if not str(source.get("source", "")).strip() or page_start < 1 or page_end < page_start:
                    errors.append(f"question {index} has an invalid source citation")
                if not str(source.get("chunk_id", "")).strip():
                    errors.append(f"question {index} has no source chunk identifier")
                if not str(source.get("corpus_version", "")).strip():
                    errors.append(f"question {index} has no corpus version")
        source_question_id = str(question.get("source_question_id", "")).strip()
        past_references = question.get("past_paper_refs", [])
        if source_question_id or past_references:
            if not source_question_id:
                errors.append(f"question {index} has no source_question_id")
            if source_question_id in used_past_questions:
                errors.append(f"question {index} repeats a past-paper source")
            used_past_questions.add(source_question_id)
            if not isinstance(past_references, list) or len(past_references) != 1:
                errors.append(f"question {index} has an invalid past-paper reference")
            else:
                reference = past_references[0]
                try:
                    reference_grade = int(reference.get("grade", 0) or 0)
                except (TypeError, ValueError):
                    reference_grade = 0
                if str(reference.get("question_id", "")) != source_question_id:
                    errors.append(f"question {index} has mismatched past-paper provenance")
                if reference_grade != expected_grade.get(qid):
                    errors.append(f"question {index} uses a past-paper source from the wrong grade")
                if str(reference.get("concept_id", "")) != qid:
                    errors.append(f"question {index} uses a past-paper source from the wrong concept")
                if str(reference.get("mapping_status", "")).lower() != "verified":
                    errors.append(f"question {index} uses an unverified question mapping")

    if errors:
        raise DiagnosticPaperError("Diagnostic paper validation failed: " + "; ".join(errors[:20]))
