"""Validated structured past-paper question bank access."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import config
from rag.diagnostic import build_diagnostic_syllabus


QUESTION_BANK_SCHEMA_VERSION = "1"
_SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}
_IGNORED_NAMES = {"question_bank.schema.json", "questions.template.json", "schema.json"}


class QuestionBankValidationError(ValueError):
    """Raised when structured source data is unsafe to use for generation."""


@dataclass(frozen=True)
class PastPaperQuestion:
    question_id: str
    year: int
    paper: str
    question_number: str
    grade: int
    concept_id: str
    concept_name: str
    stem: str
    answer: str
    solution: str
    options: tuple[str, ...]
    source_file: str
    source_page: int
    competency_level: str
    mapping_status: str
    reviewed_by: str
    reviewed_at: str
    mapping_evidence: str
    source_record_id: str = ""
    source_subquestion: str = ""
    source_dataset: str = ""
    source_record_sha256: str = ""
    original_answer: str = ""
    answer_review_evidence: str = ""
    source_verification: str = ""
    correction: str = ""
    record_review_sha256: str = ""

    @property
    def canonical_answer(self) -> str:
        letter = self.answer.strip().upper()
        if len(self.options) == 4 and letter in {"A", "B", "C", "D"}:
            return self.options["ABCD".index(letter)]
        return self.answer.strip()

    def prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "year": self.year,
            "paper": self.paper,
            "question_number": self.question_number,
            "grade": self.grade,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "question": self.stem,
            "canonical_answer": self.canonical_answer,
        }
        if self.solution:
            payload["solution"] = self.solution
        if self.options:
            payload["original_options"] = list(self.options)
        return payload

    def source_reference(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "year": self.year,
            "paper": self.paper,
            "question_number": self.question_number,
            "source": self.source_file,
            "page": self.source_page,
            "grade": self.grade,
            "concept_id": self.concept_id,
            "mapping_status": self.mapping_status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "source_record_id": self.source_record_id or self.question_id,
            "source_subquestion": self.source_subquestion,
            "source_dataset": self.source_dataset,
            "source_record_sha256": self.source_record_sha256,
            "source_verification": self.source_verification,
            "answer_review_evidence": self.answer_review_evidence,
            "correction": self.correction,
            "record_review_sha256": self.record_review_sha256,
        }


def reviewed_record_digest(row: dict[str, Any]) -> str:
    value = {key: item for key, item in row.items() if key != "record_review_sha256"}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _question_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file()
        and path.suffix.lower() in _SUPPORTED_SUFFIXES
        and path.name.lower() not in _IGNORED_NAMES
    ]


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuestionBankValidationError(
                        f"{path.name}:{line_number} must contain one JSON object."
                    )
                rows.append(value)
        return rows

    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        if "paper_meta" in value:
            raise QuestionBankValidationError(
                "Raw team datasets require `python main.py import-question-bank --source <directory>` before use."
            )
        version = _normalized_text(value.get("schema_version"))
        if version and version != QUESTION_BANK_SCHEMA_VERSION:
            raise QuestionBankValidationError(
                f"{path.name} uses unsupported schema_version '{version}'."
            )
        value = value.get("questions")
    if not isinstance(value, list):
        raise QuestionBankValidationError(
            f"{path.name} must contain a questions array."
        )
    if not all(isinstance(row, dict) for row in value):
        raise QuestionBankValidationError(
            f"{path.name} contains a question that is not an object."
        )
    return [dict(row) for row in value]


def _parse_options(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("options")
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [item.strip() for item in value.split("|")]
    if value in (None, ""):
        columns = [_normalized_text(row.get(f"option_{letter}")) for letter in "abcd"]
        value = columns if any(columns) else []
    if not isinstance(value, list):
        raise QuestionBankValidationError("options must be an array or pipe-delimited text.")
    options = tuple(_normalized_text(item) for item in value)
    if options and (
        len(options) != 4
        or any(not option for option in options)
        or len({option.casefold() for option in options}) != 4
    ):
        raise QuestionBankValidationError("options must contain four unique non-empty values.")
    return options


def _parse_reviewed_at(value: Any) -> str:
    raw = _normalized_text(value)
    if not raw:
        return ""
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone.")
    return parsed.isoformat()


def _value(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _parse_question(
    row: dict[str, Any],
    *,
    source_path: Path,
    require_reviewed: bool,
) -> PastPaperQuestion:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    review = (
        row.get("mapping_review")
        if isinstance(row.get("mapping_review"), dict)
        else {}
    )
    question_id = _normalized_text(_value(row, "question_id", "id"))
    if not question_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", question_id):
        raise QuestionBankValidationError(
            "question_id must be 3-128 characters using letters, numbers, '.', '_', ':' or '-'."
        )
    try:
        year = int(_value(row, "year"))
        grade = int(_value(row, "grade"))
        source_page = int(
            _value(row, "source_page", default=_value(source, "page"))
        )
    except (TypeError, ValueError) as exc:
        raise QuestionBankValidationError(
            "year, grade and source_page must be integers."
        ) from exc
    if not 1900 <= year <= 2100:
        raise QuestionBankValidationError("year must be between 1900 and 2100.")
    if grade not in config.DIAGNOSTIC_GRADES:
        raise QuestionBankValidationError(
            f"grade must be one of {list(config.DIAGNOSTIC_GRADES)}."
        )
    if source_page < 1:
        raise QuestionBankValidationError("source_page must be at least 1.")

    options = _parse_options(row)
    answer = _normalized_text(_value(row, "answer", "correct_answer"))
    if not answer:
        raise QuestionBankValidationError("answer is required.")
    if options and answer.upper() in {"A", "B", "C", "D"}:
        pass
    elif options and answer.casefold() not in {item.casefold() for item in options}:
        raise QuestionBankValidationError(
            "answer must be an option letter or exactly match one option."
        )

    status = _normalized_text(
        _value(row, "mapping_status", default=_value(review, "status"))
    ).lower()
    reviewed_by = _normalized_text(
        _value(row, "reviewed_by", default=_value(review, "reviewed_by"))
    )
    reviewed_at_value = _value(
        row, "reviewed_at", default=_value(review, "reviewed_at")
    )
    evidence = _normalized_text(
        _value(row, "mapping_evidence", default=_value(review, "evidence"))
    )
    try:
        reviewed_at = _parse_reviewed_at(reviewed_at_value)
    except ValueError as exc:
        raise QuestionBankValidationError(str(exc)) from exc
    if require_reviewed and (
        status != "verified" or not reviewed_by or not reviewed_at or not evidence
    ):
        raise QuestionBankValidationError(
            "mapping must be verified and include reviewed_by, reviewed_at and mapping_evidence."
        )

    paper = _normalized_text(_value(row, "paper", "paper_name"))
    question_number = _normalized_text(
        _value(row, "question_number", "number")
    )
    concept_id = _normalized_text(_value(row, "concept_id"))
    concept_name = _normalized_text(_value(row, "concept_name", "topic"))
    stem = _normalized_text(_value(row, "question", "stem"))
    source_file = Path(
        _normalized_text(
            _value(row, "source_file", default=_value(source, "file", default=source_path.name))
        )
    ).name
    required = {
        "paper": paper,
        "question_number": question_number,
        "concept_id": concept_id,
        "concept_name": concept_name,
        "question": stem,
        "source_file": source_file,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise QuestionBankValidationError(
            "missing required fields: " + ", ".join(missing)
        )

    source_hash = _normalized_text(row.get("source_record_sha256"))
    answer_evidence = _normalized_text(row.get("answer_review_evidence"))
    record_hash = _normalized_text(row.get("record_review_sha256"))
    if source_hash and record_hash != reviewed_record_digest(row):
        raise QuestionBankValidationError("The imported question changed after review. Reimport it from its reviewed source.")
    if source_hash and (
        not re.fullmatch(r"[a-f0-9]{64}", source_hash)
        or not answer_evidence
        or not _normalized_text(row.get("source_record_id"))
        or not _normalized_text(row.get("source_dataset"))
        or not _normalized_text(row.get("source_verification"))
    ):
        raise QuestionBankValidationError("Imported records require complete source identity and answer-review evidence.")

    return PastPaperQuestion(
        question_id=question_id,
        year=year,
        paper=paper,
        question_number=question_number,
        grade=grade,
        concept_id=concept_id,
        concept_name=concept_name,
        stem=stem,
        answer=answer,
        solution=_normalized_text(_value(row, "solution", "explanation")),
        options=options,
        source_file=source_file,
        source_page=source_page,
        competency_level=_normalized_text(_value(row, "competency_level")),
        mapping_status=status,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        mapping_evidence=evidence,
        source_record_id=_normalized_text(row.get("source_record_id")),
        source_subquestion=_normalized_text(row.get("source_subquestion")),
        source_dataset=_normalized_text(row.get("source_dataset")),
        source_record_sha256=source_hash,
        original_answer=_normalized_text(row.get("original_answer")),
        answer_review_evidence=answer_evidence,
        source_verification=_normalized_text(row.get("source_verification")),
        correction=_normalized_text(row.get("correction")),
        record_review_sha256=record_hash,
    )


def _concept_catalog() -> dict[str, dict[str, Any]]:
    syllabus = build_diagnostic_syllabus()
    return {
        str(mapping["topic_id"]): {
            "grade": int(mapping["grade"]),
            "name": str(mapping["topic_name"]),
            "competency_levels": {
                str(level).strip() for level in mapping.get("competency_levels", [])
            },
        }
        for mapping in syllabus["topic_mappings"]
    }


def _collect(
    root: Path,
    *,
    require_reviewed: bool,
) -> tuple[list[PastPaperQuestion], list[str], list[str]]:
    questions: list[PastPaperQuestion] = []
    errors: list[str] = []
    files = _question_files(root)
    for path in files:
        try:
            rows = _rows_from_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for row_number, row in enumerate(rows, start=1):
            try:
                questions.append(
                    _parse_question(
                        row,
                        source_path=path,
                        require_reviewed=require_reviewed,
                    )
                )
            except (QuestionBankValidationError, ValueError) as exc:
                errors.append(f"{path.name}:row {row_number}: {exc}")

    catalog = _concept_catalog()
    seen_ids: set[str] = set()
    seen_stems: set[str] = set()
    valid: list[PastPaperQuestion] = []
    for question in questions:
        if question.question_id in seen_ids:
            errors.append(f"duplicate question_id: {question.question_id}")
            continue
        seen_ids.add(question.question_id)
        stem_key = _normalized_stem(question.stem)
        if stem_key in seen_stems:
            errors.append(f"duplicate question text: {question.question_id}")
            continue
        seen_stems.add(stem_key)

        concept = catalog.get(question.concept_id)
        if concept is None:
            errors.append(
                f"{question.question_id}: unknown concept_id '{question.concept_id}'"
            )
            continue
        if question.grade != concept["grade"]:
            errors.append(
                f"{question.question_id}: grade {question.grade} does not match "
                f"{question.concept_id} (Grade {concept['grade']})"
            )
            continue
        if _normalized_stem(question.concept_name) != _normalized_stem(concept["name"]):
            errors.append(
                f"{question.question_id}: concept_name does not match topics.json"
            )
            continue
        if (
            question.competency_level
            and concept["competency_levels"]
            and question.competency_level not in concept["competency_levels"]
        ):
            errors.append(
                f"{question.question_id}: competency_level is not assigned to the concept"
            )
            continue
        valid.append(question)
    return valid, errors, [str(path) for path in files]


def validate_question_bank(
    root: Path | None = None,
    *,
    require_reviewed: bool | None = None,
) -> dict[str, Any]:
    directory = Path(root or config.QUESTION_BANK_DIR)
    reviewed = (
        config.QUESTION_BANK_REQUIRE_REVIEW
        if require_reviewed is None
        else require_reviewed
    )
    questions, errors, files = _collect(directory, require_reviewed=reviewed)
    catalog = _concept_catalog()
    mapped = sorted({question.concept_id for question in questions})
    warnings: list[str] = []
    if not files:
        warnings.append(
            "No structured past-paper question files were found."
        )
    elif not questions and not errors:
        warnings.append("The structured past-paper question bank is empty.")
    if config.QUESTION_BANK_REQUIRED and not questions:
        errors.append("A non-empty structured past-paper question bank is required.")
    return {
        "ready": bool(questions) and not errors,
        "schema_version": QUESTION_BANK_SCHEMA_VERSION,
        "directory": str(directory),
        "files": files,
        "question_count": len(questions),
        "mapped_concept_count": len(mapped),
        "mapped_concepts": mapped,
        "unmapped_concepts": sorted(set(catalog).difference(mapped)),
        "errors": errors,
        "warnings": warnings,
    }


def load_question_bank(
    root: Path | None = None,
    *,
    require_reviewed: bool | None = None,
) -> list[PastPaperQuestion]:
    directory = Path(root or config.QUESTION_BANK_DIR)
    reviewed = (
        config.QUESTION_BANK_REQUIRE_REVIEW
        if require_reviewed is None
        else require_reviewed
    )
    questions, errors, files = _collect(directory, require_reviewed=reviewed)
    if errors:
        raise QuestionBankValidationError(
            "Question bank validation failed: " + "; ".join(errors[:20])
        )
    if config.QUESTION_BANK_REQUIRED and (not files or not questions):
        raise QuestionBankValidationError(
            "A validated structured past-paper question bank is required."
        )
    return questions


def question_bank_version(questions: Iterable[PastPaperQuestion] | None = None) -> str:
    records = list(questions) if questions is not None else load_question_bank()
    if not records:
        return "none"
    encoded = json.dumps(
        [asdict(question) for question in sorted(records, key=lambda item: item.question_id)],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def questions_for_concept(
    concept_id: str,
    *,
    grade: int,
    limit: int | None = None,
    exclude_question_ids: Iterable[str] = (),
    seed: str = "",
    questions: Iterable[PastPaperQuestion] | None = None,
) -> list[PastPaperQuestion]:
    excluded = {str(item) for item in exclude_question_ids}
    records = list(questions) if questions is not None else load_question_bank()
    matches = [
        question
        for question in records
        if question.concept_id == concept_id
        and question.grade == int(grade)
        and question.question_id not in excluded
    ]
    matches.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}|{item.question_id}".encode("utf-8")
        ).hexdigest()
    )
    return matches[: (limit or config.QUESTION_BANK_MAX_EXAMPLES)]
