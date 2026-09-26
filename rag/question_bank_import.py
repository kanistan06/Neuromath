"""Audited, atomic import of the team's nested past-paper datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import config
from rag.question_bank import QuestionBankValidationError, validate_question_bank, reviewed_record_digest


REVIEW_FILE = config.DATA_DIR / "past_paper_reviews" / "ol_2024_2025.json"
BANK_FILE_NAME = "ol_2024_2025.reviewed.json"


def record_digest(meta, question):
    value = {"paper_meta": meta, "question": question}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def read_team_dataset(source):
    root = Path(source)
    if not root.is_dir():
        raise QuestionBankValidationError("Extract neuromathdataset.zip into a directory and pass that directory as --source.")
    records = []
    seen = set()
    for path in sorted(root.rglob("*.json")):
        if path.name.lower() == "schema.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict) or not isinstance(value.get("paper_meta"), dict):
            raise QuestionBankValidationError(f"{path.name}: expected a team dataset with paper_meta.")
        meta = value["paper_meta"]
        groups = [("", value["questions"])] if "questions" in value else [
            (name, part.get("questions")) for name, part in value.get("parts", {}).items()
            if isinstance(part, dict)
        ]
        if not groups:
            raise QuestionBankValidationError(f"{path.name}: no questions or paper parts found.")
        for part, questions in groups:
            if not isinstance(questions, list):
                raise QuestionBankValidationError(f"{path.name}: questions must be an array.")
            for question in questions:
                if not isinstance(question, dict) or not isinstance(question.get("question_id"), str):
                    raise QuestionBankValidationError(f"{path.name}: invalid question object.")
                identifier = question["question_id"]
                if identifier in seen:
                    raise QuestionBankValidationError(f"Duplicate source question_id: {identifier}.")
                seen.add(identifier)
                records.append((path.name, meta, part, question))
    if not records:
        raise QuestionBankValidationError("No past-paper questions were found in the source directory.")
    return records


def _original_answer(question, suffix):
    label = "(" + suffix.replace("_", ")(") + ")" if suffix else ""
    for subquestion in question.get("sub_questions", []):
        if subquestion.get("label") == label and subquestion.get("answer"):
            return str(subquestion["answer"])
    return str(question.get("answers", {}).get("final_answer", ""))


def _normalize(file_name, meta, part, question, entry, review):
    identifier = question["question_id"]
    result = []
    for record in entry["records"]:
        suffix = record["suffix"]
        result.append({
            "question_id": identifier + ("." + suffix if suffix else ""),
            "year": int(re.match(r"\d{4}", meta["examination_year"])[0]),
            "paper": " ".join([meta.get("exam", "G.C.E. O/L"), meta.get("examination_year", ""),
                                meta.get("paper", ""), part]).strip(),
            "question_number": str(question["question_number"]) + ("." + suffix if suffix else ""),
            "grade": record["grade"], "concept_id": record["concept_id"],
            "concept_name": record["concept_name"], "competency_level": "",
            "question": record["question"], "answer": record["answer"],
            "solution": record["solution"], "options": [],
            "source_file": meta["source_question_paper"],
            "source_page": question["image_references"]["question_paper_page"],
            "mapping_status": "verified", "reviewed_by": review["reviewed_by"],
            "reviewed_at": review["reviewed_at"], "mapping_evidence": record["mapping_evidence"],
            "source_record_id": identifier, "source_subquestion": suffix,
            "source_dataset": file_name, "source_record_sha256": entry["source_sha256"],
            "original_answer": _original_answer(question, suffix),
            "answer_review_evidence": record["answer_review_evidence"],
            "source_verification": review["source_verification"], "correction": record.get("correction", ""),
        })
        result[-1]["record_review_sha256"] = reviewed_record_digest(result[-1])
    return result


def import_question_bank(source, *, output_dir=None, report_path=None, review_file=None):
    output = Path(output_dir or config.QUESTION_BANK_DIR)
    if report_path and (Path(report_path).resolve().is_relative_to(output.resolve())
                        or Path(report_path).resolve().is_relative_to(Path(source).resolve())):
        raise QuestionBankValidationError("Write the import report outside the source dataset and active bank directories.")
    raw = read_team_dataset(source)
    review = json.loads(Path(review_file or REVIEW_FILE).read_text(encoding="utf-8"))
    records, audit = [], []
    for file_name, meta, part, question in raw:
        identifier = question["question_id"]
        entry = review["records"].get(identifier)
        if not entry:
            status, reason, normalized = "held", "No recorded concept and answer review exists for this question.", []
        elif entry["source_dataset"] != file_name or entry["source_sha256"] != record_digest(meta, question):
            status, reason, normalized = "held", "Source content changed after review; a new review is required.", []
        else:
            status, reason = entry["status"], entry["reason"]
            normalized = _normalize(file_name, meta, part, question, entry, review) if status == "accepted" else []
        records.extend(normalized)
        audit.append({"source_record_id": identifier, "source_dataset": file_name,
                      "status": status, "reason": reason,
                      "generated_record_ids": [record["question_id"] for record in normalized]})
    if not records:
        raise QuestionBankValidationError("No reviewed questions matched the supplied dataset; the existing bank was preserved.")
    output = Path(output_dir or config.QUESTION_BANK_DIR)
    output.mkdir(parents=True, exist_ok=True)
    target = output / BANK_FILE_NAME
    payload = {"schema_version": "1", "review_version": review["review_version"], "questions": records}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.TemporaryDirectory(prefix="neuromath-bank-") as scratch:
        stage = Path(scratch)
        for existing in output.iterdir():
            if existing.name == BANK_FILE_NAME:
                continue
            if existing.is_file():
                shutil.copy2(existing, stage / existing.name)
            elif existing.is_dir():
                shutil.copytree(existing, stage / existing.name)
        (stage / BANK_FILE_NAME).write_text(encoded, encoding="utf-8")
        validation = validate_question_bank(stage, require_reviewed=True)
        if not validation["ready"]:
            raise QuestionBankValidationError("Import validation failed; existing bank preserved: "
                                              + "; ".join(validation["errors"][:10]))
    if not target.exists() or target.read_text(encoding="utf-8") != encoded:
        descriptor, temporary = tempfile.mkstemp(prefix=".bank-", suffix=".tmp", dir=output)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
            Path(temporary).replace(target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    report = {
        "ready": True, "review_version": review["review_version"],
        "review_scope": "AI-assisted mathematical and concept audit of supplied dataset text; original PDFs and diagrams were not supplied.",
        "source_question_count": len(raw),
        "accepted_source_questions": sum(item["status"] == "accepted" for item in audit),
        "held_source_questions": sum(item["status"] != "accepted" for item in audit),
        "imported_question_count": len(records), "mapped_concept_count": validation["mapped_concept_count"],
        "unmapped_concepts": validation["unmapped_concepts"], "output": str(target),
        "source_audit": audit,
    }
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
