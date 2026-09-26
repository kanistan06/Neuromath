"""MCQ Generator module."""

import json
import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Dict, List

import config
from rag.runpod_inference import runpod_mcq_generation
from rag.answers import answer_letter_from_value, strip_option_prefix
from rag.prompts import get_mcq_prompt
from rag.quality import deduplicate_questions, validate_generated_mcq
from rag.inference_errors import access_failure, inference_budget


logger = logging.getLogger(__name__)


class MCQPromptError(RuntimeError):
    pass


def _generation_error_detail(exc: Exception) -> str:
    failure = access_failure(exc)
    detail = str(failure) if failure else f"{type(exc).__name__}: {exc}"
    if config.RUNPOD_API_KEY:
        detail = detail.replace(config.RUNPOD_API_KEY, "[REDACTED]")
    detail = re.sub(r"rpa_[A-Za-z0-9_-]+", "[REDACTED]", detail)
    return re.sub(r"\s+", " ", detail)[:500]


def _fatal_generation_error(exc: Exception) -> bool:
    if access_failure(exc) is not None:
        return True
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, MCQPromptError):
            return True
        if getattr(current, "status_code", None) in {400, 401, 402, 403, 404, 422}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (value or "").lower())).strip()


# Stems that depend on an external figure/image — drop; quiz is text-only.
_FIGURE_PHRASES = (
    "as shown in the figure",
    "as shown in figure",
    "as shown in the diagram",
    "as shown in diagram",
    "as shown in the graph",
    "as shown in the picture",
    "as shown in the image",
    "shown in the figure",
    "shown in the diagram",
    "shown in the graph",
    "see the figure",
    "see the diagram",
    "see the graph",
    "see the picture",
    "see the image",
    "see figure",
    "see diagram",
    "in the figure below",
    "in the diagram below",
    "in the graph below",
    "in the picture below",
    "according to the figure",
    "according to the diagram",
    "according to the graph",
    "refer to the figure",
    "refer to the diagram",
    "refer to the graph",
    "refer to the picture",
    "from the figure",
    "from the diagram",
    "the figure shows",
    "the diagram shows",
    "the graph shows",
    "the image shows",
    "the picture shows",
    "below shows a",
    "above shows a",
    "in the accompanying figure",
    "in the accompanying diagram",
    "use the figure",
    "use the diagram",
)


def _question_requires_external_visual(stem: str) -> bool:
    t = (stem or "").lower()
    return any(p in t for p in _FIGURE_PHRASES)


def _dedupe_questions_preserve_order(questions: List[Dict]) -> List[Dict]:
    return deduplicate_questions(
        questions,
        similarity_threshold=config.MCQ_SIMILARITY_THRESHOLD,
    )


def _is_probably_copied_question(question: str, context_chunks: List[str]) -> bool:
    """
    Heuristic guardrail:
    if a sufficiently long normalized question appears as a contiguous substring
    in any context chunk, treat it as copied and reject it.
    """
    norm_q = _normalize_text(question)
    if len(norm_q) < 40:
        return False
    for chunk in context_chunks or []:
        if norm_q in _normalize_text(chunk):
            return True
    return False


def _invoke_llm(
    prompt: str,
    *,
    strict_grounding: bool,
    use_past_questions: bool,
    question_count: int,
) -> str:
    return runpod_mcq_generation(
        prompt,
        strict_grounding=strict_grounding,
        use_past_questions=use_past_questions,
        question_count=question_count,
    )


def _extract_first_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_mcq_response(raw_text: str, *, require_correct_answer: bool = False) -> List[Dict]:
    """Parse the LLM output into structured MCQ dicts."""
    text = (raw_text or "").strip()
    if not text:
        return []

    # Remove markdown fences when present.
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

    candidates: list[str] = [text]
    array_candidate = _extract_first_json_array(text)
    if array_candidate:
        candidates.append(array_candidate)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            parsed = parsed["questions"]
        if not isinstance(parsed, list):
            continue

        out: list[Dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            options = item.get("options", [])
            if not isinstance(options, list):
                options = []
            options = [strip_option_prefix(o) for o in options]
            if len(options) != 4:
                continue
            if require_correct_answer and "correct_answer" not in item:
                continue
            raw_answer = str(item.get("answer", "")).strip()
            answer = raw_answer.upper()
            if "correct_answer" in item:
                try:
                    answer = answer_letter_from_value(
                        strip_option_prefix(item["correct_answer"]),
                        options,
                    )
                except ValueError:
                    continue
            elif answer not in {"A", "B", "C", "D"}:
                # Accept answer text and map it to option letter.
                mapped = ""
                for idx, opt in enumerate(options):
                    if raw_answer.strip().lower() == opt.strip().lower():
                        mapped = "ABCD"[idx]
                        break
                if not mapped:
                    match = re.fullmatch(
                        r"(?:option\s+)?([A-D])(?:[.)])?",
                        raw_answer,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        mapped = match.group(1).upper()
                answer = mapped
            if answer not in {"A", "B", "C", "D"}:
                continue
            question = str(item.get("question", "")).strip()
            if not question or len(options) != 4:
                continue
            row: Dict = {
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": str(item.get("explanation", "")).strip(),
            }
            if "correct_answer" in item:
                row["correct_answer"] = options["ABCD".index(answer)]
            evidence_id = str(item.get("evidence_id", "")).strip().upper()
            if evidence_id:
                row["evidence_id"] = evidence_id
            evidence = str(item.get("grounding_evidence", "")).strip()
            if evidence:
                row["grounding_evidence"] = evidence
            if "grade" in item:
                row["grade"] = item.get("grade")
            if "concept_id" in item:
                row["concept_id"] = str(item.get("concept_id", "")).strip()
            if "source" in item:
                row["source"] = Path(str(item.get("source", ""))).name
            if "page" in item:
                row["page"] = item.get("page")
            if "source_question_id" in item:
                row["source_question_id"] = str(
                    item.get("source_question_id", "")
                ).strip()
            out.append(row)
        if out:
            return out

    logger.warning("Model response did not contain a valid MCQ JSON object")
    return []


_MCQ_CONTEXT_CHAR_CAP = 3600
_EVIDENCE_MAX_CANDIDATES = 12
_EVIDENCE_CANDIDATES_PER_ATTEMPT = 4
_EVIDENCE_MIN_WORDS = 8
_EVIDENCE_MAX_WORDS = 40


def _evidence_passages(text: str) -> list[str]:
    """Return short contiguous source excerpts suitable for evidence IDs."""
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return []

    raw_segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\s*[\r\n]+\s*", compact)
        if segment.strip()
    ]
    candidates: list[str] = []
    seen: set[str] = set()

    def add_words(words: list[str]) -> None:
        if not (_EVIDENCE_MIN_WORDS <= len(words) <= _EVIDENCE_MAX_WORDS):
            return
        passage = " ".join(words).strip()
        key = _normalize_text(passage)
        if key and key not in seen:
            seen.add(key)
            candidates.append(passage)

    for segment in raw_segments:
        words = segment.split()
        if len(words) <= _EVIDENCE_MAX_WORDS:
            add_words(words)
            continue
        # Long textbook paragraphs are broken into contiguous windows. An overlap
        # keeps formulas/definitions near a boundary available to the next window.
        step = 28
        for offset in range(0, len(words), step):
            window = words[offset : offset + _EVIDENCE_MAX_WORDS]
            add_words(window)
            if offset + _EVIDENCE_MAX_WORDS >= len(words):
                break

    if not candidates:
        words = compact.split()
        if len(words) >= _EVIDENCE_MIN_WORDS:
            add_words(words[:_EVIDENCE_MAX_WORDS])
    return candidates


def _build_evidence_catalog(
    grounding_units: list[dict],
    topic_name: str,
) -> list[dict]:
    """Rank already-retrieved source excerpts without changing retrieval logic."""
    topic_terms = {
        token
        for token in _normalize_text(topic_name).split()
        if len(token) > 2
    }
    topic_phrase = _normalize_text(topic_name)
    ranked: list[tuple[float, int, dict]] = []
    order = 0

    for unit_rank, unit in enumerate(grounding_units or []):
        source_ref = dict(unit.get("source_ref", {}) or {})
        for passage in _evidence_passages(str(unit.get("text", ""))):
            normalized = _normalize_text(passage)
            passage_terms = set(normalized.split())
            coverage = len(topic_terms & passage_terms)
            phrase_bonus = 1 if topic_phrase and topic_phrase in normalized else 0
            score = (coverage * 10.0) + (phrase_bonus * 3.0) - (unit_rank * 0.05)
            ranked.append(
                (
                    score,
                    order,
                    {"text": passage, "source_ref": source_ref},
                )
            )
            order += 1

    ranked.sort(key=lambda row: (-row[0], row[1]))
    catalog: list[dict] = []
    seen: set[str] = set()
    for _score, _order, candidate in ranked:
        key = _normalize_text(candidate["text"])
        if not key or key in seen:
            continue
        seen.add(key)
        catalog.append(candidate)
        if len(catalog) >= _EVIDENCE_MAX_CANDIDATES:
            break
    return catalog


def _evidence_batch_for_attempt(
    grounding_units: list[dict],
    topic_name: str,
    attempt_index: int,
) -> list[dict]:
    """Use a small labelled evidence window and rotate it across outer retries."""
    catalog = _build_evidence_catalog(grounding_units, topic_name)
    if not catalog:
        return []

    batch_size = min(_EVIDENCE_CANDIDATES_PER_ATTEMPT, len(catalog))
    start = (max(0, int(attempt_index)) * batch_size) % len(catalog)
    selected = [catalog[(start + offset) % len(catalog)] for offset in range(batch_size)]
    return [
        {"evidence_id": f"E{index}", **candidate}
        for index, candidate in enumerate(selected, start=1)
    ]



def _source_window(parts: list[str], offset: int) -> str:
    passages = []
    for part in parts:
        text = str(part).strip()
        if len(text) <= _MCQ_CONTEXT_CHAR_CAP:
            if text:
                passages.append(text)
        else:
            passages.extend(piece.strip() for piece in re.split(r"\n\s*\n|(?<=[.!?])\s+", text)
                            if piece.strip() and len(piece.strip()) <= _MCQ_CONTEXT_CHAR_CAP)
    if not passages:
        return ""
    start = max(0, offset) % len(passages)
    selected = []
    length = 0
    for part in passages[start:] + passages[:start]:
        extra_length = len(part) + (2 if selected else 0)
        if length + extra_length > _MCQ_CONTEXT_CHAR_CAP:
            continue
        selected.append(part)
        length += extra_length
    return "\n\n".join(selected)


def generate_mcqs(
    topic_name: str,
    difficulty_level: str,
    context_chunks: List[str],
    num_questions: int = 5,
    strict_grounding: bool = False,
    grade: int | None = None,
    concept_id: str = "",
    grounding_units: list[dict] | None = None,
    past_questions: list[dict] | None = None,
    avoid_question_stems: list[str] | None = None,
    grounding_attempt: int = 0,
    rejection_feedback: list[str] | None = None,
    require_application: bool = False,
    variation_offset: int = 0,
    on_accepted=None,
    accept_limit: int | None = None,
) -> List[Dict]:
    """Generate validated JSON MCQ candidates for one curriculum concept."""
    use_past_questions = bool(past_questions)
    prompt = get_mcq_prompt(
        difficulty_level,
        strict_grounding=strict_grounding,
        use_past_questions=use_past_questions,
    )

    evidence_map: dict[str, dict] = {}
    if strict_grounding:
        evidence_batch = _evidence_batch_for_attempt(
            grounding_units or [],
            topic_name,
            grounding_attempt,
        )
        if not evidence_batch:
            logger.warning(
                "No usable 8-40 word textbook evidence passages were available for %s",
                topic_name,
            )
            return []
        evidence_map = {item["evidence_id"]: item for item in evidence_batch}
        context = "\n\n".join(
            f"[{item['evidence_id']}] {item['text']}" for item in evidence_batch
        )
    elif grounding_units:
        context_parts = []
        for unit in grounding_units:
            reference = unit.get("source_ref", {})
            context_parts.append(
                "SOURCE_FILE: {source}\nPAGE: {page}\nGRADE: {grade}\nCONTENT:\n{text}".format(
                    source=Path(str(reference.get("source", "textbook"))).name,
                    page=reference.get("page", reference.get("page_start", "")),
                    grade=reference.get("grade", grade or ""),
                    text=str(unit.get("text", "")).strip(),
                )
            )
        context = _source_window(context_parts, grounding_attempt)
    else:
        context = _source_window(context_chunks or [], grounding_attempt)

    if len(context) > _MCQ_CONTEXT_CHAR_CAP:
        context = context[:_MCQ_CONTEXT_CHAR_CAP]
    if not context.strip():
        return []

    prompt_stems = list(dict.fromkeys(avoid_question_stems or []))
    if len(prompt_stems) > 12:
        prompt_stems = prompt_stems[:8] + prompt_stems[-4:]
    try:
        formatted_prompt = prompt.format(
            topic=topic_name,
            grade=grade if grade is not None else "",
            concept_id=concept_id,
            context=context,
            past_questions=json.dumps(
                past_questions or [],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            avoid_question_stems=json.dumps(
                [str(stem)[:220] for stem in prompt_stems],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            num_questions=num_questions,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise MCQPromptError(
            "MCQ prompt formatting failed. Check rag/prompts.py placeholder and JSON braces."
        ) from exc

    directions = (
        "Apply a different supported property in a fresh self-contained problem.",
        "Choose a fresh example with different given data or conditions.",
        "Test a different supported operation, relationship, or special case.",
        "Use a different supported classification or interpretation of the concept.",
        "Choose a fresh self-contained scenario with a different requested quantity.",
        "Use another supported property to distinguish a common misconception.",
    )
    if use_past_questions:
        formatted_prompt += (
            "\nVary the narrative and distractors while preserving the selected source's "
            "mathematical givens, requested quantity, and canonical answer. Do not introduce "
            "new numerical values or choose a different task from the source problem."
        )
    else:
        formatted_prompt += (
            f"\nVariation instruction: {directions[(grounding_attempt + variation_offset) % len(directions)]}"
            " Stay at the requested Bloom level and within the supplied textbook material."
        )
    if require_application:
        formatted_prompt += (
            "\nRequire at least two connected reasoning steps in a contextual problem. "
            "No definitions, formula recall, restating a given, or bare one-step arithmetic. "
            "Use Grade-appropriate reasoning, not artificially large numbers."
        )
    if num_questions > 1:
        formatted_prompt += (
            "\nBatch diversity requirement: make every candidate use a meaningfully different "
            "task structure, representation, requested quantity, scenario, and mathematical data. "
            "Do not create near-duplicates by changing only names or numbers."
        )
    if rejection_feedback:
        formatted_prompt += "\nCorrect these issues from the previous attempt: " + "; ".join(
            list(dict.fromkeys(rejection_feedback))[-6:]
        )

    questions = parse_mcq_response(
        _invoke_llm(
            formatted_prompt,
            strict_grounding=strict_grounding,
            use_past_questions=use_past_questions,
            question_count=num_questions,
        ),
        require_correct_answer=True,
    )
    accepted: list[Dict] = []
    blocked_stems = list(avoid_question_stems or [])
    used_source_ids: set[str] = set()

    for question in questions:
        question["topic_id"] = concept_id or topic_name
        question["difficulty_level"] = difficulty_level
        if grade is not None:
            question["grade"] = grade
        if strict_grounding:
            evidence_id = str(question.get("evidence_id", "")).strip().upper()
            evidence = evidence_map.get(evidence_id)
            if evidence is None:
                logger.warning(
                    "Generated MCQ rejected for %s: invalid evidence_id %s",
                    topic_name,
                    evidence_id or "<missing>",
                    extra={"event": "mcq.grounding.rejected", "topic": topic_name},
                )
                continue
            # Trusted provenance replaces model-authored metadata. This exact
            # excerpt is known to be contiguous within the retrieved source unit.
            question["grounding_evidence"] = evidence["text"]
            question["_grounding_source_ref"] = dict(evidence["source_ref"])

        stem = str(question.get("question", ""))
        if _question_requires_external_visual(stem):
            continue
        if _is_probably_copied_question(stem, context_chunks):
            continue
        errors = validate_generated_mcq(
            question,
            past_questions=past_questions or [],
            blocked_stems=blocked_stems,
            similarity_threshold=config.MCQ_SIMILARITY_THRESHOLD,
        )
        # The author explanation is not trusted evidence. The independent reviewer
        # is deliberately blinded to it and, on success, replaces it with a clean
        # mathematical justification. Do not discard an otherwise valid stem/key
        # solely because the author's prose names a stale option/value. Key, option,
        # deterministic-math, grounding, ambiguity and duplicate failures remain
        # strict and are never filtered here.
        author_explanation_only_errors = {
            "explanation contains self-correction or unresolved reasoning",
            "answer key contradicts explanation, which identifies a different option as correct",
            "explanation states multiple different final answers",
            "answer key contradicts the sole option value stated as the explanation's conclusion",
        }
        errors = [reason for reason in errors if reason not in author_explanation_only_errors]
        if not errors:
            from rag.validation import review_mcq

            if strict_grounding:
                unit = _grounding_unit_for_evidence(question["grounding_evidence"], grounding_units or [])
                review_context = _source_window(
                    [question["grounding_evidence"], str(unit.get("text", "")) if unit else ""], 0,
                )
            else:
                review_context = context
            errors.extend(review_mcq(
                question, topic_name=topic_name, difficulty_level=difficulty_level,
                source_context=review_context, require_application=require_application,
            ))
            if not errors and use_past_questions:
                errors.extend(validate_generated_mcq(
                    question, past_questions=past_questions or [], blocked_stems=blocked_stems,
                    similarity_threshold=config.MCQ_SIMILARITY_THRESHOLD,
                ))
        if errors:
            if rejection_feedback is not None:
                rejection_feedback.extend(errors)
                if any("repeats or closely paraphrases" in reason for reason in errors):
                    rejection_feedback.append("Replace the mathematical data or task in this repeated stem: " + stem[:220])
            logger.warning(
                "Generated MCQ rejected for %s: %s",
                topic_name,
                "; ".join(errors),
                extra={
                    "event": "mcq.quality.rejected",
                    "topic": topic_name,
                    "reasons": errors,
                },
            )
            continue
        source_question_id = str(question.get("source_question_id", "")).strip()
        if source_question_id and source_question_id in used_source_ids:
            continue
        accepted.append(question)
        if on_accepted is not None:
            on_accepted(question)
        blocked_stems.append(stem)
        if source_question_id:
            used_source_ids.add(source_question_id)
        if accept_limit is not None and len(accepted) >= max(1, int(accept_limit)):
            break

    questions = _dedupe_questions_preserve_order(accepted)
    for q in questions:
        q["topic_id"] = concept_id or topic_name
        q["difficulty_level"] = difficulty_level
        q.pop("image_prompt", None)
        q.pop("image_url", None)

    output_limit = num_questions if accept_limit is None else max(1, int(accept_limit))
    return questions[:output_limit]


def _grounding_unit_for_evidence(
    evidence: str,
    grounding_units: list[dict],
) -> dict | None:
    """Return the exact textbook chunk/source that contains an evidence excerpt."""
    normalized_evidence = _normalize_text(evidence)
    word_count = len(normalized_evidence.split())
    if not normalized_evidence or word_count < 8 or word_count > 40 or len(evidence) > 500:
        return None
    for unit in grounding_units:
        text = _normalize_text(str(unit.get("text", "")))
        if normalized_evidence in text:
            return unit
    return None


def _attach_question_metadata(q, *, topic_id, mapping_metadata, source_refs, grounding_units, past_paper_refs):
    if mapping_metadata.get("adaptive_policy_version"):
        q["adaptive_policy_version"] = mapping_metadata["adaptive_policy_version"]
    expected_grade = int(mapping_metadata.get("grade", 0) or 0)
    if mapping_metadata.get("term") is not None:
        q["term"] = int(mapping_metadata["term"])
    if mapping_metadata.get("strict_textbook_grounding"):
        # Provenance comes from the evidence ID resolved inside
        # generate_mcqs, never from model-authored source metadata.
        reference = dict(q.pop("_grounding_source_ref", {}) or {})
        if not reference:
            unit = _grounding_unit_for_evidence(
                str(q.get("grounding_evidence", "")),
                grounding_units,
            )
            reference = (
                dict(unit.get("source_ref", {}) or {})
                if unit is not None
                else {}
            )
        try:
            source_grade = int(reference.get("grade", 0) or 0)
            page_start = int(
                reference.get("page_start", reference.get("page", 0)) or 0
            )
            page_end = int(
                reference.get("page_end", page_start) or page_start
            )
        except (TypeError, ValueError):
            source_grade = page_start = page_end = 0
        metadata_valid = (
            bool(reference)
            and source_grade == expected_grade
            and bool(str(reference.get("source", "")).strip())
            and page_start >= 1
            and page_end >= page_start
            and bool(str(reference.get("chunk_id", "")).strip())
            and bool(str(reference.get("corpus_version", "")).strip())
        )
        q["source_refs"] = [reference] if metadata_valid else []
    else:
        q["source_refs"] = list(source_refs)
    past_ref_by_id = {
        str(reference.get("question_id", "")): reference
        for reference in past_paper_refs
        if isinstance(reference, dict)
    }
    source_question_id = str(q.get("source_question_id", "")).strip()
    q["past_paper_refs"] = (
        [dict(past_ref_by_id[source_question_id])]
        if source_question_id in past_ref_by_id
        else []
    )
    q["topic_id"] = topic_id
    if expected_grade:
        q["grade"] = expected_grade
    q.pop("evidence_id", None)
    q.pop("concept_id", None)
    q.pop("source", None)
    q.pop("page", None)
    q.pop("_grounding_source_ref", None)
    return q


def _generate_mcqs_block(args: tuple) -> List[Dict]:
    (
        topic_id,
        topic_name,
        level_name,
        chunks,
        questions_per_level,
        mapping_metadata,
        source_refs,
        grounding_units,
        past_questions,
        past_paper_refs,
    ) = args
    last_error: Exception | None = None
    collected: list[Dict] = []
    collected_stems: list[str] = []
    used_source_ids: set[str] = set()
    base_avoid_stems = list(mapping_metadata.get("avoid_question_stems", []) or [])
    rejection_feedback: list[str] = []
    attempt_offset = int(mapping_metadata.get("generation_offset", 0))

    def checkpoint(question):
        callback = mapping_metadata.get("_on_progress")
        if callback is None:
            return
        from rag.validation import has_current_review

        candidate = _attach_question_metadata(
            json.loads(json.dumps(question)), topic_id=topic_id, mapping_metadata=mapping_metadata,
            source_refs=source_refs, grounding_units=grounding_units, past_paper_refs=past_paper_refs,
        )
        if (has_current_review(candidate)
                and (not mapping_metadata.get("strict_textbook_grounding") or candidate.get("source_refs"))
                and (not candidate.get("source_question_id") or candidate.get("past_paper_refs"))):
            callback([candidate])

    # A past-paper item is a useful seed, but a strict diagnostic must not become
    # impossible when a small model repeatedly copies that seed or cannot preserve
    # its verified answer cleanly. Give the verified source one bounded authoring
    # attempt first. If that attempt produces no accepted item, the remaining
    # retries stay fully textbook-grounded but stop forcing a past-paper transform.
    # This changes only the source constraint; every structure, answer, grounding,
    # duplication and independent-review gate below remains unchanged.
    textbook_only_after_source_failure = False

    for attempt in range(1, config.GENERATION_RETRY_ATTEMPTS + 1):
        remaining = questions_per_level - len(collected)
        if remaining <= 0:
            return collected[:questions_per_level]

        available_past_questions = [
            question
            for question in past_questions
            if str(question.get("question_id", "")).strip() not in used_source_ids
        ]
        attempt_past_questions = (
            [] if textbook_only_after_source_failure else available_past_questions
        )
        candidate_count = min(
            config.GENERATION_CANDIDATE_CAP,
            max(remaining, remaining * config.GENERATION_CANDIDATE_MULTIPLIER),
        )
        # Past-paper transformations are one-to-one with validated source items.
        # Never ask the model for more transformed questions than available sources.
        if attempt_past_questions:
            candidate_count = min(candidate_count, len(attempt_past_questions))
        candidate_count = max(remaining, candidate_count)
        checkpoint_count = 0
        accepted_before_attempt = len(collected)

        def checkpoint_candidate(question):
            nonlocal checkpoint_count
            if checkpoint_count >= remaining:
                return
            checkpoint(question)
            checkpoint_count += 1

        # When a failed past-paper transformation causes a switch to fresh
        # textbook-only authoring, remove only feedback that refers specifically
        # to preserving/copying that source. Do this in place so later validation
        # failures are appended to the same persistent feedback list and are
        # available to the next retry.
        if textbook_only_after_source_failure and rejection_feedback:
            rejection_feedback[:] = [
                reason for reason in rejection_feedback
                if "past-paper source" not in reason
                and "verified source answer" not in reason
            ]

        try:
            questions = generate_mcqs(
                topic_name=topic_name,
                difficulty_level=level_name,
                context_chunks=chunks,
                num_questions=candidate_count,
                strict_grounding=bool(mapping_metadata.get("strict_textbook_grounding")),
                grade=mapping_metadata.get("grade"),
                concept_id=topic_id,
                grounding_units=grounding_units,
                past_questions=attempt_past_questions,
                avoid_question_stems=base_avoid_stems + collected_stems,
                # generation_offset must affect strict-grounding evidence selection
                # on retries and diagnostic refill rounds too. Previously adaptive
                # diagnostic mappings ignored the offset here, so every refill saw
                # the same evidence window and reproduced the same rejection.
                grounding_attempt=attempt_offset + attempt - 1,
                rejection_feedback=rejection_feedback,
                require_application=bool(mapping_metadata.get("require_application")),
                variation_offset=attempt_offset if mapping_metadata.get("adaptive_policy_version") else 0,
                on_accepted=checkpoint_candidate,
                accept_limit=remaining,
            )
            for q in questions:
                _attach_question_metadata(q, topic_id=topic_id, mapping_metadata=mapping_metadata,
                                          source_refs=source_refs, grounding_units=grounding_units,
                                          past_paper_refs=past_paper_refs)
            structurally_valid = [
                q
                for q in questions
                if str(q.get("question", "")).strip()
                and isinstance(q.get("options"), list)
                and len(q["options"]) == 4
                and len({str(option).strip() for option in q["options"]}) == 4
                and q.get("answer") in {"A", "B", "C", "D"}
                and str(q.get("explanation", "")).strip()
                and (
                    not mapping_metadata.get("strict_textbook_grounding")
                    or (
                        str(q.get("grounding_evidence", "")).strip()
                        and bool(q.get("source_refs"))
                    )
                )
                and (
                    not str(q.get("source_question_id", "")).strip()
                    or bool(q.get("past_paper_refs"))
                )
            ]
            if structurally_valid:
                collected = _dedupe_questions_preserve_order(
                    collected + structurally_valid
                )[:questions_per_level]
                collected_stems = [
                    str(question.get("question", "")).strip()
                    for question in collected
                    if str(question.get("question", "")).strip()
                ]
                used_source_ids = {
                    str(question.get("source_question_id", "")).strip()
                    for question in collected
                    if str(question.get("source_question_id", "")).strip()
                }
            if len(collected) >= questions_per_level:
                return collected[:questions_per_level]

            # If a verified past-paper transform yielded no accepted question, do
            # not spend every remaining retry reproducing the same source-constrained
            # failure. Continue with the same concept, grade, Bloom level and strict
            # textbook evidence, but without requiring source_question_id/canonical
            # source-answer preservation. Any already accepted source-backed item is
            # retained unchanged.
            if attempt_past_questions and len(collected) == accepted_before_attempt:
                textbook_only_after_source_failure = True
                logger.info(
                    "MCQ retry switching to textbook-only grounding for %s after past-paper transform rejection",
                    topic_name,
                    extra={
                        "event": "mcq.generation.past_paper_fallback",
                        "topic": topic_name,
                        "attempt": attempt,
                    },
                )

            last_error = ValueError(
                "The model did not return enough structurally valid, evidence-grounded "
                f"questions ({len(collected)}/{questions_per_level} collected)."
            )
        except Exception as exc:
            last_error = exc
            if _fatal_generation_error(exc):
                detail = _generation_error_detail(exc)
                logger.error("MCQ generation stopped for %s: %s", topic_name, detail)
                raise RuntimeError(
                    f"MCQ generation stopped for '{topic_name}': {detail}"
                ) from exc
            logger.warning(
                "MCQ generation attempt %s/%s failed for %s: %s",
                attempt,
                config.GENERATION_RETRY_ATTEMPTS,
                topic_name,
                _generation_error_detail(exc),
                extra={
                    "attempt": attempt,
                    "event": "mcq.generation.retry",
                    "topic": topic_name,
                },
            )
        if attempt < config.GENERATION_RETRY_ATTEMPTS:
            delay = min(
                30.0,
                config.GENERATION_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            if delay > 0:
                time.sleep(delay)
    if last_error:
        logger.error(
            "MCQ generation exhausted retries for %s: %s",
            topic_name,
            _generation_error_detail(last_error),
            extra={"event": "mcq.generation.failed", "topic": topic_name},
        )
    return collected[:questions_per_level]


def _refill_mapping_questions(
    paper: List[Dict],
    *,
    mapping: dict,
    retrieved: object,
    levels_lookup: dict,
) -> List[Dict]:
    """Top up a non-diagnostic topic after final cross-batch de-duplication.

    Practice generation splits ten questions across Bloom levels.  A valid item
    from one level can still be removed later because it is too similar to an
    item produced by another level.  Refill only that final shortfall while
    blocking every already accepted stem for the topic.
    """
    topic_id = str(mapping.get("topic_id", ""))
    required = max(0, int(mapping.get("min_questions", 0) or 0))
    current = sum(1 for question in paper if str(question.get("topic_id", "")) == topic_id)
    missing = required - current
    if missing <= 0:
        return paper

    topic_name = str(mapping.get("topic_name") or topic_id or "General Mathematics")
    if isinstance(retrieved, dict):
        chunks = list(retrieved.get("chunks", []) or [])
        source_refs = list(retrieved.get("source_refs", []) or [])
        grounding_units = list(retrieved.get("grounding_units", []) or [])
        past_questions = list(retrieved.get("past_questions", []) or [])
        past_paper_refs = list(retrieved.get("past_paper_refs", []) or [])
    else:
        chunks = list(retrieved or []) if isinstance(retrieved, list) else []
        source_refs = []
        grounding_units = []
        past_questions = []
        past_paper_refs = []

    if not chunks:
        return paper

    required_levels = list(mapping.get("required_levels", [1]) or [1])
    base_count, remainder = divmod(required, len(required_levels))
    level_targets = [
        (levels_lookup.get(level_id, "Remember"), base_count + (index < remainder))
        for index, level_id in enumerate(required_levels)
    ]

    used_source_ids = {
        str(question.get("source_question_id", "")).strip()
        for question in paper
        if str(question.get("topic_id", "")) == topic_id
        and str(question.get("source_question_id", "")).strip()
    }
    remaining_past = [
        question
        for question in past_questions
        if str(question.get("question_id", "")).strip() not in used_source_ids
    ]
    past_ref_by_id = {
        str(reference.get("question_id", "")): reference
        for reference in past_paper_refs
        if isinstance(reference, dict)
    }

    def refill(task_count: int, task_questions: list[dict], level_name: str, offset: int) -> None:
        nonlocal paper
        if task_count <= 0:
            return
        existing_stems = [
            str(question.get("question", "")).strip()
            for question in paper
            if str(question.get("topic_id", "")) == topic_id
            and str(question.get("question", "")).strip()
        ]
        refill_mapping = dict(mapping)
        refill_mapping["generation_offset"] = int(mapping.get("generation_offset", 0)) + offset
        refill_mapping["avoid_question_stems"] = list(
            dict.fromkeys(
                list(mapping.get("avoid_question_stems", []) or []) + existing_stems
            )
        )
        source_ids = {
            str(question.get("question_id", ""))
            for question in task_questions
            if str(question.get("question_id", ""))
        }
        task_references = [
            past_ref_by_id[source_id]
            for source_id in source_ids
            if source_id in past_ref_by_id
        ]
        generated = _generate_mcqs_block(
            (
                topic_id,
                topic_name,
                level_name,
                list(chunks),
                task_count,
                refill_mapping,
                source_refs,
                grounding_units,
                task_questions,
                task_references,
            )
        )
        if generated:
            for question in generated:
                question.setdefault("difficulty_level", level_name)
            paper = _dedupe_questions_preserve_order(paper + generated)

    logger.info(
        "MCQ topic coverage refill started",
        extra={
            "event": "mcq.coverage.refill",
            "topic": topic_name,
            "missing_questions": missing,
        },
    )

    offset = len(required_levels)
    for level_name, target in level_targets:
        count = sum(
            1 for question in paper
            if str(question.get("topic_id", "")) == topic_id
            and question.get("difficulty_level") == level_name
        )
        for _ in range(max(0, target - count)):
            used_ids = {str(question.get("source_question_id", "")) for question in paper}
            available = [
                question for question in remaining_past
                if str(question.get("question_id", "")) not in used_ids
            ]
            refill(1, available[:1], level_name, offset)
            offset += 1

    return paper


def _generate_topic_tasks(tasks: list[tuple]) -> List[Dict]:
    collected: List[Dict] = []
    for index, task in enumerate(tasks):
        values = list(task)
        metadata = dict(values[5])
        metadata["avoid_question_stems"] = list(dict.fromkeys(
            list(metadata.get("avoid_question_stems", []) or [])
            + [str(question.get("question", "")) for question in collected]
        ))
        metadata["generation_offset"] = int(metadata.get("generation_offset", 0)) + index
        values[5] = metadata
        generated = _generate_mcqs_block(tuple(values))
        for question in generated:
            question.setdefault("difficulty_level", task[2])
        collected.extend(generated)
    return collected


def _generation_inference_call_budget(syllabus: dict) -> int:
    """Return the bounded author/reviewer call budget for one quiz request.

    A 25-concept diagnostic performs at least one author call and one independent
    review per accepted question. Rejected candidates can consume the configured
    retry/refill budget as well. The previous ``3 * question_count`` allowance was
    therefore too small for a strict 25-question diagnostic even though the same
    pipeline worked for a single-topic quiz.
    """
    try:
        total_questions = max(1, int(syllabus.get("total_questions", 10) or 10))
    except (TypeError, ValueError):
        total_questions = 10
    requested = total_questions * config.GENERATION_INFERENCE_CALLS_PER_QUESTION
    return min(
        config.GENERATION_INFERENCE_CALL_BUDGET_MAX,
        max(12, requested),
    )


def generate_paper(syllabus: dict, retrieved_content: dict, *, initial_paper=None, on_progress=None, retry_offset=0) -> List[Dict]:
    with inference_budget(_generation_inference_call_budget(syllabus)):
        return _generate_paper(syllabus, retrieved_content, initial_paper=initial_paper,
                               on_progress=on_progress, retry_offset=retry_offset)


def _generate_paper(syllabus: dict, retrieved_content: dict, *, initial_paper=None, on_progress=None, retry_offset=0) -> List[Dict]:
    """Generate a full MCQ paper based on the syllabus configuration."""
    topics_lookup = {}
    from rag.adaptive import LEVELS
    levels_lookup = dict(LEVELS)
    try:
        with open(config.TOPICS_FILE, "r", encoding="utf-8") as f:
            topics = json.load(f)
            topics_lookup = {t["id"]: t["name"] for t in topics}
    except Exception:
        pass
    try:
        with open(config.DIFFICULTY_LEVELS_FILE, "r", encoding="utf-8") as f:
            levels = json.load(f)
            levels_lookup = {l["level_id"]: l["name"] for l in levels}
    except Exception:
        pass

    from rag.validation import has_current_review

    mappings_by_id = {str(mapping["topic_id"]): mapping for mapping in syllabus.get("topic_mappings", [])}
    initial = []
    initial_counts = {}
    for question in initial_paper or []:
        if not isinstance(question, dict):
            continue
        mapping = mappings_by_id.get(str(question.get("topic_id", "")))
        if not mapping or not has_current_review(question):
            continue
        levels = list(mapping.get("required_levels", [1]) or [1])
        allowed_levels = [levels_lookup.get(level) for level in levels]
        level_name = question.get("difficulty_level")
        if level_name not in allowed_levels or question.get("grade") != mapping.get("grade"):
            continue
        base, extra = divmod(int(mapping.get("min_questions", 0)), len(levels))
        limit = base + int(allowed_levels.index(level_name) < extra)
        count_key = (str(mapping["topic_id"]), level_name)
        if initial_counts.get(count_key, 0) >= limit:
            continue
        if mapping.get("strict_textbook_grounding"):
            from rag.diagnostic import validate_diagnostic_paper, DiagnosticPaperError

            try:
                validate_diagnostic_paper([question], {"assessment_type": "diagnostic", "topic_mappings": [mapping]})
            except DiagnosticPaperError:
                continue
        retrieved = retrieved_content.get(str(mapping["topic_id"]), {})
        source_questions = retrieved.get("past_questions", []) if isinstance(retrieved, dict) else []
        if question.get("source_question_id"):
            source_questions = [source for source in source_questions
                                if str(source.get("question_id")) == str(question["source_question_id"])]
        else:
            source_questions = []
        if validate_generated_mcq(question, past_questions=source_questions,
                                  blocked_stems=mapping.get("avoid_question_stems", []),
                                  similarity_threshold=config.MCQ_SIMILARITY_THRESHOLD):
            continue
        initial.append(question)
        initial_counts[count_key] = initial_counts.get(count_key, 0) + 1
    initial = _dedupe_questions_preserve_order(initial)
    tasks: list[tuple] = []
    for mapping in syllabus.get("topic_mappings", []):
        mapping = dict(mapping)
        mapping["_on_progress"] = on_progress
        mapping["generation_offset"] = int(mapping.get("generation_offset", 0)) + retry_offset
        topic_id = mapping.get("topic_id", "")
        previous = [q for q in initial if q.get("topic_id") == topic_id]
        mapping["avoid_question_stems"] = list(mapping.get("avoid_question_stems", [])) + [q["question"] for q in previous]
        topic_name = (
            mapping.get("topic_name")
            or topics_lookup.get(topic_id, topic_id)
            or "General Mathematics"
        )
        retrieved = retrieved_content.get(topic_id, []) or []
        if isinstance(retrieved, dict):
            chunks = list(retrieved.get("chunks", []) or [])
            source_refs = list(retrieved.get("source_refs", []) or [])
            grounding_units = list(retrieved.get("grounding_units", []) or [])
            past_questions = list(retrieved.get("past_questions", []) or [])
            past_paper_refs = list(retrieved.get("past_paper_refs", []) or [])
        else:
            chunks = list(retrieved)
            source_refs = []
            grounding_units = []
            past_questions = []
            past_paper_refs = []
        if mapping.get("strict_textbook_grounding") and (
            not chunks or not source_refs or not grounding_units
        ):
            raise ValueError(
                f"Strict textbook grounding failed for '{topic_name}' ({topic_id})."
            )
        num_q = int(mapping.get("min_questions", 5))

        required_levels = list(mapping.get("required_levels", [1]) or [1])
        base_count, remainder = divmod(num_q, len(required_levels))
        level_counts = [
            base_count + (1 if index < remainder else 0)
            for index in range(len(required_levels))
        ]
        past_question_cursor = 0
        past_ref_by_id = {
            str(reference.get("question_id", "")): reference
            for reference in past_paper_refs
            if isinstance(reference, dict)
        }
        used_source_ids = {str(q.get("source_question_id", "")) for q in previous}
        past_questions = [q for q in past_questions if str(q.get("question_id", "")) not in used_source_ids]

        for level_id, questions_per_level in zip(required_levels, level_counts):
            questions_per_level -= sum(q.get("difficulty_level") == levels_lookup.get(level_id) for q in previous)
            if questions_per_level <= 0:
                continue
            level_name = levels_lookup.get(level_id)
            if level_name is None:
                raise ValueError(f"Unsupported Bloom level: {level_id}")
            available_sources = past_questions[
                past_question_cursor : past_question_cursor + questions_per_level
            ]
            past_question_cursor += len(available_sources)
            task_batches: list[tuple[int, list[dict], list[dict]]] = []
            if available_sources:
                source_ids = {
                    str(question.get("question_id", ""))
                    for question in available_sources
                }
                task_batches.append(
                    (
                        len(available_sources),
                        available_sources,
                        [
                            past_ref_by_id[source_id]
                            for source_id in source_ids
                            if source_id in past_ref_by_id
                        ],
                    )
                )
            unsourced_count = questions_per_level - len(available_sources)
            if unsourced_count:
                task_batches.append((unsourced_count, [], []))

            bounded_batches = []
            for task_count, task_questions, task_references in task_batches:
                for offset in range(0, task_count, 4):
                    batch_questions = task_questions[offset:offset + 4]
                    batch_ids = {str(row.get("question_id", "")) for row in batch_questions}
                    bounded_batches.append((
                        min(4, task_count - offset), batch_questions,
                        [ref for ref in task_references if str(ref.get("question_id", "")) in batch_ids],
                    ))
            for task_count, task_questions, task_references in bounded_batches:
                logger.info(
                    "MCQ generation task queued",
                    extra={"event": "mcq.generation.queued", "topic": topic_name},
                )
                tasks.append(
                    (
                        topic_id,
                        topic_name,
                        level_name,
                        list(chunks),
                        task_count,
                        dict(mapping),
                        source_refs,
                        grounding_units,
                        task_questions,
                        task_references,
                    )
                )

    paper: List[Dict] = list(initial)
    if tasks:
        grouped_tasks: dict[str, list[tuple]] = {}
        for task in tasks:
            grouped_tasks.setdefault(task[0], []).append(task)
        groups = list(grouped_tasks.values())
        max_workers = min(len(groups), max(1, config.GENERATION_MAX_WORKERS))
        logger.info(
            "MCQ paper generation started",
            extra={"event": "mcq.paper.started"},
        )
        if max_workers == 1:
            for group in groups:
                paper.extend(_generate_topic_tasks(group))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending = deque()
                task_iterator = iter(groups)
                try:
                    for _ in range(max_workers):
                        task = next(task_iterator, None)
                        if task is not None:
                            pending.append(executor.submit(copy_context().run, _generate_topic_tasks, task))
                    while pending:
                        paper.extend(pending.popleft().result())
                        task = next(task_iterator, None)
                        if task is not None:
                            pending.append(executor.submit(copy_context().run, _generate_topic_tasks, task))
                finally:
                    for future in pending:
                        future.cancel()

    if not paper and syllabus.get("assessment_type") != "diagnostic":
        # Final fallback: generate directly from combined retrieved context.
        combined_context: list[str] = []
        for retrieved in retrieved_content.values():
            if isinstance(retrieved, list):
                combined_context.extend(retrieved)
            elif isinstance(retrieved, dict):
                combined_context.extend(retrieved.get("chunks", []) or [])
        if combined_context:
            logger.warning("Mapped generation was empty; using combined context")
            fallback_questions = generate_mcqs(
                topic_name="Mathematics",
                difficulty_level="Apply",
                context_chunks=combined_context[:12],
                num_questions=max(8, int(syllabus.get("total_questions", 10) // 4)),
            )
            for q in fallback_questions:
                q["topic_id"] = q.get("topic_id") or "math_fallback"
            paper.extend(fallback_questions)

    raw_count = len(paper)
    cleaned: List[Dict] = []
    for q in paper:
        q.pop("image_prompt", None)
        q.pop("image_url", None)
        stem = str(q.get("question", ""))
        if _question_requires_external_visual(stem):
            continue
        cleaned.append(q)
    paper = _dedupe_questions_preserve_order(cleaned)
    dropped = raw_count - len(paper)
    if dropped:
        logger.info(
            "MCQ paper cleanup completed",
            extra={"event": "mcq.paper.cleaned"},
        )

    if syllabus.get("assessment_type") == "diagnostic":
        from rag.diagnostic import validate_diagnostic_paper

        # A 25-concept diagnostic can have a few topics whose first bounded author/review
        # attempts all fail strict validation. Refill only those missing concepts using
        # the same grounding, de-duplication and independent-review gates. The final
        # diagnostic validator remains fail-closed; this does not accept partial papers
        # or lower any quality threshold.
        for refill_round in range(config.GENERATION_COVERAGE_REFILL_ROUNDS):
            counts_now: dict[str, int] = {}
            for question in paper:
                question_topic = str(question.get("topic_id", ""))
                counts_now[question_topic] = counts_now.get(question_topic, 0) + 1

            missing_mappings = []
            for mapping_row in syllabus.get("topic_mappings", []):
                topic_id = str(mapping_row.get("topic_id", ""))
                required = max(0, int(mapping_row.get("min_questions", 0) or 0))
                if counts_now.get(topic_id, 0) < required:
                    missing_mappings.append(mapping_row)

            if not missing_mappings:
                break

            before = len(paper)
            logger.info(
                "MCQ diagnostic coverage refill started",
                extra={
                    "event": "mcq.diagnostic.coverage.refill",
                    "round": refill_round + 1,
                    "missing_topics": len(missing_mappings),
                },
            )
            for mapping_row in missing_mappings:
                topic_id = str(mapping_row.get("topic_id", ""))
                mapping = dict(mapping_row)
                mapping["_on_progress"] = on_progress
                mapping["generation_offset"] = (
                    retry_offset
                    + (refill_round + 1)
                    * max(11, int(syllabus.get("total_questions", 1) or 1))
                )
                paper = _refill_mapping_questions(
                    paper,
                    mapping=mapping,
                    retrieved=retrieved_content.get(topic_id, []) or [],
                    levels_lookup=levels_lookup,
                )

            paper = _dedupe_questions_preserve_order(paper)
            if len(paper) == before:
                logger.info(
                    "MCQ diagnostic refill round completed without a new accepted question",
                    extra={
                        "event": "mcq.diagnostic.coverage.refill.no_progress",
                        "round": refill_round + 1,
                    },
                )

        # Diagnostic coverage remains fail-closed: never silently return a partial paper.
        validate_diagnostic_paper(paper, syllabus)
    else:
        # Cross-level de-duplication can legitimately remove one or more otherwise
        # valid practice items. Refill only those final per-topic shortfalls. Use a
        # small bounded number of rounds so a rejected final candidate does not
        # force the student to restart the request manually.
        for refill_round in range(config.GENERATION_COVERAGE_REFILL_ROUNDS):
            before = len(paper)
            for mapping_row in syllabus.get("topic_mappings", []):
                topic_id = str(mapping_row.get("topic_id", ""))
                mapping = dict(mapping_row)
                mapping["_on_progress"] = on_progress
                mapping["generation_offset"] = (
                    retry_offset
                    + refill_round * max(7, int(mapping.get("min_questions", 1) or 1))
                )
                paper = _refill_mapping_questions(
                    paper,
                    mapping=mapping,
                    retrieved=retrieved_content.get(topic_id, []) or [],
                    levels_lookup=levels_lookup,
                )

            complete = True
            counts_now: dict[str, int] = {}
            for question in paper:
                question_topic = str(question.get("topic_id", ""))
                counts_now[question_topic] = counts_now.get(question_topic, 0) + 1
            for mapping_row in syllabus.get("topic_mappings", []):
                topic_id = str(mapping_row.get("topic_id", ""))
                required = max(0, int(mapping_row.get("min_questions", 0) or 0))
                if counts_now.get(topic_id, 0) < required:
                    complete = False
                    break
            if complete:
                break
            if len(paper) == before:
                logger.info(
                    "MCQ coverage refill round completed without a new accepted question",
                    extra={
                        "event": "mcq.coverage.refill.no_progress",
                        "round": refill_round + 1,
                    },
                )

        counts: dict[str, int] = {}
        for question in paper:
            question_topic = str(question.get("topic_id", ""))
            counts[question_topic] = counts.get(question_topic, 0) + 1
        incomplete: list[str] = []
        for mapping in syllabus.get("topic_mappings", []):
            topic_id = str(mapping.get("topic_id", ""))
            required = max(0, int(mapping.get("min_questions", 0) or 0))
            received = counts.get(topic_id, 0)
            if received < required:
                incomplete.append(
                    f"{topic_id}: expected {required}, received {received}"
                )
        if incomplete:
            raise ValueError(
                "Question generation did not meet the requested concept coverage: "
                + "; ".join(incomplete[:10])
            )
    return paper
