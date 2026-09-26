"""Qwen generation through RunPod Serverless OpenAI-compatible endpoints.

This module intentionally uses the RunPod's documented OpenAI-compatible HTTP contract directly
This keeps the existing generation and validation logic intact while moving only the inference transport to RunPod.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from functools import lru_cache
from typing import Any

import requests

import config
from rag.inference_errors import consume_inference_call, record_credit_failure, access_failure


logger = logging.getLogger(__name__)

RUNPOD_INFERENCE_IMPLEMENTATION = "runpod-openai-v1-length-aware"

def _chat_completions_url(endpoint_id: str) -> str:
    endpoint = (endpoint_id or "").strip()
    if not endpoint:
        raise RuntimeError("RunPod endpoint ID is required.")
    return f"https://api.runpod.ai/v2/{endpoint}/openai/v1/chat/completions"
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@lru_cache(maxsize=1)
def _http_session() -> requests.Session:
    """Create the authenticated HTTP session used for RunPod endpoints."""
    if not config.RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is required.")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )
    return session


def _routed_model_id() -> str:
    """Return the model identifier configured on the RunPod vLLM endpoint."""
    return config.RUNPOD_LLM_MODEL


def _provider_error_message(response: requests.Response) -> str:
    """Return a short provider error without exposing request/prompt contents."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
            if isinstance(value, dict):
                nested = value.get("message") or value.get("detail")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()[:500]
    return f"HTTP {response.status_code} from RunPod endpoint"


def _message_text(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract final assistant content and finish reason from router JSON."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""

    choice = choices[0]
    if not isinstance(choice, dict):
        return "", ""

    finish_reason = str(choice.get("finish_reason") or "").strip()
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip(), finish_reason
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "".join(parts).strip()
            if joined:
                return joined, finish_reason

    # OpenAI-compatible completion variants can expose text directly on choice.
    choice_text = choice.get("text")
    if isinstance(choice_text, str):
        return choice_text.strip(), finish_reason

    return "", finish_reason


def _strip_markdown_fence(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate.startswith("```"):
        return candidate

    lines = candidate.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _escape_invalid_json_backslashes(text: str) -> str:
    """Escape only backslashes that are illegal inside JSON string values."""
    out: list[str] = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        ch = text[i]

        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            i += 1
            continue

        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        next_char = text[i + 1] if i + 1 < len(text) else ""
        if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            out.append("\\")
            escaped = True
            i += 1
            continue

        if next_char == "u" and i + 5 < len(text):
            hex_part = text[i + 2 : i + 6]
            if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                out.append("\\")
                escaped = True
                i += 1
                continue

        # Preserve a literal math backslash such as \sqrt by making it valid JSON.
        out.append("\\\\")
        i += 1

    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _normalize_json_value(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"questions": value}
    return None


def _try_parse_candidate(candidate: str) -> dict[str, Any] | None:
    candidate = candidate.strip()
    if not candidate:
        return None

    variants = [candidate]
    repaired = _remove_trailing_commas(_escape_invalid_json_backslashes(candidate))
    if repaired != candidate:
        variants.append(repaired)

    decoder = json.JSONDecoder()

    for value_text in variants:
        try:
            normalized = _normalize_json_value(json.loads(value_text))
            if normalized is not None:
                return normalized
        except (json.JSONDecodeError, TypeError):
            pass

        # Accept a complete JSON object/array followed by harmless prose.
        try:
            value, _end = decoder.raw_decode(value_text)
            normalized = _normalize_json_value(value)
            if normalized is not None:
                return normalized
        except (json.JSONDecodeError, TypeError):
            pass

        # Safe fallback for model output using Python-style single quotes.
        try:
            normalized = _normalize_json_value(ast.literal_eval(value_text))
            if normalized is not None:
                return normalized
        except (ValueError, SyntaxError):
            pass

    return None


def _parse_json_object_response(text: str) -> dict[str, Any]:
    candidate = _strip_markdown_fence(text)
    if not candidate:
        raise RuntimeError("The inference provider returned no assistant content.")

    parsed = _try_parse_candidate(candidate)
    if parsed is not None:
        return parsed

    # Search for JSON embedded after a short model prefix. Each opening token is
    # attempted independently so braces in preceding prose do not break parsing.
    for index, char in enumerate(candidate):
        if char not in "[{":
            continue
        parsed = _try_parse_candidate(candidate[index:])
        if parsed is not None:
            return parsed

    raise RuntimeError("The inference provider returned assistant text that is not valid JSON.")


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True

    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS_CODES:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "loading",
            "scaling",
        )
    )


class _ProviderHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _request_completion(
    prompt: str,
    *,
    token_limit: int,
    temperature: float,
) -> tuple[str, str]:
    """Call the documented RunPod OpenAI-compatible chat-completions endpoint directly."""
    payload = {
        "model": _routed_model_id(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_limit,
        "temperature": temperature,
        "stream": False,
    }

    consume_inference_call()
    response = _http_session().post(
        _chat_completions_url(config.RUNPOD_LLM_ENDPOINT_ID),
        json=payload,
        timeout=config.RUNPOD_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        record_credit_failure(response.status_code, _provider_error_message(response))
        raise _ProviderHTTPError(
            response.status_code,
            _provider_error_message(response),
        )

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"RunPod endpoint returned HTTP {response.status_code} but the response body was not JSON."
        ) from exc

    if not isinstance(response_payload, dict):
        raise RuntimeError("RunPod endpoint returned an unexpected response shape.")

    text, finish_reason = _message_text(response_payload)
    return text, finish_reason


def _adaptive_token_limits(initial_limit: int, *, allow_expand: bool) -> list[int]:
    """Return output budgets used when a provider reports finish_reason=length.

    The configured limit remains the first request budget.  For normal calls made
    by the application (where no explicit max_new_tokens is supplied), a truncated
    response is retried with progressively larger budgets up to the maximum already
    permitted by config.py (2048 tokens).  Explicit caller limits are never changed.
    """
    limits = [max(1, int(initial_limit))]
    if not allow_expand:
        return limits

    maximum = 2048
    while limits[-1] < maximum:
        current = limits[-1]
        next_limit = min(maximum, max(current + 512, current * 2))
        if next_limit <= current:
            break
        limits.append(next_limit)
    return limits


def runpod_json_generation(
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate one validated JSON object through the configured RunPod endpoint.

    A provider response with ``finish_reason=length`` is not a valid completed
    generation: it means the assistant was cut off by the output-token budget.
    When the application uses the configured default budget, retry only that
    truncated generation with a larger budget.  This does not alter prompts,
    MCQ validation, grounding, answers, or any other application logic.
    """
    requested_limit = max_new_tokens or config.RUNPOD_LLM_MAX_NEW_TOKENS
    sampling_temperature = (
        config.RUNPOD_GENERATION_TEMPERATURE if temperature is None else temperature
    )

    # Respect explicit caller limits exactly.  The adaptive path is only for the
    # normal application call, which currently uses the configured default.
    token_limits = _adaptive_token_limits(
        requested_limit,
        allow_expand=max_new_tokens is None,
    )

    last_error: Exception | None = None
    last_finish_reason = ""
    last_text_length = 0
    last_token_limit = requested_limit

    for limit_index, token_limit in enumerate(token_limits):
        last_token_limit = token_limit
        truncated = False

        for attempt in range(config.RUNPOD_MAX_RETRIES):
            try:
                text, last_finish_reason = _request_completion(
                    prompt,
                    token_limit=token_limit,
                    temperature=sampling_temperature,
                )
                last_text_length = len(text)

                if not text:
                    raise RuntimeError(
                        "RunPod returned HTTP 200 but no assistant content"
                        + (
                            f" (finish_reason={last_finish_reason})"
                            if last_finish_reason
                            else ""
                        )
                        + "."
                    )

                # This is the exact failure shown by the current terminal log.
                # Do not attempt to repair an incomplete JSON document; request
                # the same completion again with more output room instead.
                if last_finish_reason.lower() == "length":
                    truncated = True
                    last_error = RuntimeError(
                        f"The inference provider reached the {token_limit}-token "
                        "output limit before completing the response."
                    )
                    break

                parsed = _parse_json_object_response(text)
                return json.dumps(parsed, ensure_ascii=False)

            except Exception as exc:
                last_error = exc
                if attempt + 1 >= config.RUNPOD_MAX_RETRIES or not _is_retryable_exception(exc):
                    break

                delay = min(60.0, config.RUNPOD_RETRY_BASE_SECONDS * (2**attempt))
                logger.warning(
                    "RunPod inference endpoint unavailable; retrying",
                    extra={
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "provider": "runpod",
                        "implementation": RUNPOD_INFERENCE_IMPLEMENTATION,
                    },
                )
                time.sleep(delay)

        if truncated and limit_index + 1 < len(token_limits):
            next_limit = token_limits[limit_index + 1]
            logger.warning(
                "RunPod generation hit output token limit; retrying with a larger budget",
                extra={
                    "current_max_tokens": token_limit,
                    "next_max_tokens": next_limit,
                    "finish_reason": last_finish_reason,
                    "response_chars": last_text_length,
                    "provider": "runpod",
                    "implementation": RUNPOD_INFERENCE_IMPLEMENTATION,
                },
            )
            continue

        # Non-retryable parsing/provider failures should not be repeated at
        # different token budgets; only finish_reason=length triggers expansion.
        break

    detail = str(last_error) if last_error is not None else "unknown inference error"
    finish = f" finish_reason={last_finish_reason}." if last_finish_reason else ""
    raise RuntimeError(
        f"Qwen2.5-Math generation failed [{RUNPOD_INFERENCE_IMPLEMENTATION}]: "
        f"{detail}{finish} max_tokens={last_token_limit}. "
        f"response_chars={last_text_length}."
    ) from last_error


# ---------------------------------------------------------------------------
# Structured MCQ authoring path
# ---------------------------------------------------------------------------
# Keep Qwen2.5-Math available above for mathematics solving/checking. MCQ
# authoring itself is intentionally routed through a concise non-thinking model
# with a strict JSON schema so the provider does not spend a 2048-token budget on
# hidden/verbose reasoning before finishing a small question object.
RUNPOD_MCQ_GENERATION_IMPLEMENTATION = "structured-mcq-v3-bounded"
_MCQ_PROVIDER_MAX_OUTPUT_TOKENS = 2048


def _routed_model_id_for(model: str) -> str:
    """RunPod vLLM endpoints accept the model name directly."""
    return model


def _mcq_response_format(
    *,
    strict_grounding: bool,
    use_past_questions: bool,
    max_questions: int,
) -> dict[str, Any]:
    """Return the smallest schema needed by the MCQ authoring call.

    Provenance is deliberately represented by compact IDs. Source/page/grade and
    exact textbook evidence are attached by ``rag.generator`` from trusted
    retrieval metadata after the response is parsed. This prevents the model
    from wasting output tokens copying metadata and prevents hallucinated source
    fields from entering a diagnostic paper.
    """
    question_properties: dict[str, Any] = {
        "question": {"type": "string", "minLength": 1, "maxLength": 500},
        "correct_answer": {"type": "string", "minLength": 1, "maxLength": 200},
        "options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "explanation": {"type": "string", "minLength": 1, "maxLength": 600},
    }
    required = ["question", "options", "correct_answer", "explanation"]

    if strict_grounding:
        question_properties["evidence_id"] = {
            "type": "string",
            "pattern": "^E[1-9][0-9]*$",
            "maxLength": 8,
        }
        required.append("evidence_id")

    if use_past_questions:
        question_properties["source_question_id"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
        }
        required.append("source_question_id")

    bounded_questions = max(1, min(8, int(max_questions or 1)))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "neuromath_mcq_response",
            "description": "NeuroMath MCQ generation response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": bounded_questions,
                        "items": {
                            "type": "object",
                            "properties": question_properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
        },
    }


def _request_structured_mcq_completion(
    prompt: str,
    *,
    strict_grounding: bool,
    use_past_questions: bool,
    max_questions: int,
    token_limit: int,
    temperature: float,
    response_format: dict | None = None,
) -> tuple[str, str]:
    payload = {
        "model": _routed_model_id_for(config.RUNPOD_MCQ_GENERATION_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve and review mathematics MCQs independently. Return only the schema-constrained JSON."
                    if response_format is not None else
                    "Write concise mathematics MCQs. Return only the schema-constrained "
                    "JSON. Do not output chain-of-thought, analysis, markdown, or prose "
                    "outside the JSON. Keep the answer explanation brief."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": token_limit,
        "temperature": temperature,
        "stream": False,
        "response_format": response_format or _mcq_response_format(
            strict_grounding=strict_grounding,
            use_past_questions=use_past_questions,
            max_questions=max_questions,
        ),
    }

    consume_inference_call()
    response = _http_session().post(
        _chat_completions_url(config.RUNPOD_MCQ_ENDPOINT_ID),
        json=payload,
        timeout=config.RUNPOD_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        record_credit_failure(response.status_code, _provider_error_message(response))
        raise _ProviderHTTPError(
            response.status_code,
            _provider_error_message(response),
        )

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"RunPod endpoint returned HTTP {response.status_code} "
            "but the response body was not JSON."
        ) from exc

    if not isinstance(response_payload, dict):
        raise RuntimeError("RunPod endpoint returned an unexpected response shape.")

    return _message_text(response_payload)


def runpod_mcq_review(prompt: str, *, response_format: dict) -> dict:
    """Run a separate bounded review on the existing configured MCQ provider."""
    token_limit = 1024
    expanded = False
    last_error: Exception | None = None
    for attempt in range(config.RUNPOD_MAX_RETRIES + 1):
        try:
            text, finish_reason = _request_structured_mcq_completion(
                prompt, strict_grounding=False, use_past_questions=False,
                max_questions=1, token_limit=token_limit, temperature=0.0,
                response_format=response_format,
            )
            if finish_reason.lower() == "length":
                if not expanded:
                    token_limit = _MCQ_PROVIDER_MAX_OUTPUT_TOKENS
                    expanded = True
                    continue
                raise RuntimeError("Independent MCQ review exceeded its bounded output budget.")
            if not text:
                raise RuntimeError("Independent MCQ review returned no assistant content.")
            return _parse_json_object_response(text)
        except Exception as exc:
            last_error = exc
            if attempt >= config.RUNPOD_MAX_RETRIES - 1 or not _is_retryable_exception(exc):
                break
            time.sleep(min(60.0, config.RUNPOD_RETRY_BASE_SECONDS * (2 ** attempt)))
    failure = access_failure(last_error) if last_error else None
    if failure is not None:
        if failure is last_error:
            raise failure
        raise failure from last_error
    raise RuntimeError("Independent MCQ review failed; the question was not accepted.") from last_error


def _mcq_token_limits(
    initial_limit: int,
    *,
    question_count: int,
    allow_expand: bool,
) -> list[int]:
    """Choose a bounded output budget for compact structured MCQs.

    The first budget scales with the number of requested questions. If the
    provider nevertheless reports ``finish_reason=length``, retry the same
    request once at 2048 tokens. We never keep replaying a truncated completion
    at the same limit.
    """
    count = max(1, min(8, int(question_count or 1)))
    recommended = min(
        _MCQ_PROVIDER_MAX_OUTPUT_TOKENS,
        500 + (300 * count),
    )
    first = min(
        _MCQ_PROVIDER_MAX_OUTPUT_TOKENS,
        max(256, int(initial_limit), recommended),
    )
    limits = [first]
    if allow_expand and first < _MCQ_PROVIDER_MAX_OUTPUT_TOKENS:
        limits.append(_MCQ_PROVIDER_MAX_OUTPUT_TOKENS)
    return limits


def runpod_mcq_generation(
    prompt: str,
    *,
    strict_grounding: bool,
    use_past_questions: bool = False,
    question_count: int = 1,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate compact, schema-constrained MCQ JSON.

    This path exists specifically to avoid the long-reasoning truncation seen
    when Qwen2.5-Math is used directly for MCQ authoring. The selected math model
    remains available through ``runpod_json_generation``; quiz authoring is handled
    by the configured non-thinking instruct model and trusted metadata is added
    after generation.
    """
    requested_limit = max_new_tokens or config.RUNPOD_MCQ_MAX_NEW_TOKENS
    sampling_temperature = (
        config.RUNPOD_GENERATION_TEMPERATURE if temperature is None else temperature
    )
    token_limits = _mcq_token_limits(
        requested_limit,
        question_count=question_count,
        allow_expand=max_new_tokens is None,
    )

    last_error: Exception | None = None
    last_finish_reason = ""
    last_text_length = 0
    last_token_limit = token_limits[0]

    for limit_index, token_limit in enumerate(token_limits):
        last_token_limit = token_limit
        truncated = False

        for attempt in range(config.RUNPOD_MAX_RETRIES):
            try:
                text, last_finish_reason = _request_structured_mcq_completion(
                    prompt,
                    strict_grounding=strict_grounding,
                    use_past_questions=use_past_questions,
                    max_questions=question_count,
                    token_limit=token_limit,
                    temperature=sampling_temperature,
                )
                last_text_length = len(text)

                if not text:
                    raise RuntimeError(
                        "RunPod returned HTTP 200 but no assistant content."
                    )

                if last_finish_reason.lower() == "length":
                    truncated = True
                    last_error = RuntimeError(
                        f"Structured MCQ output reached the {token_limit}-token limit."
                    )
                    break

                parsed = _parse_json_object_response(text)
                questions = parsed.get("questions")
                if not isinstance(questions, list):
                    raise RuntimeError(
                        "Structured MCQ output did not contain a questions array."
                    )
                if len(questions) > max(1, int(question_count or 1)):
                    raise RuntimeError(
                        "Structured MCQ output contained more questions than requested."
                    )
                return json.dumps(parsed, ensure_ascii=False)

            except Exception as exc:
                last_error = exc
                if attempt + 1 >= config.RUNPOD_MAX_RETRIES or not _is_retryable_exception(exc):
                    break
                delay = min(60.0, config.RUNPOD_RETRY_BASE_SECONDS * (2**attempt))
                logger.warning(
                    "RunPod MCQ endpoint unavailable; retrying",
                    extra={
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "provider": "runpod",
                        "implementation": RUNPOD_MCQ_GENERATION_IMPLEMENTATION,
                    },
                )
                time.sleep(delay)

        if truncated and limit_index + 1 < len(token_limits):
            logger.warning(
                "Structured MCQ output hit token limit; retrying once with maximum bounded budget",
                extra={
                    "current_max_tokens": token_limit,
                    "next_max_tokens": token_limits[limit_index + 1],
                    "finish_reason": last_finish_reason,
                    "response_chars": last_text_length,
                    "provider": "runpod",
                    "implementation": RUNPOD_MCQ_GENERATION_IMPLEMENTATION,
                },
            )
            continue
        break

    detail = str(last_error) if last_error is not None else "unknown inference error"
    finish = f" finish_reason={last_finish_reason}." if last_finish_reason else ""
    raise RuntimeError(
        f"MCQ generation failed [{RUNPOD_MCQ_GENERATION_IMPLEMENTATION}] "
        f"model={config.RUNPOD_MCQ_GENERATION_MODEL} "
        f"provider=runpod: "
        f"{detail}{finish} max_tokens={last_token_limit}. "
        f"response_chars={last_text_length}."
    ) from last_error
