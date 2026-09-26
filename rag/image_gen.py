"""Generate optional diagram images for MCQs via Hugging Face Inference API."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import config

_FILENAME_SAFE = re.compile(r"^[a-f0-9]{32}\.png$")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
# If the stem does not suggest a figure, question-stem fallback is skipped (reduces nonsense images).
_VISUAL_HINT = re.compile(
    r"\b("
    r"diagram|figure|triangle|quadrilateral|polygon|parallelogram|trapezoid|"
    r"prism|pyramid|cone|cylinder|sphere|circle|ellipse|arc|chord|tangent|"
    r"graph|plot|axes|axis|coordinate|grid|histogram|bar chart|scatter|"
    r"curve|parabol|asymptote|intercept|slope|derivative|integral|area under|"
    r"circuit|schematic|resistor|capacitor|battery|voltage|current|"
    r"ray diagram|light ray|refraction|reflection|lens|mirror|prism|"
    r"vector|velocity|acceleration|force|free[\s-]?body|momentum|trajectory|"
    r"angle|perpendicular|parallel|congruent|similar|rotation|symmetry|"
    r"unit circle|sine|cosine|tangent\s+line|"
    r"as shown|shown in|shown below|see the|picture of|sketch|schematic"
    r")\b",
    re.IGNORECASE,
)


def _quiz_images_dir() -> Path:
    return config.PROJECT_ROOT / "static" / "quiz_images"


def _question_stem_for_image(q: dict[str, Any]) -> str:
    text = str(q.get("question") or "").strip()
    text = _THINK_RE.sub(" ", text).strip()
    text = re.sub(r"\s+", " ", text)
    if getattr(config, "IMAGE_GEN_FALLBACK_REQUIRE_KEYWORDS", True) and not _VISUAL_HINT.search(text):
        return ""
    # Drop common MCQ wording so the image model sees mostly the geometric/setup content.
    text = re.sub(
        r"\b(which of the following|select the best|choose the correct)\b[^?.!]*[?.!]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    clipped = " ".join(parts[:2]).strip() if parts else text
    max_c = getattr(config, "IMAGE_GEN_FALLBACK_MAX_CHARS", 280)
    return (clipped or text)[:max_c] if text else ""


def _stable_seed(prompt: str) -> int:
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def _wrap_prompt_for_model(raw: str, source: Literal["llm", "question_fallback"]) -> str:
    raw = raw.strip()
    if source == "llm":
        return (
            "Exact technical line diagram as specified, minimal ink, black strokes on pure white, "
            "no artistic style, no extra objects or scenery, no text in image except simple labels "
            "mentioned in the description: "
            + raw
        )
    return (
        "Literal exam-style diagram: draw only the geometric or physical setup implied below, "
        "simple labeled line art on white, no decorative elements, no unrelated objects: "
        + raw
    )


def attach_images_to_questions(questions: list[dict[str, Any]]) -> None:
    """
    For each question, use ``image_prompt`` if present; else optionally the question stem.
    Saves PNGs under static/quiz_images/ and sets ``image_url``.
    """
    if not getattr(config, "IMAGE_GEN_ENABLED", False):
        for q in questions:
            q.pop("image_prompt", None)
        return

    if not (getattr(config, "HF_API_KEY", "") or "").strip():
        print("IMAGE_GEN_ENABLED is on but HF_API_KEY is missing; skipping image generation.")
        for q in questions:
            q.pop("image_prompt", None)
        return

    out_dir = _quiz_images_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = getattr(config, "IMAGE_GEN_MAX_PER_PAPER", 8)
    done = 0
    skipped_llm_empty = 0
    used_fallback = 0
    failed = 0

    for q in questions:
        if done >= cap:
            q.pop("image_prompt", None)
            continue

        raw = str(q.pop("image_prompt", None) or "").strip()
        source = "llm"
        if not raw and getattr(config, "IMAGE_GEN_FALLBACK_QUESTION", True):
            raw = _question_stem_for_image(q)
            source = "question_fallback"
        if not raw:
            skipped_llm_empty += 1
            continue

        raw = raw[:600]
        fname = f"{uuid.uuid4().hex}.png"
        path = out_dir / fname
        if _generate_hf_image(raw, path, source):
            q["image_url"] = f"/static/quiz_images/{fname}"
            done += 1
            if source == "question_fallback":
                used_fallback += 1
        else:
            failed += 1

    print(
        f"Quiz images: enabled=True, generated={done}/{cap} max, "
        f"via_question_fallback={used_fallback}, hf_failures={failed}, "
        f"skipped_no_prompt={skipped_llm_empty}"
    )


def _call_text_to_image(client: Any, full_prompt: str, neg: str, model_id: str, w: int, h: int) -> Any:
    """Try richer kwargs first; some Hub models reject seed or negative_prompt."""
    base: dict[str, Any] = {"model": model_id, "width": w, "height": h}
    gs = getattr(config, "HF_IMAGE_GUIDANCE_SCALE", None)
    if gs is not None:
        base["guidance_scale"] = float(gs)
    seed_mode = getattr(config, "HF_IMAGE_SEED_MODE", "stable") or "off"
    seed = _stable_seed(full_prompt) if seed_mode == "stable" else None

    attempts: list[dict[str, Any]] = []
    if seed is not None:
        attempts.append({**base, "negative_prompt": neg, "seed": seed})
    attempts.append({**base, "negative_prompt": neg})
    attempts.append(dict(base))

    last_type_err: TypeError | None = None
    for extra in attempts:
        try:
            return client.text_to_image(full_prompt, **extra)
        except TypeError as te:
            last_type_err = te
            continue
    if last_type_err:
        raise last_type_err
    return client.text_to_image(full_prompt, **base)


def _generate_hf_image(
    user_prompt: str,
    path: Path,
    source: Literal["llm", "question_fallback"] = "llm",
) -> bool:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("image_gen: huggingface_hub not available.")
        return False

    full_prompt = _wrap_prompt_for_model(user_prompt, source)
    neg = (
        "photograph, photo, blurry, cluttered, watermark, logo, text overlay, "
        "low quality, ugly, cartoon character, 3d render, surreal, abstract art, "
        "unrelated objects, fantasy scenery, meme, screenshot"
    )

    primary = getattr(config, "HF_IMAGE_MODEL", "") or "stabilityai/sd-turbo"
    fallbacks = getattr(config, "HF_IMAGE_FALLBACKS", []) or []
    model_order: list[str] = []
    for m in [primary, *fallbacks]:
        if m and m not in model_order:
            model_order.append(m)

    client = InferenceClient(
        token=config.HF_API_KEY,
        timeout=getattr(config, "HF_IMAGE_TIMEOUT_SECONDS", 120),
    )
    w = getattr(config, "HF_IMAGE_WIDTH", 512)
    h = getattr(config, "HF_IMAGE_HEIGHT", 512)

    last_err: str | None = None
    for model_id in model_order:
        try:
            image = _call_text_to_image(client, full_prompt, neg, model_id, w, h)

            path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(image, "save"):
                image.save(str(path), format="PNG")
            elif isinstance(image, (bytes, bytearray)):
                path.write_bytes(bytes(image))
            else:
                print(f"HF text_to_image: unexpected return type from {model_id}: {type(image)}")
                continue
            if model_id != primary:
                print(f"Quiz image: succeeded with fallback model {model_id}")
            return True
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            last_err = f"{model_id}: {detail}"
            print(f"HF text_to_image failed ({last_err})")

    if last_err:
        print(f"HF text_to_image: all models exhausted. Last error: {last_err}")
    return False


def is_safe_quiz_image_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not u.startswith("/static/quiz_images/"):
        return False
    name = u.rsplit("/", 1)[-1]
    return bool(_FILENAME_SAFE.match(name))
