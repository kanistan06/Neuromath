"""Structure-aware ingestion for the Grade 6-11 textbook corpus."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader
from tokenizers import Tokenizer

import config
from rag.corpus import corpus_version, source_manifest
from rag.embeddings import build_embeddings


logger = logging.getLogger(__name__)
MANIFEST_FILE_NAME = "ingest_manifest.json"


@dataclass(frozen=True)
class StructureUnit:
    text: str
    source: str
    page: int
    parent_text: str
    section: str
    block_type: str
    grade: int | None


def _extract_grade_from_filename(path: Path) -> int | None:
    match = re.search(
        r"(?:^|[^a-z0-9])(?:e?gr|g|grade)[\s_-]?(\d{1,2})(?=[^0-9]|$)",
        path.name.lower(),
    )
    if not match:
        return None
    grade = int(match.group(1))
    return grade if 6 <= grade <= 11 else None


def _clean_pdf_text(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        if "Multiple definitions in dictionary at byte" in line:
            continue
        if not line:
            if lines and not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def _load_file(path: Path) -> list[Document]:
    grade = _extract_grade_from_filename(path)
    if path.suffix.lower() in {".md", ".txt"}:
        documents = [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"page": 0},
            )
        ]
    elif path.suffix.lower() == ".pdf":
        pypdf_logger = logging.getLogger("pypdf")
        previous_level = pypdf_logger.level
        pypdf_logger.setLevel(logging.ERROR)
        try:
            reader = PdfReader(str(path))
            documents = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"page": page_number},
                )
                for page_number, page in enumerate(reader.pages)
            ]
        finally:
            pypdf_logger.setLevel(previous_level)
    else:
        return []

    result: list[Document] = []
    for document in documents:
        document.page_content = _clean_pdf_text(document.page_content)
        document.metadata["source"] = str(path.resolve())
        if grade is not None:
            document.metadata["grade"] = grade
        if len(document.page_content) >= 120:
            result.append(document)
    return result


def _reference_files(data_dirs: list[Path] | None = None) -> list[Path]:
    directories = data_dirs or config.QUIZ_REFERENCE_DIRS
    files: list[Path] = []
    for configured_dir in directories:
        base_dir = config.resolve_reference_dir(configured_dir)
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*"):
            if path.is_file() and config.include_quiz_reference_file(path):
                files.append(path)
    return sorted(set(files), key=lambda item: str(item).lower())


def load_documents(data_dirs: list[Path] | None = None) -> list[Document]:
    documents: list[Document] = []
    for path in _reference_files(data_dirs):
        documents.extend(_load_file(path))
    return documents


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    try:
        return Tokenizer.from_pretrained(
            config.CHUNK_TOKENIZER_MODEL,
                    )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer '{config.CHUNK_TOKENIZER_MODEL}'."
        ) from exc


def _encode(text: str) -> list[int]:
    return list(_tokenizer().encode(text, add_special_tokens=False).ids)


def _decode(token_ids: list[int]) -> str:
    return _tokenizer().decode(token_ids, skip_special_tokens=True).strip()


_HEADING_RE = re.compile(
    r"^(?:unit|chapter|lesson|topic|section|[0-9]+(?:\.[0-9]+){0,3})\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(r"^(?:worked\s+)?example\b", re.IGNORECASE)
_EXERCISE_RE = re.compile(r"^(?:exercise|activity|review|questions?)\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"^(?:definition|theorem|property|note|remember|summary)\b", re.IGNORECASE
)


def _line_type(line: str) -> str:
    value = line.strip()
    if _EXAMPLE_RE.match(value):
        return "example"
    if _EXERCISE_RE.match(value):
        return "exercise"
    if _DEFINITION_RE.match(value):
        return "definition"
    if _HEADING_RE.match(value) or (
        len(value) <= 90
        and len(value.split()) <= 10
        and value.isupper()
        and any(character.isalpha() for character in value)
    ):
        return "heading"
    math_symbols = sum(value.count(symbol) for symbol in "=+-×÷√∠π²³")
    if math_symbols >= 2 and len(value) <= 180:
        return "formula"
    return "prose"


def _structure_units(document: Document) -> list[StructureUnit]:
    source = str(document.metadata.get("source", "textbook"))
    page = int(document.metadata.get("page", 0)) + 1
    grade_value = document.metadata.get("grade")
    grade = int(grade_value) if isinstance(grade_value, int) else None
    parent_text = document.page_content.strip()
    current_section = Path(source).stem
    units: list[StructureUnit] = []
    buffer: list[str] = []
    buffer_type = "prose"

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            units.append(
                StructureUnit(
                    text=text,
                    source=source,
                    page=page,
                    parent_text=parent_text,
                    section=current_section,
                    block_type=buffer_type,
                    grade=grade,
                )
            )
        buffer = []

    for raw_line in parent_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            buffer_type = "prose"
            continue
        line_type = _line_type(line)
        if line_type in {"heading", "example", "exercise", "definition"}:
            flush()
            buffer_type = line_type
            if line_type == "heading":
                current_section = line[:180]
        elif line_type == "formula" and buffer_type == "prose":
            buffer_type = "formula"
        buffer.append(line)
    flush()
    return units


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "topic"


@lru_cache(maxsize=1)
def _concept_catalog() -> dict[int, list[dict[str, Any]]]:
    with config.TOPICS_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    catalog: dict[int, list[dict[str, Any]]] = {}
    for grade_row in data.get("grades", []):
        grade = int(grade_row.get("grade", 0))
        for term_row in grade_row.get("terms", []):
            term = int(term_row.get("term", 0))
            for position, row in enumerate(term_row.get("topics", []), start=1):
                name = str(row.get("topic", "")).strip()
                if not name:
                    continue
                number = int(row.get("no", position))
                words = {
                    word
                    for word in re.findall(r"[a-z0-9]+", name.lower())
                    if word not in {"a", "an", "and", "of", "the", "i", "ii"}
                }
                catalog.setdefault(grade, []).append(
                    {
                        "concept_id": f"g{grade}_t{term}_{number:02d}_{_slug(name)}",
                        "topic": name,
                        "words": words,
                        "phrase": " ".join(re.findall(r"[a-z0-9]+", name.lower())),
                    }
                )
    return catalog


def _match_concept(text: str, grade: int | None) -> tuple[str, str]:
    if grade is None:
        return "", ""
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    words = set(normalized.split())
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for concept in _concept_catalog().get(grade, []):
        concept_words = set(concept["words"])
        if not concept_words:
            continue
        overlap = len(words.intersection(concept_words)) / len(concept_words)
        phrase_bonus = 2.0 if concept["phrase"] in normalized else 0.0
        score = phrase_bonus + overlap
        if score > best[0]:
            best = (score, concept)
    if best[1] is None or best[0] < 1.0:
        return "", ""
    return str(best[1]["concept_id"]), str(best[1]["topic"])


def _split_large_unit(unit: StructureUnit) -> list[StructureUnit]:
    token_ids = _encode(unit.text)
    if len(token_ids) <= config.CHUNK_MAX_TOKENS:
        return [unit]
    pieces: list[StructureUnit] = []
    step = config.CHUNK_TARGET_TOKENS - config.CHUNK_OVERLAP_TOKENS
    for start in range(0, len(token_ids), step):
        text = _decode(token_ids[start : start + config.CHUNK_TARGET_TOKENS])
        if not text:
            continue
        pieces.append(
            StructureUnit(
                text=text,
                source=unit.source,
                page=unit.page,
                parent_text=unit.parent_text,
                section=unit.section,
                block_type=unit.block_type,
                grade=unit.grade,
            )
        )
        if start + config.CHUNK_TARGET_TOKENS >= len(token_ids):
            break
    return pieces


def _chunk_metadata(
    units: list[StructureUnit],
    text: str,
    *,
    index: int,
    corpus_id: str,
) -> dict[str, Any]:
    source = units[0].source
    grade = next((unit.grade for unit in units if unit.grade is not None), None)
    page_start = min(unit.page for unit in units)
    page_end = max(unit.page for unit in units)
    parent_pages: list[str] = []
    seen_pages: set[tuple[str, int]] = set()
    for unit in units:
        key = (unit.source, unit.page)
        if key not in seen_pages:
            seen_pages.add(key)
            parent_pages.append(unit.parent_text)
    block_priority = ["example", "exercise", "formula", "definition", "heading", "prose"]
    block_types = {unit.block_type for unit in units}
    block_type = next(kind for kind in block_priority if kind in block_types)
    section = next((unit.section for unit in reversed(units) if unit.section), Path(source).stem)
    concept_id, topic = _match_concept(text, grade)
    identity = f"{corpus_id}|{source}|{page_start}|{page_end}|{index}|{text}"
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {
        "chunk_id": chunk_id,
        "grade": grade if grade is not None else -1,
        "concept_id": concept_id,
        "topic": topic,
        "book_id": Path(source).stem,
        "source": source,
        "page": page_start - 1,
        "page_start": page_start,
        "page_end": page_end,
        "section": section[:180],
        "block_type": block_type,
        "parent_text": "\n\n".join(parent_pages),
        "corpus_version": corpus_id,
        "chunker_version": config.CHUNKER_VERSION,
        "token_count": len(_encode(text)),
    }


def split_documents(documents: list[Document], *, corpus_id: str) -> list[Document]:
    if not (
        250 <= config.CHUNK_MIN_TOKENS <= config.CHUNK_TARGET_TOKENS
        <= config.CHUNK_MAX_TOKENS <= 450
    ):
        raise ValueError("Chunk token limits must satisfy min <= target <= max <= 450.")
    if not 50 <= config.CHUNK_OVERLAP_TOKENS <= 75:
        raise ValueError("Chunk overlap must be between 50 and 75 tokens.")

    expanded_units: list[StructureUnit] = []
    for document in documents:
        for unit in _structure_units(document):
            expanded_units.extend(_split_large_unit(unit))

    chunks: list[Document] = []
    current: list[StructureUnit] = []
    current_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal current, current_tokens, chunk_index
        if not current:
            return
        text = "\n\n".join(unit.text for unit in current).strip()
        if not text:
            current = []
            current_tokens = 0
            return
        token_ids = _encode(text)
        if len(token_ids) > config.CHUNK_MAX_TOKENS:
            token_ids = token_ids[: config.CHUNK_MAX_TOKENS]
            text = _decode(token_ids)
        metadata = _chunk_metadata(
            current,
            text,
            index=chunk_index,
            corpus_id=corpus_id,
        )
        chunks.append(Document(page_content=text, metadata=metadata))
        chunk_index += 1
        overlap_ids = token_ids[-config.CHUNK_OVERLAP_TOKENS :]
        anchor = current[-1]
        overlap_text = _decode(overlap_ids)
        current = [
            StructureUnit(
                text=overlap_text,
                source=anchor.source,
                page=anchor.page,
                parent_text=anchor.parent_text,
                section=anchor.section,
                block_type=anchor.block_type,
                grade=anchor.grade,
            )
        ] if overlap_text else []
        current_tokens = len(overlap_ids)

    for unit in expanded_units:
        unit_tokens = len(_encode(unit.text))
        source_changed = bool(current and current[-1].source != unit.source)
        would_exceed = current_tokens + unit_tokens > config.CHUNK_MAX_TOKENS
        reached_target = current_tokens >= config.CHUNK_TARGET_TOKENS
        if source_changed or would_exceed or reached_target:
            flush()
            if source_changed:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    flush()

    for index, chunk in enumerate(chunks):
        if len(_encode(chunk.page_content)) >= config.CHUNK_MIN_TOKENS:
            continue
        source = chunk.metadata.get("source")
        previous = chunks[index - 1] if index > 0 else None
        following = chunks[index + 1] if index + 1 < len(chunks) else None
        use_previous = previous is not None and previous.metadata.get("source") == source
        neighbor = previous if use_previous else following
        if neighbor is None or neighbor.metadata.get("source") != source:
            raise ValueError(
                f"Source '{source}' does not contain enough text for a "
                f"{config.CHUNK_MIN_TOKENS}-token chunk."
            )
        combined_text = (
            previous.page_content + "\n\n" + chunk.page_content
            if use_previous
            else chunk.page_content + "\n\n" + following.page_content
        )
        combined = _encode(combined_text)
        window_size = min(
            len(combined),
            config.CHUNK_MIN_TOKENS + 16,
            config.CHUNK_MAX_TOKENS,
        )
        expanded = ""
        while window_size <= min(len(combined), config.CHUNK_MAX_TOKENS):
            window = combined[-window_size:] if use_previous else combined[:window_size]
            expanded = _decode(window)
            if len(_encode(expanded)) >= config.CHUNK_MIN_TOKENS:
                break
            window_size += 1
        expanded_count = len(_encode(expanded))
        if not config.CHUNK_MIN_TOKENS <= expanded_count <= config.CHUNK_MAX_TOKENS:
            raise ValueError(
                f"Could not produce an approved-size chunk for source '{source}'."
            )
        chunk.page_content = expanded
        chunk.metadata["page_start"] = min(
            int(neighbor.metadata.get("page_start", 1)),
            int(chunk.metadata.get("page_start", 1)),
        )
        chunk.metadata["page_end"] = max(
            int(neighbor.metadata.get("page_end", 1)),
            int(chunk.metadata.get("page_end", 1)),
        )
        parent_parts = (
            [neighbor.metadata.get("parent_text", ""), chunk.metadata.get("parent_text", "")]
            if use_previous
            else [chunk.metadata.get("parent_text", ""), neighbor.metadata.get("parent_text", "")]
        )
        chunk.metadata["parent_text"] = "\n\n".join(
            str(part) for part in parent_parts if str(part).strip()
        ).strip()
        chunk.metadata["token_count"] = expanded_count
        concept_id, topic = _match_concept(
            expanded,
            int(chunk.metadata["grade"]) if int(chunk.metadata["grade"]) >= 0 else None,
        )
        chunk.metadata["concept_id"] = concept_id
        chunk.metadata["topic"] = topic
        identity = (
            f"{corpus_id}|{chunk.metadata.get('source')}|"
            f"{chunk.metadata.get('page_start')}|{chunk.metadata.get('page_end')}|"
            f"{index}|{expanded}"
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    invalid_sizes = [
        int(chunk.metadata["token_count"])
        for chunk in chunks
        if not config.CHUNK_MIN_TOKENS
        <= int(chunk.metadata["token_count"])
        <= config.CHUNK_MAX_TOKENS
    ]
    if invalid_sizes:
        raise ValueError("Ingestion produced chunks outside the approved token range.")
    return chunks


def create_vector_store(chunks: list[Document]) -> None:
    embeddings = build_embeddings()
    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    if config.VECTOR_STORE_PROVIDER == "qdrant":
        raise RuntimeError("Use ingest_all to publish a versioned Qdrant index.")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False),
    )


def _manifest_path() -> Path:
    return config.INGEST_STATE_DIR / MANIFEST_FILE_NAME


def _settings_fingerprint(manifest: list[dict[str, object]]) -> str:
    payload = {
        "sources": manifest,
        "chunker_version": config.CHUNKER_VERSION,
        "chunk_min": config.CHUNK_MIN_TOKENS,
        "chunk_target": config.CHUNK_TARGET_TOKENS,
        "chunk_max": config.CHUNK_MAX_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP_TOKENS,
        "tokenizer": config.CHUNK_TOKENIZER_MODEL,
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "embedding_provider": config.HF_EMBEDDING_PROVIDER,
        "embedding_dimensions": config.HF_EMBEDDING_DIMENSIONS,
        "vector_store": config.VECTOR_STORE_PROVIDER,
        "vector_url": config.QDRANT_URL if config.VECTOR_STORE_PROVIDER == "qdrant" else "",
        "collection": (
            config.QDRANT_COLLECTION_NAME
            if config.VECTOR_STORE_PROVIDER == "qdrant"
            else config.CHROMA_COLLECTION_NAME
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_ingest_manifest() -> dict[str, Any]:
    if config.VECTOR_STORE_PROVIDER == "qdrant":
        from rag.cloud_vectors import read_manifest
        return read_manifest()
    try:
        with _manifest_path().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_ingest_manifest(data: dict[str, Any]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def vector_index_status(
    *,
    required_grades: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    grades_to_check = tuple(
        int(grade)
        for grade in (
            required_grades
            if required_grades is not None
            else config.DIAGNOSTIC_GRADES
        )
    )
    client = None
    try:
        if config.VECTOR_STORE_PROVIDER == "qdrant":
            from rag.cloud_vectors import qdrant_client, read_manifest, validate_collection
            from rag.embeddings import EMBEDDING_CONTRACT
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            client = qdrant_client()
            manifest = read_manifest(client)
            if manifest.get("status") != "ready" or manifest.get("embedding_contract") != EMBEDDING_CONTRACT:
                return {"ready": False, "reason": "The Qdrant index is incomplete or uses another embedding configuration. Finish ingestion into a new collection."}
            validate_collection(client)
            count = client.count(config.QDRANT_COLLECTION_NAME, exact=True).count - 1
            if count != manifest.get("chunk_count"):
                return {"ready": False, "reason": "Qdrant count differs from the completed index manifest.", "count": count}
            missing_grades: list[int] = []
            for grade in grades_to_check:
                conditions = [
                    FieldCondition(
                        key="metadata.grade",
                        match=MatchValue(value=grade),
                    )
                ]
                if config.CORPUS_VERSION:
                    conditions.append(
                        FieldCondition(
                            key="metadata.corpus_version",
                            match=MatchValue(value=config.CORPUS_VERSION),
                        )
                    )
                points, _ = client.scroll(
                    collection_name=config.QDRANT_COLLECTION_NAME,
                    scroll_filter=Filter(must=conditions),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if not points:
                    missing_grades.append(grade)
        else:
            import chromadb

            persist_dir = Path(config.CHROMA_PERSIST_DIR)
            if not (persist_dir / "chroma.sqlite3").exists():
                return {"ready": False, "reason": "Chroma database does not exist."}
            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                collection = client.get_collection(config.CHROMA_COLLECTION_NAME)
            except Exception:
                return {"ready": False, "reason": "Chroma collection does not exist."}
            count = int(collection.count())
            missing_grades = []
            for grade in grades_to_check:
                grade_where: dict[str, Any] = {"grade": grade}
                rows = collection.get(where=grade_where, limit=1, include=["metadatas"])
                if not rows.get("ids"):
                    missing_grades.append(grade)
                    continue

                if config.CORPUS_VERSION:
                    versioned_where: dict[str, Any] = {
                        "$and": [
                            {"grade": grade},
                            {"corpus_version": config.CORPUS_VERSION},
                        ]
                    }
                    versioned_rows = collection.get(
                        where=versioned_where, limit=1, include=["metadatas"]
                    )
                    if not versioned_rows.get("ids"):
                        # Backwards compatibility for legacy/local Chroma indexes
                        # that predate corpus_version metadata (and small isolated
                        # test collections). If version metadata exists, however, a
                        # mismatch remains a hard readiness failure.
                        metadata_rows = rows.get("metadatas") or []
                        has_version_metadata = any(
                            isinstance(metadata, dict)
                            and bool(str(metadata.get("corpus_version", "")).strip())
                            for metadata in metadata_rows
                        )
                        if has_version_metadata:
                            missing_grades.append(grade)

        if count <= 0:
            return {"ready": False, "reason": "Vector collection is empty.", "count": count}
        if missing_grades:
            grades = ", ".join(str(grade) for grade in missing_grades)
            return {
                "ready": False,
                "reason": f"Vector collection has no Grade {grades} textbook chunks.",
                "count": count,
                "missing_grades": missing_grades,
            }
        return {"ready": True, "count": count, "missing_grades": []}
    except Exception as exc:
        logger.exception("Vector index status check failed")
        return {"ready": False, "reason": f"Vector index check failed: {exc}"}
    finally:
        if config.VECTOR_STORE_PROVIDER == "qdrant" and client is not None:
            client.close()


def ensure_vector_index_ready(
    *,
    repair: bool = False,
    required_grades: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    status = vector_index_status(required_grades=required_grades)
    if status.get("ready"):
        return status
    if not repair:
        raise RuntimeError(
            f"{status.get('reason', 'Vector index is not ready')} "
            "Run `python main.py ingest --force` before loading a quiz."
        )
    ingest_all(force=True)
    repaired = vector_index_status(required_grades=required_grades)
    if not repaired.get("ready"):
        raise RuntimeError(
            "Vector index rebuild completed but validation failed: "
            f"{repaired.get('reason', 'unknown reason')}"
        )
    return repaired


def ingest_all(force: bool = False) -> dict[str, Any]:
    source_files = _reference_files()
    if not source_files:
        raise ValueError("No Grade 6-11 textbook reference files were found.")
    if config.VECTOR_STORE_PROVIDER == "qdrant" and not config.QDRANT_WRITE_API_KEY:
        raise RuntimeError("QDRANT_WRITE_API_KEY is required for ingestion.")

    manifest = source_manifest(source_files)
    corpus_id = corpus_version(manifest)
    fingerprint = _settings_fingerprint(manifest)
    previous = read_ingest_manifest()
    if not force and previous.get("fingerprint") == fingerprint:
        status = vector_index_status()
        if status.get("ready"):
            return {
                "status": "skipped",
                "message": "Ingestion skipped because the corpus is unchanged.",
                "corpus_version": corpus_id,
                "count": status.get("count", 0),
            }

    documents = load_documents()
    if not documents:
        raise ValueError("No readable Grade 6-11 textbook pages were found.")
    chunks = split_documents(documents, corpus_id=corpus_id)
    if not chunks:
        raise ValueError("The structure-aware chunker produced no textbook chunks.")

    persist_dir = Path(config.CHROMA_PERSIST_DIR).resolve()
    if config.VECTOR_STORE_PROVIDER == "chroma" and persist_dir.exists():
        if persist_dir in {Path("/").resolve(), Path.home().resolve(), config.PROJECT_ROOT.resolve()}:
            raise RuntimeError("Refusing to replace an unsafe Chroma directory.")
        shutil.rmtree(persist_dir)
    manifest_data = {
        "fingerprint": fingerprint, "corpus_version": corpus_id,
        "chunker_version": config.CHUNKER_VERSION,
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "embedding_provider": config.HF_EMBEDDING_PROVIDER,
        "embedding_dimensions": config.HF_EMBEDDING_DIMENSIONS,
        "chunk_count": len(chunks), "sources": manifest,
    }
    if config.VECTOR_STORE_PROVIDER == "qdrant":
        from rag.cloud_vectors import upload_chunks
        upload_chunks(chunks, manifest_data)
    else:
        create_vector_store(chunks)
    from rag.retriever import clear_retrieval_cache
    clear_retrieval_cache()
    _write_ingest_manifest(manifest_data)
    return {
        "status": "rebuilt",
        "message": "Ingestion complete.",
        "corpus_version": corpus_id,
        "count": len(chunks),
    }


if __name__ == "__main__":
    ingest_all()
