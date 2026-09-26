"""Grade-filtered hybrid retrieval with deterministic reranking."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config
from rag.embeddings import build_embeddings
from rag.question_bank import load_question_bank, questions_for_concept


@dataclass
class LexicalIndex:
    documents: list[Document]
    tokenized: list[list[str]]
    bm25: BM25Okapi
    loaded_at: float


_lexical_indexes: dict[tuple, LexicalIndex] = {}
_lexical_lock = threading.Lock()


def clear_retrieval_cache() -> None:
    with _lexical_lock:
        _lexical_indexes.clear()


def get_vector_store():
    embeddings = build_embeddings()
    if config.VECTOR_STORE_PROVIDER == "qdrant":
        from langchain_qdrant import QdrantVectorStore

        from rag.cloud_vectors import qdrant_client, read_manifest, validate_collection
        from rag.embeddings import EMBEDDING_CONTRACT

        client = qdrant_client()
        try:
            validate_collection(client)
            manifest = read_manifest(client)
            if manifest.get("status") != "ready" or manifest.get("embedding_contract") != EMBEDDING_CONTRACT:
                raise RuntimeError("Qdrant must contain a completed Hugging Face BGE-M3 index. Run ingestion before loading quizzes.")
            return QdrantVectorStore(
                client=client, embedding=embeddings, collection_name=config.QDRANT_COLLECTION_NAME,
                validate_embeddings=False, validate_collection_config=False,
            )
        except Exception:
            client.close()
            raise
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )


def _terms(value: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+|[=+\-*/^√π]+", str(value or "").lower())
    return [term[:-1] if len(term) > 4 and term.endswith("s") else term for term in terms]


def _document_key(document: Document) -> str:
    chunk_id = str((document.metadata or {}).get("chunk_id", ""))
    if chunk_id:
        return chunk_id
    payload = f"{document.metadata}|{document.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grade_filter(grade: int):
    if config.VECTOR_STORE_PROVIDER == "qdrant":
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key="metadata.grade", match=MatchValue(value=grade))
        ]
        if config.CORPUS_VERSION:
            conditions.append(
                FieldCondition(
                    key="metadata.corpus_version",
                    match=MatchValue(value=config.CORPUS_VERSION),
                )
            )
        return Filter(must=conditions)
    if config.CORPUS_VERSION:
        return {
            "$and": [
                {"grade": grade},
                {"corpus_version": config.CORPUS_VERSION},
            ]
        }
    return {"grade": grade}


def _load_qdrant_grade(grade: int) -> list[Document]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY or None,
        timeout=config.QDRANT_TIMEOUT_SECONDS,
    )
    conditions = [FieldCondition(key="metadata.grade", match=MatchValue(value=grade))]
    if config.CORPUS_VERSION:
        conditions.append(
            FieldCondition(
                key="metadata.corpus_version",
                match=MatchValue(value=config.CORPUS_VERSION),
            )
        )
    scroll_filter = Filter(must=conditions)
    offset = None
    documents: list[Document] = []
    try:
        while True:
            points, offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                text = str(payload.get("page_content", "")).strip()
                metadata = payload.get("metadata", {})
                if text and isinstance(metadata, dict):
                    documents.append(Document(page_content=text, metadata=metadata))
            if offset is None:
                break
        return documents
    finally:
        client.close()


def _load_chroma_grade(grade: int, vectorstore: Chroma) -> list[Document]:
    result = vectorstore.get(
        where=_grade_filter(grade),
        include=["documents", "metadatas"],
    )
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    return [
        Document(page_content=str(text), metadata=dict(metadata or {}))
        for text, metadata in zip(documents, metadatas)
        if str(text or "").strip()
    ]


def _lexical_index(grade: int, vectorstore) -> LexicalIndex:
    collection = (
        config.QDRANT_COLLECTION_NAME
        if config.VECTOR_STORE_PROVIDER == "qdrant"
        else config.CHROMA_COLLECTION_NAME
    )
    key = (config.VECTOR_STORE_PROVIDER, config.QDRANT_URL, collection, config.CORPUS_VERSION, grade)
    now = time.monotonic()
    with _lexical_lock:
        cached = _lexical_indexes.get(key)
        if cached and now - cached.loaded_at < config.LEXICAL_CACHE_SECONDS:
            return cached
        documents = (
            _load_qdrant_grade(grade)
            if config.VECTOR_STORE_PROVIDER == "qdrant"
            else _load_chroma_grade(grade, vectorstore)
        )
        if not documents:
            raise LookupError(f"No Grade {grade} textbook chunks exist in the vector index.")
        tokenized = [_terms(document.page_content) for document in documents]
        index = LexicalIndex(
            documents=documents,
            tokenized=tokenized,
            bm25=BM25Okapi(tokenized),
            loaded_at=now,
        )
        _lexical_indexes[key] = index
        return index


def _dense_candidates(query: str, grade: int, vectorstore) -> list[Document]:
    pairs = vectorstore.similarity_search_with_score(
        query,
        k=config.RETRIEVAL_DENSE_K,
        filter=_grade_filter(grade),
    )
    return [document for document, _score in pairs]


def _lexical_candidates(query: str, index: LexicalIndex) -> list[Document]:
    scores = index.bm25.get_scores(_terms(query))
    ranked = sorted(range(len(scores)), key=lambda item: float(scores[item]), reverse=True)
    return [index.documents[item] for item in ranked[: config.RETRIEVAL_LEXICAL_K]]


def _hybrid_search(
    *,
    query: str,
    topic_name: str,
    concept_id: str,
    grade: int,
    k: int,
    vectorstore,
) -> list[Document]:
    dense = _dense_candidates(query, grade, vectorstore)
    lexical_index = _lexical_index(grade, vectorstore)
    lexical = _lexical_candidates(query, lexical_index)
    candidates: dict[str, Document] = {}
    score: dict[str, float] = {}

    for rank, document in enumerate(dense, start=1):
        key = _document_key(document)
        candidates[key] = document
        score[key] = score.get(key, 0.0) + 1.0 / (60 + rank)
    for rank, document in enumerate(lexical, start=1):
        key = _document_key(document)
        candidates[key] = document
        score[key] = score.get(key, 0.0) + 1.0 / (60 + rank)

    topic_terms = set(_terms(topic_name))
    topic_phrase = " ".join(_terms(topic_name))
    for key, document in candidates.items():
        metadata = document.metadata or {}
        document_terms = set(_terms(document.page_content))
        coverage = (
            len(topic_terms.intersection(document_terms)) / len(topic_terms)
            if topic_terms
            else 0.0
        )
        normalized_document = " ".join(_terms(document.page_content))
        phrase_bonus = 0.006 if topic_phrase and topic_phrase in normalized_document else 0.0
        concept_bonus = 0.008 if metadata.get("concept_id") == concept_id else 0.0
        source_bonus = 0.002 if int(metadata.get("grade", -1)) == grade else -1.0
        score[key] += (coverage * 0.01) + phrase_bonus + concept_bonus + source_bonus

    ranked_keys = sorted(score, key=lambda item: score[item], reverse=True)
    return [candidates[key] for key in ranked_keys[:k]]


def retrieve_for_topic(
    topic_name: str,
    difficulty_level: str,
    grade: int | None = None,
    k: int = 5,
    vectorstore=None,
) -> list[str]:
    query = f"{topic_name} {difficulty_level} mathematics"
    store = vectorstore if vectorstore is not None else get_vector_store()
    if grade is None:
        documents = store.similarity_search(query, k=k)
    else:
        documents = _hybrid_search(
            query=f"Grade {grade} {query}",
            topic_name=topic_name,
            concept_id="",
            grade=grade,
            k=k,
            vectorstore=store,
        )
    return [document.page_content for document in documents]


def retrieve_grounded_topic(
    topic_name: str,
    difficulty_level: str,
    *,
    grade: int,
    concept_id: str = "",
    k: int | None = None,
    vectorstore=None,
) -> dict[str, Any]:
    final_k = k or config.RETRIEVAL_FINAL_K
    query = f"Grade {grade} {topic_name} {difficulty_level} worked mathematics"
    store = vectorstore if vectorstore is not None else get_vector_store()
    documents = _hybrid_search(
        query=query,
        topic_name=topic_name,
        concept_id=concept_id,
        grade=grade,
        k=final_k,
        vectorstore=store,
    )
    if not documents:
        raise LookupError(
            f"No Grade {grade} textbook content was retrieved for '{topic_name}'."
        )

    chunks: list[str] = []
    source_refs: list[dict[str, Any]] = []
    grounding_units: list[dict[str, Any]] = []
    seen_refs: set[tuple[str, int, int]] = set()
    for document in documents:
        metadata = document.metadata or {}
        child_text = str(document.page_content or "").strip()
        parent_text = str(metadata.get("parent_text", "")).strip()
        context_text = parent_text if parent_text and child_text in parent_text else child_text
        source_name = Path(str(metadata.get("source", "textbook"))).name
        page_start = int(metadata.get("page_start", int(metadata.get("page", 0)) + 1))
        page_end = int(metadata.get("page_end", page_start))
        source_ref = {
            "source": source_name,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "grade": grade,
            "book_id": str(metadata.get("book_id", Path(source_name).stem)),
            "section": str(metadata.get("section", "")),
            "chunk_id": str(metadata.get("chunk_id", "")),
            "corpus_version": str(metadata.get("corpus_version", "")),
        }
        if context_text:
            chunks.append(context_text)
            grounding_units.append({"text": context_text, "source_ref": source_ref})
        key = (source_name, page_start, page_end)
        if key not in seen_refs:
            seen_refs.add(key)
            source_refs.append(source_ref)
    if not chunks:
        raise LookupError(
            f"Retrieved Grade {grade} documents for '{topic_name}' contained no text."
        )
    return {
        "chunks": chunks,
        "source_refs": source_refs,
        "grounding_units": grounding_units,
    }


def retrieve_for_syllabus(
    syllabus: dict[str, Any],
    k_per_topic: int | None = None,
    *,
    exclude_question_ids_by_topic: dict[str, set[str]] | None = None,
    question_seed: str = "",
) -> dict[str, Any]:
    levels_lookup: dict[int, str] = {}
    try:
        with config.DIFFICULTY_LEVELS_FILE.open("r", encoding="utf-8") as handle:
            levels_lookup = {
                int(row["level_id"]): str(row["name"])
                for row in json.load(handle)
            }
    except (OSError, ValueError, KeyError, TypeError):
        levels_lookup = {}

    results: dict[str, Any] = {}
    question_bank = load_question_bank()
    exclusions = exclude_question_ids_by_topic or {}
    vectorstore = get_vector_store()
    for mapping in syllabus.get("topic_mappings", []):
        topic_id = str(mapping["topic_id"])
        topic_name = str(mapping.get("topic_name") or topic_id)
        grade = mapping.get("grade")
        required_levels = mapping.get("required_levels", [1])
        level_name = levels_lookup.get(max(required_levels), "Apply")
        if mapping.get("strict_textbook_grounding"):
            if not isinstance(grade, int):
                raise ValueError(f"Strictly grounded topic '{topic_id}' is missing its grade.")
            retrieved = retrieve_grounded_topic(
                topic_name,
                level_name,
                grade=grade,
                concept_id=topic_id,
                k=k_per_topic,
                vectorstore=vectorstore,
            )
        else:
            retrieved = {
                "chunks": retrieve_for_topic(
                    topic_name,
                    level_name,
                    grade=grade if isinstance(grade, int) else None,
                    k=k_per_topic or config.RETRIEVAL_FINAL_K,
                    vectorstore=vectorstore,
                ),
                "source_refs": [],
                "grounding_units": [],
            }
        past_questions = (
            questions_for_concept(
                topic_id,
                grade=grade,
                exclude_question_ids=exclusions.get(topic_id, set()),
                seed=question_seed,
                questions=question_bank,
            )
            if isinstance(grade, int)
            else []
        )
        retrieved["past_questions"] = [
            question.prompt_payload() for question in past_questions
        ]
        retrieved["past_paper_refs"] = [
            question.source_reference() for question in past_questions
        ]
        results[topic_id] = retrieved
    return results
