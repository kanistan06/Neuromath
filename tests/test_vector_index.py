import chromadb
from chromadb.config import Settings

import config
from rag.ingest import vector_index_status


def test_vector_index_requires_both_diagnostic_grades(tmp_path, monkeypatch):
    collection_name = "neuromath_test_index"
    monkeypatch.setattr(config, "VECTOR_STORE_PROVIDER", "chroma")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHROMA_COLLECTION_NAME", collection_name)
    monkeypatch.setattr(config, "DIAGNOSTIC_GRADES", (10, 11))

    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection(collection_name)
    collection.add(
        ids=["grade-10"],
        documents=["Grade 10 perimeter content"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"grade": 10}],
    )

    missing = vector_index_status()
    assert missing["ready"] is False
    assert missing["missing_grades"] == [11]

    collection.add(
        ids=["grade-11"],
        documents=["Grade 11 algebra content"],
        embeddings=[[0.0, 1.0]],
        metadatas=[{"grade": 11}],
    )
    ready = vector_index_status()
    assert ready["ready"] is True
    assert ready["count"] == 2


def test_vector_index_can_validate_one_requested_practice_grade(tmp_path, monkeypatch):
    collection_name = "neuromath_practice_grade_index"
    monkeypatch.setattr(config, "VECTOR_STORE_PROVIDER", "chroma")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHROMA_COLLECTION_NAME", collection_name)

    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection(collection_name)
    collection.add(
        ids=["grade-10"],
        documents=["Grade 10 perimeter content"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"grade": 10}],
    )

    missing = vector_index_status(required_grades=(9,))
    assert missing["ready"] is False
    assert missing["missing_grades"] == [9]

    collection.add(
        ids=["grade-9"],
        documents=["Grade 9 algebra content"],
        embeddings=[[0.0, 1.0]],
        metadatas=[{"grade": 9}],
    )
    ready = vector_index_status(required_grades=(9,))
    assert ready["ready"] is True
