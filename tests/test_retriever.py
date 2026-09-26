import time

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config
import rag.retriever as retriever


def test_hybrid_retrieval_promotes_exact_math_terms_and_concept(monkeypatch):
    dense_document = Document(
        page_content="General measurements and numerical calculations.",
        metadata={"grade": 10, "chunk_id": "dense"},
    )
    lexical_document = Document(
        page_content="The perimeter of a rectangle is twice the sum of length and breadth.",
        metadata={
            "grade": 10,
            "chunk_id": "lexical",
            "concept_id": "g10_perimeter",
        },
    )

    class VectorStore:
        def similarity_search_with_score(self, query, k, filter):
            return [(dense_document, 0.9)]

    tokenized = [retriever._terms(lexical_document.page_content)]
    index = retriever.LexicalIndex(
        documents=[lexical_document],
        tokenized=tokenized,
        bm25=BM25Okapi(tokenized),
        loaded_at=time.monotonic(),
    )
    monkeypatch.setattr(retriever, "_lexical_index", lambda grade, vectorstore: index)
    monkeypatch.setattr(config, "RETRIEVAL_DENSE_K", 12)
    monkeypatch.setattr(config, "RETRIEVAL_LEXICAL_K", 12)

    result = retriever._hybrid_search(
        query="Grade 10 perimeter rectangle",
        topic_name="Perimeter",
        concept_id="g10_perimeter",
        grade=10,
        k=2,
        vectorstore=VectorStore(),
    )

    assert [document.metadata["chunk_id"] for document in result] == [
        "lexical",
        "dense",
    ]

