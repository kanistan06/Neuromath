from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

import config
import rag.ingest as ingest


class WordTokenizer:
    def __init__(self):
        self.vocabulary = {}
        self.reverse = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            if word not in self.vocabulary:
                token_id = len(self.vocabulary) + 1
                self.vocabulary[word] = token_id
                self.reverse[token_id] = word
            ids.append(self.vocabulary[word])
        return SimpleNamespace(ids=ids)

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.reverse[token_id] for token_id in token_ids)


def test_structure_aware_chunks_keep_token_limits_and_provenance(monkeypatch):
    tokenizer = WordTokenizer()
    monkeypatch.setattr(ingest, "_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(config, "CHUNK_MIN_TOKENS", 300)
    monkeypatch.setattr(config, "CHUNK_TARGET_TOKENS", 360)
    monkeypatch.setattr(config, "CHUNK_MAX_TOKENS", 450)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 60)

    documents = [
        Document(
            page_content="UNIT 1 PERIMETER\nExample 1\n" + " ".join(["perimeter"] * 410),
            metadata={"source": "/books/maths-g-10.pdf", "page": 0, "grade": 10},
        ),
        Document(
            page_content="Worked Example 2\n" + " ".join(["rectangle"] * 410),
            metadata={"source": "/books/maths-g-10.pdf", "page": 1, "grade": 10},
        ),
    ]

    first = ingest.split_documents(documents, corpus_id="test-corpus-v1")
    second = ingest.split_documents(documents, corpus_id="test-corpus-v1")

    assert first
    assert [row.metadata["chunk_id"] for row in first] == [
        row.metadata["chunk_id"] for row in second
    ]
    assert all(300 <= row.metadata["token_count"] <= 450 for row in first)
    assert all(row.metadata["grade"] == 10 for row in first)
    assert all(row.metadata["book_id"] == "maths-g-10" for row in first)
    assert all(row.metadata["corpus_version"] == "test-corpus-v1" for row in first)
    assert all(row.metadata["chunker_version"] == config.CHUNKER_VERSION for row in first)
    assert all(row.metadata["page_start"] >= 1 for row in first)
    assert all(row.metadata["page_end"] >= row.metadata["page_start"] for row in first)
    assert any(row.metadata["block_type"] == "example" for row in first)


def test_grade_is_detected_in_existing_reference_filenames():
    assert ingest._extract_grade_from_filename(Path("maths G-6 E P-I.pdf")) == 6
    assert ingest._extract_grade_from_filename(Path("maths G-7 P-II E.pdf")) == 7
    assert ingest._extract_grade_from_filename(Path("maths G-8 P-I E.pdf")) == 8
    assert ingest._extract_grade_from_filename(Path("maths G9 P-II E.pdf")) == 9
    assert ingest._extract_grade_from_filename(Path("eGr10OM ELC.pdf")) == 10
    assert ingest._extract_grade_from_filename(Path("maths g-11 p-II E.pdf")) == 11


def test_textbook_reference_filter_includes_grades_6_to_11():
    names = [
        "maths G-6 E P-I.pdf",
        "maths G-7 P-II E.pdf",
        "maths G-8 P-I E.pdf",
        "maths G9 P-II E.pdf",
        "maths g-10 p-I E.pdf",
        "maths g-11 p-III E.pdf",
    ]

    for name in names:
        path = config.DATA_DIR / "TextBooks" / name
        assert config.include_quiz_reference_file(path)
