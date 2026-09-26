"""One-time migration of existing NeuroMath Chroma vectors to Qdrant Cloud.

This script DOES NOT call the embedding service.
It copies already-stored Chroma embeddings, documents, and metadata into
the Qdrant structure expected by the current NeuroMath application.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import config  # noqa: E402

from rag.cloud_vectors import (  # noqa: E402
    _write_manifest,
    validate_collection,
)
from rag.corpus import corpus_version, source_manifest  # noqa: E402
from rag.embeddings import EMBEDDING_CONTRACT  # noqa: E402
from rag.ingest import _reference_files, _settings_fingerprint  # noqa: E402


def _point_id(
    metadata: dict[str, Any],
    chroma_id: str,
) -> str:
    chunk_id = str(metadata.get("chunk_id") or "").strip()

    if not chunk_id:
        raise RuntimeError(
            f"Chroma record {chroma_id!r} has no chunk_id metadata."
        )

    return str(
        uuid5(
            NAMESPACE_URL,
            "neuromath:" + chunk_id,
        )
    )


def _as_vector(
    value: Any,
    *,
    chroma_id: str,
) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        value = list(value)

    if len(value) != config.HF_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Chroma record {chroma_id!r} has "
            f"{len(value)} dimensions; expected "
            f"{config.HF_EMBEDDING_DIMENSIONS}."
        )

    vector = [float(item) for item in value]

    if any(not math.isfinite(item) for item in vector):
        raise RuntimeError(
            f"Chroma record {chroma_id!r} contains "
            "non-finite vector values."
        )

    norm = math.sqrt(
        sum(
            item * item
            for item in vector
        )
    )

    if not norm:
        raise RuntimeError(
            f"Chroma record {chroma_id!r} contains "
            "a zero vector."
        )

    return vector


def _open_chroma():
    persist_dir = Path(
        config.CHROMA_PERSIST_DIR
    ).resolve()

    database_file = (
        persist_dir
        / "chroma.sqlite3"
    )

    if not database_file.exists():
        raise RuntimeError(
            "Local Chroma database was not found at "
            f"{database_file}."
        )

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False
        ),
    )

    try:
        collection = client.get_collection(
            config.CHROMA_COLLECTION_NAME
        )
    except Exception as exc:
        names = [
            item.name
            for item
            in client.list_collections()
        ]

        raise RuntimeError(
            f"Chroma collection "
            f"{config.CHROMA_COLLECTION_NAME!r} "
            f"was not found. "
            f"Available collections: {names}"
        ) from exc

    return client, collection


def _qdrant_write_client():
    key = (
        config.QDRANT_WRITE_API_KEY
        or config.QDRANT_API_KEY
    )

    if not config.QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is required."
        )

    if not key:
        raise RuntimeError(
            "QDRANT_WRITE_API_KEY is required "
            "for migration."
        )

    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=key,
        timeout=config.QDRANT_TIMEOUT_SECONDS,
    )


def _current_manifest(
    chunk_count: int,
    source_rows: list[dict[str, object]],
) -> dict[str, Any]:
    current_corpus = corpus_version(
        source_rows
    )

    return {
        "fingerprint":
            _settings_fingerprint(
                source_rows
            ),

        "corpus_version":
            current_corpus,

        "chunker_version":
            config.CHUNKER_VERSION,

        "embedding_model":
            config.HF_EMBEDDING_MODEL,

        "embedding_provider":
            config.HF_EMBEDDING_PROVIDER,

        "embedding_dimensions":
            config.HF_EMBEDDING_DIMENSIONS,

        "chunk_count":
            chunk_count,

        "sources":
            source_rows,
    }


def preflight(
    batch_size: int,
) -> dict[str, Any]:
    if (
        config.VECTOR_STORE_PROVIDER
        != "qdrant"
    ):
        raise RuntimeError(
            "Set VECTOR_STORE_PROVIDER=qdrant "
            "before running this migration."
        )

    if (
        config.HF_EMBEDDING_MODEL
        != "BAAI/bge-m3"
        or
        config.HF_EMBEDDING_DIMENSIONS
        != 1024
    ):
        raise RuntimeError(
            "Migration is approved only "
            "for the existing 1024-dimensional "
            "BAAI/bge-m3 embedding space."
        )

    source_files = _reference_files()

    if not source_files:
        raise RuntimeError(
            "No Grade 6-11 textbook "
            "source files were found."
        )

    source_rows = source_manifest(
        source_files
    )

    current_corpus = corpus_version(
        source_rows
    )

    chroma_client, collection = (
        _open_chroma()
    )

    try:
        total = int(
            collection.count()
        )

        if total <= 0:
            raise RuntimeError(
                "The Chroma collection "
                "is empty."
            )

        seen = 0
        corpus_versions: set[str] = set()
        grades: set[int] = set()

        for offset in range(
            0,
            total,
            batch_size,
        ):
            result = collection.get(
                limit=min(
                    batch_size,
                    total - offset,
                ),
                offset=offset,
                include=[
                    "documents",
                    "metadatas",
                    "embeddings",
                ],
            )

            ids = list(
                result.get("ids")
                or []
            )

            documents = list(
                result.get("documents")
                or []
            )

            metadatas = list(
                result.get("metadatas")
                or []
            )

            embeddings = (
                result.get("embeddings")
            )

            embeddings = (
                list(embeddings)
                if embeddings is not None
                else []
            )

            if not (
                len(ids)
                == len(documents)
                == len(metadatas)
                == len(embeddings)
            ):
                raise RuntimeError(
                    "Chroma returned "
                    "inconsistent record columns."
                )

            for (
                record_id,
                document,
                metadata,
                embedding,
            ) in zip(
                ids,
                documents,
                metadatas,
                embeddings,
            ):
                if not str(
                    document or ""
                ).strip():
                    raise RuntimeError(
                        f"Chroma record "
                        f"{record_id!r} "
                        "has no document text."
                    )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    raise RuntimeError(
                        f"Chroma record "
                        f"{record_id!r} "
                        "has invalid metadata."
                    )

                _point_id(
                    metadata,
                    str(record_id),
                )

                _as_vector(
                    embedding,
                    chroma_id=str(
                        record_id
                    ),
                )

                value = str(
                    metadata.get(
                        "corpus_version"
                    )
                    or ""
                ).strip()

                if value:
                    corpus_versions.add(
                        value
                    )

                grade = metadata.get(
                    "grade"
                )

                if isinstance(
                    grade,
                    int,
                ):
                    grades.add(
                        grade
                    )

                seen += 1

        if seen != total:
            raise RuntimeError(
                f"Validated {seen} "
                f"Chroma records but "
                f"collection.count() "
                f"returned {total}."
            )

        if (
            corpus_versions
            and
            corpus_versions
            != {current_corpus}
        ):
            raise RuntimeError(
                "The local Chroma index "
                "belongs to another corpus "
                "version. "
                f"Chroma="
                f"{sorted(corpus_versions)}, "
                f"current={current_corpus}. "
                "Do not migrate stale vectors "
                "into the current Qdrant "
                "collection."
            )

        return {
            "count": total,
            "corpus_version":
                current_corpus,
            "grades":
                sorted(grades),
            "sources":
                source_rows,
        }

    finally:
        del chroma_client


def migrate(
    *,
    batch_size: int,
    replace_target: bool,
) -> None:
    summary = preflight(
        batch_size
    )

    total = int(
        summary["count"]
    )

    source_rows = (
        summary["sources"]
    )

    manifest = _current_manifest(
        total,
        source_rows,
    )

    chroma_client, collection = (
        _open_chroma()
    )

    qdrant = (
        _qdrant_write_client()
    )

    try:
        exists = (
            qdrant.collection_exists(
                config.QDRANT_COLLECTION_NAME
            )
        )

        if (
            exists
            and replace_target
        ):
            print(
                "Deleting existing Qdrant "
                "collection: "
                f"{config.QDRANT_COLLECTION_NAME}"
            )

            qdrant.delete_collection(
                config.QDRANT_COLLECTION_NAME
            )

            exists = False

        if exists:
            existing_count = (
                qdrant.count(
                    config.QDRANT_COLLECTION_NAME,
                    exact=True,
                ).count
            )

            raise RuntimeError(
                "Target Qdrant collection "
                f"{config.QDRANT_COLLECTION_NAME!r} "
                "already exists with "
                f"{existing_count} point(s). "
                "Use a new "
                "QDRANT_COLLECTION_NAME, "
                "or rerun with "
                "--replace-target only if "
                "you are certain the old "
                "collection can be deleted."
            )

        qdrant.create_collection(
            collection_name=
                config.QDRANT_COLLECTION_NAME,

            vectors_config=
                models.VectorParams(
                    size=
                        config
                        .HF_EMBEDDING_DIMENSIONS,

                    distance=
                        models.Distance.COSINE,

                    on_disk=True,
                ),

            on_disk_payload=True,
        )

        validate_collection(
            qdrant
        )

        for field, schema in [
            (
                "metadata.grade",
                models
                .PayloadSchemaType
                .INTEGER,
            ),
            (
                "metadata.corpus_version",
                models
                .PayloadSchemaType
                .KEYWORD,
            ),
        ]:
            qdrant.create_payload_index(
                config.QDRANT_COLLECTION_NAME,
                field,
                field_schema=schema,
                wait=True,
            )

        building_manifest = {
            **manifest,

            "embedding_contract":
                EMBEDDING_CONTRACT,

            "status":
                "building",
        }

        _write_manifest(
            qdrant,
            building_manifest,
        )

        migrated = 0

        for offset in range(
            0,
            total,
            batch_size,
        ):
            result = collection.get(
                limit=min(
                    batch_size,
                    total - offset,
                ),
                offset=offset,
                include=[
                    "documents",
                    "metadatas",
                    "embeddings",
                ],
            )

            ids = list(
                result.get("ids")
                or []
            )

            documents = list(
                result.get("documents")
                or []
            )

            metadatas = list(
                result.get("metadatas")
                or []
            )

            embeddings = (
                result.get("embeddings")
            )

            embeddings = (
                list(embeddings)
                if embeddings is not None
                else []
            )

            points = []

            for (
                record_id,
                document,
                metadata,
                embedding,
            ) in zip(
                ids,
                documents,
                metadatas,
                embeddings,
            ):
                metadata = dict(
                    metadata or {}
                )

                metadata[
                    "embedding_contract"
                ] = EMBEDDING_CONTRACT

                points.append(
                    models.PointStruct(
                        id=_point_id(
                            metadata,
                            str(record_id),
                        ),

                        vector=_as_vector(
                            embedding,
                            chroma_id=str(
                                record_id
                            ),
                        ),

                        payload={
                            "page_content":
                                str(document),

                            "metadata":
                                metadata,
                        },
                    )
                )

            if points:
                qdrant.upsert(
                    collection_name=
                        config
                        .QDRANT_COLLECTION_NAME,

                    points=points,
                    wait=True,
                )

                migrated += len(
                    points
                )

                print(
                    f"Migrated "
                    f"{migrated}/{total}"
                )

        raw_count = (
            qdrant.count(
                config.QDRANT_COLLECTION_NAME,
                exact=True,
            ).count
        )

        # One additional point is the
        # NeuroMath index manifest.
        document_count = (
            raw_count - 1
        )

        if document_count != total:
            raise RuntimeError(
                "Qdrant verification failed: "
                f"expected {total} "
                "document points, found "
                f"{document_count} "
                f"(raw count {raw_count})."
            )

        ready_manifest = {
            **building_manifest,

            "status":
                "ready",

            "chunk_count":
                document_count,
        }

        _write_manifest(
            qdrant,
            ready_manifest,
        )

        print()
        print(
            "Migration completed "
            "successfully."
        )

        print(
            f"Chroma records:      "
            f"{total}"
        )

        print(
            f"Qdrant document pts: "
            f"{document_count}"
        )

        print(
            f"Qdrant raw points:   "
            f"{raw_count}"
        )

        print(
            f"Collection:          "
            f"{config.QDRANT_COLLECTION_NAME}"
        )

        print(
            f"Corpus version:      "
            f"{summary['corpus_version']}"
        )

        print(
            f"Embedding contract:  "
            f"{EMBEDDING_CONTRACT}"
        )

    except Exception:
        # Preserve an incomplete status
        # rather than incorrectly making
        # a partial index available.
        try:
            if qdrant.collection_exists(
                config.QDRANT_COLLECTION_NAME
            ):
                failed_manifest = {
                    **manifest,

                    "embedding_contract":
                        EMBEDDING_CONTRACT,

                    "status":
                        "failed",
                }

                _write_manifest(
                    qdrant,
                    failed_manifest,
                )

        except Exception:
            pass

        raise

    finally:
        qdrant.close()
        del chroma_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy existing NeuroMath "
            "Chroma vectors to Qdrant "
            "without re-embedding."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Perform the migration. "
            "Without this flag, "
            "validation only."
        ),
    )

    parser.add_argument(
        "--replace-target",
        action="store_true",
        help=(
            "Delete the configured "
            "Qdrant collection first. "
            "Destructive."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Records per upload batch "
            "(default: 128)."
        ),
    )

    args = parser.parse_args()

    if not (
        1
        <= args.batch_size
        <= 512
    ):
        raise SystemExit(
            "--batch-size must be "
            "between 1 and 512."
        )

    summary = preflight(
        args.batch_size
    )

    print("Preflight OK")
    print(
        "Chroma collection:",
        config.CHROMA_COLLECTION_NAME,
    )
    print(
        "Chroma count:",
        summary["count"],
    )
    print(
        "Vector dimensions:",
        config.HF_EMBEDDING_DIMENSIONS,
    )
    print(
        "Corpus version:",
        summary["corpus_version"],
    )
    print(
        "Grades found:",
        summary["grades"],
    )
    print(
        "Qdrant collection:",
        config.QDRANT_COLLECTION_NAME,
    )
    print(
        "Qdrant URL:",
        config.QDRANT_URL,
    )
    print(
        "Embedding API calls: 0"
    )

    if not args.execute:
        print()
        print(
            "Dry run only. "
            "Re-run with --execute "
            "to migrate."
        )
        return

    migrate(
        batch_size=args.batch_size,
        replace_target=
            args.replace_target,
    )


if __name__ == "__main__":
    main()