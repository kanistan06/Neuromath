"""Resumable, versioned textbook storage in Qdrant Cloud."""

from __future__ import annotations

from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

import config
from rag.embeddings import EMBEDDING_CONTRACT, build_embeddings


MANIFEST_ID = str(uuid5(NAMESPACE_URL, "neuromath:index-manifest:v1"))


def qdrant_client(*, write: bool = False):
    url = urlparse(config.QDRANT_URL)
    if url.scheme != "https" or not str(url.hostname or "").endswith(".cloud.qdrant.io"):
        raise RuntimeError("QDRANT_URL must be the HTTPS cluster endpoint from Qdrant Cloud.")
    key = config.QDRANT_WRITE_API_KEY if write else config.QDRANT_API_KEY
    if not key:
        raise RuntimeError(f"{'QDRANT_WRITE_API_KEY' if write else 'QDRANT_API_KEY'} is required.")
    return QdrantClient(url=config.QDRANT_URL, api_key=key, timeout=config.QDRANT_TIMEOUT_SECONDS)


def validate_collection(client) -> None:
    info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    params = info.config.params.vectors
    if isinstance(params, dict) or params.size != config.HF_EMBEDDING_DIMENSIONS or params.distance != models.Distance.COSINE:
        raise RuntimeError("Qdrant must use one unnamed 1024-dimensional cosine vector. Use a new collection name for a different embedding space.")


def read_manifest(client=None) -> dict:
    owned = client is None
    client = client or qdrant_client()
    try:
        if not client.collection_exists(config.QDRANT_COLLECTION_NAME):
            return {}
        points = client.retrieve(config.QDRANT_COLLECTION_NAME, [MANIFEST_ID], with_payload=True, with_vectors=False)
        return dict((points[0].payload or {}).get("index_manifest", {})) if points else {}
    finally:
        if owned:
            client.close()


def _write_manifest(client, manifest):
    client.upsert(config.QDRANT_COLLECTION_NAME, points=[
        models.PointStruct(id=MANIFEST_ID, vector={}, payload={"index_manifest": manifest}),
    ], wait=True)


def _point_id(chunk) -> str:
    return str(uuid5(NAMESPACE_URL, "neuromath:" + str(chunk.metadata["chunk_id"])))


def upload_chunks(chunks, manifest: dict) -> None:
    client = qdrant_client(write=True)
    try:
        if not client.collection_exists(config.QDRANT_COLLECTION_NAME):
            client.create_collection(config.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=config.HF_EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE, on_disk=True),
                on_disk_payload=True)
        validate_collection(client)
        previous = read_manifest(client)
        if client.count(config.QDRANT_COLLECTION_NAME, exact=True).count and not previous:
            raise RuntimeError("The existing Qdrant collection has no compatible NeuroMath manifest. Use a new QDRANT_COLLECTION_NAME.")
        if previous and (previous.get("fingerprint") != manifest["fingerprint"]
                         or previous.get("embedding_contract") != EMBEDDING_CONTRACT):
            raise RuntimeError("The corpus or embedding configuration changed. Use a new QDRANT_COLLECTION_NAME; the existing collection was preserved.")
        for field, field_schema in [("metadata.grade", models.PayloadSchemaType.INTEGER),
                                    ("metadata.corpus_version", models.PayloadSchemaType.KEYWORD)]:
            client.create_payload_index(config.QDRANT_COLLECTION_NAME, field, field_schema=field_schema, wait=True)
        manifest = {**manifest, "embedding_contract": EMBEDDING_CONTRACT, "status": "building"}
        _write_manifest(client, manifest)
        embedding = build_embeddings()
        for start in range(0, len(chunks), config.HF_EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + config.HF_EMBEDDING_BATCH_SIZE]
            identifiers = [_point_id(chunk) for chunk in batch]
            stored = {str(point.id): point for point in client.retrieve(config.QDRANT_COLLECTION_NAME, identifiers, with_payload=True, with_vectors=False)}
            pending = []
            for identifier, chunk in zip(identifiers, batch):
                metadata = {**chunk.metadata, "embedding_contract": EMBEDDING_CONTRACT}
                payload = {"page_content": chunk.page_content, "metadata": metadata}
                old = stored.get(identifier)
                if old and old.payload == payload:
                    continue
                pending.append((identifier, chunk, payload))
            if pending:
                vectors = embedding.embed_documents([chunk.page_content for _, chunk, _ in pending])
                client.upsert(config.QDRANT_COLLECTION_NAME, points=[
                    models.PointStruct(id=identifier, vector=vector, payload=payload)
                    for (identifier, _, payload), vector in zip(pending, vectors)
                ], wait=True)
        count = client.count(config.QDRANT_COLLECTION_NAME, exact=True).count - 1
        if count != len(chunks):
            raise RuntimeError("Qdrant count verification failed. The index remains unavailable until ingestion completes.")
        manifest.update(status="ready", chunk_count=count)
        _write_manifest(client, manifest)
    finally:
        client.close()
