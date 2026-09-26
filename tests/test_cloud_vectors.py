from uuid import UUID

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

import config
import rag.cloud_vectors as cloud
from rag.embeddings import EMBEDDING_CONTRACT
from rag.ingest import vector_index_status


@pytest.fixture
def cloud_store(monkeypatch):
    client = QdrantClient(':memory:')
    close = client.close
    monkeypatch.setattr(client, 'close', lambda: None)
    monkeypatch.setattr(cloud, 'qdrant_client', lambda **_kwargs: client)
    monkeypatch.setattr(config, 'VECTOR_STORE_PROVIDER', 'qdrant')
    monkeypatch.setattr(config, 'QDRANT_COLLECTION_NAME', 'hf-test-index')
    monkeypatch.setattr(config, 'HF_EMBEDDING_BATCH_SIZE', 2)
    monkeypatch.setattr(config, 'CORPUS_VERSION', '')
    yield client
    close()


def _chunks():
    return [Document(page_content=f'Textbook mathematics passage number {index}.',
                     metadata={'chunk_id': str(index) * 64, 'grade': 10 if index < 3 else 11,
                               'corpus_version': 'fixture-v1'}) for index in (1, 2, 3)]


def test_interrupted_ingestion_resumes_only_missing_vectors(cloud_store, monkeypatch):
    calls = []
    fail = [True]
    class Embeddings:
        def embed_documents(self, texts):
            calls.append(list(texts))
            if len(calls) == 2 and fail[0]:
                raise RuntimeError('Simulated HF interruption')
            return [[1.] + [0.] * 1023 for _ in texts]
    monkeypatch.setattr(cloud, 'build_embeddings', Embeddings)
    chunks = _chunks()
    with pytest.raises(RuntimeError, match='interruption'):
        cloud.upload_chunks(chunks, {'fingerprint': 'test-v1', 'corpus_version': 'fixture-v1'})
    assert cloud.read_manifest(cloud_store)['status'] == 'building'
    assert not vector_index_status()['ready']
    fail[0] = False
    cloud.upload_chunks(chunks, {'fingerprint': 'test-v1', 'corpus_version': 'fixture-v1'})
    assert [len(batch) for batch in calls] == [2, 1, 1]
    manifest = cloud.read_manifest(cloud_store)
    assert manifest['status'] == 'ready' and manifest['chunk_count'] == 3
    assert manifest['embedding_contract'] == EMBEDDING_CONTRACT
    assert vector_index_status() == {'ready': True, 'count': 3, 'missing_grades': []}
    cloud.upload_chunks(chunks, {'fingerprint': 'test-v1', 'corpus_version': 'fixture-v1'})
    assert len(calls) == 3
    points = cloud_store.query_points(config.QDRANT_COLLECTION_NAME, query=[1.] + [0.] * 1023).points
    assert len(points) == 3
    assert all(str(UUID(point.id)) == point.id for point in points)
    assert cloud.MANIFEST_ID not in {point.id for point in points}


def test_incompatible_existing_collection_is_preserved(cloud_store, monkeypatch):
    cloud_store.create_collection(config.QDRANT_COLLECTION_NAME,
                                 vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    cloud_store.upsert(config.QDRANT_COLLECTION_NAME, [models.PointStruct(id=1, vector=[1.] + [0.] * 1023, payload={'old': True})])
    monkeypatch.setattr(cloud, 'build_embeddings', lambda: pytest.fail('Unnecessary embedding request'))
    with pytest.raises(RuntimeError, match='new QDRANT_COLLECTION_NAME'):
        cloud.upload_chunks(_chunks(), {'fingerprint': 'new'})
    assert cloud_store.retrieve(config.QDRANT_COLLECTION_NAME, [1])[0].payload == {'old': True}
    assert cloud_store.count(config.QDRANT_COLLECTION_NAME).count == 1


def test_qdrant_reader_does_not_send_a_dummy_embedding_request(cloud_store, monkeypatch):
    import rag.retriever as retriever
    from langchain_core.embeddings import Embeddings
    class Stub(Embeddings):
        def embed_documents(self, texts):
            return [[1.] + [0.] * 1023 for _ in texts]
        def embed_query(self, text):
            pytest.fail('Reader setup must not call Hugging Face')
    monkeypatch.setattr(cloud, 'build_embeddings', Stub)
    cloud.upload_chunks(_chunks(), {'fingerprint': 'v1'})
    monkeypatch.setattr(config, 'HF_API_KEY', 'hf_test_only')
    monkeypatch.setattr(retriever, 'build_embeddings', Stub)
    store = retriever.get_vector_store()
    assert store.client is cloud_store
    with pytest.raises(RuntimeError, match='configuration changed'):
        cloud.upload_chunks(_chunks(), {'fingerprint': 'v2'})
    assert cloud.read_manifest(cloud_store)['status'] == 'ready'


def test_cloud_client_uses_separate_read_and_write_credentials(monkeypatch):
    monkeypatch.setattr(config, 'QDRANT_URL', 'https://unit.eu-west-1-0.aws.cloud.qdrant.io:6333')
    monkeypatch.setattr(config, 'QDRANT_API_KEY', 'read-key')
    monkeypatch.setattr(config, 'QDRANT_WRITE_API_KEY', 'write-key')
    seen = []
    monkeypatch.setattr(cloud, 'QdrantClient', lambda **kwargs: seen.append(kwargs))
    cloud.qdrant_client()
    cloud.qdrant_client(write=True)
    assert [value['api_key'] for value in seen] == ['read-key', 'write-key']
    monkeypatch.setattr(config, 'QDRANT_WRITE_API_KEY', '')
    with pytest.raises(RuntimeError, match='QDRANT_WRITE_API_KEY'):
        cloud.qdrant_client(write=True)
