from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

import config
from rag.embeddings import EmbeddingServiceError, HuggingFaceEmbeddings


class FakeClient:
    def __init__(self, payload=None):
        self.calls = []
        self.payload = payload

    def feature_extraction(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return self.payload if self.payload is not None else [[3.0, 4.0, 0.0] for _ in texts]


def make_embeddings(client, **kwargs):
    return HuggingFaceEmbeddings(api_key="hf_test_only", dimensions=3, batch_size=2,
                                client=client, **kwargs)


def test_hf_embeddings_batch_normalize_and_use_sentence_task():
    client = FakeClient()
    embeddings = make_embeddings(client)
    assert isinstance(embeddings, Embeddings)
    vectors = embeddings.embed_documents(["one", "two", "three"])
    assert embeddings.embed_query("four") == pytest.approx([0.6, 0.8, 0.0])
    assert all(vector == pytest.approx([0.6, 0.8, 0]) for vector in vectors)
    assert [len(call[0]) for call in client.calls] == [2, 1, 1]
    assert all(call[1] == {"model": "BAAI/bge-m3", "normalize": True, "truncate": False} for call in client.calls)


@pytest.mark.parametrize("payload", [[], [[1, 2]], [[[1, 0, 0], [0, 1, 0]]],
                                     [[0, 0, 0]], [[float('nan'), 1, 0]],
                                     [[float('inf'), 1, 0]], [[True, 0, 1]], [["1", 0, 1]]])
def test_malformed_embeddings_are_rejected_without_mean_pooling(payload):
    with pytest.raises(EmbeddingServiceError):
        make_embeddings(FakeClient(payload)).embed_query("query")


def test_vector_order_is_preserved_and_numpy_arrays_are_supported():
    import numpy as np
    client = FakeClient(np.array([[1., 0., 0.], [0., 1., 0.]]))
    assert make_embeddings(client).embed_documents(["first", "second"]) == [[1., 0., 0.], [0., 1., 0.]]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_permanent_hf_failure_does_not_retry_or_block_runpod(status):
    from rag.inference_errors import _blocked_until, access_failure
    class Client(FakeClient):
        def feature_extraction(self, texts, **kwargs):
            self.calls.append(texts)
            error = RuntimeError("provider response containing hf_test_only")
            error.response = SimpleNamespace(status_code=status)
            raise error
    client = Client()
    with pytest.raises(EmbeddingServiceError) as caught:
        make_embeddings(client).embed_query("text")
    assert len(client.calls) == 1
    assert "hf_test_only" not in str(caught.value)
    assert not _blocked_until
    assert access_failure(caught.value) is caught.value
    if status == 402:
        assert "Hugging Face inference credits" in str(caught.value)


def test_transient_hf_failure_retries_only_to_bound(monkeypatch):
    import rag.embeddings as module
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    class Client(FakeClient):
        def feature_extraction(self, texts, **kwargs):
            self.calls.append(texts)
            error = RuntimeError("unavailable")
            error.response = SimpleNamespace(status_code=503)
            raise error
    client = Client()
    with pytest.raises(EmbeddingServiceError):
        make_embeddings(client, max_retries=3).embed_query("text")
    assert len(client.calls) == 3 and len(sleeps) == 2


def test_sdk_authentication_and_hf_billing_are_separate_from_runpod(monkeypatch):
    import rag.embeddings as module
    arguments = {}
    def factory(**kwargs):
        arguments.update(kwargs)
        return FakeClient()
    monkeypatch.setattr(module, "InferenceClient", factory)
    HuggingFaceEmbeddings(api_key="hf_test_only", bill_to="test-org", timeout_seconds=90)
    assert arguments == {"provider": "hf-inference", "api_key": "hf_test_only", "bill_to": "test-org", "timeout": 90}


def test_empty_input_does_not_create_a_request():
    client = FakeClient()
    assert make_embeddings(client).embed_documents([]) == []
    with pytest.raises(ValueError):
        make_embeddings(client).embed_query("  ")
    assert not client.calls
