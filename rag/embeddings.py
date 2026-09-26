"""Validated BGE-M3 sentence embeddings through Hugging Face Inference Providers."""

from __future__ import annotations

import math
import time

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

import config


EMBEDDING_CONTRACT = "hf-inference-bge-m3-dense-normalized-v1"
_TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class EmbeddingServiceError(RuntimeError):
    code = "hf_embeddings_unavailable"
    status_code = 503


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, *, api_key, model_id="BAAI/bge-m3", dimensions=1024,
                 batch_size=16, timeout_seconds=120, max_retries=3,
                 retry_base_seconds=2.0, bill_to=None, client=None):
        if not api_key:
            raise EmbeddingServiceError("HF_API_KEY is required for Hugging Face embeddings.")
        if model_id != config.SELECTED_EMBEDDING_MODEL:
            raise EmbeddingServiceError("The embedding model must remain BAAI/bge-m3.")
        self.model_id = model_id
        self.dimensions = dimensions
        self.batch_size = max(1, min(32, batch_size))
        self.max_retries = max(1, min(5, max_retries))
        self.retry_base_seconds = retry_base_seconds
        self.client = client or InferenceClient(provider="hf-inference", api_key=api_key,
                                                timeout=timeout_seconds, bill_to=bill_to or None)

    def _vectors(self, payload, count):
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if count == 1 and isinstance(payload, list) and payload and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in payload):
            payload = [payload]
        if not isinstance(payload, list) or len(payload) != count:
            raise EmbeddingServiceError("Hugging Face returned an unexpected sentence-vector count.")
        vectors = []
        for row in payload:
            if not isinstance(row, list) or len(row) != self.dimensions:
                raise EmbeddingServiceError(f"Expected one {self.dimensions}-dimensional sentence vector per text; token vectors are not accepted.")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in row):
                raise EmbeddingServiceError("Hugging Face returned a malformed or non-finite embedding.")
            norm = math.hypot(*row)
            if not norm or not math.isfinite(norm):
                raise EmbeddingServiceError("Hugging Face returned an invalid zero or overflowing embedding.")
            vectors.append([value / norm for value in row])
        return vectors

    def _request(self, texts):
        for attempt in range(self.max_retries):
            try:
                result = self.client.feature_extraction(texts, model=self.model_id,
                                                        normalize=True, truncate=False)
                return self._vectors(result, len(texts))
            except EmbeddingServiceError:
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is None:
                    status = getattr(exc, "status_code", None)
                if status in _TRANSIENT and attempt + 1 < self.max_retries:
                    time.sleep(min(20.0, self.retry_base_seconds * 2**attempt))
                    continue
                hints = {
                    401: "Check HF_API_KEY.",
                    402: "Check Hugging Face inference credits and billing; RunPod credits do not pay for embeddings.",
                    403: "Enable the token's Inference Providers permission and check organization access.",
                    404: "Check BAAI/bge-m3 availability with the HF Inference provider.",
                    413: "The embedding input is too large. Check textbook chunk sizes.",
                    429: "The Hugging Face embedding rate limit was reached; retry later.",
                }
                raise EmbeddingServiceError(
                    "Hugging Face embeddings failed. " + hints.get(status, "Check provider availability and connectivity, then retry.")
                ) from exc

    def embed_documents(self, texts):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty text.")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[start:start + self.batch_size]))
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def build_embeddings():
    return HuggingFaceEmbeddings(
        api_key=config.HF_API_KEY, model_id=config.HF_EMBEDDING_MODEL,
        dimensions=config.HF_EMBEDDING_DIMENSIONS, batch_size=config.HF_EMBEDDING_BATCH_SIZE,
        timeout_seconds=config.HF_EMBEDDING_TIMEOUT_SECONDS, max_retries=config.HF_EMBEDDING_MAX_RETRIES,
        retry_base_seconds=config.HF_EMBEDDING_RETRY_BASE_SECONDS, bill_to=config.HF_BILL_TO,
    )
