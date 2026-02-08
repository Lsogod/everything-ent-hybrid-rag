from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import logger


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_documents([query])
        return vectors[0]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        expected_dim: int,
        use_dimensions: bool,
        batch_size: int = 10,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.expected_dim = expected_dim
        self.use_dimensions = use_dimensions
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.client = OpenAI(
            api_key=api_key or "EMPTY_API_KEY",
            base_url=self.base_url,
            timeout=timeout,
            max_retries=2,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            request_payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.use_dimensions:
                request_payload["dimensions"] = self.expected_dim

            try:
                response = self.client.embeddings.create(**request_payload)
            except Exception as exc:  # noqa: BLE001
                if self.use_dimensions:
                    logger.warning(
                        "embedding endpoint does not accept dimensions; retry without it: %s",
                        exc,
                    )
                    request_payload.pop("dimensions", None)
                    response = self.client.embeddings.create(**request_payload)
                else:
                    raise

            items = sorted(response.data, key=lambda x: x.index)
            vectors = [item.embedding for item in items]
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding response size mismatch")
            for vector in vectors:
                if len(vector) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected={self.expected_dim}, actual={len(vector)}"
                    )
            all_vectors.extend(vectors)

        if len(all_vectors) != len(texts):
            raise RuntimeError("Embedding response size mismatch")
        return all_vectors


class PseudoEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class EmbeddingFactory:
    @staticmethod
    def create() -> BaseEmbeddingClient:
        settings = get_settings()

        if settings.embedding_mode == "cloud":
            if settings.dashscope_api_key:
                return OpenAICompatibleEmbeddingClient(
                    base_url=settings.embedding_base_url_cloud,
                    model=settings.embedding_model_cloud,
                    api_key=settings.dashscope_api_key,
                    expected_dim=settings.embedding_dim,
                    use_dimensions=settings.embedding_use_dimensions,
                    batch_size=settings.embedding_batch_size,
                    timeout=settings.request_timeout_seconds,
                )
            logger.warning("DASHSCOPE_API_KEY is empty, fallback to pseudo embedding")
            return PseudoEmbeddingClient(settings.embedding_dim)

        return OpenAICompatibleEmbeddingClient(
            base_url=settings.embedding_base_url_local,
            model=settings.embedding_model_local,
            api_key=settings.local_embedding_api_key,
            expected_dim=settings.embedding_dim,
            use_dimensions=False,
            batch_size=settings.embedding_batch_size,
            timeout=settings.request_timeout_seconds,
        )
