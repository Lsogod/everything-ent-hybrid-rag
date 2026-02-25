from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

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


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        model: str,
        expected_dim: int,
        batch_size: int = 10,
        device: str | None = None,
        max_seq_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "missing dependency: sentence-transformers. "
                "install it in runtime image to use local sentence_transformers backend"
            ) from exc

        self.model_name = model
        self.expected_dim = expected_dim
        self.batch_size = max(1, batch_size)

        model_kwargs: dict[str, Any] = {}
        if device:
            model_kwargs["device"] = device

        # Compatibility patch for some custom model code that still calls
        # DynamicCache.get_usable_length with newer transformers releases.
        try:
            from transformers.cache_utils import DynamicCache

            if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
                def _get_usable_length(self, seq_length: int, layer_idx: int | None = None) -> int:  # noqa: ANN001
                    _ = seq_length
                    try:
                        return int(self.get_seq_length(layer_idx))
                    except TypeError:
                        return int(self.get_seq_length())

                DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

        # Qwen/GTE local models may require custom model code on HF.
        try:
            self.model = SentenceTransformer(model, trust_remote_code=True, **model_kwargs)
        except TypeError:
            self.model = SentenceTransformer(model, **model_kwargs)
        self.model.max_seq_length = max(8, int(max_seq_length))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if getattr(vectors, "ndim", 0) == 1:
            vectors = vectors.reshape(1, -1)
        output = vectors.tolist()
        if len(output) != len(texts):
            raise RuntimeError("Embedding response size mismatch")
        for vector in output:
            if len(vector) != self.expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected={self.expected_dim}, actual={len(vector)}"
                )
        return output


class EmbeddingFactory:
    _cached_client: BaseEmbeddingClient | None = None
    _cached_signature: tuple[Any, ...] | None = None

    @classmethod
    def create(cls) -> BaseEmbeddingClient:
        settings = get_settings()
        signature = (
            settings.embedding_mode,
            settings.embedding_backend_local,
            settings.embedding_dim,
            settings.embedding_model_cloud,
            settings.embedding_model_local,
            settings.embedding_base_url_cloud,
            settings.embedding_base_url_local,
            settings.embedding_device_local,
            settings.embedding_max_seq_len_local,
            settings.embedding_batch_size,
            settings.embedding_use_dimensions,
            settings.request_timeout_seconds,
            bool(settings.dashscope_api_key),
            bool(settings.local_embedding_api_key),
        )
        if cls._cached_client is not None and cls._cached_signature == signature:
            return cls._cached_client

        if settings.embedding_mode == "cloud":
            if settings.dashscope_api_key:
                client = OpenAICompatibleEmbeddingClient(
                    base_url=settings.embedding_base_url_cloud,
                    model=settings.embedding_model_cloud,
                    api_key=settings.dashscope_api_key,
                    expected_dim=settings.embedding_dim,
                    use_dimensions=settings.embedding_use_dimensions,
                    batch_size=settings.embedding_batch_size,
                    timeout=settings.request_timeout_seconds,
                )
                cls._cached_client = client
                cls._cached_signature = signature
                return client
            logger.warning("DASHSCOPE_API_KEY is empty, fallback to pseudo embedding")
            client = PseudoEmbeddingClient(settings.embedding_dim)
            cls._cached_client = client
            cls._cached_signature = signature
            return client

        if settings.embedding_backend_local == "sentence_transformers":
            client = SentenceTransformerEmbeddingClient(
                model=settings.embedding_model_local,
                expected_dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
                device=settings.embedding_device_local,
                max_seq_length=settings.embedding_max_seq_len_local,
            )
            cls._cached_client = client
            cls._cached_signature = signature
            return client

        client = OpenAICompatibleEmbeddingClient(
            base_url=settings.embedding_base_url_local,
            model=settings.embedding_model_local,
            api_key=settings.local_embedding_api_key,
            expected_dim=settings.embedding_dim,
            use_dimensions=False,
            batch_size=settings.embedding_batch_size,
            timeout=settings.request_timeout_seconds,
        )
        cls._cached_client = client
        cls._cached_signature = signature
        return client
