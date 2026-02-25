from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Everything-Ent-Hybrid"
    app_version: str = "1.1.0"
    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    api_key: str | None = None

    embedding_mode: Literal["cloud", "local"] = "cloud"
    embedding_backend_local: Literal["openai_compatible", "sentence_transformers"] = "openai_compatible"
    embedding_dim: int = 1024
    embedding_model_cloud: str = "text-embedding-v4"
    embedding_model_local: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    embedding_base_url_cloud: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_base_url_local: str = "http://host.docker.internal:8001/v1"
    embedding_device_local: str | None = None
    embedding_max_seq_len_local: int = 512
    embedding_use_dimensions: bool = True
    embedding_batch_size: int = 10
    dashscope_api_key: str | None = None
    local_embedding_api_key: str | None = None

    llm_provider: Literal["deepseek", "qwen", "custom"] = "qwen"
    llm_model: str = "qwen-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str | None = None

    es_url: str = "http://localhost:9200"
    es_index: str = "ent_kb_chunks"
    es_text_analyzer: str = "ik_max_word"
    es_search_analyzer: str | None = "ik_smart"
    es_username: str | None = None
    es_password: str | None = None
    es_verify_certs: bool = False
    es_num_candidates: int = 100
    index_with_vectors: bool = True

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    database_url: str = "sqlite:///./ent_hybrid.db"
    acl_enabled: bool = True
    acl_bootstrap: bool = True
    acl_default_admin_user: str = "admin"

    retrieval_top_k_keyword: int = 10
    retrieval_top_k_vector: int = 10
    retrieval_top_k_final: int = 5
    rrf_k: int = 60

    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    watch_extensions: str = ".pdf,.docx,.md,.txt,.py,.java,.js,.ts,.go,.sql"

    knowledge_root: str = "/data/knowledge"
    allow_external_paths: bool = False
    request_timeout_seconds: float = 30.0

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def allowed_extensions(self) -> set[str]:
        return {item.strip().lower() for item in self.watch_extensions.split(",") if item.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
