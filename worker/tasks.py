from __future__ import annotations

from app.core.logging import logger
from app.infra.queue import celery_app
from app.services.indexer import IndexerService


@celery_app.task(name="worker.tasks.index_file")
def index_file(file_path: str) -> dict:
    service = IndexerService()
    service.store.ensure_index()
    result = service.index_file(file_path)
    logger.info("index_file result=%s", result)
    return result


@celery_app.task(name="worker.tasks.delete_file")
def delete_file(file_path: str, delete_source: bool = False) -> dict:
    service = IndexerService()
    result = service.delete_file(file_path, delete_source=delete_source)
    logger.info("delete_file result=%s", result)
    return result
