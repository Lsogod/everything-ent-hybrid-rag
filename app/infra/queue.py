from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ent_hybrid",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
