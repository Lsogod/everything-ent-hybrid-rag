from __future__ import annotations

import time

from fastapi import FastAPI
from sqlalchemy.exc import OperationalError

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import configure_logging, logger
from app.infra.db import engine
from app.infra.es import ElasticsearchStore
from app.models import Base
from app.services.rbac import RBACService

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(router, prefix=settings.api_prefix)


@app.on_event("startup")
def startup_event() -> None:
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        # SQLite with multi-worker startup can race on DDL. Ignore benign "already exists".
        if "already exists" not in str(exc).lower():
            raise
        logger.warning("ignore sqlite startup ddl race: %s", exc)

    rbac = RBACService()
    rbac.bootstrap_defaults()

    store = ElasticsearchStore()
    last_error: Exception | None = None
    for _ in range(20):
        try:
            store.ensure_index()
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)

    if last_error:
        logger.warning("elasticsearch init skipped on startup: %s", last_error)

    logger.info(
        "startup completed app=%s version=%s acl_enabled=%s",
        settings.app_name,
        settings.app_version,
        settings.acl_enabled,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
