from __future__ import annotations

from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.core.config import get_settings
from app.core.logging import logger
from app.infra.queue import celery_app


class KnowledgeEventHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.settings = get_settings()

    def on_created(self, event: FileSystemEvent) -> None:
        self._submit("worker.tasks.index_file", event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._submit("worker.tasks.index_file", event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._submit("worker.tasks.delete_file", event)

    def _submit(self, task_name: str, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        file_path = str(Path(event.src_path).resolve())
        ext = Path(file_path).suffix.lower()
        if ext not in self.settings.allowed_extensions:
            return

        result = celery_app.send_task(task_name, args=[file_path])
        logger.info("watcher queued task=%s file=%s task_id=%s", task_name, file_path, result.id)


def run_watcher(path: str) -> None:
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"watch path not found: {path_obj}")

    observer = Observer()
    handler = KnowledgeEventHandler()
    observer.schedule(handler, str(path_obj), recursive=True)
    observer.start()
    logger.info("watcher started on %s", path_obj)

    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
