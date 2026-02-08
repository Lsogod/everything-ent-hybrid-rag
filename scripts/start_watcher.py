from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.watcher import run_watcher


if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    run_watcher(settings.knowledge_root)
