import logging
import logging.handlers
from pathlib import Path


def test_cron_logging_creates_matching_debug_file(monkeypatch, tmp_path):
    import hot_projects.cron_scheduled_update as cron

    monkeypatch.setattr(cron, "LOG_DIR", str(tmp_path))

    log_path = Path(cron.setup_logging())
    try:
        debug_path = tmp_path / "debug" / f"{log_path.stem}.debug.log"
        file_handlers = [
            handler
            for handler in logging.getLogger().handlers
            if hasattr(handler, "baseFilename")
        ]
        handlers_by_path = {
            Path(handler.baseFilename): handler for handler in file_handlers
        }

        assert log_path in handlers_by_path
        assert debug_path in handlers_by_path
        assert handlers_by_path[log_path].level == logging.INFO
        assert handlers_by_path[debug_path].level == logging.DEBUG
        assert not isinstance(handlers_by_path[log_path], logging.handlers.RotatingFileHandler)
        assert not isinstance(handlers_by_path[debug_path], logging.handlers.RotatingFileHandler)
        assert not hasattr(handlers_by_path[log_path], "maxBytes")
        assert not hasattr(handlers_by_path[debug_path], "maxBytes")
    finally:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
