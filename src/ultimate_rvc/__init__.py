"""The Ultimate RVC project."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ultimate_rvc.common import BASE_DIR

logger = logging.getLogger()

URVC_NO_LOGGING = os.getenv("URVC_NO_LOGGING", "0") == "1"
URVC_LOGS_DIR = Path(os.getenv("URVC_LOGS_DIR") or BASE_DIR / "logs")
URVC_CONSOLE_LOG_LEVEL = os.getenv("URVC_CONSOLE_LOG_LEVEL", "ERROR")
URVC_FILE_LOG_LEVEL = os.getenv("URVC_FILE_LOG_LEVEL", "INFO")

if URVC_NO_LOGGING:
    logging.basicConfig(handlers=[logging.NullHandler()])

else:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(URVC_CONSOLE_LOG_LEVEL)

    URVC_LOGS_DIR.mkdir(exist_ok=True, parents=True)
    file_handler = RotatingFileHandler(
        URVC_LOGS_DIR / "ultimate_rvc.log",
        mode="a",
        maxBytes=1024 * 1024 * 5,
        backupCount=1,
        encoding="utf-8",
    )
    file_handler.setLevel(URVC_FILE_LOG_LEVEL)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        style = "%",
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
    )

