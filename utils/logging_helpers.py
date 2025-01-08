"""
Helper functions to deal with how to log intermediate outputs
"""

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


LOG_DIRECTORY = os.getenv("LOG_DIRECTORY", "./logs")
log_file = f"{LOG_DIRECTORY}/{datetime.now().strftime('%Y-%m-%d')}.log"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "filename": "%(filename)s", "funcName": "%(funcName)s", "lineno": %(lineno)d, "message": "%(message)s"}'
)

file_handler = TimedRotatingFileHandler(
    filename=log_file, when="midnight", backupCount=7
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


