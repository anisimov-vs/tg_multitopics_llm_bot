import sys
import logging
import logging.handlers
import queue
import atexit
from .settings import Config


def setup_logging() -> logging.Logger:
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    listener = logging.handlers.QueueListener(log_queue, console_handler)
    listener.start()

    queue_handler = logging.handlers.QueueHandler(log_queue)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)

    logger = logging.getLogger("llm_bot")
    logger.setLevel(log_level)

    atexit.register(listener.stop)

    return logger


logger = setup_logging()
