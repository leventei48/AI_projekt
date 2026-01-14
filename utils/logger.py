# utils/logger.py
import logging
import sys

def init_logging(level=logging.DEBUG):      # <-- DEBUG-ra állítjuk
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str = None):
    return logging.getLogger(name)
