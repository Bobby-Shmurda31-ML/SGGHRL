import logging

logger = logging.getLogger("sgghrl")
logger.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO, fmt: str = None):
    """Настроить логирование библиотеки.

    Args:
        level: уровень логирования (logging.DEBUG, INFO, ...).
        fmt: формат сообщений (None — дефолтный).
    """
    if fmt is None:
        fmt = "[%(name)s %(levelname)s] %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.handlers = [handler]
    logger.setLevel(level)