import logging
import os
import structlog


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        cache_logger_on_first_use=True,
    )
