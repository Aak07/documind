"""
Structured JSON logging using structlog.

- Every log entry is a structured object (JSON in production)
- Easy to parse, filter, and send to logging systems (ELK, Datadog, etc.)
- Supports context propagation (e.g., request_id in FastAPI)

Switches automatically between:
- Pretty console logs (dev)
- JSON logs (production)
"""

import structlog
import logging
import sys
import os

def setup_logger(level: str = "INFO") -> structlog.BoundLogger:
    """Configure structured logging for the application.
        Args:
            level (str): Log level as string (e.g., "INFO", "DEBUG")
        Returns:
            structlog.BoundLogger: Configured logger instance
    """

    # Convert string level → logging constant safely
    # (handles invalid inputs gracefully)
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard Python logging
    # Required for compatibility with FastAPI, Uvicorn, Docker, etc.
    logging.basicConfig(
        format="%(message)s",     # structlog handles formatting
        stream=sys.stdout,        # Send logs to stdout (Docker-friendly)
        level=log_level,
    )

    # Environment-based renderer:
    # - Dev → human-readable logs
    # - Prod → JSON logs for machines
    is_prod = os.getenv("ENV", "dev") == "prod"

    renderer = (
        structlog.processors.JSONRenderer()
        if is_prod
        else structlog.dev.ConsoleRenderer()
    )

    # Configure structlog pipeline
    structlog.configure(
        processors=[
            # Merge context variables (useful for request_id, user_id, etc.)
            structlog.contextvars.merge_contextvars,

            # Add log level to each entry
            structlog.processors.add_log_level,

            # Add timestamp (ISO format)
            structlog.processors.TimeStamper(fmt="iso"),

            # Format exceptions as structured data (very important)
            structlog.processors.format_exc_info,

            # Final renderer (Console or JSON)
            renderer,
        ],

        # Filters logs below the configured level
        wrapper_class=structlog.make_filtering_bound_logger(log_level),

        # Use standard dict for context
        context_class=dict,

        # Use standard logging (NOT print) → production-safe
        logger_factory=structlog.stdlib.LoggerFactory(),

        # Cache logger for performance
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


# App-wide logger instance
# (Safe for now; can later move to FastAPI entrypoint if needed)
logger = setup_logger()