"""
Structured JSON logging configuration.
All logs include request-id and are formatted as JSON for easy parsing.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Outputs logs in JSON format with standard fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request_id if available from extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "request_id",  # Already handled above
            ]:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data, default=str)


def setup_logging() -> logging.Logger:
    """
    Configure and return the application logger.
    Uses JSON formatting for structured logs.
    """
    # Create logger
    logger = logging.getLogger("review_classifier")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Set JSON formatter
    json_formatter = JSONFormatter()
    console_handler.setFormatter(json_formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Global logger instance
logger = setup_logging()


# Helper functions for common logging patterns
def log_error(message: str, error: Exception = None, **kwargs):
    """Log an error with optional exception details."""
    extra = kwargs.copy()
    if error:
        extra["error_type"] = type(error).__name__
        extra["error_message"] = str(error)

    logger.error(message, extra=extra, exc_info=error is not None)


def log_info(message: str, **kwargs):
    """Log an info message with extra fields."""
    logger.info(message, extra=kwargs)


def log_warning(message: str, **kwargs):
    """Log a warning message with extra fields."""
    logger.warning(message, extra=kwargs)


def log_debug(message: str, **kwargs):
    """Log a debug message with extra fields."""
    logger.debug(message, extra=kwargs)
