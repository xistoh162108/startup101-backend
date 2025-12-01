"""
Custom exceptions for the application.
All exceptions map to standard error codes and HTTP status codes.
"""
from typing import Optional, Dict, Any

from app.schemas.error import ErrorCode


class AppException(Exception):
    """
    Base exception class for all application exceptions.
    All custom exceptions should inherit from this.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class InvalidArgumentException(AppException):
    """Raised when request arguments are invalid (422)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.INVALID_ARGUMENT, message, details)


class NotFoundException(AppException):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.NOT_FOUND, message, details)


class ConflictException(AppException):
    """Raised when there is a conflict (409)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.CONFLICT, message, details)


class ModelNotReadyException(AppException):
    """Raised when the ML model is not ready (503)."""

    def __init__(self, message: str = "Model not ready", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.MODEL_NOT_READY, message, details)


class DBUnavailableException(AppException):
    """Raised when the database is unavailable (503)."""

    def __init__(
        self, message: str = "Database unavailable", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(ErrorCode.DB_UNAVAILABLE, message, details)


class InternalException(AppException):
    """Raised for internal server errors (500)."""

    def __init__(
        self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(ErrorCode.INTERNAL, message, details)


class UnauthorizedException(AppException):
    """Raised when authentication is required (401)."""

    def __init__(self, message: str = "Unauthorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.UNAUTHORIZED, message, details)


class ForbiddenException(AppException):
    """Raised when access is forbidden (403)."""

    def __init__(self, message: str = "Forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.FORBIDDEN, message, details)


class PayloadTooLargeException(AppException):
    """Raised when request payload is too large (413)."""

    def __init__(
        self, message: str = "Payload too large", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(ErrorCode.PAYLOAD_TOO_LARGE, message, details)
