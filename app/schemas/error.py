"""
Standard error response schemas.
All API errors follow this unified format.
"""
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class ErrorCode(str, Enum):
    """Standard error codes used throughout the API."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"  # 422
    NOT_FOUND = "NOT_FOUND"  # 404
    CONFLICT = "CONFLICT"  # 409
    MODEL_NOT_READY = "MODEL_NOT_READY"  # 503
    DB_UNAVAILABLE = "DB_UNAVAILABLE"  # 503
    INTERNAL = "INTERNAL"  # 500
    UNAUTHORIZED = "UNAUTHORIZED"  # 401
    FORBIDDEN = "FORBIDDEN"  # 403
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"  # 413


class ErrorDetail(BaseModel):
    """
    Error detail object.
    Contains the error code, human-readable message, and optional additional details.
    """

    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    model_config = ConfigDict(use_enum_values=True)


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    All API errors return this structure.

    Example:
    {
        "requestId": "abc-123-def",
        "error": {
            "code": "INVALID_ARGUMENT",
            "message": "content is required",
            "details": {"field": "content"}
        }
    }
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    error: ErrorDetail = Field(..., description="Error details")

    model_config = ConfigDict(populate_by_name=True)


# HTTP Status Code mapping for error codes
ERROR_CODE_TO_HTTP_STATUS = {
    ErrorCode.INVALID_ARGUMENT: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.DB_UNAVAILABLE: 503,
    ErrorCode.INTERNAL: 500,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
}
