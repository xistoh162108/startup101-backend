"""
Custom middleware for the FastAPI application.
"""
import uuid
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextvars import ContextVar

from app.core.logging import logger

# Context variable to store request ID across async calls
request_id_var: ContextVar[str] = ContextVar("request_id", default=None)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and attach a unique request ID to each request.

    The request ID is:
    1. Generated as a UUID for each request
    2. Stored in context variable for access in other parts of the app
    3. Added to response headers (X-Request-Id)
    4. Included in all log messages
    5. Included in all API responses (requestId field)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Store in context variable for access throughout the request lifecycle
        request_id_var.set(request_id)

        # Attach to request state for easy access
        request.state.request_id = request_id

        # Record start time for latency measurement
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
            },
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            logger.error(
                f"Request failed with exception: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                },
                exc_info=True,
            )
            raise

        # Calculate latency
        latency = time.time() - start_time

        # Add request ID to response headers
        response.headers["X-Request-Id"] = request_id

        # Log completed request
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "latency_seconds": round(latency, 4),
            },
        )

        return response


def get_request_id() -> str:
    """
    Get the current request ID from context.
    Returns a default UUID if not in a request context.
    """
    request_id = request_id_var.get()
    if request_id is None:
        # Fallback if called outside request context
        return str(uuid.uuid4())
    return request_id
