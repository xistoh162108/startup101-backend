"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.core.config import settings
from app.core.middleware import RequestIdMiddleware, get_request_id
from app.core.logging import logger
from app.core.exceptions import AppException
from app.schemas.error import ErrorResponse, ErrorDetail, ErrorCode, ERROR_CODE_TO_HTTP_STATUS
from app.db.database import init_db, close_db

# Import routers
from app.api import health, classification, feedback, admin, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up Review Classification API")

    # Initialize database tables (in production, use Alembic migrations)
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        # Continue anyway - the app can still classify without DB (with persist=False)

    # TODO: Load initial model if available
    # This will be implemented in Step 5

    yield

    # Shutdown
    logger.info("Shutting down Review Classification API")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="ML-powered review advertisement classification API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(RequestIdMiddleware)

# Configure CORS
# Allow requests from frontend (localhost:3000 for dev, reviewtrust.siwon.site for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://reviewtrust.siwon.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers including X-Admin-Key
)


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """
    Handle all custom application exceptions.

    Returns standardized error response with proper HTTP status code.
    """
    request_id = get_request_id()

    logger.warning(
        f"Application exception: {exc.error_code.value}",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code.value,
            "message": exc.message,
            "details": exc.details,
        },
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=exc.error_code,
            message=exc.message,
            details=exc.details if exc.details else None,
        ),
    )

    status_code = ERROR_CODE_TO_HTTP_STATUS.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(by_alias=True, exclude_none=True),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors (422).

    Returns standardized error response.
    """
    request_id = get_request_id()

    # Extract validation error details
    errors = exc.errors()
    error_details = {
        "validation_errors": [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in errors
        ]
    }

    logger.warning(
        "Validation error",
        extra={"request_id": request_id, "errors": error_details},
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=ErrorCode.INVALID_ARGUMENT,
            message="Request validation failed",
            details=error_details,
        ),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(by_alias=True, exclude_none=True),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions.

    Returns standardized error response with 500 status.
    This ensures the server never crashes.
    """
    request_id = get_request_id()

    logger.error(
        f"Uncaught exception: {str(exc)}",
        extra={"request_id": request_id, "exception_type": type(exc).__name__},
        exc_info=True,
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=ErrorCode.INTERNAL,
            message="Internal server error",
            details={"error": str(exc)} if settings.DEBUG else None,
        ),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(by_alias=True, exclude_none=True),
    )


# Include routers
app.include_router(health.router)
app.include_router(classification.router)
app.include_router(feedback.router)
app.include_router(admin.router)
app.include_router(model.router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
