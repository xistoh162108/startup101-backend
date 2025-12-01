"""
Pydantic schemas package.
Exports all request/response models.
"""
from app.schemas.review import ReviewInput, ReviewOutput
from app.schemas.classification import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyOptions,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchClassifyResultItem,
)
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.schemas.label import AdminLabelRequest, AdminLabelResponse
from app.schemas.model import ModelInfoResponse, TrainModelRequest, TrainModelResponse
from app.schemas.error import ErrorCode, ErrorDetail, ErrorResponse, ERROR_CODE_TO_HTTP_STATUS

__all__ = [
    # Review
    "ReviewInput",
    "ReviewOutput",
    # Classification
    "ClassifyRequest",
    "ClassifyResponse",
    "ClassifyOptions",
    "BatchClassifyRequest",
    "BatchClassifyResponse",
    "BatchClassifyResultItem",
    # Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Label
    "AdminLabelRequest",
    "AdminLabelResponse",
    # Model
    "ModelInfoResponse",
    "TrainModelRequest",
    "TrainModelResponse",
    # Error
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "ERROR_CODE_TO_HTTP_STATUS",
]
