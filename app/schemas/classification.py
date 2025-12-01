"""
Pydantic schemas for Classification-related requests and responses.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.review import ReviewInput


class ClassifyOptions(BaseModel):
    """Options for classification requests."""

    persist: bool = Field(
        True, description="Whether to persist review and classification to database"
    )
    include_reasons: bool = Field(
        True, alias="includeReasons", description="Whether to include reason codes in response"
    )

    model_config = ConfigDict(populate_by_name=True)


class ClassifyRequest(BaseModel):
    """
    Request schema for single review classification.
    POST /api/reviews/classify
    """

    review: ReviewInput = Field(..., description="Review to classify")
    options: Optional[ClassifyOptions] = Field(
        default_factory=ClassifyOptions, description="Classification options"
    )


class ClassifyResponse(BaseModel):
    """
    Response schema for single review classification.
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    review_id: str = Field(..., alias="reviewId", description="Review ID (external or internal)")
    is_ad_like: bool = Field(..., alias="isAdLike", description="Whether review is ad-like")
    ad_score: float = Field(..., alias="adScore", ge=0.0, le=1.0, description="Ad probability score")
    threshold: float = Field(..., description="Threshold used for classification")
    model_version: str = Field(..., alias="modelVersion", description="Model version used")
    reasons: Optional[List[str]] = Field(None, description="Reason codes explaining classification")

    model_config = ConfigDict(populate_by_name=True)


class BatchClassifyRequest(BaseModel):
    """
    Request schema for batch review classification.
    POST /api/reviews/batch-classify
    """

    reviews: List[ReviewInput] = Field(
        ..., min_length=1, max_length=100, description="Reviews to classify (max 100)"
    )
    options: Optional[ClassifyOptions] = Field(
        default_factory=ClassifyOptions, description="Classification options"
    )


class BatchClassifyResultItem(BaseModel):
    """Individual result item in batch classification response."""

    review_id: str = Field(..., alias="reviewId", description="Review ID")
    is_ad_like: bool = Field(..., alias="isAdLike", description="Whether review is ad-like")
    ad_score: float = Field(..., alias="adScore", description="Ad probability score")
    threshold: float = Field(..., description="Threshold used")
    model_version: str = Field(..., alias="modelVersion", description="Model version")
    reasons: Optional[List[str]] = Field(None, description="Reason codes")
    error: Optional[str] = Field(None, description="Error message if classification failed")

    model_config = ConfigDict(populate_by_name=True)


class BatchClassifyResponse(BaseModel):
    """
    Response schema for batch review classification.
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    results: List[BatchClassifyResultItem] = Field(..., description="Classification results")

    model_config = ConfigDict(populate_by_name=True)
