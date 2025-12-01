"""
Pydantic schemas for Review-related requests and responses.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ReviewInput(BaseModel):
    """
    Input schema for a review (from frontend or external platforms).
    Maps to the JSON structure from the API specification.
    """

    # Use _id to match frontend naming, but map internally
    id: Optional[str] = Field(None, alias="_id", description="External review ID")
    title: str = Field(..., min_length=1, max_length=500, description="Review title")
    content: str = Field(..., min_length=1, description="Review content")
    category: Optional[str] = Field(None, description="Product category")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Rating (0-5)")

    # Author information
    author_id: Optional[str] = Field(None, alias="authorId", description="Author user ID")
    author: Optional[str] = Field(None, description="Author display name")

    # Verification and engagement
    verified_purchase: bool = Field(
        False, alias="verifiedPurchase", description="Whether purchase is verified"
    )
    helpful_votes: int = Field(0, alias="helpfulVotes", ge=0, description="Number of helpful votes")
    not_helpful_votes: int = Field(
        0, alias="notHelpfulVotes", ge=0, description="Number of not helpful votes"
    )
    trust_score: int = Field(50, alias="trustScore", ge=0, le=100, description="Trust score (0-100)")

    # Metadata
    tags: Optional[List[str]] = Field(None, description="Review tags")
    source_platform: Optional[str] = Field(
        None, alias="sourcePlatform", description="Source platform (e.g., 'ReviewTrust', 'Naver')"
    )
    is_sponsored: bool = Field(False, alias="isSponsored", description="Whether review is sponsored")

    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field name and alias
        str_strip_whitespace=True,
    )

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Validate content length (max 20,000 characters as per spec)."""
        if len(v) > 20000:
            raise ValueError("Content exceeds maximum length of 20,000 characters")
        return v


class ReviewOutput(BaseModel):
    """
    Output schema for a review.
    Used when returning review data from the database.
    """

    id: str = Field(..., description="Internal review UUID")
    external_id: Optional[str] = Field(None, description="External review ID")
    title: str
    content: str
    category: Optional[str] = None
    rating: Optional[float] = None
    author_id: Optional[str] = None
    author: Optional[str] = None
    verified_purchase: bool
    helpful_votes: int
    not_helpful_votes: int
    trust_score: int
    tags: Optional[List[str]] = None
    source_platform: Optional[str] = None
    is_sponsored: bool
    created_at: Optional[datetime] = None
    ingested_at: datetime

    model_config = ConfigDict(from_attributes=True)
