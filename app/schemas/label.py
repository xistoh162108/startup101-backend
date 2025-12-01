"""
Pydantic schemas for Admin Label (Gold Label) requests and responses.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class AdminLabelRequest(BaseModel):
    """
    Request schema for admin labeling.
    POST /api/admin/reviews/label
    """

    review_id: str = Field(..., alias="reviewId", description="Review ID to label")
    admin_id: str = Field(..., alias="adminId", description="Admin user ID")
    label: str = Field(..., description="Label: 'ad' or 'not_ad'")
    comment: Optional[str] = Field(None, description="Optional comment about the labeling decision")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label is either 'ad' or 'not_ad'."""
        if v not in ["ad", "not_ad"]:
            raise ValueError("label must be either 'ad' or 'not_ad'")
        return v


class AdminLabelResponse(BaseModel):
    """
    Response schema for admin labeling.
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    ok: bool = Field(..., description="Whether label was successfully recorded")
    latest_label: str = Field(..., alias="latestLabel", description="The label that was set")

    model_config = ConfigDict(populate_by_name=True)
