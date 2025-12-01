"""
Pydantic schemas for User Feedback requests and responses.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FeedbackRequest(BaseModel):
    """
    Request schema for user feedback submission.
    POST /api/reviews/feedback
    """

    review_id: str = Field(..., alias="reviewId", description="Review ID to provide feedback on")
    user_id: str = Field(..., alias="userId", description="User ID providing feedback")
    feedback_type: str = Field(
        ...,
        alias="feedbackType",
        description="Type of feedback: 'helpful_vote', 'ad_report', 'not_ad_report'",
    )
    vote: Optional[str] = Field(None, description="Vote value: 'helpful' or 'not_helpful'")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("feedback_type")
    @classmethod
    def validate_feedback_type(cls, v: str) -> str:
        """Validate feedback type is one of the allowed values."""
        allowed = ["helpful_vote", "ad_report", "not_ad_report"]
        if v not in allowed:
            raise ValueError(f"feedback_type must be one of: {', '.join(allowed)}")
        return v

    @field_validator("vote")
    @classmethod
    def validate_vote(cls, v: Optional[str], info) -> Optional[str]:
        """Validate vote value if feedback_type is helpful_vote."""
        # Access feedback_type from validated data
        feedback_type = info.data.get("feedback_type")
        if feedback_type == "helpful_vote":
            if v not in ["helpful", "not_helpful"]:
                raise ValueError("vote must be 'helpful' or 'not_helpful' for helpful_vote feedback")
        return v


class FeedbackResponse(BaseModel):
    """
    Response schema for feedback submission.
    """

    request_id: str = Field(..., alias="requestId", description="Request ID for tracing")
    ok: bool = Field(..., description="Whether feedback was successfully recorded")

    model_config = ConfigDict(populate_by_name=True)
