"""
User feedback API endpoints.
"""
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import get_db
from app.core.middleware import get_request_id
from app.core.exceptions import NotFoundException, ConflictException, DBUnavailableException
from app.core.logging import logger
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.models import Review, UserFeedback, FeedbackType


router = APIRouter(prefix="/api/reviews", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit user feedback for a review.

    Handles:
    - Helpful/not helpful votes
    - Ad reports
    - Not ad reports

    Args:
        request: Feedback request
        db: Database session

    Returns:
        Feedback response

    Raises:
        404: Review not found
        409: Duplicate feedback (user already submitted this type of feedback)
        422: Invalid feedback type or vote value
        503: Database unavailable
    """
    request_id = get_request_id()

    logger.info(
        "Processing feedback request",
        extra={
            "request_id": request_id,
            "review_id": request.review_id,
            "user_id": request.user_id,
            "feedback_type": request.feedback_type,
        },
    )

    try:
        # Check if review exists
        # Try to parse as UUID first, if fails, treat as external_id
        try:
            review_uuid = uuid.UUID(request.review_id)
            stmt = select(Review).where(Review.id == review_uuid)
        except ValueError:
            # Not a UUID, try external_id
            stmt = select(Review).where(Review.external_id == request.review_id)

        result = await db.execute(stmt)
        review = result.scalar_one_or_none()

        if not review:
            raise NotFoundException(
                f"Review not found: {request.review_id}",
                details={"review_id": request.review_id},
            )

        # Map feedback_type string to enum
        try:
            feedback_type_enum = FeedbackType(request.feedback_type)
        except ValueError:
            raise ConflictException(
                f"Invalid feedback_type: {request.feedback_type}",
                details={"allowed": ["helpful_vote", "ad_report", "not_ad_report"]},
            )

        # Build feedback value JSON
        feedback_value = {}
        if request.feedback_type == "helpful_vote":
            if not request.vote:
                raise ConflictException(
                    "vote is required for helpful_vote feedback",
                    details={"feedback_type": request.feedback_type},
                )
            feedback_value["vote"] = request.vote
        # For ad_report and not_ad_report, value can be empty

        # Create feedback record
        user_feedback = UserFeedback(
            review_id=review.id,
            user_id=request.user_id,
            feedback_type=feedback_type_enum,
            value=feedback_value if feedback_value else None,
        )

        db.add(user_feedback)

        try:
            await db.flush()
        except IntegrityError as e:
            # Unique constraint violation (duplicate feedback)
            await db.rollback()
            logger.warning(
                "Duplicate feedback attempt",
                extra={
                    "request_id": request_id,
                    "review_id": request.review_id,
                    "user_id": request.user_id,
                    "feedback_type": request.feedback_type,
                },
            )
            raise ConflictException(
                "User has already submitted this type of feedback for this review",
                details={
                    "review_id": request.review_id,
                    "user_id": request.user_id,
                    "feedback_type": request.feedback_type,
                },
            )

        # TODO: Update review helpful_votes/not_helpful_votes counts atomically
        # This would require additional logic to count feedbacks and update the review
        # For now, we just store the feedback

        await db.commit()

        logger.info(
            "Feedback submitted successfully",
            extra={
                "request_id": request_id,
                "review_id": str(review.id),
                "user_id": request.user_id,
                "feedback_type": request.feedback_type,
            },
        )

        return FeedbackResponse(request_id=request_id, ok=True)

    except (NotFoundException, ConflictException):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(
            f"Failed to submit feedback: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise DBUnavailableException(f"Database operation failed: {str(e)}")
