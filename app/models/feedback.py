"""
User feedback model - stores weak signals from users.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from app.db.database import Base


class FeedbackType(str, enum.Enum):
    """Feedback types from users."""

    HELPFUL_VOTE = "helpful_vote"
    AD_REPORT = "ad_report"
    NOT_AD_REPORT = "not_ad_report"


class UserFeedback(Base):
    """
    User feedback table (Weak Signals).
    Stores user votes and reports. These are NOT used as gold labels
    but can be used as features or to prioritize reviews for admin review.
    """

    __tablename__ = "user_feedback"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key to Review
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False, index=True)

    # User Information
    user_id = Column(String, nullable=False)

    # Feedback Type
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)

    # Feedback Value (flexible JSON field)
    # For helpful_vote: {"vote": "helpful"} or {"vote": "not_helpful"}
    # For ad_report: {} or {"reason": "spam"}
    value = Column(JSONB, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationship
    review = relationship("Review", back_populates="user_feedbacks")

    # Indexes and Constraints
    __table_args__ = (
        # Prevent duplicate feedback from same user
        UniqueConstraint("review_id", "user_id", "feedback_type", name="uq_user_feedback"),
        Index("idx_user_feedback_review_id", "review_id"),
        Index("idx_user_feedback_user_id", "user_id"),
        Index("idx_user_feedback_type", "feedback_type"),
        Index("idx_user_feedback_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<UserFeedback(id={self.id}, review_id={self.review_id}, user_id={self.user_id}, type={self.feedback_type.value})>"
