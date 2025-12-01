"""
Classification model - stores classification results.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.database import Base


class Classification(Base):
    """
    Classification results table.
    Stores the output of the ML model for each review.
    """

    __tablename__ = "classifications"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key to Review
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False, index=True)

    # Model Information
    model_version = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)  # Threshold used for this classification

    # Classification Results
    ad_score = Column(Float, nullable=False)  # 0.0 to 1.0
    is_ad_like = Column(Boolean, nullable=False)  # True if ad_score >= threshold

    # Explanation (reason codes)
    reasons = Column(JSONB, nullable=True)  # Array of reason codes

    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationship
    review = relationship("Review", back_populates="classifications")

    # Indexes
    __table_args__ = (
        Index("idx_classifications_review_id", "review_id"),
        Index("idx_classifications_model_version", "model_version"),
        Index("idx_classifications_is_ad_like", "is_ad_like"),
        Index("idx_classifications_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<Classification(id={self.id}, review_id={self.review_id}, ad_score={self.ad_score}, is_ad_like={self.is_ad_like})>"
