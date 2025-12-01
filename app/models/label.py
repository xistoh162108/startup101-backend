"""
Admin label model - stores gold standard labels from administrators.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.db.database import Base


class LabelType(str, enum.Enum):
    """Label types for admin labeling."""

    AD = "ad"
    NOT_AD = "not_ad"


class AdminLabel(Base):
    """
    Admin labels table (Gold Standard).
    Administrators manually label reviews as 'ad' or 'not_ad'.
    These labels are used as ground truth for model training.
    """

    __tablename__ = "admin_labels"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key to Review
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False, index=True)

    # Label
    label = Column(SQLEnum(LabelType), nullable=False)

    # Admin Information
    admin_id = Column(String, nullable=False)
    comment = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationship
    review = relationship("Review", back_populates="admin_labels")

    # Indexes
    __table_args__ = (
        Index("idx_admin_labels_review_id", "review_id"),
        Index("idx_admin_labels_label", "label"),
        Index("idx_admin_labels_admin_id", "admin_id"),
        Index("idx_admin_labels_created_at", "created_at"),
        # For getting latest label per review
        Index("idx_admin_labels_review_created", "review_id", "created_at"),
    )

    def __repr__(self):
        return f"<AdminLabel(id={self.id}, review_id={self.review_id}, label={self.label.value}, admin_id={self.admin_id})>"
