"""
Review model - stores original review data.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Boolean,
    Integer,
    DateTime,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.database import Base


class Review(Base):
    """
    Original review data table.
    Stores all review information including metadata and user engagement metrics.
    """

    __tablename__ = "reviews"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # External Reference (from frontend/other platforms)
    external_id = Column(String, unique=True, nullable=True, index=True)

    # Review Content
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    rating = Column(Float, nullable=True)

    # Author Information
    author_id = Column(String, nullable=True)
    author = Column(String, nullable=True)

    # Verification & Engagement
    verified_purchase = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    not_helpful_votes = Column(Integer, default=0)
    trust_score = Column(Integer, default=50)  # 0-100

    # Metadata
    tags = Column(JSONB, nullable=True)  # Array of tags
    source_platform = Column(String, nullable=True)  # e.g., 'ReviewTrust', 'Naver', etc.
    is_sponsored = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=True)  # Original creation time
    ingested_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )  # When ingested into our system

    # Relationships
    classifications = relationship(
        "Classification", back_populates="review", cascade="all, delete-orphan"
    )
    admin_labels = relationship(
        "AdminLabel", back_populates="review", cascade="all, delete-orphan"
    )
    user_feedbacks = relationship(
        "UserFeedback", back_populates="review", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_reviews_external_id", "external_id"),
        Index("idx_reviews_category", "category"),
        Index("idx_reviews_source_platform", "source_platform"),
        Index("idx_reviews_created_at", "created_at"),
        Index("idx_reviews_ingested_at", "ingested_at"),
    )

    def __repr__(self):
        return f"<Review(id={self.id}, external_id={self.external_id}, title={self.title[:30]}...)>"
