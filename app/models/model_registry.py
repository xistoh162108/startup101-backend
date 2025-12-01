"""
Model registry models - stores ML model versions and active model pointer.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
import enum

from app.db.database import Base


class ModelStatus(str, enum.Enum):
    """Model status in the registry."""

    STAGED = "staged"  # Trained but not active
    ACTIVE = "active"  # Currently serving predictions
    ARCHIVED = "archived"  # Old/replaced model
    FAILED = "failed"  # Training or loading failed


class ModelRegistry(Base):
    """
    Model registry table.
    Stores information about all trained models with versioning.
    """

    __tablename__ = "model_registry"

    # Primary Key (model version as string, e.g., 'clf-2025-12-01T020000Z')
    model_version = Column(String, primary_key=True)

    # Model Type
    model_type = Column(String, nullable=False)  # e.g., 'tfidf_sgd', 'multinomial_nb'

    # Artifact Storage
    artifact_uri = Column(String, nullable=False)  # Path to .joblib file

    # Performance Metrics (as JSON)
    metrics = Column(JSONB, nullable=True)  # e.g., {"accuracy": 0.95, "precision": 0.92, ...}

    # Training Data Snapshot (as JSON)
    train_data_snapshot = Column(
        JSONB, nullable=True
    )  # e.g., {"num_samples": 1000, "num_ad": 300, "date_range": "..."}

    # Status
    status = Column(SQLEnum(ModelStatus), nullable=False, default=ModelStatus.STAGED)

    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_model_registry_status", "status"),
        Index("idx_model_registry_created_at", "created_at"),
    )

    def __repr__(self):
        return f"<ModelRegistry(version={self.model_version}, type={self.model_type}, status={self.status.value})>"


class ModelState(Base):
    """
    Model state table (singleton).
    Points to the currently active model version.
    Should only ever have one row with id=1.
    """

    __tablename__ = "model_state"

    # Primary Key (fixed to 1, singleton pattern)
    id = Column(Integer, primary_key=True, default=1)

    # Active Model Version
    active_model_version = Column(
        String,
        ForeignKey("model_registry.model_version"),
        nullable=True,  # Nullable until first model is trained
    )

    # Timestamp
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ModelState(id={self.id}, active_model_version={self.active_model_version})>"
