"""
Classifier service - Core ML model service.

Handles:
- Model loading from .joblib files
- Fallback/dummy model when no trained model exists
- Hot-swapping models when a new version is detected
- Thread-safe singleton pattern
"""
import os
import threading
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack, csr_matrix

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelNotReadyException
from app.schemas.review import ReviewInput
from app.services.feature_service import feature_extractor


class ModelArtifact:
    """
    Represents a trained model artifact loaded from disk.

    Contains:
    - vectorizer: TfidfVectorizer for text features
    - classifier: SGDClassifier for classification
    - metadata: Model version, type, threshold, etc.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        classifier: SGDClassifier,
        metadata: Dict[str, Any],
    ):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.metadata = metadata
        self.loaded_at = datetime.utcnow()

    @property
    def version(self) -> str:
        """Get model version."""
        return self.metadata.get("version", "unknown")

    @property
    def threshold(self) -> float:
        """Get classification threshold."""
        return self.metadata.get("threshold", settings.AD_THRESHOLD)

    @property
    def model_type(self) -> str:
        """Get model type."""
        return self.metadata.get("model_type", "tfidf_sgd")


class DummyModel:
    """
    Dummy/fallback model when no trained model is available.

    Uses simple heuristics:
    - If review has URLs or many promo keywords -> likely ad
    - If not verified purchase and low trust score -> likely ad
    - Otherwise -> not ad
    """

    VERSION = "dummy-v1"
    THRESHOLD = 0.5

    @staticmethod
    def predict_proba(review: ReviewInput, text_stats: Dict[str, int]) -> float:
        """
        Predict ad probability using heuristics.

        Args:
            review: Review input
            text_stats: Text statistics

        Returns:
            Ad probability (0.0 to 1.0)
        """
        score = 0.0

        # URLs are strong signal for ads
        if text_stats.get("url_count", 0) > 0:
            score += 0.4

        # Promotional keywords
        promo_count = text_stats.get("promo_keyword_count", 0)
        if promo_count > 0:
            score += min(0.3, promo_count * 0.1)

        # Excessive exclamation marks
        if text_stats.get("exclamation_count", 0) > 5:
            score += 0.15

        # Not verified purchase
        if not review.verified_purchase:
            score += 0.1

        # Low trust score
        if review.trust_score < 50:
            score += 0.15

        # Explicitly sponsored
        if review.is_sponsored:
            score += 0.5

        # Clamp to [0, 1]
        return min(1.0, max(0.0, score))


class ClassifierService:
    """
    Classifier service (Singleton).

    Manages model loading, caching, and hot-swapping.
    Thread-safe using locks.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize classifier service."""
        if self._initialized:
            return

        self._model: Optional[ModelArtifact] = None
        self._model_lock = threading.Lock()
        self._active_version: Optional[str] = None
        self._artifact_dir = Path(settings.MODEL_ARTIFACT_DIR)

        # Ensure artifact directory exists
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Classifier service initialized",
            extra={"artifact_dir": str(self._artifact_dir)},
        )

        self._initialized = True

    def load_model(self, model_version: str) -> bool:
        """
        Load a model from disk.

        Args:
            model_version: Model version to load (e.g., 'clf-2025-12-01T020000Z')

        Returns:
            True if successfully loaded, False otherwise
        """
        artifact_path = self._artifact_dir / f"{model_version}.joblib"

        if not artifact_path.exists():
            logger.error(
                f"Model artifact not found: {artifact_path}",
                extra={"model_version": model_version},
            )
            return False

        try:
            logger.info(f"Loading model from {artifact_path}", extra={"model_version": model_version})

            # Load joblib artifact
            artifact_data = joblib.load(artifact_path)

            # Validate artifact structure
            required_keys = ["vectorizer", "classifier", "metadata"]
            if not all(key in artifact_data for key in required_keys):
                logger.error(
                    "Invalid artifact structure",
                    extra={"model_version": model_version, "keys": list(artifact_data.keys())},
                )
                return False

            # Create ModelArtifact
            model_artifact = ModelArtifact(
                vectorizer=artifact_data["vectorizer"],
                classifier=artifact_data["classifier"],
                metadata=artifact_data["metadata"],
            )

            # Atomically swap the model
            with self._model_lock:
                self._model = model_artifact
                self._active_version = model_version

            logger.info(
                f"Model loaded successfully",
                extra={
                    "model_version": model_version,
                    "model_type": model_artifact.model_type,
                    "threshold": model_artifact.threshold,
                },
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to load model: {str(e)}",
                extra={"model_version": model_version},
                exc_info=True,
            )
            return False

    def get_active_model(self) -> Optional[ModelArtifact]:
        """
        Get the currently active model (thread-safe).

        Returns:
            Active ModelArtifact or None
        """
        with self._model_lock:
            return self._model

    def reload_if_changed(self, new_version: str) -> bool:
        """
        Reload model if version has changed (hot-swap).

        Args:
            new_version: New model version from database

        Returns:
            True if model was reloaded, False otherwise
        """
        current_version = self._active_version

        if current_version == new_version:
            # No change
            return False

        logger.info(
            f"Model version changed, reloading",
            extra={"old_version": current_version, "new_version": new_version},
        )

        success = self.load_model(new_version)

        if not success:
            logger.error(
                "Failed to reload model, keeping old version",
                extra={"old_version": current_version, "new_version": new_version},
            )

        return success

    def classify(self, review: ReviewInput, include_reasons: bool = True) -> Dict[str, Any]:
        """
        Classify a single review.

        Args:
            review: Review input
            include_reasons: Whether to include reason codes

        Returns:
            Classification result dict with:
                - ad_score: float (0.0 to 1.0)
                - is_ad_like: bool
                - model_version: str
                - threshold: float
                - reasons: List[str] (if include_reasons=True)

        Raises:
            ModelNotReadyException: If no model is available and dummy model fails
        """
        # Extract features
        text = feature_extractor.extract_text(review)
        text_stats = feature_extractor.extract_text_statistics(text)
        metadata_features = feature_extractor.extract_metadata_features(review)

        # Get active model
        model = self.get_active_model()

        if model is None:
            # Use dummy model as fallback
            logger.warning("No trained model available, using dummy model")

            ad_score = DummyModel.predict_proba(review, text_stats)
            is_ad_like = ad_score >= DummyModel.THRESHOLD
            model_version = DummyModel.VERSION
            threshold = DummyModel.THRESHOLD

        else:
            # Use trained model
            try:
                # Vectorize text
                text_features = model.vectorizer.transform([text])

                # Metadata features
                metadata_vector = feature_extractor.metadata_to_vector(metadata_features)
                metadata_sparse = csr_matrix(metadata_vector.reshape(1, -1))

                # Combine features
                combined_features = hstack([text_features, metadata_sparse])

                # Predict
                ad_proba = model.classifier.predict_proba(combined_features)[0][1]
                ad_score = float(ad_proba)
                is_ad_like = ad_score >= model.threshold
                model_version = model.version
                threshold = model.threshold

            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}", exc_info=True)

                # Fallback to dummy model
                logger.warning("Falling back to dummy model due to prediction error")
                ad_score = DummyModel.predict_proba(review, text_stats)
                is_ad_like = ad_score >= DummyModel.THRESHOLD
                model_version = f"{DummyModel.VERSION}-fallback"
                threshold = DummyModel.THRESHOLD

        # Generate reason codes
        reasons = None
        if include_reasons:
            from app.services.reason_service import reason_generator

            reasons = reason_generator.generate_reasons(review, text_stats, metadata_features)

        result = {
            "ad_score": ad_score,
            "is_ad_like": is_ad_like,
            "model_version": model_version,
            "threshold": threshold,
        }

        if include_reasons:
            result["reasons"] = reasons

        return result


# Global singleton instance
classifier_service = ClassifierService()
