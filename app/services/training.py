"""
Model training service.

Handles:
- Loading gold label data from database
- Training TF-IDF vectorizer + SGD classifier
- Saving model artifacts (.joblib)
- Updating model registry
- Activating new models
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import InvalidArgumentException, InternalException
from app.models import Review, AdminLabel, LabelType, ModelRegistry, ModelState, ModelStatus
from app.services.feature_service import feature_extractor
from app.schemas.review import ReviewInput


class TrainingService:
    """
    Service for training and managing ML models.
    """

    @staticmethod
    async def load_training_data(
        db: AsyncSession,
    ) -> Tuple[List[ReviewInput], List[int], Dict[str, Any]]:
        """
        Load training data from database.

        Joins reviews with admin_labels (gold labels only).
        For each review, uses the LATEST admin label.

        Args:
            db: Database session

        Returns:
            Tuple of (reviews, labels, metadata)
            - reviews: List of ReviewInput objects
            - labels: List of binary labels (0=not_ad, 1=ad)
            - metadata: Training data snapshot info

        Raises:
            InvalidArgumentException: If insufficient training data
        """
        logger.info("Loading training data from database")

        # Query to get reviews with their latest admin label
        # We need to get the latest label for each review
        stmt = (
            select(Review, AdminLabel)
            .join(AdminLabel, Review.id == AdminLabel.review_id)
            .order_by(Review.id, AdminLabel.created_at.desc())
        )

        result = await db.execute(stmt)
        rows = result.all()

        if not rows:
            raise InvalidArgumentException(
                "No labeled reviews found in database",
                details={"required_minimum": settings.MIN_LABELS_FOR_TRAINING},
            )

        # Group by review_id and take latest label
        review_label_map: Dict[uuid.UUID, Tuple[Review, AdminLabel]] = {}
        for review, label in rows:
            if review.id not in review_label_map:
                review_label_map[review.id] = (review, label)
            # Since we ordered by created_at desc, first one is latest

        if len(review_label_map) < settings.MIN_LABELS_FOR_TRAINING:
            raise InvalidArgumentException(
                f"Insufficient training data: {len(review_label_map)} labeled reviews, need at least {settings.MIN_LABELS_FOR_TRAINING}",
                details={
                    "current_count": len(review_label_map),
                    "required_minimum": settings.MIN_LABELS_FOR_TRAINING,
                },
            )

        # Convert to ReviewInput and labels
        reviews = []
        labels = []

        for review, admin_label in review_label_map.values():
            # Convert SQLAlchemy model to ReviewInput
            review_input = ReviewInput(
                id=review.external_id,
                title=review.title,
                content=review.content,
                category=review.category,
                rating=review.rating,
                author_id=review.author_id,
                author=review.author,
                verified_purchase=review.verified_purchase,
                helpful_votes=review.helpful_votes,
                not_helpful_votes=review.not_helpful_votes,
                trust_score=review.trust_score,
                tags=review.tags,
                source_platform=review.source_platform,
                is_sponsored=review.is_sponsored,
            )
            reviews.append(review_input)

            # Convert label to binary (0 or 1)
            label_binary = 1 if admin_label.label == LabelType.AD else 0
            labels.append(label_binary)

        # Count labels
        num_ad = sum(labels)
        num_not_ad = len(labels) - num_ad
        created_ats = [r.created_at for r, _ in review_label_map.values() if r.created_at is not None]

        metadata = {
            "total_samples": len(labels),
            "num_ad": num_ad,
            "num_not_ad": num_not_ad,
            "review_ids": [str(rid) for rid in review_label_map.keys()],
            "date_range": {
                "earliest": min(created_ats).isoformat() if created_ats else None,
                "latest": max(created_ats).isoformat() if created_ats else None,
            },
        }

        logger.info(
            f"Loaded {len(labels)} labeled reviews",
            extra={
                "total_samples": len(labels),
                "num_ad": num_ad,
                "num_not_ad": num_not_ad,
            },
        )

        return reviews, labels, metadata

    @staticmethod
    def train_model(
        reviews: List[ReviewInput],
        labels: List[int],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[TfidfVectorizer, SGDClassifier, Dict[str, float]]:
        """
        Train TF-IDF + SGD classifier.

        Args:
            reviews: List of review inputs
            labels: List of binary labels (0 or 1)
            test_size: Fraction of data to use for validation
            random_state: Random seed

        Returns:
            Tuple of (vectorizer, classifier, metrics)

        Raises:
            InternalException: If training fails
        """
        logger.info(f"Training model with {len(reviews)} samples")

        try:
            # Extract text features
            texts = [feature_extractor.extract_text(review) for review in reviews]

            # Extract metadata features
            metadata_features_list = [
                feature_extractor.extract_metadata_features(review) for review in reviews
            ]
            metadata_vectors = np.array(
                [feature_extractor.metadata_to_vector(mf) for mf in metadata_features_list]
            )

            # Train/test split
            (
                texts_train,
                texts_test,
                metadata_train,
                metadata_test,
                y_train,
                y_test,
            ) = train_test_split(texts, metadata_vectors, labels, test_size=test_size, random_state=random_state, stratify=labels)

            logger.info(
                f"Split data: {len(y_train)} train, {len(y_test)} test",
                extra={
                    "train_ad": sum(y_train),
                    "train_not_ad": len(y_train) - sum(y_train),
                    "test_ad": sum(y_test),
                    "test_not_ad": len(y_test) - sum(y_test),
                },
            )

            # Train TF-IDF vectorizer
            logger.info("Training TF-IDF vectorizer")
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words=None,  # Could add Korean stop words
            )
            X_train_text = vectorizer.fit_transform(texts_train)
            X_test_text = vectorizer.transform(texts_test)

            # Combine text and metadata features
            X_train = hstack([X_train_text, csr_matrix(metadata_train)])
            X_test = hstack([X_test_text, csr_matrix(metadata_test)])

            logger.info(
                f"Feature matrix shape: {X_train.shape}",
                extra={"n_samples": X_train.shape[0], "n_features": X_train.shape[1]},
            )

            # Train classifier
            logger.info("Training SGD classifier")
            classifier = SGDClassifier(
                loss="log_loss",  # Logistic regression
                penalty="l2",
                alpha=0.0001,
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",  # Handle class imbalance
                n_jobs=-1,
            )
            classifier.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_samples": len(y_test),
            }

            logger.info(
                "Training completed",
                extra={
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                },
            )

            return vectorizer, classifier, metrics

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise InternalException(f"Model training failed: {str(e)}")

    @staticmethod
    def save_model_artifact(
        vectorizer: TfidfVectorizer,
        classifier: SGDClassifier,
        metadata: Dict[str, Any],
        model_version: str,
    ) -> str:
        """
        Save model artifact to disk.

        Args:
            vectorizer: Trained TF-IDF vectorizer
            classifier: Trained classifier
            metadata: Model metadata
            model_version: Model version string

        Returns:
            Artifact URI (file path)

        Raises:
            InternalException: If saving fails
        """
        try:
            artifact_dir = Path(settings.MODEL_ARTIFACT_DIR)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            artifact_path = artifact_dir / f"{model_version}.joblib"

            # Package everything together
            artifact = {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "metadata": metadata,
            }

            # Save with joblib
            joblib.dump(artifact, artifact_path)

            logger.info(
                f"Model artifact saved to {artifact_path}",
                extra={"model_version": model_version, "artifact_path": str(artifact_path)},
            )

            return str(artifact_path)

        except Exception as e:
            logger.error(f"Failed to save model artifact: {str(e)}", exc_info=True)
            raise InternalException(f"Failed to save model artifact: {str(e)}")

    @staticmethod
    async def register_model(
        db: AsyncSession,
        model_version: str,
        model_type: str,
        artifact_uri: str,
        metrics: Dict[str, float],
        train_data_snapshot: Dict[str, Any],
    ) -> ModelRegistry:
        """
        Register model in model_registry table.

        Args:
            db: Database session
            model_version: Model version
            model_type: Model type (e.g., 'tfidf_sgd')
            artifact_uri: Path to artifact file
            metrics: Performance metrics
            train_data_snapshot: Training data metadata

        Returns:
            ModelRegistry object

        Raises:
            InternalException: If registration fails
        """
        try:
            model_registry = ModelRegistry(
                model_version=model_version,
                model_type=model_type,
                artifact_uri=artifact_uri,
                metrics=metrics,
                train_data_snapshot=train_data_snapshot,
                status=ModelStatus.STAGED,
            )

            db.add(model_registry)
            await db.flush()

            logger.info(
                f"Model registered in registry",
                extra={"model_version": model_version, "status": ModelStatus.STAGED.value},
            )

            return model_registry

        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}", exc_info=True)
            raise InternalException(f"Failed to register model: {str(e)}")

    @staticmethod
    async def activate_model(db: AsyncSession, model_version: str) -> bool:
        """
        Activate a model (set it as active in model_state).

        Args:
            db: Database session
            model_version: Model version to activate

        Returns:
            True if activation succeeded

        Raises:
            InternalException: If activation fails
        """
        try:
            # Update model_registry status to active
            stmt = select(ModelRegistry).where(ModelRegistry.model_version == model_version)
            result = await db.execute(stmt)
            model_registry = result.scalar_one_or_none()

            if not model_registry:
                raise InternalException(f"Model {model_version} not found in registry")

            # Set previous active model to archived
            stmt = select(ModelRegistry).where(ModelRegistry.status == ModelStatus.ACTIVE)
            result = await db.execute(stmt)
            old_active_models = result.scalars().all()

            for old_model in old_active_models:
                old_model.status = ModelStatus.ARCHIVED

            # Set new model to active
            model_registry.status = ModelStatus.ACTIVE

            # Update model_state (singleton)
            stmt = select(ModelState).where(ModelState.id == 1)
            result = await db.execute(stmt)
            model_state = result.scalar_one_or_none()

            if model_state:
                model_state.active_model_version = model_version
                model_state.updated_at = datetime.utcnow()
            else:
                # Create model_state if it doesn't exist
                model_state = ModelState(
                    id=1,
                    active_model_version=model_version,
                )
                db.add(model_state)

            await db.flush()

            logger.info(
                f"Model activated",
                extra={"model_version": model_version},
            )

            # Trigger model reload in classifier service
            from app.services.classifier import classifier_service

            success = classifier_service.load_model(model_version)

            if not success:
                logger.error(
                    f"Failed to load activated model into memory",
                    extra={"model_version": model_version},
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to activate model: {str(e)}", exc_info=True)
            raise InternalException(f"Failed to activate model: {str(e)}")


# Global singleton instance
training_service = TrainingService()
