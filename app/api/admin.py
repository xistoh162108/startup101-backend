"""
Admin API endpoints.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, verify_admin_key
from app.core.middleware import get_request_id
from app.core.exceptions import NotFoundException, DBUnavailableException, InvalidArgumentException, InternalException
from app.core.logging import logger
from app.schemas.label import AdminLabelRequest, AdminLabelResponse
from app.schemas.model import TrainModelRequest, TrainModelResponse
from app.models import Review, AdminLabel, LabelType
from app.services.training import training_service


router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/reviews/label", response_model=AdminLabelResponse)
async def submit_admin_label(
    request: AdminLabelRequest,
    db: AsyncSession = Depends(get_db),
    admin_key: str = Depends(verify_admin_key),
):
    """
    Submit admin gold label for a review.

    Admin labels are the ground truth used for model training.

    Args:
        request: Admin label request
        db: Database session
        admin_key: Verified admin API key

    Returns:
        Admin label response

    Raises:
        401: Invalid admin API key
        404: Review not found
        503: Database unavailable
    """
    request_id = get_request_id()

    logger.info(
        "Processing admin label request",
        extra={
            "request_id": request_id,
            "review_id": request.review_id,
            "admin_id": request.admin_id,
            "label": request.label,
        },
    )

    try:
        # Find review by ID or external_id
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

        # Convert label string to enum
        label_enum = LabelType(request.label)

        # Create admin label
        # Note: If multiple labels exist for same review, latest one is used for training
        admin_label = AdminLabel(
            review_id=review.id,
            label=label_enum,
            admin_id=request.admin_id,
            comment=request.comment,
        )

        db.add(admin_label)
        await db.flush()

        logger.info(
            "Admin label submitted successfully",
            extra={
                "request_id": request_id,
                "review_id": str(review.id),
                "admin_id": request.admin_id,
                "label": request.label,
            },
        )

        await db.commit()

        return AdminLabelResponse(
            request_id=request_id,
            ok=True,
            latest_label=request.label,
        )

    except NotFoundException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to submit admin label: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise DBUnavailableException(f"Database operation failed: {str(e)}")


@router.post("/model/train", response_model=TrainModelResponse)
async def trigger_model_training(
    request: TrainModelRequest,
    db: AsyncSession = Depends(get_db),
    admin_key: str = Depends(verify_admin_key),
):
    """
    Trigger model training.

    Trains a new model using all available admin labels (gold labels).
    The training is synchronous (MVP), but could be made async with Celery.

    Args:
        request: Training request
        db: Database session
        admin_key: Verified admin API key

    Returns:
        Training response with job acceptance status

    Raises:
        401: Invalid admin API key
        409: Insufficient training data
        422: Invalid training parameters
        500: Training failed
    """
    request_id = get_request_id()

    logger.info(
        "Processing model training request",
        extra={
            "request_id": request_id,
            "admin_id": request.admin_id,
            "mode": request.mode,
            "min_new_labels": request.min_new_labels,
        },
    )

    try:
        # Step 1: Load training data
        logger.info("Loading training data from database")
        reviews, labels, train_data_snapshot = await training_service.load_training_data(db)

        # Check minimum labels requirement
        if len(labels) < request.min_new_labels:
            raise InvalidArgumentException(
                f"Insufficient labels for training: {len(labels)} available, {request.min_new_labels} required",
                details={
                    "available_labels": len(labels),
                    "required_labels": request.min_new_labels,
                },
            )

        # Step 2: Train model
        logger.info(f"Training model with {len(labels)} samples")
        vectorizer, classifier, metrics = training_service.train_model(reviews, labels)

        # Step 3: Generate model version
        model_version = f"clf-{datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ')}"

        # Step 4: Save model artifact
        logger.info(f"Saving model artifact: {model_version}")
        artifact_uri = training_service.save_model_artifact(
            vectorizer=vectorizer,
            classifier=classifier,
            metadata={
                "version": model_version,
                "model_type": "tfidf_sgd",
                "threshold": 0.8,
                "trained_at": datetime.utcnow().isoformat(),
                "trained_by": request.admin_id,
            },
            model_version=model_version,
        )

        # Step 5: Register model in database
        logger.info(f"Registering model in database")
        model_registry = await training_service.register_model(
            db=db,
            model_version=model_version,
            model_type="tfidf_sgd",
            artifact_uri=artifact_uri,
            metrics=metrics,
            train_data_snapshot=train_data_snapshot,
        )

        # Step 6: Activate model if requested (and if it's better)
        activated = False
        if request.activate_if_better:
            # For MVP, we always activate (no comparison logic yet)
            # In production, compare metrics with current active model
            logger.info(f"Activating new model: {model_version}")
            activated = await training_service.activate_model(db, model_version)

        await db.commit()

        logger.info(
            "Model training completed successfully",
            extra={
                "request_id": request_id,
                "model_version": model_version,
                "activated": activated,
                "metrics": metrics,
            },
        )

        return TrainModelResponse(
            request_id=request_id,
            accepted=True,
            job_id=model_version,  # Use model version as job ID for MVP
            message=f"Model {model_version} trained successfully. Activated: {activated}",
        )

    except (InvalidArgumentException, InternalException):
        # Re-raise custom exceptions
        raise
    except Exception as e:
        logger.error(
            f"Model training failed: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise InternalException(f"Model training failed: {str(e)}")
