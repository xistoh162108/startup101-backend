"""
Model information API endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.core.middleware import get_request_id
from app.core.config import settings
from app.core.exceptions import ModelNotReadyException
from app.core.logging import logger
from app.schemas.model import ModelInfoResponse
from app.models import ModelState, ModelRegistry, ModelStatus
from app.services.classifier import classifier_service


router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(db: AsyncSession = Depends(get_db)):
    """
    Get information about the currently active model.

    Returns:
        Model information including version, type, threshold, features, etc.

    Raises:
        503: No model is currently active
    """
    request_id = get_request_id()

    logger.info("Processing model info request", extra={"request_id": request_id})

    try:
        # Get active model version from database
        stmt = select(ModelState).where(ModelState.id == 1)
        result = await db.execute(stmt)
        model_state = result.scalar_one_or_none()

        if not model_state or not model_state.active_model_version:
            # No active model in database
            # Check if classifier service has loaded a model (shouldn't happen, but defensive)
            active_model = classifier_service.get_active_model()
            if active_model:
                return ModelInfoResponse(
                    request_id=request_id,
                    model_version=active_model.version,
                    model_type=active_model.model_type,
                    threshold=active_model.threshold,
                    last_trained_at=active_model.loaded_at,
                    features_used=[
                        "title",
                        "content",
                        "tags",
                        "rating",
                        "verified_purchase",
                        "helpful_votes",
                        "not_helpful_votes",
                        "trust_score",
                        "source_platform",
                        "is_sponsored",
                    ],
                    status="active",
                )
            else:
                raise ModelNotReadyException("No active model available")

        # Get model registry entry
        stmt = select(ModelRegistry).where(
            ModelRegistry.model_version == model_state.active_model_version
        )
        result = await db.execute(stmt)
        model_registry = result.scalar_one_or_none()

        if not model_registry:
            raise ModelNotReadyException(
                f"Active model version {model_state.active_model_version} not found in registry"
            )

        # Get loaded model from classifier service (to verify it's actually loaded)
        active_model = classifier_service.get_active_model()

        # Features used by the model (hardcoded for now, could be in metadata)
        features_used = [
            "title",
            "content",
            "tags",
            "rating",
            "verified_purchase",
            "helpful_votes",
            "not_helpful_votes",
            "trust_score",
            "source_platform",
            "is_sponsored",
        ]

        return ModelInfoResponse(
            request_id=request_id,
            model_version=model_registry.model_version,
            model_type=model_registry.model_type,
            threshold=settings.AD_THRESHOLD,  # Could also be in model metadata
            last_trained_at=model_registry.created_at,
            features_used=features_used,
            status=model_registry.status.value,
        )

    except ModelNotReadyException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get model info: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise ModelNotReadyException(f"Failed to retrieve model information: {str(e)}")
