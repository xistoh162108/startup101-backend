"""
Classification API endpoints.
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.core.middleware import get_request_id
from app.core.exceptions import (
    ModelNotReadyException,
    DBUnavailableException,
    InvalidArgumentException,
)
from app.core.logging import logger
from app.schemas.classification import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchClassifyResultItem,
)
from app.schemas.review import ReviewInput
from app.models import Review, Classification
from app.services.classifier import classifier_service


router = APIRouter(prefix="/api/reviews", tags=["classification"])


async def persist_review_and_classification(
    db: AsyncSession,
    review_input: ReviewInput,
    classification_result: dict,
) -> str:
    """
    Persist review and classification result to database.

    Args:
        db: Database session
        review_input: Review input data
        classification_result: Classification result from classifier service

    Returns:
        Internal review UUID (as string)

    Raises:
        DBUnavailableException: If database operation fails
    """
    try:
        # Check if review with external_id already exists
        review_id = None
        if review_input.id:
            stmt = select(Review).where(Review.external_id == review_input.id)
            result = await db.execute(stmt)
            existing_review = result.scalar_one_or_none()

            if existing_review:
                review_id = str(existing_review.id)
                logger.info(
                    f"Review with external_id={review_input.id} already exists",
                    extra={"review_id": review_id},
                )
            else:
                # Create new review
                review = Review(
                    external_id=review_input.id,
                    title=review_input.title,
                    content=review_input.content,
                    category=review_input.category,
                    rating=review_input.rating,
                    author_id=review_input.author_id,
                    author=review_input.author,
                    verified_purchase=review_input.verified_purchase,
                    helpful_votes=review_input.helpful_votes,
                    not_helpful_votes=review_input.not_helpful_votes,
                    trust_score=review_input.trust_score,
                    tags=review_input.tags,
                    source_platform=review_input.source_platform,
                    is_sponsored=review_input.is_sponsored,
                )
                db.add(review)
                await db.flush()  # Get the ID without committing
                review_id = str(review.id)
        else:
            # No external_id, create new review
            review = Review(
                title=review_input.title,
                content=review_input.content,
                category=review_input.category,
                rating=review_input.rating,
                author_id=review_input.author_id,
                author=review_input.author,
                verified_purchase=review_input.verified_purchase,
                helpful_votes=review_input.helpful_votes,
                not_helpful_votes=review_input.not_helpful_votes,
                trust_score=review_input.trust_score,
                tags=review_input.tags,
                source_platform=review_input.source_platform,
                is_sponsored=review_input.is_sponsored,
            )
            db.add(review)
            await db.flush()
            review_id = str(review.id)

        # Create classification record
        classification = Classification(
            review_id=uuid.UUID(review_id),
            model_version=classification_result["model_version"],
            threshold=classification_result["threshold"],
            ad_score=classification_result["ad_score"],
            is_ad_like=classification_result["is_ad_like"],
            reasons=classification_result.get("reasons"),
        )
        db.add(classification)
        await db.flush()

        logger.info(
            "Review and classification persisted",
            extra={
                "review_id": review_id,
                "external_id": review_input.id,
                "is_ad_like": classification_result["is_ad_like"],
            },
        )

        return review_id

    except Exception as e:
        logger.error(f"Failed to persist review and classification: {str(e)}", exc_info=True)
        raise DBUnavailableException(f"Database operation failed: {str(e)}")


@router.post("/classify", response_model=ClassifyResponse)
async def classify_review(
    request: ClassifyRequest,
    db: Optional[AsyncSession] = Depends(get_db),
):
    """
    Classify a single review.

    Args:
        request: Classification request
        db: Database session (optional, only needed if persist=True)

    Returns:
        Classification result

    Raises:
        422: Invalid input
        503: Model not ready or DB unavailable
    """
    request_id = get_request_id()

    logger.info(
        "Processing classification request",
        extra={
            "request_id": request_id,
            "external_id": request.review.id,
            "persist": request.options.persist,
        },
    )

    # Classify the review
    try:
        classification_result = classifier_service.classify(
            review=request.review,
            include_reasons=request.options.include_reasons,
        )
    except Exception as e:
        logger.error(
            f"Classification failed: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise ModelNotReadyException(f"Classification failed: {str(e)}")

    # Determine review_id for response
    review_id = request.review.id if request.review.id else "auto-generated"

    # Persist if requested
    if request.options.persist:
        if db is None:
            raise DBUnavailableException("Database unavailable")

        try:
            internal_review_id = await persist_review_and_classification(
                db=db,
                review_input=request.review,
                classification_result=classification_result,
            )
            # Use internal ID if no external ID
            if not request.review.id:
                review_id = internal_review_id
        except DBUnavailableException:
            # If persist fails but persist=True, we should fail
            raise

    # Build response
    response = ClassifyResponse(
        request_id=request_id,
        review_id=review_id,
        is_ad_like=classification_result["is_ad_like"],
        ad_score=classification_result["ad_score"],
        threshold=classification_result["threshold"],
        model_version=classification_result["model_version"],
        reasons=classification_result.get("reasons") if request.options.include_reasons else None,
    )

    logger.info(
        "Classification completed",
        extra={
            "request_id": request_id,
            "review_id": review_id,
            "is_ad_like": classification_result["is_ad_like"],
            "ad_score": classification_result["ad_score"],
        },
    )

    return response


@router.post("/batch-classify", response_model=BatchClassifyResponse)
async def batch_classify_reviews(
    request: BatchClassifyRequest,
    db: Optional[AsyncSession] = Depends(get_db),
):
    """
    Classify multiple reviews in batch.

    Args:
        request: Batch classification request (max 100 reviews)
        db: Database session (optional, only needed if persist=True)

    Returns:
        Batch classification results

    Raises:
        422: Invalid input or too many reviews
        503: Model not ready or DB unavailable
    """
    request_id = get_request_id()

    logger.info(
        "Processing batch classification request",
        extra={
            "request_id": request_id,
            "num_reviews": len(request.reviews),
            "persist": request.options.persist,
        },
    )

    # Validate batch size
    if len(request.reviews) > 100:
        raise InvalidArgumentException(
            "Too many reviews in batch",
            details={"max_allowed": 100, "received": len(request.reviews)},
        )

    results = []

    # Process each review
    for review in request.reviews:
        try:
            # Classify
            classification_result = classifier_service.classify(
                review=review,
                include_reasons=request.options.include_reasons,
            )

            review_id = review.id if review.id else "auto-generated"

            # Persist if requested
            if request.options.persist:
                if db is None:
                    raise DBUnavailableException("Database unavailable")

                try:
                    internal_review_id = await persist_review_and_classification(
                        db=db,
                        review_input=review,
                        classification_result=classification_result,
                    )
                    if not review.id:
                        review_id = internal_review_id
                except Exception as e:
                    # For batch, we log the error but continue
                    logger.error(
                        f"Failed to persist review in batch: {str(e)}",
                        extra={"request_id": request_id, "review_id": review_id},
                    )
                    # Add result with error
                    results.append(
                        BatchClassifyResultItem(
                            review_id=review_id,
                            is_ad_like=classification_result["is_ad_like"],
                            ad_score=classification_result["ad_score"],
                            threshold=classification_result["threshold"],
                            model_version=classification_result["model_version"],
                            reasons=classification_result.get("reasons")
                            if request.options.include_reasons
                            else None,
                            error=f"Persistence failed: {str(e)}",
                        )
                    )
                    continue

            # Add successful result
            results.append(
                BatchClassifyResultItem(
                    review_id=review_id,
                    is_ad_like=classification_result["is_ad_like"],
                    ad_score=classification_result["ad_score"],
                    threshold=classification_result["threshold"],
                    model_version=classification_result["model_version"],
                    reasons=classification_result.get("reasons")
                    if request.options.include_reasons
                    else None,
                    error=None,
                )
            )

        except Exception as e:
            # Classification failed for this review
            logger.error(
                f"Failed to classify review in batch: {str(e)}",
                extra={"request_id": request_id, "review_id": review.id},
                exc_info=True,
            )
            results.append(
                BatchClassifyResultItem(
                    review_id=review.id if review.id else "unknown",
                    is_ad_like=False,
                    ad_score=0.0,
                    threshold=0.8,
                    model_version="error",
                    reasons=None,
                    error=f"Classification failed: {str(e)}",
                )
            )

    logger.info(
        "Batch classification completed",
        extra={
            "request_id": request_id,
            "total_reviews": len(request.reviews),
            "successful": len([r for r in results if r.error is None]),
            "failed": len([r for r in results if r.error is not None]),
        },
    )

    return BatchClassifyResponse(request_id=request_id, results=results)
