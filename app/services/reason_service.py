"""
Reason code generation service.
Provides rule-based explanations for classification results.
"""
from typing import List, Dict, Any

from app.schemas.review import ReviewInput


class ReasonCodeGenerator:
    """
    Generates reason codes explaining why a review was classified as ad-like.

    Reason codes are rule-based (not from ML model) to provide interpretable
    explanations for operators and users.

    Possible reason codes:
    - contains_url: Review contains URLs
    - contains_promo_keywords: Contains promotional keywords (할인, 쿠폰, etc.)
    - excessive_exclamation: Too many exclamation marks (>5)
    - no_verified_purchase: Purchase not verified
    - low_trust_score: Trust score below 50
    - high_not_helpful_ratio: High ratio of "not helpful" votes
    - is_sponsored: Explicitly marked as sponsored
    """

    # Thresholds
    EXCLAMATION_THRESHOLD = 5
    TRUST_SCORE_THRESHOLD = 50
    NOT_HELPFUL_RATIO_THRESHOLD = 0.6

    @staticmethod
    def generate_reasons(
        review: ReviewInput,
        text_stats: Dict[str, int],
        metadata_features: Dict[str, Any],
    ) -> List[str]:
        """
        Generate reason codes for a review.

        Args:
            review: Review input
            text_stats: Text statistics from feature_service
            metadata_features: Metadata features from feature_service

        Returns:
            List of reason codes
        """
        reasons = []

        # Check for URLs
        if text_stats.get("url_count", 0) > 0:
            reasons.append("contains_url")

        # Check for promotional keywords
        if text_stats.get("promo_keyword_count", 0) > 0:
            reasons.append("contains_promo_keywords")

        # Check for excessive exclamation marks
        if text_stats.get("exclamation_count", 0) > ReasonCodeGenerator.EXCLAMATION_THRESHOLD:
            reasons.append("excessive_exclamation")

        # Check for verified purchase
        if not review.verified_purchase:
            reasons.append("no_verified_purchase")

        # Check trust score
        if review.trust_score < ReasonCodeGenerator.TRUST_SCORE_THRESHOLD:
            reasons.append("low_trust_score")

        # Check not_helpful ratio
        total_votes = review.helpful_votes + review.not_helpful_votes
        if total_votes >= 10:  # Only check if there are enough votes
            not_helpful_ratio = review.not_helpful_votes / total_votes
            if not_helpful_ratio > ReasonCodeGenerator.NOT_HELPFUL_RATIO_THRESHOLD:
                reasons.append("high_not_helpful_ratio")

        # Check if sponsored
        if review.is_sponsored:
            reasons.append("is_sponsored")

        return reasons


# Global singleton instance
reason_generator = ReasonCodeGenerator()
