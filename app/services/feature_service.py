"""
Feature extraction service for ML model.
Combines text and metadata features from reviews.
"""
import re
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.sparse import hstack, csr_matrix

from app.schemas.review import ReviewInput


class FeatureExtractor:
    """
    Feature extraction for review classification.

    Combines:
    1. Text features: title + content + tags (for TF-IDF)
    2. Metadata features: numerical and categorical features
    """

    @staticmethod
    def extract_text(review: ReviewInput) -> str:
        """
        Extract and combine text fields for TF-IDF vectorization.

        Args:
            review: Review input data

        Returns:
            Combined text string
        """
        # Combine title, content, and tags
        text_parts = []

        if review.title:
            text_parts.append(review.title.strip())

        if review.content:
            text_parts.append(review.content.strip())

        if review.tags:
            text_parts.append(" ".join(review.tags))

        combined_text = "\n".join(text_parts)

        # Limit length (max 20,000 chars as per spec)
        if len(combined_text) > 20000:
            combined_text = combined_text[:20000]

        return combined_text

    @staticmethod
    def extract_metadata_features(review: ReviewInput) -> Dict[str, Any]:
        """
        Extract metadata features (numerical and categorical).

        Args:
            review: Review input data

        Returns:
            Dictionary of metadata features
        """
        features = {}

        # Numerical features (with defaults)
        features["rating"] = review.rating if review.rating is not None else 0.0
        features["helpful_votes"] = review.helpful_votes
        features["not_helpful_votes"] = review.not_helpful_votes
        features["trust_score"] = review.trust_score

        # Boolean features (as integers)
        features["verified_purchase"] = int(review.verified_purchase)
        features["is_sponsored"] = int(review.is_sponsored)

        # Derived features
        total_votes = features["helpful_votes"] + features["not_helpful_votes"]
        features["total_votes"] = total_votes
        features["helpful_ratio"] = (
            features["helpful_votes"] / total_votes if total_votes > 0 else 0.5
        )

        # Categorical feature: source_platform (one-hot encoded)
        # Platforms: ReviewTrust (internal), Naver, Coupang, 11st, Other
        platforms = ["ReviewTrust", "Naver", "Coupang", "11st"]
        platform = review.source_platform if review.source_platform else "Other"

        for p in platforms:
            features[f"platform_{p}"] = int(platform == p)
        features["platform_Other"] = int(platform not in platforms)

        return features

    @staticmethod
    def metadata_to_vector(metadata: Dict[str, Any]) -> np.ndarray:
        """
        Convert metadata dictionary to numpy array in consistent order.

        Args:
            metadata: Metadata features dictionary

        Returns:
            Numpy array of features
        """
        # Define feature order (must be consistent!)
        feature_order = [
            "rating",
            "helpful_votes",
            "not_helpful_votes",
            "trust_score",
            "verified_purchase",
            "is_sponsored",
            "total_votes",
            "helpful_ratio",
            "platform_ReviewTrust",
            "platform_Naver",
            "platform_Coupang",
            "platform_11st",
            "platform_Other",
        ]

        vector = np.array([metadata.get(key, 0.0) for key in feature_order], dtype=np.float64)
        return vector

    @staticmethod
    def extract_text_statistics(text: str) -> Dict[str, int]:
        """
        Extract text statistics for reason code generation.

        Args:
            text: Combined review text

        Returns:
            Dictionary of text statistics
        """
        stats = {}

        # URL count
        url_pattern = r"https?://\S+|www\.\S+"
        stats["url_count"] = len(re.findall(url_pattern, text, re.IGNORECASE))

        # Promotional keywords count
        promo_keywords = [
            "할인",
            "쿠폰",
            "이벤트",
            "무료",
            "증정",
            "혜택",
            "링크",
            "클릭",
            "가입",
            "추천인",
            "discount",
            "coupon",
            "free",
            "event",
            "promo",
            "link",
            "click",
        ]
        promo_count = 0
        text_lower = text.lower()
        for keyword in promo_keywords:
            promo_count += text_lower.count(keyword)
        stats["promo_keyword_count"] = promo_count

        # Exclamation marks count
        stats["exclamation_count"] = text.count("!")

        # Character count
        stats["char_count"] = len(text)

        return stats


# Global singleton instance
feature_extractor = FeatureExtractor()
