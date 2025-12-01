"""
Script to create sample review data with admin labels for testing.

This creates dummy data in the database so you can train an initial model.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import AsyncSessionLocal, init_db
from app.models import Review, AdminLabel, LabelType


async def create_sample_data():
    """Create sample reviews and admin labels."""
    print("Creating sample data...")

    async with AsyncSessionLocal() as db:
        # Sample ad reviews
        ad_reviews = [
            {
                "title": "최고의 제품! 할인 링크 남깁니다",
                "content": "정말 좋아요! 쿠폰 받으세요!! http://bit.ly/coupon123 이벤트 중이니 서두르세요!!!",
                "category": "전자기기",
                "rating": 5.0,
                "verified_purchase": False,
                "trust_score": 30,
            },
            {
                "title": "추천합니다! 링크 클릭하세요",
                "content": "제품 정말 좋습니다. 링크로 구매하시면 할인 받으실 수 있어요! www.example.com/promo",
                "category": "화장품",
                "rating": 5.0,
                "verified_purchase": False,
                "trust_score": 25,
            },
            {
                "title": "대박 이벤트!!!",
                "content": "지금 가입하면 무료 증정!! 놓치지 마세요!!! 링크: http://promo.example.com",
                "category": "건강식품",
                "rating": 5.0,
                "verified_purchase": False,
                "trust_score": 20,
            },
        ]

        # Sample non-ad reviews
        non_ad_reviews = [
            {
                "title": "괜찮은 제품입니다",
                "content": "가격대비 나쁘지 않아요. 배송도 빠르고 포장 상태도 좋았습니다. 다만 설명서가 좀 부실한 것 같네요.",
                "category": "전자기기",
                "rating": 4.0,
                "verified_purchase": True,
                "trust_score": 85,
            },
            {
                "title": "기대 이하였어요",
                "content": "생각보다 품질이 별로였습니다. 가격은 저렴하지만 그만큼의 가치인 것 같아요.",
                "category": "생활용품",
                "rating": 2.5,
                "verified_purchase": True,
                "trust_score": 78,
            },
            {
                "title": "만족스러운 구매",
                "content": "실제로 사용해보니 리뷰들이 맞네요. 내구성도 좋고 디자인도 마음에 듭니다.",
                "category": "패션",
                "rating": 4.5,
                "verified_purchase": True,
                "trust_score": 92,
            },
        ]

        # Create reviews
        created_reviews = []

        print(f"Creating {len(ad_reviews)} ad reviews...")
        for data in ad_reviews:
            review = Review(**data)
            db.add(review)
            await db.flush()
            created_reviews.append((review, True))  # True = ad

        print(f"Creating {len(non_ad_reviews)} non-ad reviews...")
        for data in non_ad_reviews:
            review = Review(**data)
            db.add(review)
            await db.flush()
            created_reviews.append((review, False))  # False = not_ad

        # Create admin labels
        print(f"Creating admin labels...")
        for review, is_ad in created_reviews:
            label = AdminLabel(
                review_id=review.id,
                label=LabelType.AD if is_ad else LabelType.NOT_AD,
                admin_id="system",
                comment="Initial sample data",
            )
            db.add(label)

        await db.commit()

        print(f"✓ Created {len(created_reviews)} reviews with labels")
        print(f"  - {len(ad_reviews)} ad reviews")
        print(f"  - {len(non_ad_reviews)} non-ad reviews")
        print("\nYou can now train your first model!")


async def main():
    """Main entry point."""
    print("Initializing database...")
    await init_db()
    print("✓ Database initialized\n")

    await create_sample_data()


if __name__ == "__main__":
    asyncio.run(main())
