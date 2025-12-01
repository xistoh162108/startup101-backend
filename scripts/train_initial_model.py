"""
Script to train an initial model.

This script:
1. Loads labeled data from database
2. Trains a TF-IDF + SGD model
3. Saves the model artifact
4. Registers and activates the model

Usage:
    python scripts/train_initial_model.py
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import AsyncSessionLocal, init_db
from app.services.training import training_service


async def train_initial_model():
    """Train and activate an initial model."""
    print("=" * 60)
    print("Training Initial Model")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        try:
            # Step 1: Load training data
            print("\n[1/6] Loading training data from database...")
            reviews, labels, train_data_snapshot = await training_service.load_training_data(db)

            print(f"  ✓ Loaded {len(labels)} labeled reviews")
            print(f"    - Ad reviews: {sum(labels)}")
            print(f"    - Non-ad reviews: {len(labels) - sum(labels)}")

            # Step 2: Train model
            print("\n[2/6] Training model...")
            vectorizer, classifier, metrics = training_service.train_model(reviews, labels)

            print(f"  ✓ Model trained successfully")
            print(f"    - Accuracy: {metrics['accuracy']:.3f}")
            print(f"    - Precision: {metrics['precision']:.3f}")
            print(f"    - Recall: {metrics['recall']:.3f}")
            print(f"    - F1 Score: {metrics['f1']:.3f}")

            # Step 3: Generate model version
            model_version = f"clf-{datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ')}"
            print(f"\n[3/6] Model version: {model_version}")

            # Step 4: Save model artifact
            print("\n[4/6] Saving model artifact...")
            artifact_uri = training_service.save_model_artifact(
                vectorizer=vectorizer,
                classifier=classifier,
                metadata={
                    "version": model_version,
                    "model_type": "tfidf_sgd",
                    "threshold": 0.8,
                    "trained_at": datetime.utcnow().isoformat(),
                    "trained_by": "initial_script",
                },
                model_version=model_version,
            )
            print(f"  ✓ Artifact saved to: {artifact_uri}")

            # Step 5: Register model
            print("\n[5/6] Registering model in database...")
            await training_service.register_model(
                db=db,
                model_version=model_version,
                model_type="tfidf_sgd",
                artifact_uri=artifact_uri,
                metrics=metrics,
                train_data_snapshot=train_data_snapshot,
            )
            print(f"  ✓ Model registered")

            # Step 6: Activate model
            print("\n[6/6] Activating model...")
            success = await training_service.activate_model(db, model_version)

            if success:
                print(f"  ✓ Model activated successfully")
            else:
                print(f"  ✗ Failed to activate model")
                return

            await db.commit()

            print("\n" + "=" * 60)
            print("SUCCESS! Initial model is ready to use")
            print("=" * 60)
            print(f"\nModel Version: {model_version}")
            print(f"Model Type: tfidf_sgd")
            print(f"Training Samples: {len(labels)}")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"\nYou can now start the API server and classify reviews!")

        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


async def main():
    """Main entry point."""
    print("Initializing database...")
    await init_db()
    print("✓ Database initialized\n")

    await train_initial_model()


if __name__ == "__main__":
    asyncio.run(main())
