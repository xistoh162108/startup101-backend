# Model Artifacts Directory

This directory stores trained model artifacts (.joblib files).

## Structure

Each model is saved as a single `.joblib` file containing:
- `vectorizer`: TfidfVectorizer for text features
- `classifier`: SGDClassifier for classification
- `metadata`: Model version, type, threshold, training info

## Naming Convention

Model files follow this naming pattern:
```
{model_version}.joblib
```

Example:
```
clf-2025-12-01T020000Z.joblib
```

## Loading Models

Models are automatically loaded by the ClassifierService based on the active model version in the database (model_state table).

## Creating Your First Model

To create a dummy model for testing, run the training script:
```bash
python scripts/train_initial_model.py
```

This will be implemented in Step 5.
