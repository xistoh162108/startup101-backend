# Quick Start Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- (Optional) Redis for distributed locks

## Step 1: Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install fastapi uvicorn sqlalchemy asyncpg pydantic pydantic-settings \
    scikit-learn joblib numpy scipy python-dotenv
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
nano .env
```

Required environment variables:
```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/review_classifier
ADMIN_API_KEY=your-secret-admin-key-here
```

## Step 3: Initialize Database

The database will be auto-initialized on first startup, but you can also run:

```bash
# This will create all tables
python -c "import asyncio; from app.db.database import init_db; asyncio.run(init_db())"
```

## Step 4: Create Sample Data (Optional)

```bash
# Create sample reviews with labels for testing
python scripts/create_sample_data.py
```

This creates:
- 3 ad reviews (with admin labels)
- 3 non-ad reviews (with admin labels)

## Step 5: Train Initial Model

```bash
# Train your first model
python scripts/train_initial_model.py
```

This will:
1. Load labeled data from database
2. Train TF-IDF + SGD classifier
3. Save model as .joblib file
4. Register model in database
5. Activate model for serving

Expected output:
```
✓ Model trained successfully
  - Accuracy: 0.XXX
  - Precision: 0.XXX
  - Recall: 0.XXX
  - F1 Score: 0.XXX

Model Version: clf-2025-XX-XXTXXXXXX
```

## Step 6: Start API Server

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Server will be available at:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Step 7: Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

### Classify a Review

```bash
curl -X POST http://localhost:8000/api/reviews/classify \
  -H "Content-Type: application/json" \
  -d '{
    "review": {
      "title": "Great product with discount!",
      "content": "Click here for coupon: http://bit.ly/discount",
      "rating": 5.0,
      "verifiedPurchase": false,
      "helpfulVotes": 0,
      "notHelpfulVotes": 0,
      "trustScore": 30
    },
    "options": {
      "persist": false,
      "includeReasons": true
    }
  }'
```

Expected response:
```json
{
  "requestId": "...",
  "reviewId": "...",
  "isAdLike": true,
  "adScore": 0.85,
  "threshold": 0.8,
  "modelVersion": "clf-2025-XX-XXTXXXXXX",
  "reasons": ["contains_url", "contains_promo_keywords", "no_verified_purchase"]
}
```

### Submit Admin Label

```bash
curl -X POST http://localhost:8000/api/admin/reviews/label \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-secret-admin-key-here" \
  -d '{
    "reviewId": "review-id-here",
    "adminId": "admin1",
    "label": "ad",
    "comment": "Contains promotional links"
  }'
```

### Trigger Model Training

```bash
curl -X POST http://localhost:8000/api/admin/model/train \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-secret-admin-key-here" \
  -d '{
    "adminId": "admin1",
    "mode": "full",
    "minNewLabels": 6,
    "activateIfBetter": true
  }'
```

### Get Model Info

```bash
curl http://localhost:8000/api/model/info
```

## Troubleshooting

### Database Connection Error

Check your `DATABASE_URL` in `.env`:
```bash
# PostgreSQL must be running
psql -U your_user -d postgres -c "SELECT version();"

# Create database if needed
createdb review_classifier
```

### Model Not Ready Error (503)

Train a model first:
```bash
python scripts/create_sample_data.py
python scripts/train_initial_model.py
```

### Import Errors

Make sure you're in the backend directory and dependencies are installed:
```bash
cd backend
pip install -r requirements.txt  # or poetry install
```

## Next Steps

1. **Add More Training Data**: Submit more admin labels via API or database
2. **Retrain Model**: Use `/api/admin/model/train` to improve model
3. **Integrate with Frontend**: Connect to ReviewTrust frontend
4. **Production Deployment**: Use Gunicorn, setup PostgreSQL replication, add Redis for locks

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

```
┌─────────────────┐
│   Frontend      │
│  (ReviewTrust)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│   /classify     │◄─── DummyModel (fallback)
│   /feedback     │
│   /admin/*      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  ML    │ │PostgreSQL│
│ Model  │ │  + Redis │
└────────┘ └──────────┘
```

For more details, see [README.md](README.md)
