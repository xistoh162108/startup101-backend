# Review Classification API Backend

## Overview

A production-ready FastAPI backend for classifying review advertisements using machine learning.

## Features

- **ML-powered classification**: TF-IDF + SGD Classifier for ad detection
- **Model versioning**: Hot-swappable models with version management
- **Admin labeling**: Gold standard labels from administrators
- **User feedback**: Weak signals from user votes and reports
- **Async PostgreSQL**: High-performance async database operations
- **Graceful error handling**: Never crashes, always returns proper error responses

## Quick Start

### Prerequisites

- **Python 3.11** (recommended: conda)
- **Poetry** (Python dependency manager)
- **Docker** (for PostgreSQL)

### Step 0: Setup Python 3.11 Environment

```bash
# Create and activate conda environment
conda create -n startup101 python=3.11 -y
conda activate startup101

# Upgrade pip and install Poetry
python -m pip install -U pip
python -m pip install poetry
```

### Step 1: Start PostgreSQL with Docker

```bash
# Start Docker Desktop
open -a Docker

# Remove existing container if any
docker rm -f startup101-postgres 2>/dev/null || true

# Run PostgreSQL container
docker run --name startup101-postgres \
  -e POSTGRES_USER=app \
  -e POSTGRES_PASSWORD=app \
  -e POSTGRES_DB=reviewdb \
  -p 5432:5432 \
  -d postgres:16

# Verify connection
docker exec -it startup101-postgres psql -U app -d reviewdb -c "SELECT 1;"
```

### Step 2: Configure Environment

```bash
cd backend

# Copy environment template
cp .env.example .env
```

Edit `.env` file:
```bash
DATABASE_URL=postgresql+asyncpg://app:app@127.0.0.1:5432/reviewdb
ADMIN_API_KEY=your-secret-admin-key-here
```

**Important:** Clear any existing DATABASE_URL environment variable:
```bash
unset DATABASE_URL
```

### Step 3: Install Dependencies

```bash
# Use current Python from conda environment
poetry env use "$(which python)"

# Install all dependencies
poetry install
```

### Step 4: Create Sample Data & Train Initial Model

```bash
# Create sample reviews with admin labels (6 samples: 3 ad, 3 non-ad)
poetry run python scripts/create_sample_data.py

# Train your first model
poetry run python scripts/train_initial_model.py
```

Expected output:
```
[1/6] Loading training data...
  ✓ Loaded 6 labeled reviews
[2/6] Training model...
  ✓ Model trained successfully
    - Accuracy: 1.000
[6/6] Activating model...
  ✓ Model activated successfully

SUCCESS! Initial model is ready to use
Model Version: clf-2025-12-01T123456Z
```

### Step 5: Start API Server

```bash
# Development mode (with auto-reload)
poetry run uvicorn app.main:app --reload

# Production mode
poetry run gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Server will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI** (Try API in browser): http://localhost:8000/docs
- **ReDoc** (Read-only docs): http://localhost:8000/redoc

### Download OpenAPI Documentation

**Option 1: Download OpenAPI JSON (Recommended)**

This is the "source of truth" for your API documentation.

```bash
# Download OpenAPI specification
curl http://localhost:8000/openapi.json -o openapi.json

# You can convert to YAML if needed
# (requires yq: brew install yq)
yq -P openapi.json > openapi.yaml
```

**Option 2: Save Swagger HTML**

This works online but may break offline due to CDN dependencies.

```bash
curl http://localhost:8000/docs -o swagger.html
```

**Option 3: Generate Static Documentation (Offline-ready)**

Create a single HTML file with embedded assets for offline use.

```bash
# Using Redocly (install once)
npm install -g @redocly/cli

# Generate static documentation
redocly build-docs openapi.json -o redoc-static.html

# Now you can open redoc-static.html in any browser, even offline!
```

This creates a beautiful, fully self-contained documentation page.

## Architecture

```
app/
├── main.py              # FastAPI entry point
├── api/                 # API routers
├── schemas/             # Pydantic models
├── models/              # SQLAlchemy models
├── services/            # Business logic
│   ├── classifier.py    # ML model service
│   ├── feedback.py      # Feedback handling
│   └── training.py      # Model training
├── core/                # Core utilities
└── db/                  # Database setup
```

## Development

```bash
# Run tests
pytest

# Format code
black app/

# Lint
ruff check app/

# Type check
mypy app/
```
