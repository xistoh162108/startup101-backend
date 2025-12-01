"""
Core configuration module using Pydantic Settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/review_classifier"

    # Redis
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"

    # Model Storage
    MODEL_ARTIFACT_DIR: str = "./models"

    # API Configuration
    AD_THRESHOLD: float = 0.8
    ADMIN_API_KEY: str = "change-me-in-production"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Logging
    LOG_LEVEL: str = "INFO"

    # Application
    APP_NAME: str = "Review Classification API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Batch limits
    MAX_BATCH_SIZE: int = 100
    MAX_TEXT_LENGTH: int = 20000

    # Training
    MIN_LABELS_FOR_TRAINING: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Global settings instance
settings = Settings()
