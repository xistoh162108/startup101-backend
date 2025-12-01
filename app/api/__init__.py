"""
API routers package.
"""
from app.api import health, classification, feedback

__all__ = [
    "health",
    "classification",
    "feedback",
]
