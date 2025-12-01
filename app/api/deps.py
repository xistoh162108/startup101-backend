"""
API dependencies for dependency injection.
"""
from typing import AsyncGenerator
from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.core.config import settings
from app.core.exceptions import UnauthorizedException


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Usage:
        @router.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def verify_admin_key(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> str:
    """
    Dependency for verifying admin API key.

    Usage:
        @router.post("/admin/endpoint")
        async def admin_endpoint(admin_key: str = Depends(verify_admin_key)):
            ...

    Raises:
        UnauthorizedException: If admin key is invalid
    """
    if x_admin_key != settings.ADMIN_API_KEY:
        raise UnauthorizedException("Invalid admin API key")

    return x_admin_key
