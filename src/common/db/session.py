from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from common.config import get_settings


@lru_cache(maxsize=1)
def _get_engine() -> AsyncEngine:
    """
    Create database engine (cached).
    """
    s = get_settings()
    return create_async_engine(
        s.POSTGRES_URL,
        echo=s.DB_ECHO,
        pool_size=s.DB_POOL_SIZE,
        max_overflow=s.DB_MAX_OVERFLOW,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Create session factory (cached).
    """
    return async_sessionmaker(_get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """
    Dependency that provides a database session.
    """
    async with _get_session_factory()() as session:
        async with session.begin():
            yield session
