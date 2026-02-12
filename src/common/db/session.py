from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from common.config import settings

# Single engine instance
engine = create_async_engine(
    settings.POSTGRES_URL,
    echo=settings.DB_ECHO,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# Single session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """
    Dependency that provides a database session.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session
