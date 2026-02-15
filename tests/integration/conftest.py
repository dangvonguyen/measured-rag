from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from pydantic import PostgresDsn
from sqlalchemy import NullPool, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from common.config import settings
from common.db.base import Base
from rag.db import models  # noqa: F401
from rag.services.vector_store import PGVectorStore

_TEST_DB_NAME = f"{settings.POSTGRES_DB}_test"
_TEST_DB_URL = PostgresDsn.build(
    scheme="postgresql+asyncpg",
    username=settings.POSTGRES_USER,
    password=settings.POSTGRES_PASSWORD,
    host=settings.POSTGRES_HOST,
    port=settings.POSTGRES_PORT,
    path=_TEST_DB_NAME,
).encoded_string()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def test_database() -> AsyncGenerator[None]:
    """Create test DB before session, drop it after."""
    admin_engine = create_async_engine(
        settings.POSTGRES_URL, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin_engine.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DB_NAME}"))
        await conn.execute(text(f"CREATE DATABASE {_TEST_DB_NAME}"))
    await admin_engine.dispose()

    yield

    admin_engine = create_async_engine(
        settings.POSTGRES_URL, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin_engine.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_TEST_DB_NAME}"))
    await admin_engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def db_schema(test_database: None) -> AsyncGenerator[None]:  # noqa: ARG001
    """Create tables once per session, drop them on teardown."""
    engine = create_async_engine(_TEST_DB_URL, poolclass=NullPool)

    schemas = {t.schema for t in Base.metadata.tables.values() if t.schema}
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for schema in schemas:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    yield

    engine = create_async_engine(_TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_schema: None) -> AsyncGenerator[AsyncSession]:  # noqa: ARG001
    """Function-scoped session with transactional rollback."""
    engine = create_async_engine(_TEST_DB_URL, poolclass=NullPool)
    async with engine.connect() as conn:
        transaction = await conn.begin()
        session = AsyncSession(conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    await engine.dispose()


@pytest.fixture
def vector_store(db_session: AsyncSession) -> PGVectorStore:
    return PGVectorStore(db_session)
