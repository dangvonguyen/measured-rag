import asyncio
import re
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.config import settings
from common.db import base

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target_metadata to Base.metadata (includes all imported models)
target_metadata = base.Base.metadata


def get_url() -> str:
    return settings.POSTGRES_URL


def get_next_revision_number() -> str:
    """
    Generate the next zero-padded numeric revision ID (001, 002, etc.).
    """
    versions_dir = Path(__file__).resolve().parent / "versions"

    if not versions_dir.exists():
        return "001"

    revision_numbers = []

    for file in versions_dir.iterdir():
        match = re.match(r"^(\d+)_", file.name)
        if match:
            revision_numbers.append(int(match.group(1)))

    if not revision_numbers:
        return "001"

    next_number = max(revision_numbers) + 1
    return f"{next_number:03d}"


def process_revision_directives(context, revision, directives) -> None:  # type: ignore # noqa
    """
    Override Alembic's revision ID with sequential numbers.
    """
    if not directives:
        return

    script = directives[0]

    # Override revision ID with sequential number
    script.rev_id = get_next_revision_number()


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (generate SQL without executing).
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (execute against live DB).
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
