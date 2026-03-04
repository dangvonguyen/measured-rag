"""Create initial model

Revision ID: 001
Revises:
Create Date: 2026-02-12 16:06:37.014942

"""

from collections.abc import Sequence
from typing import Union

import pgvector.sqlalchemy  # type:ignore
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS rag")
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("source_name", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "embedding_vector",
            pgvector.sqlalchemy.vector.VECTOR(dim=1536),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
        schema="rag",
    )
    op.create_index(
        op.f("ix_rag_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
        schema="rag",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_rag_document_chunks_document_id"),
        table_name="document_chunks",
        schema="rag",
    )
    op.drop_table("document_chunks", schema="rag")
    op.execute("DROP SCHEMA IF EXISTS rag")
