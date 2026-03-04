"""Update table schemas

Revision ID: 003
Revises: 002
Create Date: 2026-03-05 02:35:51.684792

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, Sequence[str], None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS chat")
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="chat",
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("USER", "ASSISTANT", name="message_role", schema="chat"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["chat.conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="chat",
    )
    op.create_index(
        op.f("ix_chat_messages_conversation_id"),
        "messages",
        ["conversation_id"],
        unique=False,
        schema="chat",
    )
    op.create_table(
        "rag_responses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column(
            "retrieved_chunks", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["chat.messages.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", name="uq_rag_response_message"),
        schema="chat",
    )
    op.drop_table("rag_responses", schema="rag")
    op.drop_index(
        op.f("ix_rag_messages_conversation_id"), table_name="messages", schema="rag"
    )
    op.drop_table("messages", schema="rag")
    op.drop_table("conversations", schema="rag")
    op.execute("DROP TYPE IF EXISTS rag.messagerole")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("conversations_pkey")),
        schema="rag",
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("conversation_id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM("USER", "ASSISTANT", name="messagerole", schema="rag"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("content", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["rag.conversations.id"],
            name=op.f("messages_conversation_id_fkey"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("messages_pkey")),
        schema="rag",
    )
    op.create_index(
        op.f("ix_rag_messages_conversation_id"),
        "messages",
        ["conversation_id"],
        unique=False,
        schema="rag",
    )
    op.create_table(
        "rag_responses",
        sa.Column("id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("message_id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("query", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "retrieved_chunks",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("prompt_tokens", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "completion_tokens", sa.INTEGER(), autoincrement=False, nullable=False
        ),
        sa.Column(
            "latency_ms",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["rag.messages.id"],
            name=op.f("rag_responses_message_id_fkey"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("rag_responses_pkey")),
        sa.UniqueConstraint(
            "message_id",
            name=op.f("uq_rag_response_message"),
            postgresql_include=[],
            postgresql_nulls_not_distinct=False,
        ),
        schema="rag",
    )
    op.drop_table("rag_responses", schema="chat")
    op.drop_index(
        op.f("ix_chat_messages_conversation_id"), table_name="messages", schema="chat"
    )
    op.drop_table("messages", schema="chat")
    op.drop_table("conversations", schema="chat")
    op.execute("DROP TYPE IF EXISTS chat.message_role")
    op.execute("DROP SCHEMA IF EXISTS chat")
