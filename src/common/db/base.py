from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all ORM models."""

    pass


class CreatedAtMixin:
    """Add created_at to models."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class UpdatedAtMixin:
    """Add updated_at to models."""

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TimestampMixin(CreatedAtMixin, UpdatedAtMixin):
    """Add created_at, updated_at to models."""

    pass


class SoftDeleteMixin:
    """Add soft delete capability."""

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
