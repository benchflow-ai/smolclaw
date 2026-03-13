"""Calendar ACL rule model."""

from __future__ import annotations

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class AclRule(Base):
    __tablename__ = "acl_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    calendar_id: Mapped[str] = mapped_column(
        ForeignKey("calendars.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scope_type: Mapped[str] = mapped_column(String, default="user")
    scope_value: Mapped[str] = mapped_column(String, default="")
    role: Mapped[str] = mapped_column(String, default="reader")
    etag: Mapped[str] = mapped_column(String, default="")

    calendar = relationship("Calendar", back_populates="acl_rules")
