"""Calendar model."""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Calendar(Base):
    __tablename__ = "calendars"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str] = mapped_column(String, default="")
    timezone: Mapped[str] = mapped_column(String, default="America/Los_Angeles")
    access_role: Mapped[str] = mapped_column(String, default="owner")
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    selected: Mapped[bool] = mapped_column(Boolean, default=True)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_override: Mapped[str] = mapped_column(String, default="")
    auto_accept_invitations: Mapped[bool] = mapped_column(Boolean, default=False)
    color_id: Mapped[str] = mapped_column(String, default="9")

    user = relationship("User", back_populates="calendars")
    events = relationship("Event", back_populates="calendar", cascade="all, delete-orphan")
    acl_rules = relationship("AclRule", back_populates="calendar", cascade="all, delete-orphan")
