"""Calendar event model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    calendar_id: Mapped[str] = mapped_column(ForeignKey("calendars.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="confirmed")
    start_dt: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_dt: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attendees_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    etag: Mapped[str] = mapped_column(String, default="")
    i_cal_uid: Mapped[str] = mapped_column(String, default="")
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    recurrence_json: Mapped[str] = mapped_column(Text, default="[]")
    recurring_event_id: Mapped[str] = mapped_column(String, default="")
    original_start_time: Mapped[str] = mapped_column(String, default="")

    calendar = relationship("Calendar", back_populates="events")
    user = relationship("User", back_populates="events")
