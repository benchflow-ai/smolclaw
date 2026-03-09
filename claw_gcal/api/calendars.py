"""Calendar list and calendar resource endpoints."""

from __future__ import annotations

import hashlib
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, User

from .deps import get_db, resolve_actor_user_id, resolve_user_id
from .schemas import (
    CalendarInsertRequest,
    CalendarListEntry,
    CalendarListResponse,
    CalendarResource,
    ConferenceProperties,
    ReminderOverride,
    NotificationRule,
    NotificationSettings,
)

router = APIRouter()


def _etag_for_calendar(calendar: Calendar) -> str:
    raw = f"{calendar.id}:{calendar.summary}:{calendar.description}:{calendar.timezone}:{calendar.selected}"
    return f'"{hashlib.md5(raw.encode("utf-8")).hexdigest()}"'


def _default_notification_settings() -> NotificationSettings:
    return NotificationSettings(
        notifications=[
            NotificationRule(type="eventCreation", method="email"),
            NotificationRule(type="eventChange", method="email"),
            NotificationRule(type="eventCancellation", method="email"),
            NotificationRule(type="eventResponse", method="email"),
        ]
    )


def _to_calendar_entry(calendar: Calendar) -> CalendarListEntry:
    return CalendarListEntry(
        etag=_etag_for_calendar(calendar),
        id=calendar.id,
        summary=calendar.summary,
        timeZone=calendar.timezone,
        accessRole=calendar.access_role,
        primary=calendar.is_primary,
        selected=calendar.selected,
        colorId="14" if calendar.is_primary else "9",
        backgroundColor="#9fe1e7" if calendar.is_primary else "#7ae7bf",
        foregroundColor="#000000",
        conferenceProperties=ConferenceProperties(),
        defaultReminders=[ReminderOverride(method="popup", minutes=10)],
        notificationSettings=_default_notification_settings(),
    )


def _to_calendar_resource(calendar: Calendar, *, data_owner: str | None = None) -> CalendarResource:
    return CalendarResource(
        etag=_etag_for_calendar(calendar),
        id=calendar.id,
        summary=calendar.summary,
        timeZone=calendar.timezone,
        conferenceProperties=ConferenceProperties(),
        dataOwner=data_owner,
    )


def _resolve_calendar(db: Session, user_id: str, calendar_id: str) -> Calendar:
    if calendar_id == "primary":
        cal = db.query(Calendar).filter(
            Calendar.user_id == user_id,
            Calendar.is_primary.is_(True),
        ).first()
    else:
        cal = db.query(Calendar).filter(
            Calendar.id == calendar_id,
            Calendar.user_id == user_id,
        ).first()

    if not cal:
        raise HTTPException(404, "Calendar not found")
    return cal


@router.get(
    "/users/{userId}/calendarList",
    response_model=CalendarListResponse,
    response_model_exclude_none=True,
)
def calendar_list(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendars = (
        db.query(Calendar)
        .filter(Calendar.user_id == _user_id)
        .order_by(Calendar.is_primary.desc(), Calendar.summary.asc())
        .all()
    )

    items = [_to_calendar_entry(c) for c in calendars]

    list_etag_raw = "|".join([item.etag for item in items])
    next_sync_token = hashlib.md5(
        f"{_user_id}:{len(items)}:{list_etag_raw}".encode("utf-8")
    ).hexdigest()
    return CalendarListResponse(
        etag=f'"{hashlib.md5(list_etag_raw.encode("utf-8")).hexdigest()}"',
        items=items,
        nextSyncToken=next_sync_token,
    )


@router.get(
    "/users/{userId}/calendarList/{calendarId}",
    response_model=CalendarListEntry,
    response_model_exclude_none=True,
)
def calendar_list_get(
    userId: str,
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    return _to_calendar_entry(calendar)


@router.get(
    "/calendars/{calendarId}",
    response_model=CalendarResource,
    response_model_exclude_none=True,
)
def calendars_get(
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    return _to_calendar_resource(calendar)


@router.post("/calendars", response_model=CalendarResource, response_model_exclude_none=True)
def calendars_insert(
    body: CalendarInsertRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    summary = body.summary.strip()
    if not summary:
        raise HTTPException(400, "Missing required field: summary")

    calendar = Calendar(
        id=f"cal_{uuid.uuid4().hex[:12]}",
        user_id=_user_id,
        summary=summary,
        description=body.description,
        timezone=body.timeZone,
        access_role="owner",
        is_primary=False,
        selected=body.selected,
    )
    db.add(calendar)
    db.commit()
    db.refresh(calendar)

    user = db.query(User).filter(User.id == _user_id).first()
    return _to_calendar_resource(calendar, data_owner=user.email_address if user else None)


@router.delete("/calendars/{calendarId}", status_code=status.HTTP_204_NO_CONTENT)
def calendars_delete(
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    if calendar.is_primary:
        raise HTTPException(400, "Primary calendar cannot be deleted")
    db.delete(calendar)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
