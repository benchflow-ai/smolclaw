"""Calendar event endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, Event

from .deps import get_db, resolve_actor_user_id
from .schemas import (
    EventActor,
    EventDateTime,
    EventListResponse,
    EventPatchRequest,
    EventReminders,
    EventResource,
    EventWriteRequest,
    ReminderOverride,
)

router = APIRouter()


def _parse_rfc3339(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid RFC3339 datetime: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_event_etag(event: Event) -> str:
    raw = f"{event.id}:{event.summary}:{event.status}:{event.updated_at.isoformat()}:{event.sequence}"
    return f'"{hashlib.md5(raw.encode("utf-8")).hexdigest()}"'


def _resolve_calendar_for_actor(db: Session, calendar_id: str, actor_user_id: str) -> Calendar:
    if calendar_id == "primary":
        calendar = db.query(Calendar).filter(
            Calendar.user_id == actor_user_id,
            Calendar.is_primary.is_(True),
        ).first()
    else:
        calendar = db.query(Calendar).filter(
            Calendar.id == calendar_id,
            Calendar.user_id == actor_user_id,
        ).first()

    if not calendar:
        raise HTTPException(404, "Calendar not found")
    return calendar


def _to_event_resource(event: Event) -> EventResource:
    tz = event.calendar.timezone if event.calendar else "UTC"
    actor_email = event.user.email_address if event.user else ""

    return EventResource(
        etag=event.etag or _compute_event_etag(event),
        id=event.id,
        status=event.status,
        htmlLink=f"https://www.google.com/calendar/event?eid={event.id}",
        created=_iso(event.created_at),
        updated=_iso(event.updated_at),
        summary=event.summary or None,
        description=event.description or None,
        location=event.location or None,
        iCalUID=event.i_cal_uid,
        sequence=event.sequence,
        start=EventDateTime(dateTime=_iso(event.start_dt), timeZone=tz),
        end=EventDateTime(dateTime=_iso(event.end_dt), timeZone=tz),
        creator=EventActor(email=actor_email, self=True) if actor_email else None,
        organizer=EventActor(email=actor_email, self=True) if actor_email else None,
        reminders=EventReminders(useDefault=True),
        eventType="default",
    )


@router.get(
    "/calendars/{calendarId}/events",
    response_model=EventListResponse,
    response_model_exclude_none=True,
)
def events_list(
    calendarId: str,
    maxResults: int = Query(250, ge=1, le=2500),
    q: str | None = Query(None),
    timeMin: str | None = Query(None),
    timeMax: str | None = Query(None),
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    query = db.query(Event).filter(Event.calendar_id == calendar.id)

    if q:
        query = query.filter((Event.summary.contains(q)) | (Event.description.contains(q)))
    if timeMin:
        query = query.filter(Event.end_dt >= _parse_rfc3339(timeMin))
    if timeMax:
        query = query.filter(Event.start_dt <= _parse_rfc3339(timeMax))

    total_count = query.count()
    events = query.order_by(Event.start_dt.asc()).limit(maxResults).all()
    items = [_to_event_resource(e) for e in events]

    list_etag_raw = "|".join([item.etag for item in items])
    updated = _iso(events[-1].updated_at) if events else _iso(datetime.now(timezone.utc))
    token_seed = f"{calendar.id}:{len(events)}:{updated}"
    next_sync_token = _md5_hex(token_seed)
    next_page_token = str(maxResults) if total_count > maxResults else None

    return EventListResponse(
        etag=f'"{hashlib.md5(list_etag_raw.encode("utf-8")).hexdigest()}"',
        summary=calendar.summary,
        description=calendar.description or "",
        timeZone=calendar.timezone,
        updated=updated,
        accessRole=calendar.access_role,
        defaultReminders=[ReminderOverride(method="popup", minutes=10)],
        nextPageToken=next_page_token,
        nextSyncToken=None if next_page_token else next_sync_token,
        items=items,
    )


@router.get(
    "/calendars/{calendarId}/events/{eventId}",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_get(
    calendarId: str,
    eventId: str,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    event = db.query(Event).filter(
        Event.id == eventId,
        Event.calendar_id == calendar.id,
    ).first()
    if not event:
        raise HTTPException(404, "Event not found")
    return _to_event_resource(event)


@router.post(
    "/calendars/{calendarId}/events",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_insert(
    calendarId: str,
    body: EventWriteRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    start_dt = _parse_rfc3339(body.start.dateTime)
    end_dt = _parse_rfc3339(body.end.dateTime)
    if end_dt <= start_dt:
        raise HTTPException(400, "Event end must be after start")

    now = datetime.now(timezone.utc)
    event = Event(
        id=f"evt_{uuid.uuid4().hex[:12]}",
        calendar_id=calendar.id,
        user_id=_actor_user_id,
        summary=body.summary,
        description=body.description,
        location=body.location,
        status="confirmed",
        start_dt=start_dt,
        end_dt=end_dt,
        attendees_json="[]",
        created_at=now,
        updated_at=now,
        etag="",
        i_cal_uid=f"{uuid.uuid4().hex}@google.com",
        sequence=0,
    )
    event.etag = _compute_event_etag(event)

    db.add(event)
    db.commit()
    db.refresh(event)

    return _to_event_resource(event)


@router.patch(
    "/calendars/{calendarId}/events/{eventId}",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_patch(
    calendarId: str,
    eventId: str,
    body: EventPatchRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    event = db.query(Event).filter(
        Event.id == eventId,
        Event.calendar_id == calendar.id,
    ).first()
    if not event:
        raise HTTPException(404, "Event not found")

    if body.summary is not None:
        event.summary = body.summary
    if body.description is not None:
        event.description = body.description
    if body.location is not None:
        event.location = body.location
    if body.status is not None:
        event.status = body.status
    if body.start is not None:
        event.start_dt = _parse_rfc3339(body.start.dateTime)
    if body.end is not None:
        event.end_dt = _parse_rfc3339(body.end.dateTime)
    if event.end_dt <= event.start_dt:
        raise HTTPException(400, "Event end must be after start")

    event.sequence += 1
    event.updated_at = datetime.now(timezone.utc)
    event.etag = _compute_event_etag(event)

    db.commit()
    db.refresh(event)

    return _to_event_resource(event)


@router.delete("/calendars/{calendarId}/events/{eventId}", status_code=status.HTTP_204_NO_CONTENT)
def events_delete(
    calendarId: str,
    eventId: str,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    event = db.query(Event).filter(
        Event.id == eventId,
        Event.calendar_id == calendar.id,
    ).first()
    if not event:
        raise HTTPException(404, "Event not found")

    db.delete(event)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _md5_hex(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return digest
