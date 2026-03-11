"""Calendar event endpoints."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, Event
from claw_gcal.state.channels import channel_registry

from .deps import get_db, resolve_actor_user_id
from .schemas import (
    ChannelRequest,
    ChannelResponse,
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


def _parse_event_datetime(dt_value: EventDateTime, tz_name: str) -> datetime:
    if dt_value.dateTime:
        return _parse_rfc3339(dt_value.dateTime)
    if dt_value.date:
        # Store all-day values as midnight in calendar timezone (serialized in UTC).
        day = datetime.fromisoformat(dt_value.date).date()
        return datetime.combine(day, time.min, tzinfo=timezone.utc)
    raise HTTPException(400, "Missing event datetime value")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_event_etag(event: Event) -> str:
    raw = f"{event.id}:{event.summary}:{event.status}:{event.updated_at.isoformat()}:{event.sequence}"
    return f'"{hashlib.md5(raw.encode("utf-8")).hexdigest()}"'


def _md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


def _deserialize_recurrence(raw: str) -> list[str] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) and data else None


def _to_event_resource(event: Event) -> EventResource:
    tz = event.calendar.timezone if event.calendar else "UTC"
    actor_email = event.user.email_address if event.user else ""

    recurrence = _deserialize_recurrence(event.recurrence_json)
    original_start = (
        EventDateTime(dateTime=event.original_start_time, timeZone=tz)
        if event.original_start_time
        else None
    )

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
        recurrence=recurrence,
        recurringEventId=event.recurring_event_id or None,
        originalStartTime=original_start,
    )


def _query_events_base(
    *,
    db: Session,
    calendar: Calendar,
    q: str | None,
    timeMin: str | None,
    timeMax: str | None,
    showDeleted: bool,
) -> tuple[list[Event], int]:
    query = db.query(Event).filter(Event.calendar_id == calendar.id)
    if not showDeleted:
        query = query.filter(Event.status != "cancelled")
    if q:
        query = query.filter((Event.summary.contains(q)) | (Event.description.contains(q)))
    if timeMin:
        query = query.filter(Event.end_dt >= _parse_rfc3339(timeMin))
    if timeMax:
        query = query.filter(Event.start_dt <= _parse_rfc3339(timeMax))
    total_count = query.count()
    return query, total_count


@router.get(
    "/calendars/{calendarId}/events",
    response_model=EventListResponse,
    response_model_exclude_none=True,
)
def events_list(
    calendarId: str,
    maxResults: int = Query(250, ge=1, le=2500),
    pageToken: str | None = Query(None),
    syncToken: str | None = Query(None),
    q: str | None = Query(None),
    timeMin: str | None = Query(None),
    timeMax: str | None = Query(None),
    orderBy: str | None = Query(None),
    showDeleted: bool = Query(False),
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    query, total_count = _query_events_base(
        db=db,
        calendar=calendar,
        q=q,
        timeMin=timeMin,
        timeMax=timeMax,
        showDeleted=showDeleted,
    )

    if orderBy == "updated":
        query = query.order_by(Event.updated_at.asc())
    else:
        query = query.order_by(Event.start_dt.asc())

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError:
            raise HTTPException(400, "Invalid pageToken")
    events = query.offset(offset).limit(maxResults).all()
    items = [_to_event_resource(e) for e in events]

    list_etag_raw = "|".join(item.etag for item in items)
    updated = _iso(events[-1].updated_at) if events else _iso(datetime.now(timezone.utc))
    token_seed = f"{calendar.id}:{total_count}:{updated}:{syncToken or ''}"
    incompatible_sync_filters = any(
        [
            bool(q),
            bool(timeMin),
            bool(timeMax),
            bool(orderBy),
            bool(pageToken),
        ]
    )
    next_sync_token = None if incompatible_sync_filters else _md5_hex(token_seed)
    next_page_token = str(offset + maxResults) if total_count > (offset + maxResults) else None

    return EventListResponse(
        etag=f'"{_md5_hex(list_etag_raw)}"',
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


def _create_event_from_body(
    *,
    calendar: Calendar,
    actor_user_id: str,
    body: EventWriteRequest,
) -> Event:
    start_dt = _parse_event_datetime(body.start, calendar.timezone)
    end_dt = _parse_event_datetime(body.end, calendar.timezone)
    if end_dt <= start_dt:
        raise HTTPException(400, "Event end must be after start")

    now = datetime.now(timezone.utc)
    event = Event(
        id=f"evt_{uuid.uuid4().hex[:12]}",
        calendar_id=calendar.id,
        user_id=actor_user_id,
        summary=body.summary,
        description=body.description,
        location=body.location,
        status=body.status or "confirmed",
        start_dt=start_dt,
        end_dt=end_dt,
        attendees_json="[]",
        created_at=now,
        updated_at=now,
        etag="",
        i_cal_uid=body.iCalUID or f"{uuid.uuid4().hex}@google.com",
        sequence=0,
        recurrence_json=json.dumps(body.recurrence or []),
        recurring_event_id="",
        original_start_time="",
    )
    event.etag = _compute_event_etag(event)
    return event


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
    event = _create_event_from_body(calendar=calendar, actor_user_id=_actor_user_id, body=body)
    db.add(event)
    db.commit()
    db.refresh(event)
    return _to_event_resource(event)


@router.post(
    "/calendars/{calendarId}/events/import",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_import(
    calendarId: str,
    body: EventWriteRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    event = _create_event_from_body(calendar=calendar, actor_user_id=_actor_user_id, body=body)
    if not body.iCalUID:
        event.i_cal_uid = f"{uuid.uuid4().hex}@import.calendar.google.com"
    event.etag = _compute_event_etag(event)
    db.add(event)
    db.commit()
    db.refresh(event)
    return _to_event_resource(event)


@router.put(
    "/calendars/{calendarId}/events/{eventId}",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_update(
    calendarId: str,
    eventId: str,
    body: EventWriteRequest,
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

    start_dt = _parse_event_datetime(body.start, calendar.timezone)
    end_dt = _parse_event_datetime(body.end, calendar.timezone)
    if end_dt <= start_dt:
        raise HTTPException(400, "Event end must be after start")

    event.summary = body.summary
    event.description = body.description
    event.location = body.location
    event.status = body.status or "confirmed"
    event.start_dt = start_dt
    event.end_dt = end_dt
    if body.iCalUID:
        event.i_cal_uid = body.iCalUID
    event.recurrence_json = json.dumps(body.recurrence or [])
    event.sequence += 1
    event.updated_at = datetime.now(timezone.utc)
    event.etag = _compute_event_etag(event)

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
        event.start_dt = _parse_event_datetime(body.start, calendar.timezone)
    if body.end is not None:
        event.end_dt = _parse_event_datetime(body.end, calendar.timezone)
    if body.recurrence is not None:
        event.recurrence_json = json.dumps(body.recurrence)
    if event.end_dt <= event.start_dt:
        raise HTTPException(400, "Event end must be after start")

    event.sequence += 1
    event.updated_at = datetime.now(timezone.utc)
    event.etag = _compute_event_etag(event)

    db.commit()
    db.refresh(event)

    return _to_event_resource(event)


@router.get(
    "/calendars/{calendarId}/events/{eventId}/instances",
    response_model=EventListResponse,
    response_model_exclude_none=True,
)
def events_instances(
    calendarId: str,
    eventId: str,
    maxResults: int = Query(250, ge=1, le=2500),
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

    recurrence = _deserialize_recurrence(event.recurrence_json)
    # Recurrence expansion is intentionally simplified. For non-recurring events
    # real API returns an empty items list with collection metadata.
    items: list[EventResource] = []
    if recurrence:
        # Emit the master event as a single instance to keep deterministic behavior.
        items = [_to_event_resource(event)]
        items = items[:maxResults]

    list_etag_raw = "|".join(i.etag for i in items)
    return EventListResponse(
        etag=f'"{_md5_hex(list_etag_raw)}"',
        summary=calendar.summary,
        description=calendar.description or "",
        timeZone=calendar.timezone,
        updated=_iso(event.updated_at),
        accessRole=calendar.access_role,
        defaultReminders=[ReminderOverride(method="popup", minutes=10)],
        items=items,
        nextSyncToken=_md5_hex(f"{calendar.id}:{event.id}:{event.updated_at.isoformat()}"),
    )


@router.post(
    "/calendars/{calendarId}/events/{eventId}/move",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_move(
    calendarId: str,
    eventId: str,
    destination: str = Query(...),
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    source_calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    destination_calendar = _resolve_calendar_for_actor(db, destination, _actor_user_id)
    event = db.query(Event).filter(
        Event.id == eventId,
        Event.calendar_id == source_calendar.id,
    ).first()
    if not event:
        raise HTTPException(404, "Not Found")

    event.calendar_id = destination_calendar.id
    event.updated_at = datetime.now(timezone.utc)
    event.sequence += 1
    event.etag = _compute_event_etag(event)
    db.commit()
    db.refresh(event)
    return _to_event_resource(event)


@router.post(
    "/calendars/{calendarId}/events/quickAdd",
    response_model=EventResource,
    response_model_exclude_none=True,
)
def events_quick_add(
    calendarId: str,
    text: str = Query(...),
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    start_dt = datetime.now(timezone.utc) + timedelta(hours=1)
    end_dt = start_dt + timedelta(hours=1)
    body = EventWriteRequest(
        summary=text,
        description="",
        location="",
        start=EventDateTime(dateTime=_iso(start_dt)),
        end=EventDateTime(dateTime=_iso(end_dt)),
    )
    event = _create_event_from_body(calendar=calendar, actor_user_id=_actor_user_id, body=body)
    db.add(event)
    db.commit()
    db.refresh(event)
    return _to_event_resource(event)


@router.post(
    "/calendars/{calendarId}/events/watch",
    response_model=ChannelResponse,
    response_model_exclude_none=True,
)
def events_watch(
    calendarId: str,
    body: ChannelRequest,
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    resource_uri = f"https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events?alt=json"
    return channel_registry.register(
        resource_uri=resource_uri,
        channel_id=body.id,
        address=body.address,
        token=body.token,
        channel_type=body.type,
        payload=body.payload,
        params=body.params,
        expiration=body.expiration,
    )


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
