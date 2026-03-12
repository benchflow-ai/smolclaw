"""Calendar event endpoints."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterator

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


def _parse_event_datetime(dt_value: EventDateTime, tz_name: str) -> tuple[datetime, bool]:
    if dt_value.dateTime:
        return _parse_rfc3339(dt_value.dateTime), False
    if dt_value.date:
        # Store all-day values as midnight in calendar timezone (serialized in UTC).
        day = datetime.fromisoformat(dt_value.date).date()
        return datetime.combine(day, time.min, tzinfo=timezone.utc), True
    raise HTTPException(400, "Missing event datetime value")


def _iso(dt: datetime) -> str:
    return _as_aware_utc(dt).isoformat().replace("+00:00", "Z")


def _as_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_date_str(dt: datetime) -> str:
    return _as_aware_utc(dt).date().isoformat()


def _event_datetime_resource(
    *,
    dt: datetime,
    is_date: bool,
    tz_name: str,
) -> EventDateTime:
    if is_date:
        return EventDateTime(date=_event_date_str(dt))
    return EventDateTime(dateTime=_iso(dt), timeZone=tz_name)


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


_WEEKDAY_MAP: dict[str, int] = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}


def _parse_until(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
            return dt.replace(tzinfo=timezone.utc)
        if "T" in value:
            dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        dt = datetime.strptime(value, "%Y%m%d")
        return datetime.combine(dt.date(), time.max, tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_rrule(recurrence: list[str] | None) -> dict[str, Any] | None:
    if not recurrence:
        return None

    rule_str = next(
        (entry[len("RRULE:") :] for entry in recurrence if isinstance(entry, str) and entry.startswith("RRULE:")),
        None,
    )
    if not rule_str:
        return None

    parts: dict[str, str] = {}
    for raw_part in rule_str.split(";"):
        if "=" not in raw_part:
            continue
        key, value = raw_part.split("=", 1)
        parts[key.upper()] = value

    freq = (parts.get("FREQ") or "").upper()
    if freq not in {"DAILY", "WEEKLY"}:
        return None

    interval = 1
    try:
        interval = max(1, int(parts.get("INTERVAL", "1")))
    except ValueError:
        interval = 1

    count: int | None = None
    if parts.get("COUNT"):
        try:
            count = max(1, int(parts["COUNT"]))
        except ValueError:
            count = None

    until = _parse_until(parts.get("UNTIL", ""))

    byday: list[int] | None = None
    if parts.get("BYDAY"):
        parsed_days: list[int] = []
        for token in parts["BYDAY"].split(","):
            token = token.strip().upper()
            code = token[-2:] if len(token) >= 2 else token
            day = _WEEKDAY_MAP.get(code)
            if day is not None:
                parsed_days.append(day)
        if parsed_days:
            byday = sorted(set(parsed_days))

    return {
        "freq": freq,
        "interval": interval,
        "count": count,
        "until": until,
        "byday": byday,
    }


def _iter_weekly_starts(
    *,
    start_dt: datetime,
    interval: int,
    byday: list[int] | None,
    count: int | None,
    until: datetime | None,
    max_emits: int,
) -> Iterator[datetime]:
    start_dt = _as_aware_utc(start_dt)
    emitted = 0
    max_occurrences = count if count is not None else max_emits

    if byday:
        anchor_week_start = start_dt.date() - timedelta(days=start_dt.weekday())
        week_index = 0
        while emitted < max_occurrences and week_index < max_emits:
            week_start = anchor_week_start + timedelta(weeks=week_index * interval)
            for day_index in byday:
                candidate_date = week_start + timedelta(days=day_index)
                candidate = datetime.combine(candidate_date, start_dt.timetz())
                if candidate.tzinfo is None:
                    candidate = candidate.replace(tzinfo=timezone.utc)
                if candidate < start_dt:
                    continue
                if until and candidate > until:
                    return
                yield candidate
                emitted += 1
                if emitted >= max_occurrences:
                    return
            week_index += 1
        return

    index = 0
    while emitted < max_occurrences and index < max_emits:
        candidate = start_dt + timedelta(weeks=index * interval)
        if until and candidate > until:
            break
        yield candidate
        emitted += 1
        index += 1


def _iter_recurrence_starts(
    *,
    start_dt: datetime,
    rule: dict[str, Any],
    max_emits: int,
) -> Iterator[datetime]:
    start_dt = _as_aware_utc(start_dt)
    freq = rule["freq"]
    interval = rule["interval"]
    count = rule["count"]
    until = rule["until"]
    byday = rule["byday"]
    max_occurrences = count if count is not None else max_emits

    if freq == "DAILY":
        emitted = 0
        index = 0
        while emitted < max_occurrences and index < max_emits:
            candidate = start_dt + timedelta(days=index * interval)
            index += 1
            if until and candidate > until:
                break
            if byday and candidate.weekday() not in byday:
                continue
            yield candidate
            emitted += 1
        return

    if freq == "WEEKLY":
        yield from _iter_weekly_starts(
            start_dt=start_dt,
            interval=interval,
            byday=byday,
            count=count,
            until=until,
            max_emits=max_emits,
        )


def _to_instance_resource(
    *,
    event: Event,
    start_dt: datetime,
    end_dt: datetime,
    time_zone: str,
) -> EventResource:
    start_dt = _as_aware_utc(start_dt)
    end_dt = _as_aware_utc(end_dt)
    base = _to_event_resource(event)
    start_iso = _iso(start_dt)
    end_iso = _iso(end_dt)
    instance_id = f"{event.id}_{start_dt.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    instance_etag = f'"{_md5_hex(f"{event.id}:{start_iso}:{end_iso}:{event.updated_at.isoformat()}:{event.sequence}")}"'
    start_value = _event_datetime_resource(
        dt=start_dt,
        is_date=bool(event.start_is_date),
        tz_name=time_zone,
    )
    end_value = _event_datetime_resource(
        dt=end_dt,
        is_date=bool(event.end_is_date),
        tz_name=time_zone,
    )
    original_start = (
        EventDateTime(date=_event_date_str(start_dt))
        if event.start_is_date
        else EventDateTime(dateTime=start_iso, timeZone=time_zone)
    )
    return base.model_copy(
        update={
            "id": instance_id,
            "etag": instance_etag,
            "start": start_value,
            "end": end_value,
            "recurrence": None,
            "recurringEventId": event.id,
            "originalStartTime": original_start,
        }
    )


def _to_event_resource(event: Event) -> EventResource:
    tz = event.calendar.timezone if event.calendar else "UTC"
    actor_email = event.user.email_address if event.user else ""
    organizer_name = event.calendar.summary if event.calendar else None

    recurrence = _deserialize_recurrence(event.recurrence_json)
    original_start = None
    if event.original_start_time:
        if event.start_is_date:
            original_start = EventDateTime(date=event.original_start_time)
        else:
            original_start = EventDateTime(dateTime=event.original_start_time, timeZone=tz)

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
        start=_event_datetime_resource(
            dt=event.start_dt,
            is_date=bool(event.start_is_date),
            tz_name=tz,
        ),
        end=_event_datetime_resource(
            dt=event.end_dt,
            is_date=bool(event.end_is_date),
            tz_name=tz,
        ),
        creator=EventActor(email=actor_email, self=True) if actor_email else None,
        organizer=EventActor(email=actor_email, self=True, displayName=organizer_name) if actor_email else None,
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
    include_recurrence_parents: bool = False,
) -> tuple[Any, int]:
    query = db.query(Event).filter(Event.calendar_id == calendar.id)
    if not showDeleted:
        query = query.filter(Event.status != "cancelled")
    if q:
        query = query.filter((Event.summary.contains(q)) | (Event.description.contains(q)))
    # For singleEvents expansion we must not filter recurring masters by their
    # original start/end, otherwise later instances can be dropped.
    if timeMin and not include_recurrence_parents:
        query = query.filter(Event.end_dt >= _parse_rfc3339(timeMin))
    if timeMax and not include_recurrence_parents:
        query = query.filter(Event.start_dt <= _parse_rfc3339(timeMax))
    total_count = query.count()
    return query, total_count


def _event_in_window(
    *,
    start_dt: datetime,
    end_dt: datetime,
    time_min_dt: datetime | None,
    time_max_dt: datetime | None,
) -> bool:
    start_dt = _as_aware_utc(start_dt)
    end_dt = _as_aware_utc(end_dt)
    if time_min_dt and end_dt < time_min_dt:
        return False
    if time_max_dt and start_dt >= time_max_dt:
        return False
    return True


def _sort_event_items(items: list[EventResource], order_by: str | None) -> list[EventResource]:
    if order_by == "updated":
        return sorted(items, key=lambda item: (item.updated, item.id))

    return sorted(
        items,
        key=lambda item: (
            item.start.dateTime or f"{item.start.date}T00:00:00Z",
            item.id,
        ),
    )


def _expand_single_events(
    *,
    events: list[Event],
    calendar: Calendar,
    time_min_dt: datetime | None,
    time_max_dt: datetime | None,
    emit_hint: int,
) -> list[EventResource]:
    expanded: list[EventResource] = []

    for event in events:
        recurrence = _deserialize_recurrence(event.recurrence_json)
        rule = _parse_rrule(recurrence)
        if not rule:
            if _event_in_window(
                start_dt=event.start_dt,
                end_dt=event.end_dt,
                time_min_dt=time_min_dt,
                time_max_dt=time_max_dt,
            ):
                expanded.append(_to_event_resource(event))
            continue

        duration = _as_aware_utc(event.end_dt) - _as_aware_utc(event.start_dt)
        max_emits = max(emit_hint * 10, 1000)
        for instance_start in _iter_recurrence_starts(
            start_dt=event.start_dt,
            rule=rule,
            max_emits=max_emits,
        ):
            instance_start = _as_aware_utc(instance_start)
            instance_end = _as_aware_utc(instance_start + duration)

            if time_max_dt and instance_start >= time_max_dt:
                break
            if not _event_in_window(
                start_dt=instance_start,
                end_dt=instance_end,
                time_min_dt=time_min_dt,
                time_max_dt=time_max_dt,
            ):
                continue

            expanded.append(
                _to_instance_resource(
                    event=event,
                    start_dt=instance_start,
                    end_dt=instance_end,
                    time_zone=calendar.timezone,
                )
            )

    return expanded


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
    singleEvents: bool = Query(False),
    q: str | None = Query(None),
    timeMin: str | None = Query(None),
    timeMax: str | None = Query(None),
    orderBy: str | None = Query(None),
    showDeleted: bool = Query(False),
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError:
            raise HTTPException(400, "Invalid pageToken")

    time_min_dt = _parse_rfc3339(timeMin) if timeMin else None
    time_max_dt = _parse_rfc3339(timeMax) if timeMax else None
    if time_min_dt and time_max_dt and time_max_dt <= time_min_dt:
        raise HTTPException(400, "timeMax must be greater than timeMin")

    query, total_count = _query_events_base(
        db=db,
        calendar=calendar,
        q=q,
        timeMin=timeMin,
        timeMax=timeMax,
        showDeleted=showDeleted,
        include_recurrence_parents=singleEvents,
    )

    if singleEvents:
        source_events = query.order_by(Event.start_dt.asc()).all()
        all_items = _expand_single_events(
            events=source_events,
            calendar=calendar,
            time_min_dt=time_min_dt,
            time_max_dt=time_max_dt,
            emit_hint=offset + maxResults,
        )
        ordered_items = _sort_event_items(all_items, orderBy)
        total_count = len(ordered_items)
        items = ordered_items[offset : offset + maxResults]
    else:
        if orderBy == "updated":
            query = query.order_by(Event.updated_at.asc())
        else:
            query = query.order_by(Event.start_dt.asc())
        events = query.offset(offset).limit(maxResults).all()
        items = [_to_event_resource(e) for e in events]

    list_etag_raw = "|".join(item.etag for item in items)
    updated = items[-1].updated if items else _iso(datetime.now(timezone.utc))
    token_seed = f"{calendar.id}:{total_count}:{updated}:{syncToken or ''}"
    incompatible_sync_filters = any(
        [
            bool(singleEvents),
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
    start_dt, start_is_date = _parse_event_datetime(body.start, calendar.timezone)
    end_dt, end_is_date = _parse_event_datetime(body.end, calendar.timezone)
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
        start_is_date=start_is_date,
        end_is_date=end_is_date,
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

    start_dt, start_is_date = _parse_event_datetime(body.start, calendar.timezone)
    end_dt, end_is_date = _parse_event_datetime(body.end, calendar.timezone)
    if end_dt <= start_dt:
        raise HTTPException(400, "Event end must be after start")

    event.summary = body.summary
    event.description = body.description
    event.location = body.location
    event.status = body.status or "confirmed"
    event.start_dt = start_dt
    event.end_dt = end_dt
    event.start_is_date = start_is_date
    event.end_is_date = end_is_date
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
        start_dt, start_is_date = _parse_event_datetime(body.start, calendar.timezone)
        event.start_dt = start_dt
        event.start_is_date = start_is_date
    if body.end is not None:
        end_dt, end_is_date = _parse_event_datetime(body.end, calendar.timezone)
        event.end_dt = end_dt
        event.end_is_date = end_is_date
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
    pageToken: str | None = Query(None),
    timeMin: str | None = Query(None),
    timeMax: str | None = Query(None),
    originalStart: str | None = Query(None),
    showDeleted: bool = Query(False),
    timeZone: str | None = Query(None),
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

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError as exc:
            raise HTTPException(400, "Invalid pageToken") from exc

    time_min_dt = _parse_rfc3339(timeMin) if timeMin else None
    time_max_dt = _parse_rfc3339(timeMax) if timeMax else None
    if time_min_dt and time_max_dt and time_max_dt <= time_min_dt:
        raise HTTPException(400, "timeMax must be greater than timeMin")
    original_start_dt = _parse_rfc3339(originalStart) if originalStart else None
    if original_start_dt is not None:
        original_start_dt = _as_aware_utc(original_start_dt)

    recurrence = _deserialize_recurrence(event.recurrence_json)
    rule = _parse_rrule(recurrence)
    items: list[EventResource] = []

    if rule and (showDeleted or event.status != "cancelled"):
        duration = event.end_dt - event.start_dt
        needed_matches = offset + maxResults
        matched: list[EventResource] = []
        has_more = False
        # Keep bounded for unbounded recurrences while still supporting paging/filtering.
        max_emits = max(needed_matches * 10, 1000)

        for start_dt in _iter_recurrence_starts(
            start_dt=event.start_dt,
            rule=rule,
            max_emits=max_emits,
        ):
            start_dt = _as_aware_utc(start_dt)
            end_dt = _as_aware_utc(start_dt + duration)

            if time_min_dt and end_dt < time_min_dt:
                continue
            if time_max_dt and start_dt >= time_max_dt:
                break
            if original_start_dt and start_dt != original_start_dt:
                continue

            matched.append(
                _to_instance_resource(
                    event=event,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    time_zone=timeZone or calendar.timezone,
                )
            )
            if len(matched) > needed_matches:
                has_more = True
                break

        items = matched[offset : offset + maxResults]
        next_page_token = str(offset + maxResults) if has_more else None
    else:
        next_page_token = None

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
        nextPageToken=next_page_token,
        nextSyncToken=None if next_page_token else _md5_hex(f"{calendar.id}:{event.id}:{event.updated_at.isoformat()}"),
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
