"""Calendar freebusy endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, Event

from .events import (
    _as_aware_utc,
    _deserialize_recurrence,
    _iter_recurrence_starts,
    _parse_rrule,
    _recurrence_emit_hint,
)
from .deps import get_db, resolve_actor_user_id
from .schemas import (
    BusyTimeRange,
    FreeBusyCalendarResponse,
    FreeBusyRequest,
    FreeBusyResponse,
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


def _as_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_calendar(db: Session, calendar_id: str, actor_user_id: str) -> Calendar | None:
    if calendar_id == "primary":
        return db.query(Calendar).filter(
            Calendar.user_id == actor_user_id,
            Calendar.is_primary.is_(True),
        ).first()
    return db.query(Calendar).filter(
        Calendar.id == calendar_id,
        Calendar.user_id == actor_user_id,
    ).first()


def _busy_ranges_for_event(
    *,
    event: Event,
    time_min: datetime,
    time_max: datetime,
) -> list[BusyTimeRange]:
    if event.status == "cancelled":
        return []

    recurrence = _deserialize_recurrence(event.recurrence_json)
    rule = _parse_rrule(recurrence)
    if not rule:
        start_dt = _as_aware_utc(event.start_dt)
        end_dt = _as_aware_utc(event.end_dt)
        if end_dt <= time_min or start_dt >= time_max:
            return []
        return [
            BusyTimeRange(
                start=_iso(max(start_dt, time_min)),
                end=_iso(min(end_dt, time_max)),
            )
        ]

    duration = _as_aware_utc(event.end_dt) - _as_aware_utc(event.start_dt)
    max_emits = _recurrence_emit_hint(
        start_dt=event.start_dt,
        rule=rule,
        target_dt=time_max + duration + timedelta(days=1),
    )

    busy: list[BusyTimeRange] = []
    for start_dt in _iter_recurrence_starts(
        start_dt=event.start_dt,
        rule=rule,
        max_emits=max_emits,
    ):
        start_dt = _as_aware_utc(start_dt)
        end_dt = _as_aware_utc(start_dt + duration)
        if end_dt <= time_min:
            continue
        if start_dt >= time_max:
            break
        busy.append(
            BusyTimeRange(
                start=_iso(max(start_dt, time_min)),
                end=_iso(min(end_dt, time_max)),
            )
        )

    return busy


@router.post("/freeBusy", response_model=FreeBusyResponse, response_model_exclude_none=True)
def freebusy_query(
    body: FreeBusyRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    time_min = _parse_rfc3339(body.timeMin)
    time_max = _parse_rfc3339(body.timeMax)
    if time_max <= time_min:
        raise HTTPException(400, "timeMax must be greater than timeMin")

    calendars: dict[str, FreeBusyCalendarResponse] = {}
    for item in body.items:
        cal = _resolve_calendar(db, item.id, _actor_user_id)
        if not cal:
            calendars[item.id] = FreeBusyCalendarResponse(
                errors=[{"domain": "global", "reason": "notFound"}],
                busy=[],
            )
            continue

        events = db.query(Event).filter(
            Event.calendar_id == cal.id,
        ).order_by(Event.start_dt.asc()).all()
        busy: list[BusyTimeRange] = []
        for event in events:
            busy.extend(
                _busy_ranges_for_event(
                    event=event,
                    time_min=time_min,
                    time_max=time_max,
                )
            )
        busy.sort(key=lambda slot: (slot.start, slot.end))
        calendars[item.id] = FreeBusyCalendarResponse(busy=busy)

    return FreeBusyResponse(
        timeMin=_iso(time_min),
        timeMax=_iso(time_max),
        calendars=calendars,
    )
