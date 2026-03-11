"""Calendar freebusy endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, Event

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
            Event.status != "cancelled",
            Event.end_dt > time_min,
            Event.start_dt < time_max,
        ).order_by(Event.start_dt.asc()).all()
        busy = [
            BusyTimeRange(
                start=_iso(max(_as_aware_utc(e.start_dt), time_min)),
                end=_iso(min(_as_aware_utc(e.end_dt), time_max)),
            )
            for e in events
        ]
        calendars[item.id] = FreeBusyCalendarResponse(busy=busy)

    return FreeBusyResponse(
        timeMin=_iso(time_min),
        timeMax=_iso(time_max),
        calendars=calendars,
    )
