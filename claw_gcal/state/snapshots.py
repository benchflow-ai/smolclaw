"""State snapshots, reset, and diff functionality."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from claw_gcal.models import AclRule, Calendar, Event, User, get_session_factory

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent.parent / ".data" / "snapshots_gcal"


def _serialize_events(db: Session, user_id: str) -> list[dict]:
    events = db.query(Event).filter(Event.user_id == user_id).all()
    return [
        {
            "id": e.id,
            "calendarId": e.calendar_id,
            "summary": e.summary,
            "description": e.description,
            "location": e.location,
            "status": e.status,
            "start": e.start_dt.isoformat(),
            "end": e.end_dt.isoformat(),
            "startIsDate": e.start_is_date,
            "endIsDate": e.end_is_date,
            "attendees": e.attendees_json,
            "etag": e.etag,
            "iCalUID": e.i_cal_uid,
            "sequence": e.sequence,
            "recurrence": e.recurrence_json,
            "recurringEventId": e.recurring_event_id,
            "originalStartTime": e.original_start_time,
            # Preserve original timestamps for round-trip fidelity.
            "created": e.created_at.isoformat(),
            "updated": e.updated_at.isoformat(),
        }
        for e in events
    ]


def _serialize_acls(db: Session, user_id: str) -> list[dict]:
    rows = (
        db.query(AclRule)
        .join(Calendar, Calendar.id == AclRule.calendar_id)
        .filter(Calendar.user_id == user_id)
        .all()
    )
    return [
        {
            "id": r.id,
            "calendarId": r.calendar_id,
            "scopeType": r.scope_type,
            "scopeValue": r.scope_value,
            "role": r.role,
            "etag": r.etag,
        }
        for r in rows
    ]


def _serialize_user(db: Session, user: User) -> dict:
    calendars = db.query(Calendar).filter(Calendar.user_id == user.id).all()
    return {
        "user": {
            "id": user.id,
            "email": user.email_address,
            "displayName": user.display_name,
            "timezone": user.timezone,
            "historyId": user.history_id,
        },
        "calendars": [
            {
                "id": c.id,
                "summary": c.summary,
                "description": c.description,
                "location": c.location,
                "timeZone": c.timezone,
                "accessRole": c.access_role,
                "primary": c.is_primary,
                "selected": c.selected,
                "hidden": c.hidden,
                "summaryOverride": c.summary_override,
                "autoAcceptInvitations": c.auto_accept_invitations,
                "colorId": c.color_id,
            }
            for c in calendars
        ],
        "acls": _serialize_acls(db, user.id),
        "events": _serialize_events(db, user.id),
    }


def get_state_dump() -> dict:
    """Get full state dump for all users."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        users = db.query(User).all()
        return {
            "users": {u.id: _serialize_user(db, u) for u in users},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        db.close()


def take_snapshot(name: str) -> Path:
    """Save current state to a JSON snapshot."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    state = get_state_dump()
    path = SNAPSHOTS_DIR / f"{name}.json"
    path.write_text(json.dumps(state, indent=2))
    return path


def restore_snapshot(name: str) -> bool:
    """Restore DB from a snapshot. Returns True if successful."""
    path = SNAPSHOTS_DIR / f"{name}.json"
    if not path.exists():
        return False

    state = json.loads(path.read_text())
    _restore_from_state(state)
    return True


def _parse_iso_datetime(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return fallback
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _restore_from_state(state: dict):
    """Rebuild DB from a state dict."""
    from claw_gcal.models import Base, get_engine

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    session_factory = get_session_factory()
    db = session_factory()
    try:
        for user_id, user_data in state.get("users", {}).items():
            user = user_data["user"]
            db.add(
                User(
                    id=user["id"],
                    email_address=user["email"],
                    display_name=user["displayName"],
                    timezone=user.get("timezone", "America/Los_Angeles"),
                    history_id=user.get("historyId", 1),
                )
            )

            for cal in user_data.get("calendars", []):
                db.add(
                    Calendar(
                        id=cal["id"],
                        user_id=user_id,
                        summary=cal["summary"],
                        description=cal.get("description", ""),
                        location=cal.get("location", ""),
                        timezone=cal.get("timeZone", "America/Los_Angeles"),
                        access_role=cal.get("accessRole", "owner"),
                        is_primary=cal.get("primary", False),
                        selected=cal.get("selected", True),
                        hidden=cal.get("hidden", False),
                        summary_override=cal.get("summaryOverride", ""),
                        auto_accept_invitations=cal.get("autoAcceptInvitations", False),
                        color_id=cal.get("colorId", "9"),
                    )
                )

            for acl in user_data.get("acls", []):
                db.add(
                    AclRule(
                        id=acl["id"],
                        calendar_id=acl["calendarId"],
                        scope_type=acl.get("scopeType", "user"),
                        scope_value=acl.get("scopeValue", ""),
                        role=acl.get("role", "reader"),
                        etag=acl.get("etag", ""),
                    )
                )

            for event in user_data.get("events", []):
                fallback_now = datetime.now(timezone.utc)
                db.add(
                    Event(
                        id=event["id"],
                        calendar_id=event["calendarId"],
                        user_id=user_id,
                        summary=event.get("summary", ""),
                        description=event.get("description", ""),
                        location=event.get("location", ""),
                        status=event.get("status", "confirmed"),
                        start_dt=_parse_iso_datetime(event.get("start"), fallback_now),
                        end_dt=_parse_iso_datetime(event.get("end"), fallback_now),
                        start_is_date=bool(event.get("startIsDate", False)),
                        end_is_date=bool(event.get("endIsDate", False)),
                        attendees_json=event.get("attendees", "[]"),
                        created_at=_parse_iso_datetime(event.get("created"), fallback_now),
                        updated_at=_parse_iso_datetime(event.get("updated"), fallback_now),
                        etag=event.get("etag", ""),
                        i_cal_uid=event.get("iCalUID", ""),
                        sequence=event.get("sequence", 0),
                        recurrence_json=event.get("recurrence", "[]"),
                        recurring_event_id=event.get("recurringEventId", ""),
                        original_start_time=event.get("originalStartTime", ""),
                    )
                )

        db.commit()
    finally:
        db.close()


def _index_by_id(items: list[dict]) -> dict[str, dict]:
    return {str(item.get("id")): item for item in items}


def _diff_items(initial_items: list[dict], current_items: list[dict]) -> dict:
    initial_idx = _index_by_id(initial_items)
    current_idx = _index_by_id(current_items)

    added = [i for i in current_items if str(i.get("id")) not in initial_idx]
    deleted = [i for i in initial_items if str(i.get("id")) not in current_idx]

    updated = []
    for item_id, curr in current_idx.items():
        init = initial_idx.get(item_id)
        if init is not None and curr != init:
            updated.append(curr)

    return {
        "added": added,
        "updated": updated,
        "deleted": deleted,
    }


def get_diff() -> dict:
    """Compute diff versus initial snapshot."""
    initial_path = SNAPSHOTS_DIR / "initial.json"
    if not initial_path.exists():
        return {"error": "No initial snapshot found"}

    initial_state = json.loads(initial_path.read_text())
    current_state = get_state_dump()

    diff = {"users": {}}

    all_user_ids = set(initial_state.get("users", {}).keys()) | set(
        current_state.get("users", {}).keys()
    )

    for user_id in sorted(all_user_ids):
        init_user = initial_state.get("users", {}).get(user_id, {})
        curr_user = current_state.get("users", {}).get(user_id, {})

        diff["users"][user_id] = {
            "calendars": _diff_items(
                init_user.get("calendars", []),
                curr_user.get("calendars", []),
            ),
            "acls": _diff_items(
                init_user.get("acls", []),
                curr_user.get("acls", []),
            ),
            "events": _diff_items(
                init_user.get("events", []),
                curr_user.get("events", []),
            ),
        }

    return diff
