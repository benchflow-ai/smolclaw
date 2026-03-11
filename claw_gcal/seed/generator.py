"""Generate deterministic Calendar seed data."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from claw_gcal.models import (
    AclRule,
    Calendar,
    Event,
    User,
    get_session_factory,
    init_db,
    reset_engine,
)
from claw_gcal.state.snapshots import take_snapshot


def _mk_event_id(user_idx: int, event_idx: int) -> str:
    return f"evt_u{user_idx:02d}_{event_idx:04d}"


def _mk_calendar_id(user_email: str, suffix: str = "primary") -> str:
    if suffix == "primary":
        return user_email
    return f"{suffix}-{user_email}"


def _mk_acl_id(scope_type: str, scope_value: str) -> str:
    if scope_type == "default":
        return "default"
    return f"{scope_type}:{scope_value or ''}"


def seed_database(
    scenario: str = "default",
    seed: int = 42,
    db_path: str | None = None,
    num_users: int = 1,
) -> dict:
    """Seed database with deterministic calendar data.

    Args:
        scenario: default | long_context
        seed: random seed for deterministic content
        db_path: sqlite db path
        num_users: number of users to seed
    """
    if scenario not in {"default", "long_context"}:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. Available: default, long_context"
        )

    reset_engine()
    init_db(db_path)

    rng = random.Random(seed)
    session_factory = get_session_factory(db_path)
    db = session_factory()

    try:
        users = []
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        per_user_events = 8 if scenario == "default" else 800

        for i in range(num_users):
            idx = i + 1
            user_id = f"user{idx}"
            email = f"alex{idx if idx > 1 else ''}@nexusai.com"
            user = User(
                id=user_id,
                email_address=email,
                display_name=f"Alex {idx}",
                timezone="America/Los_Angeles",
                history_id=1,
            )
            db.add(user)
            users.append(user)

            primary_calendar_id = _mk_calendar_id(email, "primary")
            team_calendar_id = _mk_calendar_id(email, "team")

            primary = Calendar(
                id=primary_calendar_id,
                user_id=user_id,
                summary="Primary Calendar",
                description="Main personal calendar",
                location="",
                timezone="America/Los_Angeles",
                access_role="owner",
                is_primary=True,
                selected=True,
                hidden=False,
                summary_override="",
                auto_accept_invitations=False,
                color_id="14",
            )
            team = Calendar(
                id=team_calendar_id,
                user_id=user_id,
                summary="Team Calendar",
                description="Shared team events",
                location="",
                timezone="America/Los_Angeles",
                access_role="owner",
                is_primary=False,
                selected=True,
                hidden=False,
                summary_override="",
                auto_accept_invitations=False,
                color_id="9",
            )

            db.add(primary)
            db.add(team)

            # Owner ACL for each owned calendar (mirrors real owner access).
            for cal_id in (primary_calendar_id, team_calendar_id):
                rule_id = _mk_acl_id("user", email)
                db.add(
                    AclRule(
                        id=f"{cal_id}:{rule_id}",
                        calendar_id=cal_id,
                        scope_type="user",
                        scope_value=email,
                        role="owner",
                        etag=f'"{cal_id}:{rule_id}:owner"',
                    )
                )

            # Primary events
            for j in range(per_user_events):
                start = now + timedelta(hours=(j * 2) - 8)
                duration_h = rng.choice([1, 1, 2])
                end = start + timedelta(hours=duration_h)
                event_id = _mk_event_id(idx, j)
                db.add(
                    Event(
                        id=event_id,
                        calendar_id=primary_calendar_id,
                        user_id=user_id,
                        summary=f"Focus Block {j + 1}",
                        description="Deep work session",
                        location="",
                        status="confirmed",
                        start_dt=start,
                        end_dt=end,
                        attendees_json="[]",
                        created_at=now,
                        updated_at=now,
                        etag=f'"{event_id}-v1"',
                        i_cal_uid=f"{event_id}@smolclaw.local",
                        sequence=0,
                        recurrence_json="[]",
                        recurring_event_id="",
                        original_start_time="",
                    )
                )

            # One team sync event for quick coverage
            db.add(
                Event(
                    id=f"evt_team_u{idx:02d}",
                    calendar_id=team_calendar_id,
                    user_id=user_id,
                    summary="Team Sync",
                    description="Weekly team sync",
                    location="Zoom",
                    status="confirmed",
                    start_dt=now + timedelta(days=1, hours=1),
                    end_dt=now + timedelta(days=1, hours=2),
                    attendees_json="[]",
                    created_at=now,
                    updated_at=now,
                    etag=f'"evt_team_u{idx:02d}-v1"',
                    i_cal_uid=f"evt_team_u{idx:02d}@smolclaw.local",
                    sequence=0,
                    recurrence_json="[]",
                    recurring_event_id="",
                    original_start_time="",
                )
            )

        db.commit()

        take_snapshot("initial")

        calendar_count = db.query(Calendar).count()
        event_count = db.query(Event).count()

        return {
            "users": len(users),
            "calendars": calendar_count,
            "events": event_count,
            "scenario": scenario,
        }
    finally:
        db.close()
