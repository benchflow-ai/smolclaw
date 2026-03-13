"""Deterministic seed data generation for Calendar."""

from __future__ import annotations

import pathlib
import random

from sqlalchemy.orm import Session

from claw_gcal.models import (
    AclRule,
    Calendar,
    Event,
    User,
    get_session_factory,
    init_db,
    reset_engine,
)
from claw_gcal.seed.content import (
    CALENDAR_TEMPLATES,
    DEFAULT_DISTRIBUTION,
    DEFAULT_TARGET_EVENTS,
    SCENARIO_DEFINITIONS,
    LONG_CONTEXT_DISTRIBUTION,
    LONG_CONTEXT_TARGET_EVENTS,
)
from claw_gcal.seed.long_context import (
    seed_distribution_scenario,
    seed_long_context_scenario as _seed_long_context_base,
)
from claw_gcal.seed.task_seed import seed_task_scenario
from claw_gcal.state.snapshots import take_snapshot


def _calendar_id_for_key(user_email: str, key: str) -> str:
    if key == "primary":
        return user_email
    return f"{key}-{user_email}"


def _public_acl_rule_id(scope_type: str, scope_value: str) -> str:
    if scope_type == "default":
        return "default"
    return f"{scope_type}:{scope_value or ''}"


def _storage_acl_rule_id(calendar_id: str, scope_type: str, scope_value: str) -> str:
    return f"{calendar_id}:{_public_acl_rule_id(scope_type, scope_value)}"


def _acl_etag(calendar_id: str, scope_type: str, scope_value: str, role: str) -> str:
    public_id = _public_acl_rule_id(scope_type, scope_value)
    return f'"{calendar_id}:{public_id}:{role}"'


def _create_calendar(
    db: Session,
    *,
    user: User,
    template: dict,
) -> Calendar:
    key = str(template["key"])
    calendar_id = _calendar_id_for_key(user.email_address, key)
    summary = str(template.get("summary", "")).format(
        email=user.email_address,
        name=user.display_name,
    )

    calendar = Calendar(
        id=calendar_id,
        user_id=user.id,
        summary=summary,
        description=str(template.get("description", "")),
        location=str(template.get("location", "")),
        timezone=str(template.get("timezone", user.timezone)),
        access_role=str(template.get("accessRole", "owner")),
        is_primary=bool(template.get("primary", False)),
        selected=bool(template.get("selected", True)),
        hidden=bool(template.get("hidden", False)),
        summary_override=str(template.get("summaryOverride", "")),
        auto_accept_invitations=bool(template.get("autoAcceptInvitations", False)),
        color_id=str(template.get("colorId", "9")),
    )
    db.add(calendar)

    # Owner or reader ACL representing the current actor's direct access.
    actor_role = calendar.access_role
    db.add(
        AclRule(
            id=_storage_acl_rule_id(calendar_id, "user", user.email_address),
            calendar_id=calendar_id,
            scope_type="user",
            scope_value=user.email_address,
            role=actor_role,
            etag=_acl_etag(calendar_id, "user", user.email_address, actor_role),
        )
    )

    for rule in template.get("acl_rules", []):
        scope_type = str(rule.get("scopeType", "default"))
        scope_value = str(rule.get("scopeValue", ""))
        role = str(rule.get("role", "reader"))
        db.add(
            AclRule(
                id=_storage_acl_rule_id(calendar_id, scope_type, scope_value),
                calendar_id=calendar_id,
                scope_type=scope_type,
                scope_value=scope_value,
                role=role,
                etag=_acl_etag(calendar_id, scope_type, scope_value, role),
            )
        )

    return calendar


def _create_user(idx: int) -> User:
    if idx == 1:
        return User(
            id="user1",
            email_address="alex@nexusai.com",
            display_name="Alex Chen",
            timezone="America/Los_Angeles",
            history_id=1,
        )

    return User(
        id=f"user{idx}",
        email_address=f"alex{idx}@nexusai.com",
        display_name=f"Alex {idx}",
        timezone="America/Los_Angeles",
        history_id=1,
    )


def seed_default_scenario(db: Session, user: User, calendars_by_key: dict[str, Calendar], rng) -> int:
    config = SCENARIO_DEFINITIONS["default"]
    return seed_distribution_scenario(
        db,
        user,
        calendars_by_key,
        rng,
        target_events=config["target_events"],
        distribution=config["distribution"],
        needle_events=config["needle_events"],
        recurring_needles=config["recurring_needles"],
        include_needles=config["include_needles"],
    )


def seed_launch_crunch_scenario(
    db: Session,
    user: User,
    calendars_by_key: dict[str, Calendar],
    rng,
) -> int:
    config = SCENARIO_DEFINITIONS["launch_crunch"]
    return seed_distribution_scenario(
        db,
        user,
        calendars_by_key,
        rng,
        target_events=config["target_events"],
        distribution=config["distribution"],
        needle_events=config["needle_events"],
        recurring_needles=config["recurring_needles"],
        include_needles=config["include_needles"],
    )


def seed_travel_heavy_scenario(
    db: Session,
    user: User,
    calendars_by_key: dict[str, Calendar],
    rng,
) -> int:
    config = SCENARIO_DEFINITIONS["travel_heavy"]
    return seed_distribution_scenario(
        db,
        user,
        calendars_by_key,
        rng,
        target_events=config["target_events"],
        distribution=config["distribution"],
        needle_events=config["needle_events"],
        recurring_needles=config["recurring_needles"],
        include_needles=config["include_needles"],
    )


def seed_long_context_scenario(
    db: Session,
    user: User,
    calendars_by_key: dict[str, Calendar],
    rng,
) -> int:
    config = SCENARIO_DEFINITIONS["long_context"]
    return _seed_long_context_base(
        db,
        user,
        calendars_by_key,
        rng,
        target_events=config["target_events"],
        distribution=config["distribution"],
        needle_events=config["needle_events"],
        recurring_needles=config["recurring_needles"],
        include_needles=config["include_needles"],
    )


SCENARIOS = {
    "default": seed_default_scenario,
    "launch_crunch": seed_launch_crunch_scenario,
    "travel_heavy": seed_travel_heavy_scenario,
    "long_context": seed_long_context_scenario,
}


_harbor_dir = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "harbor"


def _make_task_scenario(task_dir_name: str):
    def _scenario(db: Session, user: User, calendars_by_key: dict[str, Calendar], rng) -> int:
        return seed_task_scenario(db, user, calendars_by_key, rng, task_dir_name)

    _scenario.__name__ = f"seed_task_{task_dir_name.replace('-', '_')}"
    _scenario.__doc__ = f"Per-task seed scenario for {task_dir_name}"
    return _scenario


if _harbor_dir.is_dir():
    for _task_dir in sorted(_harbor_dir.iterdir()):
        if _task_dir.is_dir() and (_task_dir / "data" / "needles.py").exists():
            SCENARIOS[f"task:{_task_dir.name}"] = _make_task_scenario(_task_dir.name)


def seed_database(
    scenario: str = "default",
    seed: int = 42,
    db_path: str | None = None,
    num_users: int = 1,
) -> dict:
    """Seed database with deterministic Calendar data."""
    scenario_fn = SCENARIOS.get(scenario)
    if scenario_fn is None:
        raise ValueError(f"Unknown scenario: {scenario!r}. Available: {list(SCENARIOS.keys())}")

    reset_engine()
    init_db(db_path)

    rng = random.Random(seed)
    session_factory = get_session_factory(db_path)
    db = session_factory()

    try:
        user_calendars: list[tuple[User, dict[str, Calendar]]] = []

        for i in range(num_users):
            user = _create_user(i + 1)
            db.add(user)

            calendars_by_key: dict[str, Calendar] = {}
            for template in CALENDAR_TEMPLATES:
                calendar = _create_calendar(db, user=user, template=template)
                calendars_by_key[str(template["key"])] = calendar

            user_calendars.append((user, calendars_by_key))

        db.flush()

        for user, calendars_by_key in user_calendars:
            scenario_fn(db, user, calendars_by_key, rng)

        db.commit()
        take_snapshot("initial")

        return {
            "users": db.query(User).count(),
            "calendars": db.query(Calendar).count(),
            "events": db.query(Event).count(),
            "scenario": scenario,
        }
    finally:
        db.close()
