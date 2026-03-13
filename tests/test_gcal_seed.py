"""Tests for gcal seed skeleton and default distributions."""

from __future__ import annotations

from datetime import timedelta

from claw_gcal.models import Calendar, Event, get_session_factory, init_db, reset_engine
from claw_gcal.seed.content import (
    CALENDAR_TEMPLATES,
    DEFAULT_TARGET_EVENTS,
    EVENT_POOLS,
    LONG_CONTEXT_TARGET_EVENTS,
    PERSONAS,
    RECURRING_NEEDLES,
    SCENARIO_DEFINITIONS,
)
from claw_gcal.seed.generator import SCENARIOS, seed_database


def _open_db(db_path: str):
    reset_engine()
    init_db(db_path)
    return get_session_factory(db_path)()


def test_scenarios_include_default_and_long_context():
    assert "default" in SCENARIOS
    assert "launch_crunch" in SCENARIOS
    assert "travel_heavy" in SCENARIOS
    assert "long_context" in SCENARIOS


def test_seed_content_is_richer_and_more_modular():
    assert len(PERSONAS) >= 19
    assert "security" in EVENT_POOLS
    assert len(EVENT_POOLS["security"]) >= 5
    assert sum(len(pool) for pool in EVENT_POOLS.values()) >= 31
    assert {"default", "launch_crunch", "travel_heavy", "long_context"} <= set(SCENARIO_DEFINITIONS)


def test_default_seed_distribution_and_coverage(tmp_path):
    db_path = str(tmp_path / "gcal_default_seed.db")
    reset_engine()
    result = seed_database(scenario="default", seed=42, db_path=db_path, num_users=1)

    assert result["users"] == 1
    assert result["calendars"] == len(CALENDAR_TEMPLATES)
    assert result["events"] == DEFAULT_TARGET_EVENTS

    db = _open_db(db_path)
    try:
        primary = (
            db.query(Calendar)
            .filter(Calendar.user_id == "user1", Calendar.is_primary.is_(True))
            .one()
        )
        events = db.query(Event).all()

        recurring = [e for e in events if e.recurrence_json != "[]"]
        cancelled = [e for e in events if e.status == "cancelled"]
        all_day = [
            e
            for e in events
            if e.start_dt.hour == 0
            and e.start_dt.minute == 0
            and (e.end_dt - e.start_dt) >= timedelta(days=1)
        ]
        non_primary = [e for e in events if e.calendar_id != primary.id]
        covered_calendar_ids = {e.calendar_id for e in events}
        security_flavored = [
            e
            for e in events
            if any(token in (e.summary or "") for token in ("Security", "Access", "Compliance", "Vendor"))
        ]

        assert len(recurring) >= len(RECURRING_NEEDLES)
        assert len(cancelled) >= 1
        assert len(all_day) >= 1
        assert len(non_primary) >= 1
        assert len(covered_calendar_ids) == len(CALENDAR_TEMPLATES)
        assert len(security_flavored) >= 1
    finally:
        db.close()
        reset_engine()


def test_named_scenarios_hit_target_counts(tmp_path):
    for scenario_name in ("default", "launch_crunch", "travel_heavy"):
        db_path = str(tmp_path / f"{scenario_name}.db")
        reset_engine()
        result = seed_database(scenario=scenario_name, seed=11, db_path=db_path, num_users=1)

        assert result["events"] == SCENARIO_DEFINITIONS[scenario_name]["target_events"]

        db = _open_db(db_path)
        try:
            assert db.query(Event).count() == SCENARIO_DEFINITIONS[scenario_name]["target_events"]
            assert db.query(Event).filter(Event.recurrence_json != "[]").count() >= len(
                SCENARIO_DEFINITIONS[scenario_name]["recurring_needles"]
            )
        finally:
            db.close()
            reset_engine()


def test_long_context_seed_target_count(tmp_path):
    db_path = str(tmp_path / "gcal_long_seed.db")
    reset_engine()
    result = seed_database(scenario="long_context", seed=7, db_path=db_path, num_users=1)

    assert result["users"] == 1
    assert result["events"] == LONG_CONTEXT_TARGET_EVENTS
    assert result["events"] > DEFAULT_TARGET_EVENTS

    db = _open_db(db_path)
    try:
        recurring_count = db.query(Event).filter(Event.recurrence_json != "[]").count()
        assert recurring_count >= len(RECURRING_NEEDLES)
    finally:
        db.close()
        reset_engine()


def test_default_seed_scales_with_user_count(tmp_path):
    db_path = str(tmp_path / "gcal_multi_user_seed.db")
    reset_engine()
    result = seed_database(scenario="default", seed=99, db_path=db_path, num_users=2)

    assert result["users"] == 2
    assert result["calendars"] == len(CALENDAR_TEMPLATES) * 2
    assert result["events"] == DEFAULT_TARGET_EVENTS * 2
    reset_engine()
