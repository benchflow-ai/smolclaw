#!/usr/bin/env python3
"""Validate that Calendar seed data preserves key mock invariants.

Usage:
    python scripts/validate_gcal_seed.py
    python scripts/validate_gcal_seed.py --scenario default
    python scripts/validate_gcal_seed.py --db existing.db --scenario long_context
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))

from claw_gcal.models.base import resolve_db_path
from claw_gcal.seed.content import (
    CALENDAR_TEMPLATES,
    SCENARIO_DEFINITIONS,
    RECURRING_NEEDLES,
)


def _load_summary(db_path: str) -> dict[str, int | str]:
    path = resolve_db_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        user = conn.execute("SELECT * FROM users LIMIT 1").fetchone()
        counts = {
            "users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "calendars": conn.execute("SELECT COUNT(*) FROM calendars").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "recurring": conn.execute(
                "SELECT COUNT(*) FROM events WHERE recurrence_json != '[]'"
            ).fetchone()[0],
            "cancelled": conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'cancelled'"
            ).fetchone()[0],
            "all_day": conn.execute(
                "SELECT COUNT(*) FROM events WHERE start_is_date = 1 AND end_is_date = 1"
            ).fetchone()[0],
            "non_primary": conn.execute(
                """
                SELECT COUNT(*)
                FROM events e
                JOIN calendars c ON c.id = e.calendar_id
                WHERE c.is_primary = 0
                """
            ).fetchone()[0],
            "calendar_coverage": conn.execute(
                "SELECT COUNT(DISTINCT calendar_id) FROM events"
            ).fetchone()[0],
            "email": user["email_address"] if user else "",
        }
    finally:
        conn.close()

    return counts


def validate(db_path: str, scenario: str) -> bool:
    summary = _load_summary(db_path)
    scenario_config = SCENARIO_DEFINITIONS[scenario]

    print(f"Scenario: {scenario}")
    print(f"User email: {summary['email']}")
    print(f"Users: {summary['users']}")
    print(f"Calendars: {summary['calendars']}")
    print(f"Events: {summary['events']}")
    print(f"Recurring: {summary['recurring']}")
    print(f"Cancelled: {summary['cancelled']}")
    print(f"All-day: {summary['all_day']}")
    print(f"Non-primary events: {summary['non_primary']}")
    print(f"Calendar coverage: {summary['calendar_coverage']}")

    errors: list[str] = []
    expected_events = int(scenario_config["target_events"])
    needle_events = scenario_config["needle_events"] if scenario_config["include_needles"] else []
    recurring_needles = (
        scenario_config["recurring_needles"] if scenario_config["include_needles"] else []
    )
    expected_cancelled = sum(1 for event in needle_events if event.get("status") == "cancelled")
    expected_all_day = sum(1 for event in needle_events if event.get("all_day"))
    if summary["users"] != 1:
        errors.append(f"users={summary['users']} != 1")
    if summary["calendars"] != len(CALENDAR_TEMPLATES):
        errors.append(f"calendars={summary['calendars']} != {len(CALENDAR_TEMPLATES)}")
    if summary["events"] != expected_events:
        errors.append(f"events={summary['events']} != {expected_events}")
    if summary["recurring"] < len(recurring_needles):
        errors.append(
            f"recurring={summary['recurring']} < {len(recurring_needles)} recurring needles"
        )
    if summary["all_day"] < expected_all_day:
        errors.append(f"all_day={summary['all_day']} < {expected_all_day}")
    if summary["non_primary"] < 1:
        errors.append("non_primary=0")
    if summary["calendar_coverage"] != len(CALENDAR_TEMPLATES):
        errors.append(
            f"calendar_coverage={summary['calendar_coverage']} != {len(CALENDAR_TEMPLATES)}"
        )
    if summary["email"] != "alex@nexusai.com":
        errors.append(f"email={summary['email']} != alex@nexusai.com")

    if expected_cancelled and summary["cancelled"] < expected_cancelled:
        errors.append(f"cancelled={summary['cancelled']} < {expected_cancelled}")

    if errors:
        print(f"\nFAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nALL CHECKS PASSED")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Existing db to validate (skip seeding)")
    parser.add_argument(
        "--scenario",
        default="long_context",
        choices=sorted(SCENARIO_DEFINITIONS.keys()),
        help="Scenario to seed or validate",
    )
    args = parser.parse_args()

    if args.db:
        ok = validate(args.db, args.scenario)
        sys.exit(0 if ok else 1)

    db_name = f"validate_gcal_{args.scenario}.db"
    path = resolve_db_path(db_name)
    for suffix in ("", "-shm", "-wal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()

    print(f"Seeding {db_name}...")
    from claw_gcal.models import reset_engine
    from claw_gcal.seed.generator import seed_database

    reset_engine()
    seed_database(scenario=args.scenario, db_path=db_name)
    print()
    ok = validate(db_name, args.scenario)

    for suffix in ("", "-shm", "-wal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
