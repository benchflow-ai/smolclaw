"""Operational event templates for seed filling."""

from __future__ import annotations

from claw_gcal.seed.content_library.templates import (
    INCIDENT_DRILL_TEMPLATE,
    LAUNCH_WAR_ROOM_TEMPLATE,
)

OPS_EVENT_POOL: list[dict] = [
    {
        "summary": "Incident drill: {project}",
        "calendar": "product",
        "description": INCIDENT_DRILL_TEMPLATE,
        "location": "War Room",
        "days_from_now_range": (-40, 35),
        "start_hour_choices": [10, 14, 16],
        "duration_hours_choices": [1, 2],
        "attendees_pool": ["marcus", "priya", "james", "lisa"],
        "attendees_count_range": (2, 4),
        "cancelled_ratio": 0.08,
    },
    {
        "summary": "On-call handoff",
        "calendar": "team",
        "description": "Weekly handoff with incident context, escalations, and pending risks.",
        "location": "Slack Huddle",
        "days_from_now_range": (-60, 30),
        "start_hour_choices": [8, 9, 17],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["marcus", "priya", "james"],
        "attendees_count_range": (1, 2),
    },
    {
        "summary": "Support capacity review",
        "calendar": "team",
        "description": "Review queue health, staffing gaps, and escalation trends for the week.",
        "location": "Zoom",
        "days_from_now_range": (-35, 28),
        "start_hour_choices": [9, 15],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["lisa", "marcus", "sarah"],
        "attendees_count_range": (2, 3),
    },
    {
        "summary": "Launch readiness checkpoint",
        "calendar": "product",
        "description": LAUNCH_WAR_ROOM_TEMPLATE,
        "location": "War Room",
        "days_from_now_range": (-14, 20),
        "start_hour_choices": [11, 14, 16],
        "duration_hours_choices": [1, 1.5],
        "attendees_pool": ["sarah", "omar", "lisa", "helen"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Change freeze review",
        "calendar": "team",
        "description": "Confirm exceptions, rollback coverage, and the owners approved to ship during freeze.",
        "location": "Zoom",
        "days_from_now_range": (-12, 18),
        "start_hour_choices": [9, 12, 16],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["lisa", "marcus", "priya", "james"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Rollback practice: {project}",
        "calendar": "product",
        "description": INCIDENT_DRILL_TEMPLATE,
        "location": "War Room",
        "days_from_now_range": (-24, 20),
        "start_hour_choices": [10, 15],
        "duration_hours_choices": [1, 1.5],
        "attendees_pool": ["james", "marcus", "priya", "lisa"],
        "attendees_count_range": (2, 4),
    },
]
