"""Work, leadership, and recruiting event templates."""

from __future__ import annotations

from claw_gcal.seed.content_library.templates import (
    DESIGN_REVIEW_TEMPLATE,
    INTERVIEW_PANEL_TEMPLATE,
    LEADERSHIP_BRIEF_TEMPLATE,
    WORK_AGENDA_TEMPLATE,
)

WORK_EVENT_POOL: list[dict] = [
    {
        "summary": "{project}: sprint planning",
        "calendar": "team",
        "description": WORK_AGENDA_TEMPLATE,
        "location": "{room}",
        "days_from_now_range": (-21, 45),
        "start_hour_choices": [9, 10, 11, 14, 15],
        "duration_hours_choices": [1, 2],
        "attendees_pool": ["sarah", "marcus", "priya", "james"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Design review: {topic}",
        "calendar": "product",
        "description": DESIGN_REVIEW_TEMPLATE,
        "location": "{room}",
        "days_from_now_range": (-30, 50),
        "start_hour_choices": [10, 13, 15],
        "duration_hours_choices": [1, 1.5],
        "attendees_pool": ["sarah", "marcus", "priya", "elena", "omar"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Leadership prep: {project}",
        "calendar": "leadership",
        "description": LEADERSHIP_BRIEF_TEMPLATE,
        "location": "Board Room C",
        "days_from_now_range": (-20, 35),
        "start_hour_choices": [8, 8.5, 17],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["sarah", "omar", "victor", "helen"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Customer checkpoint: {project}",
        "calendar": "product",
        "description": "Review launch blockers, customer asks, and enablement readiness for {project}.",
        "location": "Zoom",
        "days_from_now_range": (-15, 40),
        "start_hour_choices": [9, 12, 16],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["omar", "carlos", "sarah", "marcus"],
        "attendees_count_range": (2, 3),
    },
    {
        "summary": "Interview loop: {project} candidate",
        "calendar": "recruiting",
        "description": INTERVIEW_PANEL_TEMPLATE,
        "location": "Meet Room A",
        "days_from_now_range": (-20, 30),
        "start_hour_choices": [10, 11, 13, 14],
        "duration_hours_choices": [1, 2],
        "attendees_pool": ["maya", "marcus", "priya", "james", "elena"],
        "attendees_count_range": (2, 4),
    },
]

LAUNCH_CRUNCH_NEEDLE_EVENTS: list[dict] = [
    {
        "summary": "Launch War Room",
        "calendar": "product",
        "description": "Monitor rollout metrics, support load, and rollback triggers in real time.",
        "location": "War Room",
        "days_from_now": 0,
        "start_hour": 9,
        "duration_hours": 3,
        "attendees": ["sarah", "omar", "lisa", "marcus", "priya"],
    },
    {
        "summary": "Executive Status Brief",
        "calendar": "leadership",
        "description": "Share risk summary and launch readiness decision points with leadership.",
        "location": "Board Room C",
        "days_from_now": 0,
        "start_hour": 13,
        "duration_hours": 1,
        "attendees": ["victor", "sarah", "omar", "helen"],
    },
    {
        "summary": "Final Support Coverage Check",
        "calendar": "team",
        "description": "Confirm launch-day staffing, escalation rotations, and incident ownership.",
        "location": "Zoom",
        "days_from_now": -1,
        "start_hour": 17,
        "duration_hours": 0.5,
        "attendees": ["lisa", "marcus", "priya", "james"],
    },
]

LAUNCH_CRUNCH_RECURRING_NEEDLES: list[dict] = [
    {
        "summary": "Launch Command Sync",
        "calendar": "product",
        "description": "Daily launch sync for risks, customers, and dependencies.",
        "location": "War Room",
        "days_from_now": -2,
        "start_hour": 9.5,
        "duration_hours": 0.5,
        "recurrence": ["RRULE:FREQ=DAILY;BYDAY=MO,TU,WE,TH,FR;COUNT=8"],
        "attendees": ["sarah", "omar", "marcus", "lisa"],
    },
]
