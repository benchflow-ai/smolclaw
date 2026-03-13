"""Personal and family event templates."""

from __future__ import annotations

from claw_gcal.seed.content_library.templates import DEEP_WORK_TEMPLATE, FAMILY_PLAN_TEMPLATE

PERSONAL_EVENT_POOL: list[dict] = [
    {
        "summary": "Coffee with {persona}",
        "calendar": "family",
        "description": "Catch up and life updates.",
        "location": "{city}",
        "days_from_now_range": (-14, 30),
        "start_hour_choices": [8, 9, 17, 18],
        "duration_hours_choices": [1, 1.5],
        "attendees_pool": ["amy", "david", "zoe"],
        "attendees_count_range": (1, 1),
    },
    {
        "summary": "Deep work: {topic}",
        "calendar": "primary",
        "description": DEEP_WORK_TEMPLATE,
        "location": "Home Office",
        "days_from_now_range": (-20, 40),
        "start_hour_choices": [7, 8, 13],
        "duration_hours_choices": [1, 2, 3],
    },
    {
        "summary": "Family logistics check-in",
        "calendar": "family",
        "description": FAMILY_PLAN_TEMPLATE,
        "location": "Home",
        "days_from_now_range": (-10, 30),
        "start_hour_choices": [18, 19],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["zoe", "amy"],
        "attendees_count_range": (1, 2),
    },
    {
        "summary": "School pickup coverage",
        "calendar": "family",
        "description": FAMILY_PLAN_TEMPLATE,
        "location": "Mission District",
        "days_from_now_range": (-7, 21),
        "start_hour_choices": [15.5, 16, 17],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["zoe"],
        "attendees_count_range": (1, 1),
    },
    {
        "summary": "Dentist appointment",
        "calendar": "primary",
        "description": "Routine check-up. Leave transit buffer before and after.",
        "location": "{city}",
        "days_from_now_range": (-12, 35),
        "start_hour_choices": [8, 9, 14],
        "duration_hours_choices": [1],
    },
    {
        "summary": "Date night",
        "calendar": "family",
        "description": "Keep the evening clear and avoid back-to-back work obligations.",
        "location": "{city}",
        "days_from_now_range": (-10, 40),
        "start_hour_choices": [18.5, 19, 20],
        "duration_hours_choices": [2, 3],
        "attendees_pool": ["amy", "zoe"],
        "attendees_count_range": (1, 2),
    },
]
