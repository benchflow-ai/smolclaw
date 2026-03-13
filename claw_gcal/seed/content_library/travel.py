"""Travel-focused event templates and scenario extras."""

from __future__ import annotations

from claw_gcal.seed.content_library.templates import CONFERENCE_DAY_TEMPLATE, TRAVEL_BRIEF_TEMPLATE

TRAVEL_EVENT_POOL: list[dict] = [
    {
        "summary": "Travel: {city}",
        "calendar": "travel",
        "description": TRAVEL_BRIEF_TEMPLATE,
        "location": "Airport",
        "days_from_now_range": (-90, 90),
        "start_hour_choices": [6, 7, 8, 9, 20],
        "duration_hours_choices": [2, 4, 6],
    },
    {
        "summary": "Conference day: {topic}",
        "calendar": "travel",
        "description": CONFERENCE_DAY_TEMPLATE,
        "location": "Convention Center",
        "days_from_now_range": (-90, 90),
        "all_day": True,
        "duration_days": 1,
    },
    {
        "summary": "Hotel check-in: {city}",
        "calendar": "travel",
        "description": TRAVEL_BRIEF_TEMPLATE,
        "location": "{city}",
        "days_from_now_range": (-60, 75),
        "start_hour_choices": [15, 16, 17, 18],
        "duration_hours_choices": [0.5, 1],
    },
    {
        "summary": "Customer dinner: {city}",
        "calendar": "travel",
        "description": "Relationship-building dinner tied to current travel and customer follow-ups.",
        "location": "{city}",
        "days_from_now_range": (-30, 60),
        "start_hour_choices": [18, 19, 20],
        "duration_hours_choices": [1.5, 2],
        "attendees_pool": ["carlos", "omar", "nina"],
        "attendees_count_range": (1, 3),
    },
    {
        "summary": "Airport buffer: {city}",
        "calendar": "travel",
        "description": "Protect buffer time for traffic, check-in, and delayed boarding.",
        "location": "Airport",
        "days_from_now_range": (-45, 60),
        "start_hour_choices": [5, 6, 19, 20],
        "duration_hours_choices": [1, 1.5],
    },
    {
        "summary": "Partner breakfast: {city}",
        "calendar": "travel",
        "description": "Early relationship touchpoint before a packed day of travel or meetings.",
        "location": "{city}",
        "days_from_now_range": (-20, 45),
        "start_hour_choices": [7, 8],
        "duration_hours_choices": [1],
        "attendees_pool": ["carlos", "nina", "kevin"],
        "attendees_count_range": (1, 2),
    },
]

TRAVEL_HEAVY_NEEDLE_EVENTS: list[dict] = [
    {
        "summary": "Conference Keynote Dry Run",
        "calendar": "travel",
        "description": "Final slides, timing, and AV check before the keynote.",
        "location": "Convention Center",
        "days_from_now": 6,
        "start_hour": 17,
        "duration_hours": 1,
        "attendees": ["omar", "carlos"],
    },
    {
        "summary": "Customer Roadshow Dinner",
        "calendar": "travel",
        "description": "Relationship dinner after the customer roadshow.",
        "location": "New York",
        "days_from_now": 8,
        "start_hour": 19,
        "duration_hours": 2,
        "attendees": ["carlos", "nina"],
    },
    {
        "summary": "Expense Report Catch-up",
        "calendar": "primary",
        "description": "Block time to reconcile receipts and outstanding travel expenses.",
        "location": "Home Office",
        "days_from_now": 10,
        "start_hour": 8,
        "duration_hours": 1,
        "attendees": ["helen"],
    },
]

TRAVEL_HEAVY_RECURRING_NEEDLES: list[dict] = [
    {
        "summary": "Trip Logistics Review",
        "calendar": "travel",
        "description": "Weekly review of flights, hotels, and customer-facing prep.",
        "location": "Zoom",
        "days_from_now": -7,
        "start_hour": 15,
        "duration_hours": 0.5,
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO;COUNT=8"],
        "attendees": ["carlos", "helen"],
    },
]
