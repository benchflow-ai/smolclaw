"""Security, compliance, and vendor-risk event templates."""

from __future__ import annotations

from claw_gcal.seed.content_library.templates import (
    ACCESS_REVIEW_TEMPLATE,
    COMPLIANCE_CHECK_TEMPLATE,
    SECURITY_DRILL_TEMPLATE,
    VENDOR_RISK_TEMPLATE,
)

SECURITY_EVENT_POOL: list[dict] = [
    {
        "summary": "Security review: {project}",
        "calendar": "team",
        "description": SECURITY_DRILL_TEMPLATE,
        "location": "War Room",
        "days_from_now_range": (-35, 28),
        "start_hour_choices": [9, 11, 15],
        "duration_hours_choices": [1, 1.5],
        "attendees_pool": ["ian", "marcus", "priya", "james"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Access review: {topic}",
        "calendar": "leadership",
        "description": ACCESS_REVIEW_TEMPLATE,
        "location": "Board Room C",
        "days_from_now_range": (-21, 35),
        "start_hour_choices": [8, 10, 16],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["ian", "helen", "victor", "rachel"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Compliance evidence check: {project}",
        "calendar": "product",
        "description": COMPLIANCE_CHECK_TEMPLATE,
        "location": "Zoom",
        "days_from_now_range": (-30, 45),
        "start_hour_choices": [9, 13, 16],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["rachel", "tanya", "helen", "omar"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Vendor access checkpoint",
        "calendar": "product",
        "description": VENDOR_RISK_TEMPLATE,
        "location": "Zoom",
        "days_from_now_range": (-18, 32),
        "start_hour_choices": [10, 14, 17],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["ian", "nina", "rachel", "omar"],
        "attendees_count_range": (2, 4),
    },
    {
        "summary": "Phishing simulation retro",
        "calendar": "team",
        "description": "Review the latest phishing simulation, missed detections, and follow-up training.",
        "location": "Zoom",
        "days_from_now_range": (-28, 25),
        "start_hour_choices": [11, 15],
        "duration_hours_choices": [0.5, 1],
        "attendees_pool": ["ian", "marcus", "tanya"],
        "attendees_count_range": (2, 3),
    },
]

SECURITY_NEEDLE_EVENTS: list[dict] = [
    {
        "summary": "Quarterly Access Audit",
        "calendar": "leadership",
        "description": "Validate admin roles, shared inbox access, and vendor exceptions before sign-off.",
        "location": "Board Room C",
        "days_from_now": 4,
        "start_hour": 9,
        "duration_hours": 1,
        "attendees": ["ian", "rachel", "helen", "victor"],
    },
    {
        "summary": "SOC 2 Evidence Freeze",
        "calendar": "product",
        "description": "Lock the audit evidence set and note any missing approvals before handoff.",
        "location": "Zoom",
        "days_from_now": 6,
        "start_hour": 14,
        "duration_hours": 1,
        "attendees": ["rachel", "tanya", "helen"],
    },
]

SECURITY_RECURRING_NEEDLES: list[dict] = [
    {
        "summary": "Weekly Security Triage",
        "calendar": "team",
        "description": "Review new findings, high-risk dependencies, and follow-up owners.",
        "location": "War Room",
        "days_from_now": -2,
        "start_hour": 13,
        "duration_hours": 0.5,
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=TH;COUNT=10"],
        "attendees": ["ian", "marcus", "priya"],
    },
]
