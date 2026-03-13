"""Reusable description snippets for event templates."""

from __future__ import annotations

WORK_AGENDA_TEMPLATE = (
    "Agenda:\n"
    "- Review status for {project}\n"
    "- Discuss blockers related to {topic}\n"
    "- Confirm owners and next steps\n"
    "- Capture follow-ups in the team tracker"
)

DESIGN_REVIEW_TEMPLATE = (
    "Review goals:\n"
    "- Validate the current proposal for {topic}\n"
    "- Align on tradeoffs, scope, and launch timing\n"
    "- Confirm what needs an async follow-up before sign-off"
)

LEADERSHIP_BRIEF_TEMPLATE = (
    "Prep notes:\n"
    "- Summarize current momentum on {project}\n"
    "- Call out staffing, risk, and dependency changes\n"
    "- Bring a clear recommendation for the decision owner"
)

INTERVIEW_PANEL_TEMPLATE = (
    "Loop plan:\n"
    "- Review role rubric and scorecard expectations\n"
    "- Assign interview focus areas\n"
    "- Reserve 15 minutes for a same-day debrief"
)

INCIDENT_DRILL_TEMPLATE = (
    "Exercise plan:\n"
    "- Trigger scenario for {project}\n"
    "- Confirm escalation path and rollback readiness\n"
    "- Document gaps in runbooks, ownership, or tooling"
)

SECURITY_DRILL_TEMPLATE = (
    "Security checklist:\n"
    "- Review high-signal alerts tied to {project}\n"
    "- Confirm escalation paths, access scope, and customer impact\n"
    "- Capture follow-up remediation owners before closing the review"
)

ACCESS_REVIEW_TEMPLATE = (
    "Access review:\n"
    "- Inspect admin roles and stale shared access related to {topic}\n"
    "- Verify approvals for exceptions, vendors, and temporary grants\n"
    "- Document anything that must be revoked this week"
)

COMPLIANCE_CHECK_TEMPLATE = (
    "Audit prep:\n"
    "- Check evidence completeness for {project}\n"
    "- Confirm sign-offs, screenshots, and linked tickets are current\n"
    "- Flag any control that still needs an owner before audit handoff"
)

VENDOR_RISK_TEMPLATE = (
    "Vendor checkpoint:\n"
    "- Review open security questionnaires and shared data paths\n"
    "- Confirm who still needs access to {project} environments\n"
    "- Capture contract or remediation blockers for leadership follow-up"
)

LAUNCH_WAR_ROOM_TEMPLATE = (
    "Launch checklist:\n"
    "- Review traffic ramp and support coverage\n"
    "- Confirm comms owners and rollback criteria\n"
    "- Track risks, mitigations, and external updates live"
)

DEEP_WORK_TEMPLATE = (
    "Focus block:\n"
    "- Protect time for strategy work on {topic}\n"
    "- Silence notifications and defer meetings when possible\n"
    "- Leave notes for the next unblocker before wrapping"
)

FAMILY_PLAN_TEMPLATE = (
    "Personal logistics:\n"
    "- Confirm location, timing, and shared responsibilities\n"
    "- Leave buffer for transit and follow-up errands\n"
    "- Capture anything that needs rescheduling"
)

TRAVEL_BRIEF_TEMPLATE = (
    "Travel brief:\n"
    "- Confirm departure, arrival, and hotel timing for {city}\n"
    "- Keep attendee notes and customer context handy\n"
    "- Leave space for transit delays or venue changes"
)

CONFERENCE_DAY_TEMPLATE = (
    "On-site plan:\n"
    "- Prioritize high-signal sessions tied to {topic}\n"
    "- Reserve time for customer meetings and follow-ups\n"
    "- Capture quick notes before the end of the day"
)
