"""Curated seed content for Calendar scenarios.

This module is a compatibility facade over the richer content library so the
rest of the seed system can keep importing a stable set of names.
"""

from __future__ import annotations

from claw_gcal.seed.content_library.calendars import CALENDAR_TEMPLATES
from claw_gcal.seed.content_library.needles import NEEDLE_EVENTS, RECURRING_NEEDLES
from claw_gcal.seed.content_library.operations import OPS_EVENT_POOL
from claw_gcal.seed.content_library.parameters import (
    CITY_NAMES,
    MEETING_TOPICS,
    PROJECT_NAMES,
    ROOM_NAMES,
)
from claw_gcal.seed.content_library.personal import PERSONAL_EVENT_POOL
from claw_gcal.seed.content_library.personas import PERSONAS
from claw_gcal.seed.content_library.scenarios import (
    DEFAULT_DISTRIBUTION,
    DEFAULT_TARGET_EVENTS,
    LAUNCH_CRUNCH_DISTRIBUTION,
    LAUNCH_CRUNCH_TARGET_EVENTS,
    LONG_CONTEXT_DISTRIBUTION,
    LONG_CONTEXT_TARGET_EVENTS,
    SCENARIO_DEFINITIONS,
    TRAVEL_HEAVY_DISTRIBUTION,
    TRAVEL_HEAVY_TARGET_EVENTS,
)
from claw_gcal.seed.content_library.travel import TRAVEL_EVENT_POOL
from claw_gcal.seed.content_library.work import WORK_EVENT_POOL

EVENT_POOLS: dict[str, list[dict]] = {
    "work": WORK_EVENT_POOL,
    "ops": OPS_EVENT_POOL,
    "personal": PERSONAL_EVENT_POOL,
    "travel": TRAVEL_EVENT_POOL,
}

__all__ = [
    "CALENDAR_TEMPLATES",
    "CITY_NAMES",
    "DEFAULT_DISTRIBUTION",
    "DEFAULT_TARGET_EVENTS",
    "EVENT_POOLS",
    "LAUNCH_CRUNCH_DISTRIBUTION",
    "LAUNCH_CRUNCH_TARGET_EVENTS",
    "LONG_CONTEXT_DISTRIBUTION",
    "LONG_CONTEXT_TARGET_EVENTS",
    "MEETING_TOPICS",
    "NEEDLE_EVENTS",
    "OPS_EVENT_POOL",
    "PERSONAL_EVENT_POOL",
    "PERSONAS",
    "PROJECT_NAMES",
    "RECURRING_NEEDLES",
    "ROOM_NAMES",
    "SCENARIO_DEFINITIONS",
    "TRAVEL_EVENT_POOL",
    "TRAVEL_HEAVY_DISTRIBUTION",
    "TRAVEL_HEAVY_TARGET_EVENTS",
    "WORK_EVENT_POOL",
]
