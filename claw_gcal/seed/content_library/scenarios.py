"""Named scenario presets for Calendar seeding."""

from __future__ import annotations

from claw_gcal.seed.content_library.needles import NEEDLE_EVENTS, RECURRING_NEEDLES
from claw_gcal.seed.content_library.travel import (
    TRAVEL_HEAVY_NEEDLE_EVENTS,
    TRAVEL_HEAVY_RECURRING_NEEDLES,
)
from claw_gcal.seed.content_library.work import (
    LAUNCH_CRUNCH_NEEDLE_EVENTS,
    LAUNCH_CRUNCH_RECURRING_NEEDLES,
)

DEFAULT_TARGET_EVENTS = 72
DEFAULT_DISTRIBUTION: dict[str, float] = {
    "work": 0.40,
    "ops": 0.20,
    "personal": 0.22,
    "travel": 0.18,
}

LONG_CONTEXT_TARGET_EVENTS = 1400
LONG_CONTEXT_DISTRIBUTION: dict[str, float] = {
    "work": 0.38,
    "ops": 0.24,
    "personal": 0.18,
    "travel": 0.20,
}

LAUNCH_CRUNCH_TARGET_EVENTS = 96
LAUNCH_CRUNCH_DISTRIBUTION: dict[str, float] = {
    "work": 0.44,
    "ops": 0.28,
    "personal": 0.10,
    "travel": 0.18,
}

TRAVEL_HEAVY_TARGET_EVENTS = 88
TRAVEL_HEAVY_DISTRIBUTION: dict[str, float] = {
    "work": 0.18,
    "ops": 0.10,
    "personal": 0.18,
    "travel": 0.54,
}

SCENARIO_DEFINITIONS: dict[str, dict] = {
    "default": {
        "description": "Balanced work/personal calendar with recurring meetings and a few edge cases.",
        "target_events": DEFAULT_TARGET_EVENTS,
        "distribution": DEFAULT_DISTRIBUTION,
        "needle_events": NEEDLE_EVENTS,
        "recurring_needles": RECURRING_NEEDLES,
        "include_needles": True,
    },
    "launch_crunch": {
        "description": "Launch-heavy week with leadership syncs, war rooms, and extra operational load.",
        "target_events": LAUNCH_CRUNCH_TARGET_EVENTS,
        "distribution": LAUNCH_CRUNCH_DISTRIBUTION,
        "needle_events": NEEDLE_EVENTS + LAUNCH_CRUNCH_NEEDLE_EVENTS,
        "recurring_needles": RECURRING_NEEDLES + LAUNCH_CRUNCH_RECURRING_NEEDLES,
        "include_needles": True,
    },
    "travel_heavy": {
        "description": "Conference and customer travel with denser logistics and follow-up work.",
        "target_events": TRAVEL_HEAVY_TARGET_EVENTS,
        "distribution": TRAVEL_HEAVY_DISTRIBUTION,
        "needle_events": NEEDLE_EVENTS + TRAVEL_HEAVY_NEEDLE_EVENTS,
        "recurring_needles": RECURRING_NEEDLES + TRAVEL_HEAVY_RECURRING_NEEDLES,
        "include_needles": True,
    },
    "long_context": {
        "description": "High-volume calendar with dense event history and recurring patterns.",
        "target_events": LONG_CONTEXT_TARGET_EVENTS,
        "distribution": LONG_CONTEXT_DISTRIBUTION,
        "needle_events": NEEDLE_EVENTS,
        "recurring_needles": RECURRING_NEEDLES,
        "include_needles": True,
    },
}
