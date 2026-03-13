"""Per-task seed scenario for Calendar.

Task-specific seed definitions live in:
    tasks/harbor/<task-name>/data/needles.py

Supported module-level fields in needles.py:
    NEEDLE_EVENTS / NEEDLES      list[dict] or dict[str, dict]
    RECURRING_NEEDLES            list[dict] or dict[str, dict]
    FILL_CONFIG                  {
        "target_count": int,
        "distribution": dict[str, float],
        "include_needles": bool,
    }
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping

from sqlalchemy.orm import Session

from claw_gcal.models import Calendar, User
from claw_gcal.seed.content import (
    DEFAULT_DISTRIBUTION,
    DEFAULT_TARGET_EVENTS,
)
from claw_gcal.seed.long_context import seed_distribution_scenario

_HARBOR_DIR = Path(__file__).resolve().parents[2] / "tasks" / "harbor"


def _load_needles_module(task_dir_name: str):
    needles_path = _HARBOR_DIR / task_dir_name / "data" / "needles.py"
    if not needles_path.exists():
        raise FileNotFoundError(f"Task needles not found: {needles_path}")

    module_name = f"gcal_task_needles_{task_dir_name.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, needles_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load task needles: {needles_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _as_event_list(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        ordered_keys = sorted(raw.keys(), key=lambda key: str(key))
        values = [raw[key] for key in ordered_keys]
    elif isinstance(raw, list):
        values = raw
    else:
        return []

    items: list[dict] = []
    for item in values:
        if isinstance(item, dict):
            items.append(dict(item))
    return items


def _read_task_config(task_dir_name: str) -> tuple[list[dict], list[dict], dict]:
    module = _load_needles_module(task_dir_name)
    needle_events = _as_event_list(
        getattr(module, "NEEDLE_EVENTS", getattr(module, "NEEDLES", None))
    )
    recurring_needles = _as_event_list(getattr(module, "RECURRING_NEEDLES", None))
    fill_config = getattr(module, "FILL_CONFIG", {}) or {}

    return needle_events, recurring_needles, fill_config


def get_task_data_summary(task_dir_name: str) -> dict:
    """Return a compact summary used by admin/debug tools."""
    needles_path = _HARBOR_DIR / task_dir_name / "data" / "needles.py"
    if not needles_path.exists():
        return {"has_per_task_data": False}

    needle_events, recurring_needles, fill_config = _read_task_config(task_dir_name)
    return {
        "has_per_task_data": True,
        "needle_event_count": len(needle_events),
        "recurring_needle_count": len(recurring_needles),
        "fill_config": fill_config,
    }


def seed_task_scenario(
    db: Session,
    user: User,
    calendars_by_key: Mapping[str, Calendar],
    rng,
    task_dir_name: str,
) -> int:
    """Seed a task-specific scenario: task needles + shared distribution fill."""
    needle_events, recurring_needles, fill_config = _read_task_config(task_dir_name)

    target_count = int(fill_config.get("target_count", DEFAULT_TARGET_EVENTS))
    raw_distribution = fill_config.get("distribution", DEFAULT_DISTRIBUTION)
    distribution = raw_distribution if isinstance(raw_distribution, dict) else DEFAULT_DISTRIBUTION
    include_needles = bool(fill_config.get("include_needles", True))

    return seed_distribution_scenario(
        db,
        user,
        calendars_by_key,
        rng,
        target_events=target_count,
        distribution=distribution,
        needle_events=needle_events,
        recurring_needles=recurring_needles,
        include_needles=include_needles,
    )
