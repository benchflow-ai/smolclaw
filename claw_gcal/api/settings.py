"""Calendar settings endpoints."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from claw_gcal.models import User
from claw_gcal.state.channels import channel_registry

from .deps import get_db, resolve_user_id
from .schemas import (
    CalendarSetting,
    CalendarSettingsListResponse,
    ChannelRequest,
    ChannelResponse,
)

router = APIRouter()

_SETTINGS_TEMPLATE: list[tuple[str, str]] = [
    ("autoAddHangouts", "true"),
    ("defaultEventLength", "60"),
    ("dateFieldOrder", "MDY"),
    ("weekStart", "0"),
    ("format24HourTime", "false"),
    ("hideInvitations", "false"),
    ("hideInvitationsSetting", "NONE"),
    ("locale", "en"),
    ("remindOnRespondedEventsOnly", "false"),
    ("showDeclinedEvents", "true"),
    ("timezone", "UTC"),
    ("useKeyboardShortcuts", "true"),
    ("hideWeekends", "false"),
]


def _md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _settings_for_user(user: User | None) -> list[CalendarSetting]:
    timezone_value = user.timezone if user and user.timezone else "UTC"
    items: list[CalendarSetting] = []
    for setting_id, default_value in _SETTINGS_TEMPLATE:
        value = timezone_value if setting_id == "timezone" else default_value
        etag = f'"{_md5_hex(f"{setting_id}:{value}")}"'
        items.append(
            CalendarSetting(
                etag=etag,
                id=setting_id,
                value=value,
            )
        )
    return items


@router.get(
    "/users/{userId}/settings",
    response_model=CalendarSettingsListResponse,
    response_model_exclude_none=True,
)
def settings_list(
    userId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    items = _settings_for_user(user)
    etag_raw = "|".join(i.etag for i in items)
    return CalendarSettingsListResponse(
        etag=f'"{_md5_hex(etag_raw)}"',
        items=items,
        nextSyncToken=_md5_hex(f"{_user_id}:{etag_raw}"),
    )


@router.get(
    "/users/{userId}/settings/{setting}",
    response_model=CalendarSetting,
    response_model_exclude_none=True,
)
def settings_get(
    userId: str,
    setting: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    user = db.query(User).filter(User.id == _user_id).first()
    settings = {s.id: s for s in _settings_for_user(user)}
    item = settings.get(setting)
    if not item:
        raise HTTPException(404, "Setting not found")
    return item


@router.post(
    "/users/{userId}/settings/watch",
    response_model=ChannelResponse,
    response_model_exclude_none=True,
)
def settings_watch(
    userId: str,
    body: ChannelRequest,
    _user_id: str = Depends(resolve_user_id),
):
    resource_uri = "https://www.googleapis.com/calendar/v3/users/me/settings?alt=json"
    return channel_registry.register(
        resource_uri=resource_uri,
        channel_id=body.id,
        address=body.address,
        token=body.token,
        channel_type=body.type,
        payload=body.payload,
        params=body.params,
        expiration=body.expiration,
    )
