"""Calendar list and calendar resource endpoints."""

from __future__ import annotations

import hashlib
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from claw_gcal.models import AclRule, Calendar, Event, User
from claw_gcal.state.channels import channel_registry

from .deps import get_db, resolve_actor_user_id, resolve_user_id
from .schemas import (
    CalendarInsertRequest,
    CalendarListEntry,
    CalendarListMutationRequest,
    CalendarListResponse,
    CalendarPatchRequest,
    CalendarResource,
    CalendarUpdateRequest,
    ChannelRequest,
    ChannelResponse,
    ConferenceProperties,
    NotificationRule,
    NotificationSettings,
    ReminderOverride,
)

router = APIRouter()

_CALENDAR_COLOR_MAP: dict[str, tuple[str, str]] = {
    "9": ("#7bd148", "#1d1d1d"),
    "14": ("#9fe1e7", "#1d1d1d"),
}


def _md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _etag_for_calendar(calendar: Calendar) -> str:
    raw = (
        f"{calendar.id}:{calendar.summary}:{calendar.description}:"
        f"{calendar.location}:{calendar.timezone}:{calendar.selected}:"
        f"{calendar.summary_override}:{calendar.hidden}:{calendar.color_id}"
    )
    return f'"{_md5_hex(raw)}"'


def _calendar_colors(calendar: Calendar) -> tuple[str, str]:
    return _CALENDAR_COLOR_MAP.get(calendar.color_id, ("#7bd148", "#1d1d1d"))


def _default_notification_settings() -> NotificationSettings:
    return NotificationSettings(
        notifications=[
            NotificationRule(type="eventCreation", method="email"),
            NotificationRule(type="eventChange", method="email"),
            NotificationRule(type="eventCancellation", method="email"),
            NotificationRule(type="eventResponse", method="email"),
        ]
    )


def _to_calendar_entry(
    calendar: Calendar,
    *,
    data_owner: str | None = None,
    include_empty_description: bool = False,
) -> CalendarListEntry:
    bg, fg = _calendar_colors(calendar)
    summary_override = calendar.summary_override if calendar.summary_override else None
    include_data_owner = (not calendar.is_primary) and bool(data_owner)
    include_description = (
        not calendar.is_primary
        and (bool(calendar.description) or include_empty_description)
    )

    return CalendarListEntry(
        etag=_etag_for_calendar(calendar),
        id=calendar.id,
        summary=calendar.summary,
        description=calendar.description if include_description else None,
        location=calendar.location or None,
        timeZone=calendar.timezone,
        accessRole=calendar.access_role,
        primary=True if calendar.is_primary else None,
        selected=True if calendar.selected else None,
        hidden=calendar.hidden if calendar.hidden else None,
        autoAcceptInvitations=calendar.auto_accept_invitations or None,
        colorId=calendar.color_id,
        backgroundColor=bg,
        foregroundColor=fg,
        conferenceProperties=ConferenceProperties(),
        defaultReminders=(
            [ReminderOverride(method="popup", minutes=10)] if calendar.is_primary else []
        ),
        notificationSettings=_default_notification_settings() if calendar.is_primary else None,
        dataOwner=data_owner if include_data_owner else None,
        summaryOverride=summary_override,
    )


def _to_calendar_resource(
    calendar: Calendar,
    *,
    data_owner: str | None = None,
    include_empty_description: bool = False,
) -> CalendarResource:
    include_data_owner = (not calendar.is_primary) and bool(data_owner)
    include_description = (
        not calendar.is_primary
        and (bool(calendar.description) or include_empty_description)
    )
    return CalendarResource(
        etag=_etag_for_calendar(calendar),
        id=calendar.id,
        summary=calendar.summary,
        description=calendar.description if include_description else None,
        location=calendar.location or None,
        timeZone=calendar.timezone,
        conferenceProperties=ConferenceProperties(),
        dataOwner=data_owner if include_data_owner else None,
        autoAcceptInvitations=calendar.auto_accept_invitations or None,
    )


def _resolve_calendar(db: Session, user_id: str, calendar_id: str) -> Calendar:
    if calendar_id == "primary":
        cal = db.query(Calendar).filter(
            Calendar.user_id == user_id,
            Calendar.is_primary.is_(True),
        ).first()
    else:
        cal = db.query(Calendar).filter(
            Calendar.id == calendar_id,
            Calendar.user_id == user_id,
        ).first()

    if not cal:
        raise HTTPException(404, "Calendar not found")
    return cal


def _owner_email(db: Session, user_id: str) -> str | None:
    user = db.query(User).filter(User.id == user_id).first()
    return user.email_address if user else None


def _owner_acl_id(calendar_id: str, owner_email: str) -> str:
    return f"{calendar_id}:user:{owner_email}"


@router.get(
    "/users/{userId}/calendarList",
    response_model=CalendarListResponse,
    response_model_exclude_none=True,
)
def calendar_list(
    userId: str,
    maxResults: int = Query(100, ge=1, le=250),
    pageToken: str | None = Query(None),
    minAccessRole: str | None = Query(None),
    showHidden: bool = Query(False),
    syncToken: str | None = Query(None),
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    query = db.query(Calendar).filter(Calendar.user_id == _user_id)
    if not showHidden:
        query = query.filter(Calendar.hidden.is_(False))
    calendars = query.order_by(Calendar.is_primary.desc(), Calendar.summary.asc()).all()

    if minAccessRole:
        role_rank = {
            "none": 0,
            "freeBusyReader": 1,
            "reader": 2,
            "writer": 3,
            "owner": 4,
        }
        min_rank = role_rank.get(minAccessRole)
        if min_rank is not None:
            calendars = [
                c for c in calendars if role_rank.get(c.access_role, 0) >= min_rank
            ]

    offset = 0
    if pageToken:
        try:
            offset = int(pageToken)
        except ValueError as exc:
            raise HTTPException(400, "Invalid pageToken") from exc

    total_count = len(calendars)
    sliced = calendars[offset : offset + maxResults]

    owner = _owner_email(db, _user_id)
    items = [_to_calendar_entry(c, data_owner=owner) for c in sliced]

    list_etag_raw = "|".join(item.etag for item in items)
    next_page_token = str(offset + maxResults) if total_count > (offset + maxResults) else None
    next_sync_token = (
        _md5_hex(f"{_user_id}:{total_count}:{list_etag_raw}:{syncToken or ''}")
        if not next_page_token
        else None
    )
    return CalendarListResponse(
        etag=f'"{_md5_hex(list_etag_raw)}"',
        items=items,
        nextPageToken=next_page_token,
        nextSyncToken=next_sync_token,
    )


@router.get(
    "/users/{userId}/calendarList/{calendarId}",
    response_model=CalendarListEntry,
    response_model_exclude_none=True,
)
def calendar_list_get(
    userId: str,
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    return _to_calendar_entry(calendar, data_owner=_owner_email(db, _user_id))


@router.post(
    "/users/{userId}/calendarList",
    response_model=CalendarListEntry,
    response_model_exclude_none=True,
)
def calendar_list_insert(
    userId: str,
    body: CalendarListMutationRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    if not body.id:
        raise HTTPException(400, "Missing required field: id")
    calendar = _resolve_calendar(db, _user_id, body.id)
    if body.id == "primary" or calendar.is_primary:
        raise HTTPException(400, "Invalid resource id value.")

    calendar.hidden = False
    if body.selected is None:
        calendar.selected = True
    else:
        calendar.selected = body.selected
    if body.summaryOverride is not None:
        calendar.summary_override = body.summaryOverride
    if body.colorId is not None:
        calendar.color_id = body.colorId

    db.commit()
    db.refresh(calendar)
    return _to_calendar_entry(calendar, data_owner=_owner_email(db, _user_id))


def _apply_calendar_list_mutation(
    calendar: Calendar,
    body: CalendarListMutationRequest,
    *,
    is_update: bool,
):
    if is_update:
        calendar.selected = bool(body.selected) if body.selected is not None else True
        calendar.summary_override = body.summaryOverride or ""
        calendar.hidden = bool(body.hidden) if body.hidden is not None else False
        if body.colorId is not None:
            calendar.color_id = body.colorId
    else:
        if body.selected is not None:
            calendar.selected = body.selected
        if body.summaryOverride is not None:
            calendar.summary_override = body.summaryOverride
        if body.hidden is not None:
            calendar.hidden = body.hidden
        if body.colorId is not None:
            calendar.color_id = body.colorId


@router.patch(
    "/users/{userId}/calendarList/{calendarId}",
    response_model=CalendarListEntry,
    response_model_exclude_none=True,
)
def calendar_list_patch(
    userId: str,
    calendarId: str,
    body: CalendarListMutationRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    _apply_calendar_list_mutation(calendar, body, is_update=False)
    db.commit()
    db.refresh(calendar)
    return _to_calendar_entry(
        calendar,
        data_owner=_owner_email(db, _user_id),
        include_empty_description=not calendar.is_primary,
    )


@router.put(
    "/users/{userId}/calendarList/{calendarId}",
    response_model=CalendarListEntry,
    response_model_exclude_none=True,
)
def calendar_list_update(
    userId: str,
    calendarId: str,
    body: CalendarListMutationRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    _apply_calendar_list_mutation(calendar, body, is_update=True)
    db.commit()
    db.refresh(calendar)
    return _to_calendar_entry(
        calendar,
        data_owner=_owner_email(db, _user_id),
        include_empty_description=not calendar.is_primary,
    )


@router.delete("/users/{userId}/calendarList/{calendarId}", status_code=status.HTTP_204_NO_CONTENT)
def calendar_list_delete(
    userId: str,
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)

    # Real API forbids removing calendars owned by current user.
    if calendar.access_role == "owner":
        raise HTTPException(
            403,
            "The data owner of a calendar cannot remove such a calendar from their calendar list.",
        )

    calendar.hidden = True
    calendar.selected = False
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/users/{userId}/calendarList/watch",
    response_model=ChannelResponse,
    response_model_exclude_none=True,
)
def calendar_list_watch(
    userId: str,
    body: ChannelRequest,
    _user_id: str = Depends(resolve_user_id),
):
    resource_uri = "https://www.googleapis.com/calendar/v3/users/me/calendarList?alt=json"
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


@router.get(
    "/calendars/{calendarId}",
    response_model=CalendarResource,
    response_model_exclude_none=True,
)
def calendars_get(
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    return _to_calendar_resource(calendar, data_owner=_owner_email(db, _user_id))


@router.post("/calendars", response_model=CalendarResource, response_model_exclude_none=True)
def calendars_insert(
    body: CalendarInsertRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    summary = body.summary.strip()
    if not summary:
        raise HTTPException(400, "Missing required field: summary")

    calendar = Calendar(
        id=f"cal_{uuid.uuid4().hex[:12]}",
        user_id=_user_id,
        summary=summary,
        description=body.description,
        location=body.location,
        timezone=body.timeZone,
        access_role="owner",
        is_primary=False,
        selected=body.selected,
        hidden=False,
        summary_override="",
        auto_accept_invitations=False,
        color_id="9",
    )
    db.add(calendar)
    db.commit()
    db.refresh(calendar)

    owner = _owner_email(db, _user_id) or ""
    db.add(
        AclRule(
            id=_owner_acl_id(calendar.id, owner),
            calendar_id=calendar.id,
            scope_type="user",
            scope_value=owner,
            role="owner",
            etag=f'"{calendar.id}:owner:{_md5_hex(owner)}"',
        )
    )
    db.commit()

    return _to_calendar_resource(calendar, data_owner=owner)


@router.patch("/calendars/{calendarId}", response_model=CalendarResource, response_model_exclude_none=True)
def calendars_patch(
    calendarId: str,
    body: CalendarPatchRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    if body.summary is not None:
        calendar.summary = body.summary
    if body.description is not None:
        calendar.description = body.description
    if body.location is not None:
        calendar.location = body.location
    if body.timeZone is not None:
        calendar.timezone = body.timeZone
    if body.autoAcceptInvitations is not None:
        calendar.auto_accept_invitations = body.autoAcceptInvitations
    db.commit()
    db.refresh(calendar)
    return _to_calendar_resource(
        calendar,
        data_owner=_owner_email(db, _user_id),
        include_empty_description=not calendar.is_primary,
    )


@router.put("/calendars/{calendarId}", response_model=CalendarResource, response_model_exclude_none=True)
def calendars_update(
    calendarId: str,
    body: CalendarUpdateRequest,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    summary = body.summary.strip()
    if not summary:
        raise HTTPException(400, "Missing required field: summary")
    calendar.summary = summary
    calendar.description = body.description
    calendar.location = body.location
    calendar.timezone = body.timeZone
    calendar.auto_accept_invitations = body.autoAcceptInvitations
    db.commit()
    db.refresh(calendar)
    return _to_calendar_resource(calendar, data_owner=_owner_email(db, _user_id))


@router.post("/calendars/{calendarId}/clear", status_code=status.HTTP_204_NO_CONTENT)
def calendars_clear(
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    if not calendar.is_primary:
        raise HTTPException(400, "Invalid Value")
    db.query(Event).filter(Event.calendar_id == calendar.id).delete()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/calendars/{calendarId}", status_code=status.HTTP_204_NO_CONTENT)
def calendars_delete(
    calendarId: str,
    db: Session = Depends(get_db),
    _user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar(db, _user_id, calendarId)
    if calendar.is_primary:
        raise HTTPException(400, "Primary calendar cannot be deleted")
    db.delete(calendar)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
