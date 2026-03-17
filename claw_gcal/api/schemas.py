"""Pydantic schemas mirroring Google Calendar API response format."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConferenceProperties(BaseModel):
    allowedConferenceSolutionTypes: list[str] = Field(
        default_factory=lambda: ["hangoutsMeet"]
    )


class ReminderOverride(BaseModel):
    method: str
    minutes: int


class NotificationRule(BaseModel):
    type: str
    method: str


class NotificationSettings(BaseModel):
    notifications: list[NotificationRule] = Field(default_factory=list)


# --- Calendars ---
class CalendarListEntry(BaseModel):
    model_config = {"exclude_none": True}

    kind: Literal["calendar#calendarListEntry"] = "calendar#calendarListEntry"
    etag: str
    id: str
    summary: str
    timeZone: str
    accessRole: str
    primary: bool | None = None
    selected: bool | None = True
    colorId: str | None = None
    backgroundColor: str | None = None
    foregroundColor: str | None = None
    conferenceProperties: ConferenceProperties | None = None
    defaultReminders: list[ReminderOverride] | None = None
    notificationSettings: NotificationSettings | None = None
    dataOwner: str | None = None
    summaryOverride: str | None = None
    description: str | None = None
    location: str | None = None
    hidden: bool | None = None
    deleted: bool | None = None
    autoAcceptInvitations: bool | None = None


class CalendarResource(BaseModel):
    model_config = {"exclude_none": True}

    kind: Literal["calendar#calendar"] = "calendar#calendar"
    etag: str
    id: str
    summary: str
    timeZone: str
    conferenceProperties: ConferenceProperties | None = None
    dataOwner: str | None = None
    description: str | None = None
    location: str | None = None
    autoAcceptInvitations: bool | None = None


class CalendarListResponse(BaseModel):
    kind: Literal["calendar#calendarList"] = "calendar#calendarList"
    etag: str
    items: list[CalendarListEntry]
    nextPageToken: str | None = None
    nextSyncToken: str | None = None


class CalendarInsertRequest(BaseModel):
    summary: str = ""
    description: str = ""
    location: str = ""
    timeZone: str = "America/Los_Angeles"
    selected: bool = True


class CalendarPatchRequest(BaseModel):
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    timeZone: str | None = None
    autoAcceptInvitations: bool | None = None


class CalendarUpdateRequest(BaseModel):
    summary: str = ""
    description: str = ""
    location: str = ""
    timeZone: str = "America/Los_Angeles"
    autoAcceptInvitations: bool = False


class CalendarListMutationRequest(BaseModel):
    selected: bool | None = None
    summaryOverride: str | None = None
    colorId: str | None = None
    hidden: bool | None = None
    defaultReminders: list[ReminderOverride] | None = None
    notificationSettings: NotificationSettings | None = None
    id: str | None = None


# --- Events ---
class EventDateTime(BaseModel):
    dateTime: str | None = None
    date: str | None = None
    timeZone: str | None = None


class EventActor(BaseModel):
    model_config = {"exclude_none": True}

    email: str
    self: bool = True
    displayName: str | None = None


class EventReminders(BaseModel):
    useDefault: bool = True


class EventAttendee(BaseModel):
    model_config = {"exclude_none": True}

    email: str
    displayName: str | None = None
    responseStatus: str = "needsAction"
    self: bool | None = None
    optional: bool | None = None
    organizer: bool | None = None
    resource: bool | None = None


class EventResource(BaseModel):
    model_config = {"exclude_none": True}

    kind: Literal["calendar#event"] = "calendar#event"
    etag: str
    id: str
    status: str
    htmlLink: str = ""
    created: str
    updated: str
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    iCalUID: str
    sequence: int = 0
    start: EventDateTime
    end: EventDateTime
    attendees: list[EventAttendee] | None = None
    creator: EventActor | None = None
    organizer: EventActor | None = None
    reminders: EventReminders | None = None
    eventType: str | None = None
    recurrence: list[str] | None = None
    recurringEventId: str | None = None
    originalStartTime: EventDateTime | None = None


class EventListResponse(BaseModel):
    kind: Literal["calendar#events"] = "calendar#events"
    etag: str
    summary: str
    timeZone: str
    items: list[EventResource]
    description: str = ""
    updated: str | None = None
    accessRole: str | None = None
    defaultReminders: list[ReminderOverride] | None = None
    nextPageToken: str | None = None
    nextSyncToken: str | None = None


class EventWriteRequest(BaseModel):
    summary: str = ""
    description: str = ""
    location: str = ""
    start: EventDateTime
    end: EventDateTime
    iCalUID: str | None = None
    recurrence: list[str] | None = None
    status: str | None = None
    attendees: list[EventAttendee] | None = None


class EventPatchRequest(BaseModel):
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    status: str | None = None
    start: EventDateTime | None = None
    end: EventDateTime | None = None
    recurrence: list[str] | None = None
    attendees: list[EventAttendee] | None = None


# --- ACL ---
class AclScope(BaseModel):
    type: str
    value: str | None = None


class AclRuleResource(BaseModel):
    kind: Literal["calendar#aclRule"] = "calendar#aclRule"
    etag: str
    id: str
    role: str
    scope: AclScope


class AclListResponse(BaseModel):
    kind: Literal["calendar#acl"] = "calendar#acl"
    etag: str
    items: list[AclRuleResource]
    nextPageToken: str | None = None
    nextSyncToken: str | None = None


class AclRuleWriteRequest(BaseModel):
    role: str = "reader"
    scope: AclScope = Field(default_factory=lambda: AclScope(type="default"))


class AclRulePatchRequest(BaseModel):
    role: str | None = None
    scope: AclScope | None = None


# --- Channels/Watch ---
class ChannelRequest(BaseModel):
    id: str | None = None
    kind: str | None = None
    resourceId: str | None = None
    resourceUri: str | None = None
    token: str | None = None
    type: str | None = None
    address: str | None = None
    expiration: str | None = None
    payload: bool | None = None
    params: dict[str, Any] | None = None


class ChannelResponse(BaseModel):
    model_config = {"exclude_none": True}

    kind: Literal["api#channel"] = "api#channel"
    id: str
    resourceId: str
    resourceUri: str
    expiration: str | None = None
    token: str | None = None
    payload: bool | None = None
    params: dict[str, Any] | None = None


# --- Colors ---
class ColorDef(BaseModel):
    background: str
    foreground: str


class ColorsResponse(BaseModel):
    kind: Literal["calendar#colors"] = "calendar#colors"
    updated: str
    calendar: dict[str, ColorDef]
    event: dict[str, ColorDef]


# --- Freebusy ---
class FreeBusyCalendarItem(BaseModel):
    id: str


class FreeBusyRequest(BaseModel):
    timeMin: str
    timeMax: str
    timeZone: str | None = None
    items: list[FreeBusyCalendarItem]
    groupExpansionMax: int | None = None
    calendarExpansionMax: int | None = None


class BusyTimeRange(BaseModel):
    start: str
    end: str


class FreeBusyCalendarResponse(BaseModel):
    errors: list[dict[str, Any]] | None = None
    busy: list[BusyTimeRange]


class FreeBusyResponse(BaseModel):
    kind: Literal["calendar#freeBusy"] = "calendar#freeBusy"
    timeMin: str
    timeMax: str
    calendars: dict[str, FreeBusyCalendarResponse]
    groups: dict[str, Any] | None = None


# --- Settings ---
class CalendarSetting(BaseModel):
    kind: Literal["calendar#setting"] = "calendar#setting"
    etag: str
    id: str
    value: str


class CalendarSettingsListResponse(BaseModel):
    kind: Literal["calendar#settings"] = "calendar#settings"
    etag: str
    items: list[CalendarSetting]
    nextPageToken: str | None = None
    nextSyncToken: str | None = None


# --- Profile ---
class Profile(BaseModel):
    emailAddress: str
    displayName: str
    calendarsTotal: int = 0
    eventsTotal: int = 0
    historyId: str = "1"
