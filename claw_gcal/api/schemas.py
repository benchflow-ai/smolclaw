"""Pydantic schemas for Calendar API responses."""

from __future__ import annotations

from typing import Literal

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


class CalendarListEntry(BaseModel):
    kind: Literal["calendar#calendarListEntry"] = "calendar#calendarListEntry"
    etag: str
    id: str
    summary: str
    timeZone: str
    accessRole: str
    primary: bool = False
    selected: bool = True
    colorId: str | None = None
    backgroundColor: str | None = None
    foregroundColor: str | None = None
    conferenceProperties: ConferenceProperties | None = None
    defaultReminders: list[ReminderOverride] | None = None
    notificationSettings: NotificationSettings | None = None


class CalendarResource(BaseModel):
    kind: Literal["calendar#calendar"] = "calendar#calendar"
    etag: str
    id: str
    summary: str
    timeZone: str
    conferenceProperties: ConferenceProperties | None = None
    dataOwner: str | None = None


class CalendarListResponse(BaseModel):
    kind: Literal["calendar#calendarList"] = "calendar#calendarList"
    etag: str
    items: list[CalendarListEntry]
    nextPageToken: str | None = None
    nextSyncToken: str | None = None


class CalendarInsertRequest(BaseModel):
    summary: str = ""
    description: str = ""
    timeZone: str = "America/Los_Angeles"
    selected: bool = True


class EventDateTime(BaseModel):
    dateTime: str
    timeZone: str | None = None


class EventActor(BaseModel):
    email: str
    self: bool = True


class EventReminders(BaseModel):
    useDefault: bool = True


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
    creator: EventActor | None = None
    organizer: EventActor | None = None
    reminders: EventReminders | None = None
    eventType: str | None = None


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


class EventPatchRequest(BaseModel):
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    status: str | None = None
    start: EventDateTime | None = None
    end: EventDateTime | None = None


class Profile(BaseModel):
    emailAddress: str
    displayName: str
    calendarsTotal: int = 0
    eventsTotal: int = 0
    historyId: str = "1"
