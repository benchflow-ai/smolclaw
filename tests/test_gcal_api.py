"""Tests for the Calendar REST API mock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from claw_gcal.models import init_db, reset_engine
from claw_gcal.seed.generator import seed_database


@pytest.fixture
def gcal_db_path(tmp_path):
    path = str(tmp_path / "test_gcal.db")
    yield path
    reset_engine()


@pytest.fixture
def gcal_seeded_db(gcal_db_path):
    reset_engine()
    seed_database(scenario="default", seed=42, db_path=gcal_db_path)
    return gcal_db_path


@pytest.fixture
def gcal_client(gcal_seeded_db):
    reset_engine()
    init_db(gcal_seeded_db)
    from claw_gcal.api.app import app

    with TestClient(app) as client:
        yield client
    reset_engine()


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TestCalendarList:
    def test_list_and_get(self, gcal_client):
        resp = gcal_client.get("/calendar/v3/users/me/calendarList")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data and len(data["items"]) >= 1
        cal_id = data["items"][0]["id"]

        get_resp = gcal_client.get(f"/calendar/v3/users/me/calendarList/{cal_id}")
        assert get_resp.status_code == 200
        got = get_resp.json()
        assert got["id"] == cal_id
        assert "accessRole" in got

    def test_list_pagination_uses_next_page_token(self, gcal_client):
        resp = gcal_client.get("/calendar/v3/users/me/calendarList", params={"maxResults": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "nextPageToken" in data
        assert "nextSyncToken" not in data

    def test_non_primary_entry_omits_primary_and_notification_settings(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "NonPrimary"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        get_resp = gcal_client.get(f"/calendar/v3/users/me/calendarList/{cal_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert "primary" not in data
        assert "notificationSettings" not in data
        assert data["defaultReminders"] == []

    def test_insert_patch_update(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "CL-Insert"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        ins = gcal_client.post("/calendar/v3/users/me/calendarList", json={"id": cal_id})
        assert ins.status_code == 200
        assert ins.json()["id"] == cal_id

        patch = gcal_client.patch(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": False, "summaryOverride": "Override"},
        )
        assert patch.status_code == 200
        pdata = patch.json()
        assert "summaryOverride" in pdata

        put = gcal_client.put(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": True, "summaryOverride": "Override2"},
        )
        assert put.status_code == 200
        assert put.json()["selected"] is True

    def test_insert_requires_id(self, gcal_client):
        resp = gcal_client.post("/calendar/v3/users/me/calendarList", json={})
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Missing required field: id"
        assert resp.json()["error"]["reason"] == "badRequest"

    def test_patch_update_preserve_description_key_for_secondary(self, gcal_client):
        c = gcal_client.post(
            "/calendar/v3/calendars",
            json={"summary": "HasDesc", "description": "Secondary description"},
        )
        assert c.status_code == 200
        cal_id = c.json()["id"]

        patch = gcal_client.patch(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": False},
        )
        assert patch.status_code == 200
        assert patch.json()["description"] == "Secondary description"

        put = gcal_client.put(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": True},
        )
        assert put.status_code == 200
        assert put.json()["description"] == "Secondary description"

    def test_delete_owned_calendar_list_entry_forbidden(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "OwnedCL"})
        cal_id = c.json()["id"]
        resp = gcal_client.delete(f"/calendar/v3/users/me/calendarList/{cal_id}")
        assert resp.status_code == 403

    def test_list_invalid_page_token_returns_google_error(self, gcal_client):
        resp = gcal_client.get(
            "/calendar/v3/users/me/calendarList",
            params={"pageToken": "not-an-offset"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Invalid pageToken"
        assert resp.json()["error"]["reason"] == "badRequest"

    def test_watch(self, gcal_client):
        resp = gcal_client.post(
            "/calendar/v3/users/me/calendarList/watch",
            json={"id": "ch-test", "type": "web_hook", "address": "https://example.com/hook"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["kind"] == "api#channel"
        assert data["id"] == "ch-test"
        assert "resourceId" in data
        assert data["resourceUri"].endswith("/users/me/calendarList?alt=json")
        assert "type" not in data
        assert "address" not in data


class TestCalendars:
    def test_get_patch_update_delete(self, gcal_client):
        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Cal CRUD"})
        assert create.status_code == 200
        cal_id = create.json()["id"]

        get_resp = gcal_client.get(f"/calendar/v3/calendars/{cal_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == cal_id

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "Cal Patch", "description": "D1"},
        )
        assert patch.status_code == 200
        assert patch.json()["summary"] == "Cal Patch"

        put = gcal_client.put(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "Cal Update", "description": "D2", "timeZone": "UTC"},
        )
        assert put.status_code == 200
        assert put.json()["summary"] == "Cal Update"

        delete = gcal_client.delete(f"/calendar/v3/calendars/{cal_id}")
        assert delete.status_code == 204

    def test_patch_preserves_description_key_for_secondary(self, gcal_client):
        create = gcal_client.post(
            "/calendar/v3/calendars",
            json={"summary": "Cal Desc", "description": "Calendar description"},
        )
        assert create.status_code == 200
        cal_id = create.json()["id"]

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "Cal Desc Patched"},
        )
        assert patch.status_code == 200
        assert patch.json()["description"] == "Calendar description"

    def test_insert_requires_summary(self, gcal_client):
        resp = gcal_client.post("/calendar/v3/calendars", json={"summary": "   "})
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Missing required field: summary"

    def test_update_requires_summary(self, gcal_client):
        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Needs Update"})
        assert create.status_code == 200
        cal_id = create.json()["id"]

        resp = gcal_client.put(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "   ", "description": "", "location": "", "timeZone": "UTC"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Missing required field: summary"

    def test_clear_primary(self, gcal_client):
        resp = gcal_client.post("/calendar/v3/calendars/primary/clear")
        assert resp.status_code == 204

    def test_clear_non_primary_returns_invalid_value(self, gcal_client):
        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Secondary"})
        cal_id = create.json()["id"]
        resp = gcal_client.post(f"/calendar/v3/calendars/{cal_id}/clear")
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Invalid Value"

    def test_delete_primary_rejected(self, gcal_client):
        resp = gcal_client.delete("/calendar/v3/calendars/primary")
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Primary calendar cannot be deleted"


class TestAcl:
    def test_acl_crud(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "ACL Cal"})
        cal_id = c.json()["id"]

        acl_list_resp = gcal_client.get(f"/calendar/v3/calendars/{cal_id}/acl")
        assert acl_list_resp.status_code == 200
        assert "items" in acl_list_resp.json()

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "default"}, "role": "reader"},
        )
        assert ins.status_code == 200
        rule_id = ins.json()["id"]

        g = gcal_client.get(f"/calendar/v3/calendars/{cal_id}/acl/{rule_id}")
        assert g.status_code == 200
        assert g.json()["id"] == rule_id

        p = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}/acl/{rule_id}",
            json={"role": "reader"},
        )
        assert p.status_code == 200

        u = gcal_client.put(
            f"/calendar/v3/calendars/{cal_id}/acl/{rule_id}",
            json={"scope": {"type": "default"}, "role": "reader"},
        )
        assert u.status_code == 200

        d = gcal_client.delete(f"/calendar/v3/calendars/{cal_id}/acl/{rule_id}")
        assert d.status_code == 204

    def test_acl_patch_default_writer_rejected(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "ACL Rule"})
        cal_id = c.json()["id"]
        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "default"}, "role": "reader"},
        )
        rule_id = ins.json()["id"]
        p = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}/acl/{rule_id}",
            json={"role": "writer"},
        )
        assert p.status_code == 400

    def test_acl_insert_duplicate_conflict(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "ACL Duplicate"})
        cal_id = c.json()["id"]

        first = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "default"}, "role": "reader"},
        )
        assert first.status_code == 200

        duplicate = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "default"}, "role": "reader"},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["reason"] == "conflict"

    def test_acl_update_scope_conflict(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "ACL Conflict"})
        cal_id = c.json()["id"]

        first = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "user", "value": "one@example.com"}, "role": "reader"},
        )
        second = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/acl",
            json={"scope": {"type": "user", "value": "two@example.com"}, "role": "reader"},
        )
        assert first.status_code == 200
        assert second.status_code == 200

        conflict = gcal_client.put(
            f"/calendar/v3/calendars/{cal_id}/acl/{first.json()['id']}",
            json={"scope": {"type": "user", "value": "two@example.com"}, "role": "reader"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["reason"] == "conflict"

    def test_acl_watch(self, gcal_client):
        r = gcal_client.post(
            "/calendar/v3/calendars/primary/acl/watch",
            json={"id": "acl-watch", "type": "web_hook", "address": "https://example.com/hook"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["kind"] == "api#channel"
        assert data["resourceUri"].endswith("/calendars/primary/acl?alt=json")
        assert "type" not in data
        assert "address" not in data


class TestEvents:
    def test_update_import_instances_move_quickadd_delete(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Evt Cal", "timeZone": "UTC"})
        cal_id = c.json()["id"]
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={"summary": "Evt 1", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert ins.status_code == 200
        event_id = ins.json()["id"]

        upd = gcal_client.put(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}",
            json={"summary": "Evt 2", "description": "D", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert upd.status_code == 200
        assert upd.json()["summary"] == "Evt 2"

        imp = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events/import",
            json={
                "summary": "Imp",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "iCalUID": "imp-1@local",
            },
        )
        assert imp.status_code == 200
        assert imp.json()["iCalUID"] == "imp-1@local"

        inst = gcal_client.get(f"/calendar/v3/calendars/{cal_id}/events/{event_id}/instances")
        assert inst.status_code == 200
        assert "items" in inst.json()

        move = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}/move",
            params={"destination": "primary"},
        )
        assert move.status_code == 200
        assert move.json()["status"] == "cancelled"

        quick = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events/quickAdd",
            params={"text": "Lunch tomorrow noon"},
        )
        assert quick.status_code == 200
        assert quick.json()["summary"] == "Lunch"
        assert quick.json()["start"]["dateTime"].endswith("T12:00:00Z")

        moved_get = gcal_client.get(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert moved_get.status_code == 200
        assert moved_get.json()["status"] == "confirmed"
        assert moved_get.json()["sequence"] == 0
        delete = gcal_client.delete(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert delete.status_code == 204

    def test_import_requires_icaluid(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Import Needs UID"})
        cal_id = c.json()["id"]
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))

        resp = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events/import",
            json={
                "summary": "Imp Missing UID",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
            },
        )
        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["message"] == "Missing iCalUID."
        assert error["reason"] == "required"

    def test_watch(self, gcal_client):
        r = gcal_client.post(
            "/calendar/v3/calendars/primary/events/watch",
            json={"id": "evt-watch", "type": "web_hook", "address": "https://example.com/hook"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "evt-watch"
        assert data["resourceUri"].endswith("/calendars/primary/events?alt=json")
        assert "type" not in data
        assert "address" not in data

    def test_list_with_filters_omits_next_sync_token(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Evt List"})
        cal_id = c.json()["id"]
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={"summary": "E1", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        resp = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events",
            params={
                "singleEvents": True,
                "timeMin": _rfc3339(now),
                "timeMax": _rfc3339(now + timedelta(days=10)),
                "orderBy": "startTime",
            },
        )
        assert resp.status_code == 200
        assert "nextSyncToken" not in resp.json()

    def test_list_single_events_expands_recurrence(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Single Events"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "Recurring List Event",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "recurrence": ["RRULE:FREQ=DAILY;COUNT=3"],
            },
        )
        assert ins.status_code == 200
        event_id = ins.json()["id"]

        listed = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events",
            params={"singleEvents": True, "orderBy": "startTime"},
        )
        assert listed.status_code == 200
        data = listed.json()
        assert len(data["items"]) == 3
        assert "nextSyncToken" not in data
        assert all(item["recurringEventId"] == event_id for item in data["items"])
        assert all("recurrence" not in item for item in data["items"])

    def test_list_invalid_page_token_returns_google_error(self, gcal_client):
        resp = gcal_client.get(
            "/calendar/v3/calendars/primary/events",
            params={"pageToken": "bad-token"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Invalid pageToken"
        assert resp.json()["error"]["reason"] == "badRequest"

    def test_list_invalid_time_window_returns_google_error(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        resp = gcal_client.get(
            "/calendar/v3/calendars/primary/events",
            params={
                "timeMin": _rfc3339(now + timedelta(days=2)),
                "timeMax": _rfc3339(now + timedelta(days=1)),
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "timeMax must be greater than timeMin"

    def test_insert_rejects_end_before_start(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1) - timedelta(minutes=30))
        resp = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={"summary": "Backwards", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Event end must be after start"

    def test_patch_rejects_end_before_start_without_mutating(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        created = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={"summary": "Patch Guard", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert created.status_code == 200
        event_id = created.json()["id"]

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/primary/events/{event_id}",
            json={"end": {"dateTime": _rfc3339(now + timedelta(hours=1))}},
        )
        assert patch.status_code == 400
        assert patch.json()["error"]["message"] == "Event end must be after start"

        fetched = gcal_client.get(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert fetched.status_code == 200
        assert fetched.json()["summary"] == "Patch Guard"
        assert fetched.json()["end"]["dateTime"] == end

    def test_show_deleted_returns_cancelled_events(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        created = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={"summary": "Cancelled Visible", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        event_id = created.json()["id"]

        cancelled = gcal_client.patch(
            f"/calendar/v3/calendars/primary/events/{event_id}",
            json={"status": "cancelled"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        hidden = gcal_client.get("/calendar/v3/calendars/primary/events")
        hidden_ids = {item["id"] for item in hidden.json()["items"]}
        assert event_id not in hidden_ids

        shown = gcal_client.get(
            "/calendar/v3/calendars/primary/events",
            params={"showDeleted": True},
        )
        shown_items = {item["id"]: item for item in shown.json()["items"]}
        assert shown_items[event_id]["status"] == "cancelled"

        fetched = gcal_client.get(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "cancelled"

    def test_update_increments_sequence_and_preserves_created(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        created = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={"summary": "Sequence Check", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert created.status_code == 200
        event = created.json()

        updated = gcal_client.put(
            f"/calendar/v3/calendars/primary/events/{event['id']}",
            json={
                "summary": "Sequence Updated",
                "description": "desc",
                "location": "Room 1",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
            },
        )
        assert updated.status_code == 200
        body = updated.json()
        assert body["sequence"] == event["sequence"] + 1
        assert body["created"] == event["created"]
        assert body["updated"] != event["updated"]

    def test_list_single_events_weekly_byday_recurrence(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Weekly BYDAY"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        base = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)
        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "Weekly Standup",
                "start": {"dateTime": _rfc3339(base)},
                "end": {"dateTime": _rfc3339(base + timedelta(minutes=30))},
                "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO,WE;COUNT=4"],
            },
        )
        assert ins.status_code == 200

        listed = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events",
            params={"singleEvents": True, "orderBy": "startTime"},
        )
        assert listed.status_code == 200
        starts = [item["start"]["dateTime"] for item in listed.json()["items"]]
        assert starts == [
            "2026-03-16T09:00:00Z",
            "2026-03-18T09:00:00Z",
            "2026-03-23T09:00:00Z",
            "2026-03-25T09:00:00Z",
        ]

    def test_get_recurring_instance_by_instance_id(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Instance Lookup"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "Recurring Instance Lookup",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "recurrence": ["RRULE:FREQ=DAILY;COUNT=3"],
            },
        )
        assert ins.status_code == 200
        parent_id = ins.json()["id"]

        listed = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events",
            params={"singleEvents": True, "orderBy": "startTime"},
        )
        assert listed.status_code == 200
        instance = listed.json()["items"][1]

        fetched = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events/{instance['id']}",
        )
        assert fetched.status_code == 200
        data = fetched.json()
        assert data["id"] == instance["id"]
        assert data["recurringEventId"] == parent_id
        assert data["originalStartTime"] == instance["originalStartTime"]

    def test_all_day_events_use_date_fields(self, gcal_client):
        create = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={
                "summary": "All Day Event",
                "start": {"date": "2026-03-20"},
                "end": {"date": "2026-03-21"},
            },
        )
        assert create.status_code == 200
        created = create.json()
        event_id = created["id"]

        assert created["start"] == {"date": "2026-03-20"}
        assert created["end"] == {"date": "2026-03-21"}

        fetched = gcal_client.get(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert fetched.status_code == 200
        got = fetched.json()
        assert got["start"] == {"date": "2026-03-20"}
        assert got["end"] == {"date": "2026-03-21"}

    def test_instances_expands_recurrence_and_supports_pagination(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Recurring"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "Recurring Event",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "recurrence": ["RRULE:FREQ=DAILY;COUNT=5"],
            },
        )
        assert ins.status_code == 200
        event_id = ins.json()["id"]

        page1 = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}/instances",
            params={"maxResults": 2},
        )
        assert page1.status_code == 200
        body1 = page1.json()
        assert len(body1["items"]) == 2
        assert "nextPageToken" in body1
        assert "nextSyncToken" not in body1
        assert body1["items"][0]["recurringEventId"] == event_id
        assert "originalStartTime" in body1["items"][0]

        page2 = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}/instances",
            params={"maxResults": 2, "pageToken": body1["nextPageToken"]},
        )
        assert page2.status_code == 200
        body2 = page2.json()
        assert len(body2["items"]) >= 1

        original_start = body1["items"][0]["originalStartTime"]["dateTime"]
        single = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}/instances",
            params={"originalStart": original_start},
        )
        assert single.status_code == 200
        assert len(single.json()["items"]) == 1

    def test_instances_respect_time_zone_override(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Instances TZ"})
        cal_id = c.json()["id"]
        base = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "TZ Series",
                "start": {"dateTime": _rfc3339(base)},
                "end": {"dateTime": _rfc3339(base + timedelta(hours=1))},
                "recurrence": ["RRULE:FREQ=DAILY;COUNT=2"],
            },
        )
        event_id = ins.json()["id"]

        resp = gcal_client.get(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}/instances",
            params={"timeZone": "America/Los_Angeles"},
        )
        assert resp.status_code == 200
        item = resp.json()["items"][0]
        assert item["start"]["timeZone"] == "America/Los_Angeles"
        assert item["originalStartTime"]["timeZone"] == "America/Los_Angeles"

    def test_snapshot_restore_preserves_event_timestamps(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Snapshot TS"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=2))
        end = _rfc3339(now + timedelta(days=2, hours=1))

        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={"summary": "Before Snapshot", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        assert ins.status_code == 200
        event = ins.json()
        event_id = event["id"]
        created_before = event["created"]
        updated_before = event["updated"]

        snap = gcal_client.post("/_admin/snapshot/ts_preserve")
        assert snap.status_code == 200

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}/events/{event_id}",
            json={"summary": "After Snapshot"},
        )
        assert patch.status_code == 200
        assert patch.json()["summary"] == "After Snapshot"
        assert patch.json()["updated"] != updated_before

        restore = gcal_client.post("/_admin/restore/ts_preserve")
        assert restore.status_code == 200

        after = gcal_client.get(f"/calendar/v3/calendars/{cal_id}/events/{event_id}")
        assert after.status_code == 200
        restored = after.json()
        assert restored["summary"] == "Before Snapshot"
        assert restored["created"] == created_before
        assert restored["updated"] == updated_before


class TestSettingsColorsFreebusyChannels:
    def test_settings_list_get_watch(self, gcal_client):
        settings_list_resp = gcal_client.get("/calendar/v3/users/me/settings")
        assert settings_list_resp.status_code == 200
        assert "items" in settings_list_resp.json()

        g = gcal_client.get("/calendar/v3/users/me/settings/timezone")
        assert g.status_code == 200
        assert g.json()["id"] == "timezone"

        w = gcal_client.post(
            "/calendar/v3/users/me/settings/watch",
            json={"id": "settings-watch", "type": "web_hook", "address": "https://example.com/hook"},
        )
        assert w.status_code == 200
        ch = w.json()
        assert ch["id"] == "settings-watch"
        assert ch["resourceUri"].endswith("/users/me/settings?alt=json")
        assert "type" not in ch
        assert "address" not in ch

        stop = gcal_client.post(
            "/calendar/v3/channels/stop",
            json={"id": ch["id"], "resourceId": ch["resourceId"]},
        )
        assert stop.status_code == 204

    def test_settings_get_unknown_returns_google_error(self, gcal_client):
        resp = gcal_client.get("/calendar/v3/users/me/settings/not-a-setting")
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == "Setting not found"
        assert resp.json()["error"]["reason"] == "notFound"

    def test_settings_watch_preserves_optional_channel_fields(self, gcal_client):
        resp = gcal_client.post(
            "/calendar/v3/users/me/settings/watch",
            json={
                "id": "settings-opts",
                "type": "web_hook",
                "address": "https://example.com/hook",
                "token": "verify-token",
                "payload": True,
                "params": {"ttl": "600"},
                "expiration": "9999999999999",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["token"] == "verify-token"
        assert body["payload"] is True
        assert body["params"] == {"ttl": "600"}
        assert body["expiration"] == "9999999999999"

    def test_colors_and_freebusy(self, gcal_client):
        colors = gcal_client.get("/calendar/v3/colors")
        assert colors.status_code == 200
        assert colors.json()["kind"] == "calendar#colors"

        now = datetime.now(timezone.utc).replace(microsecond=0)
        resp = gcal_client.post(
            "/calendar/v3/freeBusy",
            json={
                "timeMin": _rfc3339(now),
                "timeMax": _rfc3339(now + timedelta(days=2)),
                "items": [{"id": "primary"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "calendars" in data and "primary" in data["calendars"]

    def test_freebusy_expands_recurring_events(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Busy Recurrence"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        ins = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events",
            json={
                "summary": "Recurring Busy Block",
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "recurrence": ["RRULE:FREQ=DAILY;COUNT=3"],
            },
        )
        assert ins.status_code == 200

        resp = gcal_client.post(
            "/calendar/v3/freeBusy",
            json={
                "timeMin": _rfc3339(now),
                "timeMax": _rfc3339(now + timedelta(days=5)),
                "items": [{"id": cal_id}],
            },
        )
        assert resp.status_code == 200
        busy = resp.json()["calendars"][cal_id]["busy"]
        assert len(busy) == 3
        assert busy[0]["start"] == start
        assert busy[-1]["end"] == _rfc3339(now + timedelta(days=3, hours=1))

    def test_freebusy_invalid_time_window_returns_google_error(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        resp = gcal_client.post(
            "/calendar/v3/freeBusy",
            json={
                "timeMin": _rfc3339(now + timedelta(days=2)),
                "timeMax": _rfc3339(now + timedelta(days=1)),
                "items": [{"id": "primary"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "timeMax must be greater than timeMin"

    def test_freebusy_missing_calendar_reports_not_found(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        resp = gcal_client.post(
            "/calendar/v3/freeBusy",
            json={
                "timeMin": _rfc3339(now),
                "timeMax": _rfc3339(now + timedelta(days=1)),
                "items": [{"id": "missing-calendar"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["calendars"]["missing-calendar"]["busy"] == []
        assert body["calendars"]["missing-calendar"]["errors"][0]["reason"] == "notFound"

    def test_freebusy_ignores_cancelled_events(self, gcal_client):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = _rfc3339(now + timedelta(days=1))
        end = _rfc3339(now + timedelta(days=1, hours=1))
        created = gcal_client.post(
            "/calendar/v3/calendars/primary/events",
            json={"summary": "Cancelled Busy", "start": {"dateTime": start}, "end": {"dateTime": end}},
        )
        event_id = created.json()["id"]

        cancelled = gcal_client.patch(
            f"/calendar/v3/calendars/primary/events/{event_id}",
            json={"status": "cancelled"},
        )
        assert cancelled.status_code == 200

        resp = gcal_client.post(
            "/calendar/v3/freeBusy",
            json={
                "timeMin": _rfc3339(now),
                "timeMax": _rfc3339(now + timedelta(days=2)),
                "items": [{"id": "primary"}],
            },
        )
        assert resp.status_code == 200
        busy = resp.json()["calendars"]["primary"]["busy"]
        assert all(slot["start"] != start for slot in busy)

    def test_channels_stop_unknown(self, gcal_client):
        resp = gcal_client.post(
            "/calendar/v3/channels/stop",
            json={"id": "does-not-exist", "resourceId": "unknown"},
        )
        assert resp.status_code == 404

    def test_channels_stop_missing_fields_returns_google_error(self, gcal_client):
        resp = gcal_client.post("/calendar/v3/channels/stop", json={"id": "missing-resource"})
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Missing required field: id/resourceId"
        assert resp.json()["error"]["reason"] == "badRequest"


class TestHealthProfileAdmin:
    def test_health_and_profile(self, gcal_client):
        health = gcal_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        profile = gcal_client.get("/calendar/v3/users/me/profile")
        assert profile.status_code == 200
        data = profile.json()
        assert data["emailAddress"] == "alex@nexusai.com"
        assert data["displayName"] == "Alex Chen"
        assert data["calendarsTotal"] > 0
        assert data["eventsTotal"] > 0
        assert data["historyId"].isdigit()

    def test_admin_state_diff_action_log_snapshot_restore_reset(self, gcal_client):
        state = gcal_client.get("/_admin/state")
        assert state.status_code == 200
        state_data = state.json()
        assert "users" in state_data
        assert "user1" in state_data["users"]

        diff = gcal_client.get("/_admin/diff")
        assert diff.status_code == 200
        assert "users" in diff.json()

        # Generate activity then verify audit trail.
        gcal_client.get("/calendar/v3/users/me/calendarList")
        log = gcal_client.get("/_admin/action_log")
        assert log.status_code == 200
        log_data = log.json()
        assert "entries" in log_data
        assert log_data["count"] >= 1

        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Admin Snapshot"})
        assert create.status_code == 200
        cal_id = create.json()["id"]

        snap = gcal_client.post("/_admin/snapshot/gcal_admin_test")
        assert snap.status_code == 200
        assert snap.json()["status"] == "ok"

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "Mutated"},
        )
        assert patch.status_code == 200
        assert patch.json()["summary"] == "Mutated"

        restore = gcal_client.post("/_admin/restore/gcal_admin_test")
        assert restore.status_code == 200
        assert restore.json()["status"] == "ok"

        after = gcal_client.get(f"/calendar/v3/calendars/{cal_id}")
        assert after.status_code == 200
        assert after.json()["summary"] == "Admin Snapshot"

        tasks = gcal_client.get("/_admin/tasks")
        assert tasks.status_code == 200
        assert "tasks" in tasks.json()

        reset = gcal_client.post("/_admin/reset")
        assert reset.status_code == 200
        assert reset.json()["status"] == "ok"

    def test_admin_reset_clears_channel_registry(self, gcal_client):
        watch = gcal_client.post(
            "/calendar/v3/calendars/primary/events/watch",
            json={"id": "reset-watch", "type": "web_hook", "address": "https://example.com/hook"},
        )
        assert watch.status_code == 200
        channel = watch.json()

        reset = gcal_client.post("/_admin/reset")
        assert reset.status_code == 200
        assert reset.json()["status"] == "ok"

        stop = gcal_client.post(
            "/calendar/v3/channels/stop",
            json={"id": channel["id"], "resourceId": channel["resourceId"]},
        )
        assert stop.status_code == 404
