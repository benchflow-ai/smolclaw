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

    def test_patch_update_include_description_key_for_secondary(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "NoDesc"})
        assert c.status_code == 200
        cal_id = c.json()["id"]

        patch = gcal_client.patch(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": False},
        )
        assert patch.status_code == 200
        assert "description" in patch.json()

        put = gcal_client.put(
            f"/calendar/v3/users/me/calendarList/{cal_id}",
            json={"selected": True},
        )
        assert put.status_code == 200
        assert "description" in put.json()

    def test_delete_owned_calendar_list_entry_forbidden(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "OwnedCL"})
        cal_id = c.json()["id"]
        resp = gcal_client.delete(f"/calendar/v3/users/me/calendarList/{cal_id}")
        assert resp.status_code == 403

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

    def test_patch_includes_description_key_for_secondary(self, gcal_client):
        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Cal Desc"})
        assert create.status_code == 200
        cal_id = create.json()["id"]

        patch = gcal_client.patch(
            f"/calendar/v3/calendars/{cal_id}",
            json={"summary": "Cal Desc Patched"},
        )
        assert patch.status_code == 200
        assert "description" in patch.json()

    def test_clear_primary(self, gcal_client):
        resp = gcal_client.post("/calendar/v3/calendars/primary/clear")
        assert resp.status_code == 204

    def test_clear_non_primary_returns_invalid_value(self, gcal_client):
        create = gcal_client.post("/calendar/v3/calendars", json={"summary": "Secondary"})
        cal_id = create.json()["id"]
        resp = gcal_client.post(f"/calendar/v3/calendars/{cal_id}/clear")
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Invalid Value"


class TestAcl:
    def test_acl_crud(self, gcal_client):
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "ACL Cal"})
        cal_id = c.json()["id"]

        l = gcal_client.get(f"/calendar/v3/calendars/{cal_id}/acl")
        assert l.status_code == 200
        assert "items" in l.json()

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
        c = gcal_client.post("/calendar/v3/calendars", json={"summary": "Evt Cal"})
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

        quick = gcal_client.post(
            f"/calendar/v3/calendars/{cal_id}/events/quickAdd",
            params={"text": "Lunch tomorrow noon"},
        )
        assert quick.status_code == 200
        assert quick.json()["summary"] == "Lunch tomorrow noon"

        delete = gcal_client.delete(f"/calendar/v3/calendars/primary/events/{event_id}")
        assert delete.status_code in (204, 404)

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
        l = gcal_client.get("/calendar/v3/users/me/settings")
        assert l.status_code == 200
        assert "items" in l.json()

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

    def test_channels_stop_unknown(self, gcal_client):
        resp = gcal_client.post(
            "/calendar/v3/channels/stop",
            json={"id": "does-not-exist", "resourceId": "unknown"},
        )
        assert resp.status_code == 404


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
