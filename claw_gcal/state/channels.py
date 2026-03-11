"""In-memory watch channel registry for Calendar watch/stop endpoints."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone


class ChannelRegistry:
    def __init__(self):
        self._channels: dict[tuple[str, str], dict] = {}

    def register(
        self,
        *,
        resource_uri: str,
        channel_id: str | None = None,
        address: str | None = None,
        token: str | None = None,
        channel_type: str | None = None,
        payload: bool | None = None,
        params: dict | None = None,
        expiration: str | None = None,
    ) -> dict:
        channel_id = channel_id or f"chan_{uuid.uuid4().hex[:16]}"
        resource_id = hashlib.md5(resource_uri.encode("utf-8")).hexdigest()
        if expiration:
            expiration_ms = str(expiration)
        else:
            exp = datetime.now(timezone.utc) + timedelta(days=7)
            expiration_ms = str(int(exp.timestamp() * 1000))

        channel = {
            "kind": "api#channel",
            "id": channel_id,
            "resourceId": resource_id,
            "resourceUri": resource_uri,
            "expiration": expiration_ms,
        }
        # Real API often omits optional fields when absent.
        if token:
            channel["token"] = token
        if payload is not None:
            channel["payload"] = payload
        if params:
            channel["params"] = params

        self._channels[(channel_id, resource_id)] = channel
        return channel

    def stop(self, channel_id: str, resource_id: str) -> bool:
        return self._channels.pop((channel_id, resource_id), None) is not None

    def clear(self):
        self._channels.clear()


channel_registry = ChannelRegistry()
