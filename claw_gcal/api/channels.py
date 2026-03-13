"""Calendar channels endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from claw_gcal.state.channels import channel_registry

from .schemas import ChannelRequest

router = APIRouter()


@router.post("/channels/stop", status_code=status.HTTP_204_NO_CONTENT)
def channels_stop(body: ChannelRequest):
    if not body.id or not body.resourceId:
        raise HTTPException(400, "Missing required field: id/resourceId")
    ok = channel_registry.stop(body.id, body.resourceId)
    if not ok:
        raise HTTPException(404, f"Channel '{body.id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
