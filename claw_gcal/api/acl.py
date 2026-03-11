"""Calendar ACL endpoints."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from claw_gcal.models import AclRule, Calendar
from claw_gcal.state.channels import channel_registry

from .deps import get_db, resolve_actor_user_id
from .schemas import (
    AclListResponse,
    AclRulePatchRequest,
    AclRuleResource,
    AclRuleWriteRequest,
    AclScope,
    ChannelRequest,
    ChannelResponse,
)

router = APIRouter()

_VALID_ROLES = {"none", "freeBusyReader", "reader", "writer", "owner"}


def _md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _resolve_calendar_for_actor(db: Session, calendar_id: str, actor_user_id: str) -> Calendar:
    if calendar_id == "primary":
        calendar = db.query(Calendar).filter(
            Calendar.user_id == actor_user_id,
            Calendar.is_primary.is_(True),
        ).first()
    else:
        calendar = db.query(Calendar).filter(
            Calendar.id == calendar_id,
            Calendar.user_id == actor_user_id,
        ).first()

    if not calendar:
        raise HTTPException(404, "Calendar not found")
    return calendar


def _storage_rule_id(calendar_id: str, rule_id: str) -> str:
    return f"{calendar_id}:{rule_id}"


def _public_rule_id(rule: AclRule) -> str:
    if ":" in rule.id:
        return rule.id.split(":", 1)[1]
    return rule.id


def _rule_to_resource(rule: AclRule) -> AclRuleResource:
    return AclRuleResource(
        etag=rule.etag,
        id=_public_rule_id(rule),
        role=rule.role,
        scope=AclScope(type=rule.scope_type, value=rule.scope_value or None),
    )


def _build_rule_id(scope: AclScope) -> str:
    if scope.type == "default":
        return "default"
    if not scope.value:
        raise HTTPException(400, "Missing scope.value")
    return f"{scope.type}:{scope.value}"


def _validate_scope_role(scope: AclScope, role: str):
    if role not in _VALID_ROLES:
        raise HTTPException(400, "Invalid ACL role")
    if scope.type == "default" and role in {"writer", "owner"}:
        raise HTTPException(
            400,
            "The default scope cannot be granted a write or owners access role to a calendar.",
        )


@router.get(
    "/calendars/{calendarId}/acl",
    response_model=AclListResponse,
    response_model_exclude_none=True,
)
def acl_list(
    calendarId: str,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    rows = db.query(AclRule).filter(AclRule.calendar_id == calendar.id).order_by(AclRule.id.asc()).all()
    items = [_rule_to_resource(r) for r in rows]
    etag_raw = "|".join(i.etag for i in items)
    return AclListResponse(
        etag=f'"{_md5_hex(etag_raw)}"',
        items=items,
        nextSyncToken=_md5_hex(f"{calendar.id}:{len(items)}:{etag_raw}"),
    )


@router.get(
    "/calendars/{calendarId}/acl/{ruleId}",
    response_model=AclRuleResource,
    response_model_exclude_none=True,
)
def acl_get(
    calendarId: str,
    ruleId: str,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    rule = db.query(AclRule).filter(
        AclRule.id == _storage_rule_id(calendar.id, ruleId),
        AclRule.calendar_id == calendar.id,
    ).first()
    if not rule:
        raise HTTPException(404, "ACL rule not found")
    return _rule_to_resource(rule)


@router.post(
    "/calendars/{calendarId}/acl",
    response_model=AclRuleResource,
    response_model_exclude_none=True,
)
def acl_insert(
    calendarId: str,
    body: AclRuleWriteRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    public_rule_id = _build_rule_id(body.scope)
    _validate_scope_role(body.scope, body.role)

    storage_id = _storage_rule_id(calendar.id, public_rule_id)
    existing = db.query(AclRule).filter(
        AclRule.id == storage_id,
        AclRule.calendar_id == calendar.id,
    ).first()
    if existing:
        raise HTTPException(409, "ACL rule already exists")

    rule = AclRule(
        id=storage_id,
        calendar_id=calendar.id,
        scope_type=body.scope.type,
        scope_value=body.scope.value or "",
        role=body.role,
        etag=f'"{_md5_hex(f"{storage_id}:{body.role}:{body.scope.type}:{body.scope.value or ""}")}"',
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return _rule_to_resource(rule)


@router.patch(
    "/calendars/{calendarId}/acl/{ruleId}",
    response_model=AclRuleResource,
    response_model_exclude_none=True,
)
def acl_patch(
    calendarId: str,
    ruleId: str,
    body: AclRulePatchRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    rule = db.query(AclRule).filter(
        AclRule.id == _storage_rule_id(calendar.id, ruleId),
        AclRule.calendar_id == calendar.id,
    ).first()
    if not rule:
        raise HTTPException(404, "ACL rule not found")

    scope = AclScope(type=rule.scope_type, value=rule.scope_value or None)
    role = rule.role
    if body.scope is not None:
        scope = body.scope
    if body.role is not None:
        role = body.role
    _validate_scope_role(scope, role)

    # Scope changes can change rule id.
    new_public_rule_id = _build_rule_id(scope)
    new_storage_rule_id = _storage_rule_id(calendar.id, new_public_rule_id)
    if new_storage_rule_id != rule.id:
        conflict = db.query(AclRule).filter(
            AclRule.id == new_storage_rule_id,
            AclRule.calendar_id == calendar.id,
        ).first()
        if conflict:
            raise HTTPException(409, "ACL rule already exists")
        rule.id = new_storage_rule_id

    rule.scope_type = scope.type
    rule.scope_value = scope.value or ""
    rule.role = role
    rule.etag = f'"{_md5_hex(f"{rule.id}:{rule.role}:{rule.scope_type}:{rule.scope_value}")}"'

    db.commit()
    db.refresh(rule)
    return _rule_to_resource(rule)


@router.put(
    "/calendars/{calendarId}/acl/{ruleId}",
    response_model=AclRuleResource,
    response_model_exclude_none=True,
)
def acl_update(
    calendarId: str,
    ruleId: str,
    body: AclRuleWriteRequest,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    rule = db.query(AclRule).filter(
        AclRule.id == _storage_rule_id(calendar.id, ruleId),
        AclRule.calendar_id == calendar.id,
    ).first()
    if not rule:
        raise HTTPException(404, "ACL rule not found")

    _validate_scope_role(body.scope, body.role)
    new_public_rule_id = _build_rule_id(body.scope)
    new_storage_rule_id = _storage_rule_id(calendar.id, new_public_rule_id)
    if new_storage_rule_id != rule.id:
        conflict = db.query(AclRule).filter(
            AclRule.id == new_storage_rule_id,
            AclRule.calendar_id == calendar.id,
        ).first()
        if conflict:
            raise HTTPException(409, "ACL rule already exists")
        rule.id = new_storage_rule_id

    rule.scope_type = body.scope.type
    rule.scope_value = body.scope.value or ""
    rule.role = body.role
    rule.etag = f'"{_md5_hex(f"{rule.id}:{rule.role}:{rule.scope_type}:{rule.scope_value}")}"'
    db.commit()
    db.refresh(rule)
    return _rule_to_resource(rule)


@router.delete("/calendars/{calendarId}/acl/{ruleId}", status_code=204)
def acl_delete(
    calendarId: str,
    ruleId: str,
    db: Session = Depends(get_db),
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    calendar = _resolve_calendar_for_actor(db, calendarId, _actor_user_id)
    rule = db.query(AclRule).filter(
        AclRule.id == _storage_rule_id(calendar.id, ruleId),
        AclRule.calendar_id == calendar.id,
    ).first()
    if not rule:
        raise HTTPException(404, "ACL rule not found")
    db.delete(rule)
    db.commit()
    return Response(status_code=204)


@router.post(
    "/calendars/{calendarId}/acl/watch",
    response_model=ChannelResponse,
    response_model_exclude_none=True,
)
def acl_watch(
    calendarId: str,
    body: ChannelRequest,
    _actor_user_id: str = Depends(resolve_actor_user_id),
):
    resource_uri = f"https://www.googleapis.com/calendar/v3/calendars/{calendarId}/acl?alt=json"
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
