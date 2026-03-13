"""Shared dependencies for API routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from claw_gcal.models import User, get_session_factory


def get_db() -> Session:
    """Yield a DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _resolve_header_user(
    db: Session,
    x_claw_gcal_user: str | None,
    x_mock_gcal_user: str | None = None,
) -> str | None:
    candidate = x_claw_gcal_user or x_mock_gcal_user
    if candidate:
        user = db.query(User).filter(
            (User.id == candidate) | (User.email_address == candidate)
        ).first()
        if user:
            return user.id
    return None


def resolve_user_id(
    userId: str,
    x_claw_gcal_user: str | None = Header(None),
    x_mock_gcal_user: str | None = Header(None),
    db: Session = Depends(get_db),
) -> str:
    """Resolve 'me' to the actual user ID.

    Priority: userId path param -> X-Claw-Gcal-User header -> first user in DB.
    """
    if userId != "me":
        user = db.query(User).filter(User.id == userId).first()
        if not user:
            raise HTTPException(404, f"User {userId!r} not found")
        return userId

    resolved = _resolve_header_user(db, x_claw_gcal_user, x_mock_gcal_user)
    if resolved:
        return resolved

    # Fallback: first user
    user = db.query(User).first()
    if not user:
        raise HTTPException(404, "No users in database. Run `smolclaw-gcal seed` first.")
    return user.id


def resolve_actor_user_id(
    x_claw_gcal_user: str | None = Header(None),
    x_mock_gcal_user: str | None = Header(None),
    db: Session = Depends(get_db),
) -> str:
    """Resolve actor for endpoints without userId path params."""
    resolved = _resolve_header_user(db, x_claw_gcal_user, x_mock_gcal_user)
    if resolved:
        return resolved

    # Fallback: first user
    user = db.query(User).first()
    if not user:
        raise HTTPException(404, "No users in database. Run `smolclaw-gcal seed` first.")
    return user.id
