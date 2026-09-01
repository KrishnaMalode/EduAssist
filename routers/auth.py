"""
routers/auth.py — Authentication endpoints for EduAssist.

Routes
------
POST /auth/register       Register a new student account → StudentResponse + JWT
POST /auth/login          Student login (OAuth2 form) → JWT
POST /auth/admin/login    Admin login via .env credentials → admin JWT

JWT tokens use HS256 with a 24-hour expiry. The "role" claim is embedded
in the payload so downstream dependencies can distinguish student vs admin.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from gamification.engine import apply_decay
from gamification.models_ext import StudentProfile
from routers.dependencies import ALGORITHM, SECRET_KEY
from routers.dependencies import get_current_student

router = APIRouter(prefix="/auth", tags=["Auth"])

# ─── Constants ────────────────────────────────────────────────────────────────

ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24   # 24 hours


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored bcrypt *hashed* string."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Sign and return a JWT access token.

    Args:
        data:          Payload claims (must include "sub" and "role").
        expires_delta: Optional custom expiry; defaults to 24 hours.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=schemas.RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new student account",
)
def register(
    payload: schemas.StudentCreate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Create a new student account.

    - Validates that the e-mail address is not already taken (409 Conflict).
    - Hashes the plain-text password with bcrypt before persisting.
    - Issues a 24-hour JWT so the client can proceed without a separate login step.

    Returns:
        RegisterResponse: nested student profile + access_token.

    Raises:
        HTTP 409 — e-mail already registered.
    """
    existing = (
        db.query(models.Student)
        .filter(models.Student.email == payload.email)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"E-mail '{payload.email}' is already registered.",
        )

    student = models.Student(
        name=payload.name,
        email=payload.email,
        hashed_password=_hash_password(payload.password),
        subjects=payload.subjects,
        daily_hours=payload.daily_hours,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    token = create_access_token({"sub": str(student.id), "role": "student"})
    return {"student": student, "access_token": token, "token_type": "bearer"}


@router.post(
    "/login",
    response_model=schemas.StudentLoginResponse,
    summary="Student login — returns JWT access token",
)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Authenticate a student using e-mail + password.

    The OAuth2 password form uses the ``username`` field to carry the e-mail
    address so this endpoint is compatible with Swagger's built-in
    *Authorize* dialog and standard OAuth2 clients.

    Returns:
        StudentLoginResponse: access_token, token_type, student_id, name.

    Raises:
        HTTP 401 — incorrect e-mail or password.
    """
    student = (
        db.query(models.Student)
        .filter(models.Student.email == form_data.username)
        .first()
    )
    if not student or not _verify_password(form_data.password, student.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect e-mail or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    apply_decay(student.id, db)
    profile = db.query(StudentProfile).filter_by(student_id=student.id).first()
    if not profile:
        db.add(StudentProfile(student_id=student.id))
        db.commit()

    token = create_access_token({"sub": str(student.id), "role": "student"})
    return {
        "access_token": token,
        "token_type": "bearer",
        "student_id": student.id,
        "name": student.name,
    }


@router.post(
    "/admin/login",
    response_model=schemas.AdminLoginResponse,
    summary="Admin login — returns admin JWT",
)
def admin_login(
    payload: schemas.LoginRequest,
) -> dict[str, Any]:
    """
    Authenticate as the platform administrator.

    Credentials are checked against the ``ADMIN_EMAIL`` and
    ``ADMIN_PASSWORD`` environment variables — no database lookup is
    performed.  The returned JWT carries ``role: "admin"`` which grants
    access to all ``/admin/*`` endpoints.

    Returns:
        AdminLoginResponse: access_token, token_type, role.

    Raises:
        HTTP 401 — invalid admin credentials.
    """
    admin_email: str = os.getenv("ADMIN_EMAIL", "")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")

    if not admin_email or not admin_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin credentials are not configured on this server.",
        )

    if payload.email != admin_email or payload.password != admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        {"sub": "admin", "role": "admin", "email": admin_email}
    )
    return {"access_token": token, "token_type": "bearer", "role": "admin"}
