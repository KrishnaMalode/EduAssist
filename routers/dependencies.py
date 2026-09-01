"""
routers/dependencies.py — Shared FastAPI dependency functions.

Provides:
  - get_current_student()  Decode JWT, verify role == "student", return Student ORM object.
  - get_current_admin()    Decode JWT, verify role == "admin", return raw payload dict.

Both are used as FastAPI Depends() parameters across all routers.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from database import get_db
import models

# ─── JWT settings (mirrors what auth.py uses to sign tokens) ─────────────────

SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM: str = "HS256"

# tokenUrl must match the login endpoint path so Swagger's "Authorize" button works
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ─── Student dependency ───────────────────────────────────────────────────────

def get_current_student(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db),
) -> models.Student:
    """
    Decode the Bearer JWT and return the corresponding Student ORM object.

    Raises:
        HTTP 401 — if the token is missing, expired, or has an invalid signature.
        HTTP 401 — if the payload's role is not "student".
        HTTP 401 — if the student_id in the token no longer exists in the DB.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        student_id: str | None = payload.get("sub")
        role: str = payload.get("role", "student")

        if student_id is None:
            raise credentials_exc
        if role != "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint requires a student token.",
            )
    except JWTError:
        raise credentials_exc

    student: models.Student | None = db.get(models.Student, int(student_id))
    if student is None:
        raise credentials_exc
    return student


# ─── Admin dependency ─────────────────────────────────────────────────────────

def get_current_admin(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, Any]:
    """
    Decode the Bearer JWT and verify the caller is an admin.

    Returns the raw decoded payload (contains "role", "email", "sub").

    Raises:
        HTTP 401 — if the token is missing, expired, or invalid.
        HTTP 403 — if the token's role is not "admin".
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate admin credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise credentials_exc

    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )
    return payload
