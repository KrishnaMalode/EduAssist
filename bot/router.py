"""bot/router.py — FastAPI router for the EduAssist chat bot."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
from bot.chain import run_chat
from bot.context import build_context
from database import get_db
from routers.auth import get_current_student

router = APIRouter(tags=["Chat Bot"])

StudentDep = Annotated[models.Student, Depends(get_current_student)]


class ChatRequest(BaseModel):
    student_id: int
    message: str = Field(..., min_length=1, max_length=500)


@router.post("/chat", status_code=status.HTTP_200_OK)
async def chat(
    payload: ChatRequest,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != payload.student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    today_start = datetime.combine(date.today(), datetime.min.time())
    sent_today = (
        db.query(models.ChatMessage)
        .filter(
            models.ChatMessage.student_id == payload.student_id,
            models.ChatMessage.timestamp >= today_start,
        )
        .count()
    )
    if sent_today >= 30:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Daily message limit reached")

    return await run_chat(payload.student_id, payload.message, db)


@router.get("/chat/history/{student_id}")
async def get_history(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    student: models.Student | None = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    cleared_at = getattr(student, "chat_cleared_at", None) or student.cleared_at
    query = db.query(models.ChatMessage).filter(models.ChatMessage.student_id == student_id)
    if cleared_at:
        query = query.filter(models.ChatMessage.timestamp > cleared_at)

    total = query.count()
    messages = (
        query
        .order_by(models.ChatMessage.timestamp.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "messages": messages,
        "total": total,
        "has_more": (offset + len(messages)) < total,
    }


@router.delete("/chat/history/{student_id}")
async def clear_history(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    student: models.Student | None = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    now = datetime.utcnow()
    if hasattr(student, "chat_cleared_at"):
        setattr(student, "chat_cleared_at", now)
    else:
        student.cleared_at = now
    db.commit()

    return {"cleared": True, "cleared_at": now}


@router.get("/chat/context/{student_id}")
async def get_context(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    ctx = build_context(student_id, db)
    return asdict(ctx)
