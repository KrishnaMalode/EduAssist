from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import models
from database import get_db
from gamification.engine import apply_decay, get_topic_mastery_map
from gamification.models_ext import Badge, StudentProfile
from routers.auth import get_current_student

router = APIRouter(tags=["Gamification"])

StudentDep = Annotated[models.Student, Depends(get_current_student)]


@router.get("/gamification/profile/{student_id}")
def get_profile(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    profile = db.query(StudentProfile).filter(StudentProfile.student_id == student_id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    try:
        badge_ids = json.loads(profile.badges or "[]")
    except json.JSONDecodeError:
        badge_ids = []

    badge_rows = db.query(Badge).filter(Badge.id.in_(badge_ids)).all() if badge_ids else []

    return {
        "profile": profile,
        "badges": [
            {"id": b.id, "name": b.name, "description": b.description, "icon_key": b.icon_key}
            for b in badge_rows
        ],
    }


@router.get("/gamification/mastery/{student_id}")
def mastery_map(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    items = get_topic_mastery_map(student_id, db)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["subject"], []).append(item)

    return grouped


@router.get("/gamification/mastery/{student_id}/{subject}")
def mastery_subject(
    student_id: int,
    subject: str,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    items = [i for i in get_topic_mastery_map(student_id, db) if i["subject"].lower() == subject.lower()]
    return {subject: items}


@router.post("/gamification/decay/apply/{student_id}")
def decay_apply(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if current.id != student_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return apply_decay(student_id, db)


@router.get("/gamification/leaderboard")
def leaderboard(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (
        db.query(StudentProfile, models.Student)
        .join(models.Student, StudentProfile.student_id == models.Student.id)
        .order_by(StudentProfile.total_xp.desc())
        .limit(10)
        .all()
    )

    results: list[dict[str, Any]] = []
    for idx, (profile, student) in enumerate(rows, start=1):
        try:
            badges = json.loads(profile.badges or "[]")
            badges_count = len(badges) if isinstance(badges, list) else 0
        except json.JSONDecodeError:
            badges_count = 0

        results.append(
            {
                "rank": idx,
                "name": student.name,
                "total_xp": profile.total_xp,
                "global_level": profile.global_level,
                "title": profile.title,
                "badges_count": badges_count,
            }
        )

    return results
