from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

import models
from services.analytics import get_weak_topics


@dataclass
class StudentContext:
    student_id: int
    name: str
    weak_topics: list[str]
    todays_topics: list[str]
    recent_scores: list[float]
    is_struggling: bool


def build_context(student_id: int, db: Session) -> StudentContext:
    """
    Fetch all fields from DB.
    - weak_topics: call analytics.get_weak_topics(), extract topic names
    - todays_topics: load student's current StudyPlan, parse schedule_json,
      find entries where day matches today's weekday
    - recent_scores: last 5 TestResult.score for this student, newest first
    - is_struggling: True if len(recent_scores) >= 3 and all(
        s < 50 for s in recent_scores[:3])
    """
    student: models.Student | None = db.get(models.Student, student_id)
    if not student:
        raise ValueError(f"Student {student_id} not found")

    weak_raw = get_weak_topics(student_id, db)
    weak_topics = [item["subject"] for item in weak_raw]

    today_name = date.today().strftime("%A")
    plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    todays_topics: list[str] = []
    if plan and plan.schedule_json:
        todays_topics = [
            entry.get("topic", "")
            for entry in plan.schedule_json
            if entry.get("day", "").lower() == today_name.lower()
        ]
        todays_topics = [t for t in todays_topics if t]

    recent_scores = [
        r.score
        for r in (
            db.query(models.TestResult)
            .filter(models.TestResult.student_id == student_id)
            .order_by(models.TestResult.taken_at.desc())
            .limit(5)
            .all()
        )
    ]

    last_3 = recent_scores[:3]
    is_struggling = len(last_3) >= 3 and all(s < 50 for s in last_3)

    return StudentContext(
        student_id=student_id,
        name=student.name,
        weak_topics=weak_topics,
        todays_topics=todays_topics,
        recent_scores=recent_scores,
        is_struggling=is_struggling,
    )
