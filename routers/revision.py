"""
routers/revision.py — Spaced-repetition revision recommendations & daily notifications.

Routes
------
GET  /revision/today/{student_id}   SM-2 revision list for today (top 5)
POST /notifications/daily           Admin: compute + store revision for all students

APScheduler Integration
-----------------------
To call POST /notifications/daily automatically at 07:00 every morning,
add the following to main.py *before* `app = FastAPI(...)`:

    from apscheduler.schedulers.background import BackgroundScheduler
    import httpx

    scheduler = BackgroundScheduler()

    def _trigger_daily_notifications():
        # Use a pre-generated admin token or an internal secret header
        try:
            httpx.post(
                "http://localhost:8000/notifications/daily",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                timeout=120,
            )
        except Exception as e:
            logger.error("Daily notification job failed: %s", e)

    scheduler.add_job(
        _trigger_daily_notifications,
        trigger="cron",
        hour=7,
        minute=0,
        id="daily_notifications",
        replace_existing=True,
    )
    scheduler.start()

Install APScheduler:  pip install apscheduler
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from gamification.engine import award_xp
from routers.dependencies import get_current_admin, get_current_student
from services.analytics import get_weak_topics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Revision"])

StudentDep = Annotated[models.Student, Depends(get_current_student)]
AdminDep   = Annotated[dict[str, Any], Depends(get_current_admin)]


# ─── SM-2 priority constants ─────────────────────────────────────────────────

_CRITICAL_DAYS    = 3      # not seen for > 3 days AND score < 50%  → "critical"
_REVIEW_DAYS      = 1      # not seen for > 1 day  AND score < 65%  → "review"
_CRITICAL_THRESH  = 50.0   # score threshold for critical
_REVIEW_THRESH    = 65.0   # score threshold for review

_PRIORITY_ORDER = {"critical": 0, "scheduled": 1, "review": 2}
_ESTIMATED_MINS = {"critical": 45, "scheduled": 60, "review": 30}


# ─── Revision computation ─────────────────────────────────────────────────────

def _last_seen_date(topic: str, subject: str, student_id: int, db: Session) -> date | None:
    """
    Return the most recent date on which the student took a test covering
    *topic* in *subject*, or None if never tested.
    """
    result: models.TestResult | None = (
        db.query(models.TestResult)
        .join(models.Test, models.TestResult.test_id == models.Test.id)
        .filter(
            models.TestResult.student_id == student_id,
            models.Test.topic == topic,
            models.Test.subject == subject,
        )
        .order_by(models.TestResult.taken_at.desc())
        .first()
    )
    return result.taken_at.date() if result else None


def _avg_score_for_topic(topic: str, subject: str, student_id: int, db: Session) -> float:
    """
    Return the student's average score (0-100) for *topic* / *subject*.
    Returns 0.0 if never tested.
    """
    results: list[models.TestResult] = (
        db.query(models.TestResult)
        .join(models.Test, models.TestResult.test_id == models.Test.id)
        .filter(
            models.TestResult.student_id == student_id,
            models.Test.topic == topic,
            models.Test.subject == subject,
        )
        .all()
    )
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results), 2)


def _todays_scheduled_topics(student_id: int, db: Session) -> list[dict[str, str]]:
    """
    Pull today's topics from the most recently created StudyPlan.

    Returns a list of {topic, subject} dicts, or [] if no plan exists.
    """
    today_name = date.today().strftime("%A")   # e.g. "Monday"
    plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    if not plan or not plan.schedule_json:
        return []
    return [
        {"topic": s["topic"], "subject": s["subject"]}
        for s in plan.schedule_json
        if s.get("day", "").lower() == today_name.lower()
    ]


def _compute_revision(student_id: int, db: Session) -> list[dict[str, Any]]:
    """
    Core SM-2 revision algorithm.

    Steps:
    1. Fetch weak topics (avg < 65%) from the analytics service.
    2. Fetch today's scheduled topics from the current study plan.
    3. Apply priority rules:
       - ``critical``  — weak topic not seen in > 3 days OR never seen, score < 50%.
       - ``review``    — weak topic not seen in > 1 day, score 50-64%.
       - ``scheduled`` — topic appears in today's plan (regardless of score).
    4. Deduplicate by (topic, subject); highest-urgency priority wins.
    5. Sort by priority order, return top 5.

    Returns a list of RevisionTopic-compatible dicts.
    """
    today = date.today()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}   # key = (topic, subject)

    def _upsert(topic: str, subject: str, priority: str, reason: str) -> None:
        key = (topic, subject)
        existing = candidates.get(key)
        if existing is None or (
            _PRIORITY_ORDER[priority] < _PRIORITY_ORDER[existing["priority"]]
        ):
            candidates[key] = {
                "topic": topic,
                "subject": subject,
                "priority": priority,
                "reason": reason,
                "estimated_mins": _ESTIMATED_MINS[priority],
            }

    # ── Step 1: Weak topics → critical / review ───────────────────────────────
    weak: list[dict[str, Any]] = get_weak_topics(student_id, db)
    for wt in weak:
        subject: str = wt["subject"]
        avg: float = wt["avg_score"]

        # We don't have per-topic breakdown from analytics; query directly
        # for each unique (topic, subject) the student has been tested on
        tested_topics: list[str] = [
            row[0]
            for row in (
                db.query(models.Test.topic)
                .filter(
                    models.Test.student_id == student_id,
                    models.Test.subject == subject,
                )
                .distinct()
                .all()
            )
        ]

        for topic in tested_topics:
            topic_avg = _avg_score_for_topic(topic, subject, student_id, db)
            last_seen = _last_seen_date(topic, subject, student_id, db)
            days_since = (today - last_seen).days if last_seen else 999

            if topic_avg < _CRITICAL_THRESH and days_since > _CRITICAL_DAYS:
                _upsert(
                    topic, subject, "critical",
                    f"Score {topic_avg:.0f}% — not practised in {days_since} days",
                )
            elif topic_avg < _REVIEW_THRESH and days_since > _REVIEW_DAYS:
                _upsert(
                    topic, subject, "review",
                    f"Score {topic_avg:.0f}% — due for a review session",
                )

    # ── Step 2: Today's schedule → scheduled ─────────────────────────────────
    for entry in _todays_scheduled_topics(student_id, db):
        _upsert(
            entry["topic"],
            entry["subject"],
            "scheduled",
            "On today's study plan",
        )

    # ── Step 3: Sort and cap ──────────────────────────────────────────────────
    ranked = sorted(
        candidates.values(),
        key=lambda x: _PRIORITY_ORDER[x["priority"]],
    )
    return ranked[:5]


# ─── GET /revision/today/{student_id} ────────────────────────────────────────

@router.get(
    "/revision/today/{student_id}",
    response_model=schemas.RevisionResponse,
    summary="Today's SM-2 spaced-repetition revision list (top 5)",
)
def revision_today(
    student_id: int,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Compute and return the student's top 5 revision topics for today.

    **SM-2 Priority rules applied:**

    | Priority | Condition |
    |---|---|
    | ``critical`` | avg score < 50% **AND** not seen in > 3 days |
    | ``review``   | avg score 50-64% **AND** not seen in > 1 day |
    | ``scheduled``| topic appears in today's AI-generated study plan |

    Topics are deduplicated — if a topic qualifies for multiple levels the
    highest urgency wins.  The list is capped at 5 items.

    Returns:
        RevisionResponse: student_id, today's date, and up to 5 ranked topics.

    Raises:
        HTTP 403 — student requesting another student's revision list.
        HTTP 404 — student not found.
    """
    if current.id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own revision list.",
        )

    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found.",
        )

    topics = _compute_revision(student_id, db)
    today_str = date.today().isoformat()
    existing: models.DailyRevision | None = (
        db.query(models.DailyRevision)
        .filter(
            models.DailyRevision.student_id == student_id,
            models.DailyRevision.date == today_str,
        )
        .first()
    )
    if not existing:
        db.add(models.DailyRevision(student_id=student_id, date=today_str, topics_json=topics))
        award_xp(student_id, subject="general", topic="daily_revision", action_key="daily_revision_done", db=db)
        db.commit()
    return {
        "student_id": student_id,
        "date": date.today().isoformat(),
        "topics": topics,
    }


# ─── POST /notifications/daily ────────────────────────────────────────────────

@router.post(
    "/notifications/daily",
    response_model=schemas.DailyNotificationResponse,
    summary="[Admin] Compute and store daily revision for all students",
)
def send_daily_notifications(
    _admin: AdminDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Batch job — compute today's revision list for every registered student
    and persist each one to the ``daily_revisions`` table.

    Intended to be called automatically at 07:00 each morning by APScheduler
    (see module docstring for setup instructions).  Can also be triggered
    manually by an admin.

    Behaviour:
    - Skips students who already have a DailyRevision record for today
      (idempotent — safe to call multiple times).
    - Errors for individual students are caught, logged, and included in
      the ``errors`` list so the job continues for remaining students.

    Returns:
        DailyNotificationResponse: count of students processed + error list.

    Raises:
        HTTP 403 — caller is not an admin.

    APScheduler setup (add to main.py):
    ::

        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            lambda: httpx.post("http://localhost:8000/notifications/daily",
                               headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}),
            trigger="cron", hour=7, minute=0,
        )
        scheduler.start()
    """
    today_str = date.today().isoformat()
    students: list[models.Student] = db.query(models.Student).all()

    processed = 0
    errors: list[str] = []

    for student in students:
        # Idempotency check — skip if already delivered today
        existing: models.DailyRevision | None = (
            db.query(models.DailyRevision)
            .filter(
                models.DailyRevision.student_id == student.id,
                models.DailyRevision.date == today_str,
            )
            .first()
        )
        if existing:
            logger.debug(
                "Student %d already has revision for %s — skipping.", student.id, today_str
            )
            processed += 1
            continue

        try:
            topics = _compute_revision(student.id, db)
            revision = models.DailyRevision(
                student_id=student.id,
                date=today_str,
                topics_json=topics,
            )
            db.add(revision)
            db.commit()
            processed += 1
            logger.info(
                "DailyRevision stored for student %d (%d topics)", student.id, len(topics)
            )
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            msg = f"Student {student.id} ({student.email}): {exc}"
            logger.error("Daily notification error — %s", msg)
            errors.append(msg)

    return {"processed": processed, "errors": errors}
