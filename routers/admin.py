"""
routers/admin.py — Admin-only management endpoints.

All routes require a valid admin JWT (role == "admin") obtained from
POST /auth/admin/login.  Any student token will receive HTTP 403.

Routes
------
GET  /admin/overview                   Platform-level KPIs
GET  /admin/students                   Paginated student list with flags
GET  /admin/students/{id}              Full student dossier
GET  /admin/students/{id}/progress     8-week weekly progress series
POST /admin/students/{id}/note         Add an admin note
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
from routers.dependencies import get_current_admin
from services.analytics import get_summary, get_weak_topics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# Convenience alias — all admin routes share the same dependency
AdminDep = Annotated[dict[str, Any], Depends(get_current_admin)]


# ─── Helper: fetch student or raise 404 ──────────────────────────────────────

def _get_student_or_404(student_id: int, db: Session) -> models.Student:
    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found.",
        )
    return student


# ─── GET /admin/overview ──────────────────────────────────────────────────────

@router.get(
    "/overview",
    response_model=schemas.AdminOverview,
    summary="Platform-level KPI dashboard",
)
def admin_overview(
    _admin: AdminDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Return high-level platform metrics visible on the admin dashboard.

    Metrics:
    - **total_students** — total registered student count.
    - **avg_score_all** — mean score across every TestResult in the DB.
    - **active_today** — students who submitted at least one test today.
    - **most_failed_topics** — top 5 topics with the lowest average score,
      computed across all students and all test results.

    Raises:
        HTTP 403 — caller is not an admin.
    """
    today = date.today()

    # total students
    total_students: int = db.query(models.Student).count()

    # avg score across all results
    all_results: list[models.TestResult] = db.query(models.TestResult).all()
    avg_score_all = (
        round(sum(r.score for r in all_results) / len(all_results), 2)
        if all_results
        else 0.0
    )

    # active today: distinct students with a result taken today
    active_ids: set[int] = {
        r.student_id
        for r in all_results
        if r.taken_at.date() == today
    }
    active_today = len(active_ids)

    # most failed topics: aggregate (topic, subject) → [scores]
    topic_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in all_results:
        test: models.Test | None = db.get(models.Test, result.test_id)
        if test:
            topic_scores[(test.topic, test.subject)].append(result.score)

    topic_stats: list[dict[str, Any]] = [
        {
            "topic": topic,
            "subject": subject,
            "avg_score": round(sum(scores) / len(scores), 2),
            "attempts": len(scores),
        }
        for (topic, subject), scores in topic_scores.items()
    ]
    # Sort ascending by avg_score (worst first), take top 5
    most_failed = sorted(topic_stats, key=lambda x: x["avg_score"])[:5]

    return {
        "total_students": total_students,
        "avg_score_all": avg_score_all,
        "active_today": active_today,
        "most_failed_topics": most_failed,
    }


# ─── GET /admin/students ──────────────────────────────────────────────────────

@router.get(
    "/students",
    response_model=list[schemas.AdminStudentListItem],
    summary="List all students with lightweight performance summary",
)
def admin_list_students(
    _admin: AdminDep,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """
    Return a paginated list of all students, sorted by most recently active.

    Each row includes:
    - ``avg_score`` — mean score across all tests (0.0 if none taken).
    - ``last_active`` — timestamp of the most recent test result.
    - ``tests_taken`` — total number of tests submitted.
    - ``has_flag`` — True if any AdminNote exists for this student.

    Raises:
        HTTP 403 — caller is not an admin.
    """
    students: list[models.Student] = (
        db.query(models.Student).offset(skip).limit(limit).all()
    )

    rows: list[dict[str, Any]] = []
    for s in students:
        results: list[models.TestResult] = (
            db.query(models.TestResult)
            .filter(models.TestResult.student_id == s.id)
            .order_by(models.TestResult.taken_at.desc())
            .all()
        )
        avg_score = (
            round(sum(r.score for r in results) / len(results), 2)
            if results
            else 0.0
        )
        last_active: datetime | None = results[0].taken_at if results else None
        has_flag: bool = (
            db.query(models.AdminNote)
            .filter(models.AdminNote.student_id == s.id)
            .first()
        ) is not None

        rows.append(
            {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "avg_score": avg_score,
                "last_active": last_active,
                "tests_taken": len(results),
                "has_flag": has_flag,
            }
        )

    # Sort by last_active descending (None values pushed to end)
    rows.sort(key=lambda r: r["last_active"] or datetime.min, reverse=True)
    return rows


# ─── GET /admin/students/{id} ─────────────────────────────────────────────────

@router.get(
    "/students/{student_id}",
    response_model=schemas.AdminStudentDetail,
    summary="Full student dossier — plan, tests, chat, notes, analytics",
)
def admin_student_detail(
    student_id: int,
    _admin: AdminDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Return a comprehensive view of a single student for the admin panel.

    Includes:
    - Basic student profile.
    - Most recent study plan (or None if not yet generated).
    - Last 5 tests (questions included).
    - Last 10 chat messages (chronological).
    - All admin notes (newest first).
    - Full analytics summary (from analytics service).

    Raises:
        HTTP 403 — caller is not an admin.
        HTTP 404 — student not found.
    """
    student = _get_student_or_404(student_id, db)

    latest_plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )

    last_5_tests: list[models.Test] = (
        db.query(models.Test)
        .filter(models.Test.student_id == student_id)
        .order_by(models.Test.created_at.desc())
        .limit(5)
        .all()
    )

    last_10_messages: list[models.ChatMessage] = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.student_id == student_id)
        .order_by(models.ChatMessage.timestamp.desc())
        .limit(10)
        .all()[::-1]   # return in chronological order
    )

    admin_notes: list[models.AdminNote] = (
        db.query(models.AdminNote)
        .filter(models.AdminNote.student_id == student_id)
        .order_by(models.AdminNote.created_at.desc())
        .all()
    )

    analytics = get_summary(student_id, db)
    # Coerce to AnalyticsSummary-compatible dict (scores_by_day already matches)

    return {
        "student": student,
        "latest_study_plan": latest_plan,
        "last_5_tests": last_5_tests,
        "last_10_messages": last_10_messages,
        "admin_notes": admin_notes,
        "analytics": analytics,
    }


# ─── GET /admin/students/{id}/progress ───────────────────────────────────────

@router.get(
    "/students/{student_id}/progress",
    response_model=list[schemas.AdminProgressWeek],
    summary="8-week weekly progress series for chart rendering",
)
def admin_student_progress(
    student_id: int,
    _admin: AdminDep,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """
    Return per-week aggregated performance data for the last 8 calendar weeks.

    Each element represents one ISO week and contains:
    - ``week_start`` — Monday of that week (ISO format ``YYYY-MM-DD``).
    - ``avg_score``  — mean score for all tests taken in that week.
    - ``tests_taken`` — number of tests submitted that week.
    - ``topics_covered`` — deduplicated list of topics tested that week.

    Weeks with no activity are included with ``avg_score=0`` so the chart
    always renders a full 8-point series.

    Raises:
        HTTP 403 — caller is not an admin.
        HTTP 404 — student not found.
    """
    _get_student_or_404(student_id, db)

    # Build week buckets for the last 8 weeks (most recent first)
    today = date.today()
    # Start of current ISO week (Monday)
    this_monday = today - timedelta(days=today.weekday())
    week_starts: list[date] = [this_monday - timedelta(weeks=i) for i in range(8)]

    # Fetch all results for the student with their linked test
    results: list[models.TestResult] = (
        db.query(models.TestResult)
        .filter(models.TestResult.student_id == student_id)
        .all()
    )

    # Index results by their ISO week Monday
    def _monday(dt: datetime) -> date:
        d = dt.date()
        return d - timedelta(days=d.weekday())

    bucket: dict[date, list[models.TestResult]] = defaultdict(list)
    for r in results:
        bucket[_monday(r.taken_at)].append(r)

    weeks: list[dict[str, Any]] = []
    for ws in week_starts:
        week_results = bucket.get(ws, [])
        if week_results:
            avg = round(sum(r.score for r in week_results) / len(week_results), 2)
            topics: list[str] = []
            for r in week_results:
                test = db.get(models.Test, r.test_id)
                if test and test.topic not in topics:
                    topics.append(test.topic)
        else:
            avg = 0.0
            topics = []

        weeks.append(
            {
                "week_start": ws.isoformat(),
                "avg_score": avg,
                "tests_taken": len(week_results),
                "topics_covered": topics,
            }
        )

    # Return chronological order (oldest first = better for charting)
    weeks.reverse()
    return weeks


# ─── POST /admin/students/{id}/note ──────────────────────────────────────────

@router.post(
    "/students/{student_id}/note",
    response_model=schemas.AdminNoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an admin note about a student",
)
def add_note(
    student_id: int,
    payload: schemas.AdminNoteBody,
    _admin: AdminDep,
    db: Session = Depends(get_db),
) -> models.AdminNote:
    """
    Create a new administrative note about a student.

    Notes are displayed in the admin student detail view and can be used
    to flag at-risk students or record observations.

    Returns:
        AdminNoteResponse: the newly persisted note record.

    Raises:
        HTTP 403 — caller is not an admin.
        HTTP 404 — student not found.
    """
    _get_student_or_404(student_id, db)

    note = models.AdminNote(student_id=student_id, note=payload.note)
    db.add(note)
    db.commit()
    db.refresh(note)

    logger.info("Admin note %d added for student %d", note.id, student_id)
    return note
