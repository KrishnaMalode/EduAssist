"""
services/analytics.py — Student performance analytics.

Provides:
  - get_summary()     — aggregated performance metrics
  - get_weak_topics() — subjects where the student underperforms (<65%)

All computations are done in-process over SQLAlchemy query results.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from models import Test, TestResult

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _results_for_student(student_id: int, db: Session) -> list[TestResult]:
    """Return all TestResult rows for a student, ordered by taken_at."""
    return (
        db.query(TestResult)
        .filter(TestResult.student_id == student_id)
        .order_by(TestResult.taken_at)
        .all()
    )


def _compute_study_streak(results: list[TestResult]) -> int:
    """
    Count the number of consecutive calendar days (ending today) on which
    the student took at least one test.

    Returns:
        Integer streak count (0 if no test taken today or yesterday).
    """
    if not results:
        return 0

    # Collect unique activity dates (date objects)
    active_dates: set[date] = {r.taken_at.date() for r in results}
    today = date.today()

    streak = 0
    current = today
    while current in active_dates:
        streak += 1
        current -= timedelta(days=1)

    return streak


def _scores_by_subject(results: list[TestResult], db: Session) -> dict[str, list[float]]:
    """Return a mapping of subject → [scores] using the related Test.subject."""
    subject_scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        test: Test | None = db.get(Test, result.test_id)
        if test:
            subject_scores[test.subject].append(result.score)
    return dict(subject_scores)


# ─── Public API ───────────────────────────────────────────────────────────────

def get_summary(student_id: int, db: Session) -> dict[str, Any]:
    """
    Compute aggregated performance metrics for a student.

    Returns:
        A dict with keys:
          - tests_taken     (int)
          - avg_score       (float, 0-100)
          - study_streak    (int, consecutive days)
          - best_subject    (str | None)
          - worst_subject   (str | None)
          - scores_by_day   list of {"date": str, "score": float}
    """
    results = _results_for_student(student_id, db)

    if not results:
        return {
            "tests_taken": 0,
            "avg_score": 0.0,
            "study_streak": 0,
            "best_subject": None,
            "worst_subject": None,
            "scores_by_day": [],
        }

    tests_taken = len(results)
    avg_score = round(sum(r.score for r in results) / tests_taken, 2)
    study_streak = _compute_study_streak(results)

    # Per-subject averages
    subject_scores = _scores_by_subject(results, db)
    subject_avgs: dict[str, float] = {
        subj: round(sum(scores) / len(scores), 2)
        for subj, scores in subject_scores.items()
    }

    best_subject: str | None = None
    worst_subject: str | None = None
    if subject_avgs:
        best_subject = max(subject_avgs, key=lambda s: subject_avgs[s])
        worst_subject = min(subject_avgs, key=lambda s: subject_avgs[s])

    # Scores grouped by calendar date (use the last score of each day)
    scores_by_day_map: dict[str, float] = {}
    for result in results:
        day_key = result.taken_at.date().isoformat()
        if day_key not in scores_by_day_map:
            scores_by_day_map[day_key] = result.score
        else:
            # Keep average of all tests on the same day
            existing = scores_by_day_map[day_key]
            scores_by_day_map[day_key] = round((existing + result.score) / 2, 2)

    scores_by_day = [
        {"date": d, "score": s} for d, s in sorted(scores_by_day_map.items())
    ]

    return {
        "tests_taken": tests_taken,
        "avg_score": avg_score,
        "study_streak": study_streak,
        "best_subject": best_subject,
        "worst_subject": worst_subject,
        "scores_by_day": scores_by_day,
    }


def get_weak_topics(student_id: int, db: Session) -> list[dict[str, Any]]:
    """
    Identify subjects where the student's average score is below 65%.

    Returns:
        A list of dicts sorted by severity (worst first):
          - subject     (str)
          - avg_score   (float)
          - tests_taken (int)
          - severity    "critical" (<40%) | "moderate" (40-54%) | "mild" (55-64%)
    """
    results = _results_for_student(student_id, db)
    if not results:
        return []

    subject_scores = _scores_by_subject(results, db)
    weak: list[dict[str, Any]] = []

    for subject, scores in subject_scores.items():
        avg = round(sum(scores) / len(scores), 2)
        if avg < 65.0:
            if avg < 40.0:
                severity = "critical"
            elif avg < 55.0:
                severity = "moderate"
            else:
                severity = "mild"

            weak.append(
                {
                    "subject": subject,
                    "avg_score": avg,
                    "tests_taken": len(scores),
                    "severity": severity,
                }
            )

    # Sort by avg_score ascending (worst performance first)
    weak.sort(key=lambda x: x["avg_score"])

    logger.debug("Weak topics for student %d: %s", student_id, weak)
    return weak
