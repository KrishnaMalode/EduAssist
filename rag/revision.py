"""
rag/revision.py — Spaced repetition engine using a simplified SM-2 algorithm.

SM-2 Overview
-------------
SM-2 (SuperMemo 2) schedules review intervals based on performance quality.
The core formula adjusts the *easiness factor* (EF) after each answer:

    EF = max(1.3, EF + 0.1 − (5 − q) × (0.08 + (5 − q) × 0.02))

Where *q* is quality (0–5).  We map score_pct to quality:

    score ≥ 90%  → q = 5
    score ≥ 75%  → q = 4
    score ≥ 60%  → q = 3
    score ≥ 45%  → q = 2
    score ≥ 25%  → q = 1
    else         → q = 0

Intervals grow as:  rep 1 → 1 day, rep 2 → 6 days, rep n → prev × EF.
A *negative* days_until_review means the topic is overdue.

Public API
----------
compute_sm2(history)             → float days_until_review
get_revision_plan(student_id, db) → list[dict]  ranked revision items
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

import models

logger = logging.getLogger(__name__)

# ─── SM-2 constants ───────────────────────────────────────────────────────────

SM2_EASINESS_FACTOR: float = 2.5   # default starting EF
SM2_EF_MIN: float = 1.3            # floor for EF

_INITIAL_INTERVALS = [1, 6]        # days for repetition 1 and 2


# ─── Score → quality mapping ─────────────────────────────────────────────────

def _score_to_quality(score_pct: float) -> int:
    """
    Map a percentage score (0–100) to an SM-2 quality value (0–5).

    Args:
        score_pct: Percentage score, e.g. 72.5.

    Returns:
        Integer quality in range [0, 5].
    """
    if score_pct >= 90:
        return 5
    if score_pct >= 75:
        return 4
    if score_pct >= 60:
        return 3
    if score_pct >= 45:
        return 2
    if score_pct >= 25:
        return 1
    return 0


# ─── Public: compute_sm2 ──────────────────────────────────────────────────────

def compute_sm2(history: list[dict[str, Any]]) -> float:
    """
    Run the simplified SM-2 algorithm over a topic's attempt history and
    return how many days from *today* the next review should occur.

    Args:
        history: List of attempt dicts in chronological order (oldest first).
                 Each dict must contain:
                 - ``score_pct``  (float) — percentage score 0–100
                 - ``taken_at``   (datetime | str) — when the test was taken

    Returns:
        days_until_review (float).
        - **Positive** → topic is up-to-date, review due in N days.
        - **Negative** → topic is overdue by abs(N) days.
        - Returns ``0.0`` for empty history (review immediately).

    Example::

        history = [
            {"score_pct": 40.0, "taken_at": datetime(2025, 1, 1)},
            {"score_pct": 68.0, "taken_at": datetime(2025, 1, 3)},
        ]
        days = compute_sm2(history)
    """
    if not history:
        return 0.0

    ef = SM2_EASINESS_FACTOR
    interval: float = 0.0
    last_date: date = date.today()

    for rep_idx, attempt in enumerate(history):
        # Resolve taken_at to a date object
        taken_at = attempt.get("taken_at", datetime.utcnow())
        if isinstance(taken_at, datetime):
            last_date = taken_at.date()
        elif isinstance(taken_at, date):
            last_date = taken_at
        else:
            try:
                last_date = datetime.fromisoformat(str(taken_at)).date()
            except (ValueError, TypeError):
                logger.warning("Could not parse taken_at='%s'; using today.", taken_at)
                last_date = date.today()

        q = _score_to_quality(float(attempt.get("score_pct", 0)))
        ef = max(SM2_EF_MIN, ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

        if rep_idx < len(_INITIAL_INTERVALS):
            interval = float(_INITIAL_INTERVALS[rep_idx])
        else:
            interval = max(1.0, interval * ef)

    days_since_last = (date.today() - last_date).days
    days_until_review = interval - days_since_last

    logger.debug(
        "SM-2: %d attempts | EF=%.2f | interval=%.1f | days_since=%d | due_in=%.1f",
        len(history), ef, interval, days_since_last, days_until_review,
    )
    return round(days_until_review, 2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _todays_plan_topics(student_id: int, db: Session) -> set[tuple[str, str]]:
    """
    Return a set of (topic, subject) tuples from today's study plan.

    Returns an empty set if the student has no current plan.
    """
    today_name = date.today().strftime("%A")
    plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    if not plan or not plan.schedule_json:
        return set()

    return {
        (s["topic"], s["subject"])
        for s in plan.schedule_json
        if s.get("day", "").lower() == today_name.lower()
    }


def _estimated_mins(days_overdue: float, avg_score: float) -> int:
    """
    Heuristic estimate of how many minutes a revision session should take.

    Overdue critical topics get 45 min; upcoming topics 30 min;
    high-scorers always get shorter sessions.
    """
    if avg_score >= 80:
        return 20
    if days_overdue > 0:
        return 45
    return 30


def _revision_reason(days_overdue: float, avg_score: float, is_scheduled: bool) -> str:
    """Build a human-readable reason string for this revision item."""
    parts: list[str] = []
    if is_scheduled:
        parts.append("on today's study plan")
    if days_overdue > 0:
        parts.append(f"overdue by {days_overdue:.0f} day(s)")
    if avg_score < 50:
        parts.append(f"low avg score ({avg_score:.0f}%)")
    elif avg_score < 65:
        parts.append(f"needs improvement ({avg_score:.0f}%)")
    return "; ".join(parts) if parts else "scheduled review"


# ─── Public: get_revision_plan ────────────────────────────────────────────────

def get_revision_plan(student_id: int, db: Session) -> list[dict[str, Any]]:
    """
    Generate a personalised revision plan for the student using SM-2 scheduling.

    Algorithm:
    1. Fetch all TestResults for the student, joined with the Test table
       to obtain topic and subject.
    2. Group attempts by (topic, subject).
    3. Run ``compute_sm2()`` on each group's history (sorted oldest-first).
    4. Compute ``days_overdue = -days_until_review`` (positive = overdue).
    5. Set priority:
       - ``"critical"``  if days_overdue > 0
       - ``"upcoming"``  otherwise
    6. Overlay today's StudyPlan: any topic scheduled for today is promoted
       to ``"critical"`` if already overdue, else tagged with ``"scheduled"``
       as part of its reason string.
    7. Sort by days_overdue descending (most urgent first).

    Args:
        student_id: The student's database ID.
        db:         Active SQLAlchemy session.

    Returns:
        A list of revision item dicts, each containing:
        - ``topic``          (str)
        - ``subject``        (str)
        - ``days_overdue``   (float — negative means days_until_review)
        - ``avg_score``      (float — percentage)
        - ``priority``       (str — ``"critical"`` | ``"upcoming"``)
        - ``estimated_mins`` (int)
        - ``reason``         (str)
    """
    # Fetch all results with test metadata
    results: list[models.TestResult] = (
        db.query(models.TestResult)
        .filter(models.TestResult.student_id == student_id)
        .order_by(models.TestResult.taken_at.asc())
        .all()
    )

    if not results:
        logger.info("No test results for student %d — empty revision plan.", student_id)
        return []

    # Group by (topic, subject)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        test: models.Test | None = db.get(models.Test, result.test_id)
        if not test:
            continue
        key = (test.topic, test.subject)
        groups[key].append(
            {"score_pct": result.score, "taken_at": result.taken_at}
        )

    scheduled_today = _todays_plan_topics(student_id, db)

    plan: list[dict[str, Any]] = []
    for (topic, subject), history in groups.items():
        avg_score = round(sum(h["score_pct"] for h in history) / len(history), 2)
        days_until = compute_sm2(history)            # negative = overdue
        days_overdue = round(-days_until, 2)         # positive = overdue
        is_scheduled = (topic, subject) in scheduled_today

        priority = "critical" if days_overdue > 0 else "upcoming"
        reason = _revision_reason(days_overdue, avg_score, is_scheduled)
        mins = _estimated_mins(days_overdue, avg_score)

        plan.append(
            {
                "topic": topic,
                "subject": subject,
                "days_overdue": days_overdue,
                "avg_score": avg_score,
                "priority": priority,
                "estimated_mins": mins,
                "reason": reason,
            }
        )

    # Most overdue first, then alphabetical for stability
    plan.sort(key=lambda x: (-x["days_overdue"], x["topic"]))

    logger.info(
        "Revision plan for student %d: %d topics (%d critical).",
        student_id,
        len(plan),
        sum(1 for p in plan if p["priority"] == "critical"),
    )
    return plan
