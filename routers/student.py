"""
routers/student.py — Student self-service endpoints.

All routes are protected by get_current_student; each student can only
access their own data.

Routes
------
GET  /student/me            Full profile + current study plan
GET  /student/tests         Paginated test history with scores
GET  /student/plan          Current week's study plan JSON
POST /student/plan/generate Trigger the AI scheduler, return new plan
POST /test/submit           Grade submitted answers, return detailed result
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from gamification.engine import award_xp
from routers.dependencies import get_current_student
from services.scheduler import generate_study_plan

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Student"])

# Convenience alias
StudentDep = Annotated[models.Student, Depends(get_current_student)]


# ─── GET /student/me ──────────────────────────────────────────────────────────

@router.get(
    "/student/me",
    response_model=schemas.StudentFullProfile,
    summary="Full student profile and current study plan",
)
def my_profile(
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Return the authenticated student's profile combined with their most
    recently generated study plan.

    The ``current_plan`` field is ``null`` if the student has not yet
    generated a plan — prompt them to call POST /student/plan/generate.

    Returns:
        StudentFullProfile: student info + current_plan (nullable).
    """
    plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == current.id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    return {"student": current, "current_plan": plan}


# ─── GET /student/tests ───────────────────────────────────────────────────────

@router.get(
    "/student/tests",
    response_model=list[schemas.TestResultResponse],
    summary="All tests taken by the student with their scores",
)
def my_tests(
    current: StudentDep,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[models.TestResult]:
    """
    Return a paginated list of every graded test result for the authenticated
    student, ordered most recent first.

    Each result includes the raw ``answers_json`` and the auto-calculated
    percentage ``score``.

    Returns:
        list[TestResultResponse]: graded result records.
    """
    return (
        db.query(models.TestResult)
        .filter(models.TestResult.student_id == current.id)
        .order_by(models.TestResult.taken_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# ─── GET /student/plan ────────────────────────────────────────────────────────

@router.get(
    "/student/plan",
    response_model=schemas.StudyPlanResponse,
    summary="Current week's AI study plan",
)
def my_plan(
    current: StudentDep,
    db: Session = Depends(get_db),
) -> models.StudyPlan:
    """
    Return the student's most recent study plan.

    The schedule_json contains a 7-day array produced by the AI scheduler.
    If no plan exists yet the client should call POST /student/plan/generate.

    Returns:
        StudyPlanResponse: the plan including the full schedule_json.

    Raises:
        HTTP 404 — no study plan found for this student yet.
    """
    plan: models.StudyPlan | None = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == current.id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No study plan found. Generate one via POST /student/plan/generate.",
        )
    return plan


# ─── POST /student/plan/generate ─────────────────────────────────────────────

@router.post(
    "/student/plan/generate",
    response_model=schemas.StudyPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new AI weekly study plan",
)
def generate_plan(
    current: StudentDep,
    db: Session = Depends(get_db),
) -> models.StudyPlan:
    """
    Invoke the AI scheduler to produce a personalised 7-day study plan.

    The scheduler (services/scheduler.py) queries the student's subjects,
    daily_hours, and current weak topics then prompts Gemini 1.5 Flash to
    generate a structured JSON schedule.  The result is persisted and
    returned immediately.

    Returns:
        StudyPlanResponse: newly created plan with full schedule_json.

    Raises:
        HTTP 422 — LLM returned unparseable JSON after two attempts.
    """
    try:
        plan = generate_study_plan(student=current, db=db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    logger.info("Student %d generated new study plan %d", current.id, plan.id)
    return plan


# ─── POST /test/submit ────────────────────────────────────────────────────────

@router.post(
    "/test/submit",
    response_model=schemas.TestSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit answers for a test — auto-graded with per-question feedback",
)
def submit_test(
    payload: schemas.TestSubmitRequest,
    current: StudentDep,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Grade the student's submitted answers and return a detailed result.

    Grading logic:
    1. Load the Test record; verify it belongs to the authenticated student.
    2. For each question compare the submitted answer to ``correct_answer``.
    3. Score = (correct_count / total_questions) × 100, rounded to 2 dp.
    4. Persist a TestResult row with the raw answers and computed score.
    5. Return full per-question feedback including explanations.

    Body:
        - ``test_id`` — ID of the Test to grade.
        - ``answers`` — mapping of question id (str) → option letter (A/B/C/D).

    Returns:
        TestSubmitResponse: score, correct_count, total_questions, result_id,
        and a ``details`` list with per-question correct answer + explanation.

    Raises:
        HTTP 404 — test not found.
        HTTP 403 — the test belongs to a different student.
        HTTP 422 — the test has no questions (data integrity error).
    """
    test: models.Test | None = db.get(models.Test, payload.test_id)
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test {payload.test_id} not found.",
        )
    if test.student_id != current.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This test does not belong to you.",
        )

    questions: list[dict[str, Any]] = test.questions_json or []
    total = len(questions)
    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Test has no questions — cannot be graded.",
        )

    # Grade and build per-question detail
    details: list[dict[str, Any]] = []
    correct_count = 0

    for q in questions:
        qid = str(q["id"])
        chosen = payload.answers.get(qid)
        correct = q["correct_answer"]
        is_correct = chosen == correct
        if is_correct:
            correct_count += 1

        details.append(
            {
                "question_id": qid,
                "question": q["question"],
                "your_answer": chosen,
                "correct_answer": correct,
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
            }
        )

    score = round((correct_count / total) * 100, 2)

    # Persist result
    result = models.TestResult(
        student_id=current.id,
        test_id=payload.test_id,
        answers_json=payload.answers,
        score=score,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    xp_result: dict[str, Any] | None = None
    for _ in range(correct_count):
        xp_result = award_xp(current.id, test.subject, test.topic, "test_correct_answer", db)

    if score == 100:
        xp_result = award_xp(current.id, test.subject, test.topic, "test_perfect_score", db)
    if score >= 80:
        xp_result = award_xp(current.id, test.subject, test.topic, "test_above_80", db)

    topic_test_ids = [
        row[0]
        for row in (
            db.query(models.Test.id)
            .filter(models.Test.subject == test.subject, models.Test.topic == test.topic)
            .all()
        )
    ]
    first_test_on_topic = (
        db.query(models.TestResult)
        .filter(
            models.TestResult.student_id == current.id,
            models.TestResult.test_id.in_(topic_test_ids),
        )
        .count()
        == 1
    )
    if first_test_on_topic:
        xp_result = award_xp(current.id, test.subject, test.topic, "ncert_topic_first", db)

    logger.info(
        "Test submitted: student=%d test=%d score=%.1f%% (%d/%d)",
        current.id, payload.test_id, score, correct_count, total,
    )

    return {
        "score": score,
        "correct_count": correct_count,
        "total_questions": total,
        "result_id": result.id,
        "details": details,
        "xp_result": xp_result,
    }
