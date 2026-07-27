"""
main.py — EduAssist FastAPI application entry point.

Startup:
  - Creates all DB tables via SQLAlchemy (lifespan event).
  - Applies CORS middleware (allow all origins — for development).

Routers registered
------------------
  /auth/*              routers/auth.py       — register, login, admin login
  /admin/*             routers/admin.py      — admin dashboard & management
  /student/*           routers/student.py    — student self-service
  /test/submit         routers/student.py    — detailed test grading
  /revision/*          routers/revision.py   — SM-2 daily revision
  /notifications/*     routers/revision.py   — bulk notification job
  /syllabus/*          rag_router.py         — PDF/text ingest, subject listing
  /chat (POST)         bot/router.py         — AI tutor (rate-limited)
  /chat/history/*      bot/router.py         — paginated history + soft-delete
  /chat/context/*      bot/router.py         — student context inspector

Legacy inline routes (kept for backwards compatibility):
  /students/*          CRUD on Student table
  /study-plans/*       Study plan generation & retrieval
  /tests/*             Test generation & result listing
  /analytics/*         Analytics summary & weak topics
  /chat/*              AI tutoring chat
  /health              Liveness check
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session

import models
import schemas
from database import Base, engine, get_db, SessionLocal
from gamification.engine import BADGE_SEEDS
from gamification.models_ext import Badge
from gamification.router import router as gamification_router
from ncert.router import router as ncert_router
from routers import auth as auth_router
from routers import admin as admin_router
from routers import student as student_router
from routers import revision as revision_router
from rag_router import router as rag_router
from bot.router import router as bot_router
from routers.dependencies import get_current_student  # shared dependency
from services.analytics import get_summary, get_weak_topics
from services.assessment import generate_test
from services.chatbot import chat as chatbot_chat
from services.scheduler import generate_study_plan

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eduassist")


def _get_allowed_origins() -> list[str]:
    """Read CORS origins from ALLOWED_ORIGINS env var (comma-separated)."""
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]

# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all DB tables on startup; log shutdown on exit."""
    logger.info("Creating database tables …")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for entry in BADGE_SEEDS:
            db.merge(Badge(**entry))
        db.commit()
        logger.info("EduAssist started. Badges seeded: %d", len(BADGE_SEEDS))
    finally:
        db.close()
    logger.info("Database ready.")
    yield
    logger.info("EduAssist shutting down.")


# ─── App instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduAssist API",
    description=(
        "AI-powered student tutoring platform backend. "
        "Use POST /auth/login (student) or POST /auth/admin/login (admin) "
        "to get a Bearer token, then click the 🔒 Authorize button."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ─── CORS (all origins — development only) ────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register new routers ─────────────────────────────────────────────────────

app.include_router(auth_router.router)       # /auth/*
app.include_router(admin_router.router)      # /admin/*
app.include_router(student_router.router)    # /student/*, /test/submit
app.include_router(revision_router.router)   # /revision/*, /notifications/*
app.include_router(rag_router)               # /syllabus/*
app.include_router(bot_router)               # /chat (POST), /chat/history/*, /chat/context/*
app.include_router(ncert_router)             # /ncert/*
app.include_router(gamification_router)      # /gamification/*


# ══════════════════════════════════════════════════════════════════════════════
# STUDENTS  /students  (legacy CRUD — kept for backwards compatibility)
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/students",
    response_model=list[schemas.StudentResponse],
    tags=["Students (legacy)"],
    summary="List all students",
)
def list_students(
    skip: int = 0,
    limit: int = 50,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[models.Student]:
    """Return a paginated list of all registered students."""
    return db.query(models.Student).offset(skip).limit(limit).all()


@app.get(
    "/students/{student_id}",
    response_model=schemas.StudentResponse,
    tags=["Students (legacy)"],
    summary="Get student by ID",
)
def get_student(
    student_id: int,
    _current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.Student:
    """Return a student's profile by their database ID."""
    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")
    return student


@app.patch(
    "/students/{student_id}",
    response_model=schemas.StudentResponse,
    tags=["Students (legacy)"],
    summary="Update name, subjects, or daily_hours",
)
def update_student(
    student_id: int,
    payload: schemas.StudentUpdate,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.Student:
    """Partially update a student's own profile."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="You can only update your own profile.")

    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(student, field, value)

    db.commit()
    db.refresh(student)
    return student


@app.delete(
    "/students/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Students (legacy)"],
    summary="Delete a student and all related data",
)
def delete_student(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> None:
    """Permanently delete a student account (cascade-deletes all related records)."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="You can only delete your own account.")

    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")

    db.delete(student)
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# STUDY PLANS  /study-plans
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/study-plans/{student_id}/generate",
    response_model=schemas.StudyPlanResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Study Plans"],
    summary="Generate an AI weekly study plan",
)
def generate_plan_legacy(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.StudyPlan:
    """Invoke the AI scheduler for a personalised 7-day study plan."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    student = db.get(models.Student, student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")

    try:
        return generate_study_plan(student, db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get(
    "/study-plans/{student_id}",
    response_model=list[schemas.StudyPlanResponse],
    tags=["Study Plans"],
    summary="List all study plans for a student",
)
def list_study_plans(
    student_id: int,
    skip: int = 0,
    limit: int = 10,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[models.StudyPlan]:
    """Return all study plans for the given student, newest first."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    return (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.get(
    "/study-plans/{student_id}/latest",
    response_model=schemas.StudyPlanResponse,
    tags=["Study Plans"],
    summary="Get the most recent study plan",
)
def get_latest_plan(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.StudyPlan:
    """Return the student's most recently created study plan."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    plan = (
        db.query(models.StudyPlan)
        .filter(models.StudyPlan.student_id == student_id)
        .order_by(models.StudyPlan.created_at.desc())
        .first()
    )
    if not plan:
        raise HTTPException(status_code=404, detail="No study plan found. Generate one first.")
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# TESTS  /tests
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/tests/{student_id}/generate",
    response_model=schemas.TestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tests"],
    summary="Generate an AI MCQ test",
)
def create_test(
    student_id: int,
    payload: schemas.TestCreate,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.Test:
    """Generate a new MCQ test (RAG if FAISS index exists, otherwise pure LLM)."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    try:
        return generate_test(
            subject=payload.subject,
            topic=payload.topic,
            difficulty=payload.difficulty,
            n_questions=payload.n_questions,
            student_id=student_id,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get(
    "/tests/{student_id}",
    response_model=list[schemas.TestResponse],
    tags=["Tests"],
    summary="List all tests for a student",
)
def list_tests(
    student_id: int,
    skip: int = 0,
    limit: int = 20,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[models.Test]:
    """Return all tests generated for the student, newest first."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    return (
        db.query(models.Test)
        .filter(models.Test.student_id == student_id)
        .order_by(models.Test.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.get(
    "/tests/detail/{test_id}",
    response_model=schemas.TestResponse,
    tags=["Tests"],
    summary="Get a specific test by ID",
)
def get_test(
    test_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> models.Test:
    """Return a test's full question set by its database ID."""
    test = db.get(models.Test, test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found.")
    if test.student_id != current.id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return test


@app.get(
    "/tests/{student_id}/results",
    response_model=list[schemas.TestResultResponse],
    tags=["Tests"],
    summary="List all graded test results for a student",
)
def list_results(
    student_id: int,
    skip: int = 0,
    limit: int = 50,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[models.TestResult]:
    """Return all graded results for the student, most recent first."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    return (
        db.query(models.TestResult)
        .filter(models.TestResult.student_id == student_id)
        .order_by(models.TestResult.taken_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS  /analytics
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/analytics/{student_id}/summary",
    response_model=schemas.AnalyticsSummary,
    tags=["Analytics"],
    summary="Aggregated performance metrics",
)
def analytics_summary(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return tests_taken, avg_score, streak, best/worst subject, and daily scores."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return get_summary(student_id, db)


@app.get(
    "/analytics/{student_id}/weak-topics",
    response_model=list[schemas.WeakTopic],
    tags=["Analytics"],
    summary="Subjects scoring below 65%",
)
def analytics_weak_topics(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return weak subjects sorted by severity: critical (<40%), moderate, mild."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return get_weak_topics(student_id, db)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT  /chat
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/chat/{student_id}",
    response_model=schemas.ChatResponse,
    tags=["Chat"],
    summary="Send a message to the AI tutor",
)
def send_chat_message(
    student_id: int,
    payload: schemas.ChatMessageCreate,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run one conversational turn via ConversationChain; return reply + suggested topics."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    try:
        return chatbot_chat(student_id=student_id, message=payload.message, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/chat/{student_id}/history",
    response_model=list[schemas.ChatMessageResponse],
    tags=["Chat"],
    summary="Paginated chat history",
)
def chat_history(
    student_id: int,
    skip: int = 0,
    limit: int = 50,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> list[models.ChatMessage]:
    """Return stored chat messages for the student, oldest first."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.student_id == student_id)
        .order_by(models.ChatMessage.timestamp.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.delete(
    "/chat/{student_id}/history",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Chat"],
    summary="Clear chat history",
)
def clear_chat_history(
    student_id: int,
    current: models.Student = Depends(get_current_student),
    db: Session = Depends(get_db),
) -> None:
    """Permanently delete all chat messages for the student."""
    if current.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    db.query(models.ChatMessage).filter(
        models.ChatMessage.student_id == student_id
    ).delete(synchronize_session=False)
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"], summary="Liveness check")
def health() -> dict[str, str]:
    """Return 200 OK — used by load balancers / health probes."""
    return {"status": "ok", "service": "EduAssist API", "version": "2.0.0"}


# ─── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
