"""
services/scheduler.py — AI-powered weekly study plan generator.

Uses LangChain + Gemini (gemini-1.5-flash) to produce a personalised
7-day schedule in JSON format, then persists it to the StudyPlan table.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from models import Student, StudyPlan
from services.analytics import get_weak_topics

logger = logging.getLogger(__name__)

# ─── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini 1.5 Flash LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        temperature=0.4,
    )


# ─── Prompt template ──────────────────────────────────────────────────────────

_SCHEDULE_PROMPT = PromptTemplate(
    input_variables=["subjects", "weak_topics", "daily_hours", "week_start"],
    template="""You are an expert academic planner for a student tutoring platform.

Student profile:
- Subjects: {subjects}
- Weak topics (needs extra attention): {weak_topics}
- Available study time per day: {daily_hours} hours
- Week starting: {week_start}

Generate a 7-day personalised study schedule.

Rules:
1. Cover ALL subjects across the week, allocating more time to weak topics.
2. Each day should have 1-3 study sessions fitting within the daily hours.
3. Each session: exactly one subject and one specific topic.
4. Priority must be "high", "medium", or "low".
5. duration_mins must be a multiple of 15 and between 30 and 120.

Respond with ONLY a valid JSON array — no markdown fences, no extra text.
Format:
[
  {{"day": "Monday", "subject": "Math", "topic": "Quadratic Equations", "duration_mins": 60, "priority": "high"}},
  ...
]
""",
)


# ─── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_schedule_json(raw: str) -> list[dict]:
    """
    Extract and validate the JSON array from the LLM response.

    Strips markdown code fences if the model includes them, then parses.
    Raises ValueError if the result is not a list.
    """
    # Strip ```json ... ``` or ``` ... ``` wrappers
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")

    required_keys = {"day", "subject", "topic", "duration_mins", "priority"}
    for i, entry in enumerate(parsed):
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(f"Entry {i} missing keys: {missing}")

    return parsed


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_study_plan(student: Student, db: Session) -> StudyPlan:
    """
    Generate and persist a 7-day AI study plan for the given student.

    Steps:
      1. Fetch the student's weak topics from the analytics service.
      2. Build and run a LangChain LLMChain with the schedule prompt.
      3. Parse the JSON output; retry once on parse failure.
      4. Persist to the StudyPlan table and return the ORM object.

    Args:
        student: The SQLAlchemy Student ORM instance.
        db:      An active SQLAlchemy Session.

    Returns:
        The newly created StudyPlan ORM object (already committed).

    Raises:
        ValueError: If the LLM returns unparseable JSON after two attempts.
    """
    llm = _build_llm()
    chain = LLMChain(llm=llm, prompt=_SCHEDULE_PROMPT)

    # Gather context
    weak_topics_data: list[dict] = get_weak_topics(student.id, db)
    weak_topics_str = (
        ", ".join(f"{wt['subject']} ({wt['avg_score']:.0f}%)" for wt in weak_topics_data)
        if weak_topics_data
        else "None identified yet"
    )
    subjects_str = ", ".join(student.subjects) if student.subjects else "General Studies"
    week_start_str = date.today().isoformat()

    prompt_inputs = {
        "subjects": subjects_str,
        "weak_topics": weak_topics_str,
        "daily_hours": student.daily_hours,
        "week_start": week_start_str,
    }

    # Attempt 1
    raw_output: str = chain.run(**prompt_inputs)
    schedule: list[dict] | None = None

    for attempt in range(1, 3):
        try:
            schedule = _parse_schedule_json(raw_output)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Schedule parse attempt %d failed: %s", attempt, exc)
            if attempt == 2:
                raise ValueError(
                    f"LLM returned invalid schedule JSON after 2 attempts. "
                    f"Last error: {exc}"
                ) from exc
            # Retry with a stricter prompt
            retry_prompt = PromptTemplate(
                input_variables=["raw"],
                template=(
                    "The following text is supposed to be a JSON array of study schedule entries "
                    "but it is malformed. Fix it and return ONLY valid JSON, nothing else.\n\n{raw}"
                ),
            )
            retry_chain = LLMChain(llm=llm, prompt=retry_prompt)
            raw_output = retry_chain.run(raw=raw_output)

    # Persist
    plan = StudyPlan(
        student_id=student.id,
        week_start=week_start_str,
        schedule_json=schedule,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    logger.info(
        "Generated study plan %d for student %d (%d sessions)",
        plan.id,
        student.id,
        len(schedule),
    )
    return plan
