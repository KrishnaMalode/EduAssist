"""
services/chatbot.py — AI tutoring chatbot with memory and topic extraction.

Pipeline per chat() call:
  1. Load the student's last 10 messages from DB.
  2. Inject them into a ConversationBufferWindowMemory.
  3. Build a system prompt with the student's name, weak topics, and today's schedule.
  4. Run a LangChain ConversationChain → receive the AI reply.
  5. Run a second LLM call to extract 1-2 suggested review topics from the reply.
  6. Persist both the user message and AI reply to the ChatMessage table.
  7. Return {reply, suggested_topics}.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any

from langchain_classic.chains import ConversationChain, LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from models import ChatMessage, Student, StudyPlan
from services.analytics import get_weak_topics

logger = logging.getLogger(__name__)

# ─── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        temperature=temperature,
    )


# ─── Context helpers ──────────────────────────────────────────────────────────

def _load_chat_history(student_id: int, db: Session) -> list[ChatMessage]:
    """Return the most recent 10 messages for the student, oldest first."""
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.student_id == student_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(10)
        .all()[::-1]   # reverse to chronological order
    )


def _get_todays_schedule(student_id: int, db: Session) -> str:
    """
    Return today's study sessions from the most recent study plan, as a
    human-readable string. Returns "No sessions scheduled for today." if none.
    """
    today_name = date.today().strftime("%A")   # e.g. "Monday"
    plan: StudyPlan | None = (
        db.query(StudyPlan)
        .filter(StudyPlan.student_id == student_id)
        .order_by(StudyPlan.created_at.desc())
        .first()
    )
    if not plan or not plan.schedule_json:
        return "No sessions scheduled for today."

    todays = [
        s for s in plan.schedule_json
        if s.get("day", "").lower() == today_name.lower()
    ]
    if not todays:
        return "No sessions scheduled for today."

    parts = [
        f"{s['subject']} — {s['topic']} ({s['duration_mins']} min, {s['priority']} priority)"
        for s in todays
    ]
    return "; ".join(parts)


def _build_system_prompt(student: Student, db: Session) -> str:
    """Construct the system prompt injected before the conversation."""
    weak_topics = get_weak_topics(student.id, db)
    weak_str = (
        ", ".join(
            f"{wt['subject']} ({wt['avg_score']:.0f}%, {wt['severity']})"
            for wt in weak_topics
        )
        if weak_topics
        else "None identified yet"
    )
    todays_schedule = _get_todays_schedule(student.id, db)

    return (
        f"You are EduAssist, a friendly and knowledgeable AI tutor.\n"
        f"You are currently helping {student.name}.\n\n"
        f"Student context:\n"
        f"- Subjects: {', '.join(student.subjects or ['General'])}\n"
        f"- Weak topics: {weak_str}\n"
        f"- Today's schedule: {todays_schedule}\n\n"
        f"Guidelines:\n"
        f"- Give clear, step-by-step explanations.\n"
        f"- Use examples and analogies appropriate for a student.\n"
        f"- If the student struggles, gently redirect to their weak topics.\n"
        f"- Keep responses concise (under 300 words) unless detail is explicitly needed.\n"
        f"- Be encouraging and supportive."
    )


# ─── Topic extraction ─────────────────────────────────────────────────────────

_TOPIC_EXTRACT_PROMPT = PromptTemplate(
    input_variables=["reply"],
    template="""You are a curriculum analyst. Read the AI tutor's reply below and extract
1 to 2 academic topics that the student should review or practice next.

Tutor reply:
\"\"\"
{reply}
\"\"\"

Respond with ONLY a valid JSON array (no markdown fences, no extra text).
Each element: {{"topic": "<specific topic>", "subject": "<subject area>"}}

If no specific topic is clearly mentioned, infer from context.
Example: [{{"topic": "Newton's Second Law", "subject": "Physics"}}]
""",
)


def _extract_suggested_topics(reply: str, llm: ChatGoogleGenerativeAI) -> list[dict[str, str]]:
    """
    Use a second LLM call to parse 1-2 suggested topics from the AI reply.

    Falls back to an empty list if JSON parsing fails.
    """
    chain = LLMChain(llm=llm, prompt=_TOPIC_EXTRACT_PROMPT)
    raw: str = chain.run(reply=reply)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    try:
        topics = json.loads(text)
        if isinstance(topics, list):
            # Validate and cap at 2
            valid = [
                {"topic": t["topic"], "subject": t["subject"]}
                for t in topics
                if isinstance(t, dict) and "topic" in t and "subject" in t
            ]
            return valid[:2]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to extract suggested topics: %s | raw: %s", exc, raw)

    return []


# ─── Public API ───────────────────────────────────────────────────────────────

def chat(student_id: int, message: str, db: Session) -> dict[str, Any]:
    """
    Process one student chat turn and return the AI reply.

    Steps:
      1. Look up the student; raise ValueError if not found.
      2. Load the last 10 chat messages into ConversationBufferWindowMemory.
      3. Inject student context into a system prefix.
      4. Run ConversationChain with the new message.
      5. Extract suggested topics from the reply via a second LLM call.
      6. Persist both messages to the database.
      7. Return {reply, suggested_topics}.

    Args:
        student_id: The student's ID.
        message:    The student's new message text.
        db:         Active SQLAlchemy session.

    Returns:
        dict with keys "reply" (str) and "suggested_topics" (list[dict]).

    Raises:
        ValueError: If the student is not found.
    """
    student: Student | None = db.get(Student, student_id)
    if not student:
        raise ValueError(f"Student {student_id} not found")

    llm = _build_llm(temperature=0.7)
    llm_low_temp = _build_llm(temperature=0.1)   # deterministic for topic extraction

    # ── Step 1: Build memory from DB history ─────────────────────────────────
    history = _load_chat_history(student_id, db)
    memory = ConversationBufferWindowMemory(k=10, return_messages=True)

    for msg in history:
        if msg.role == "user":
            memory.chat_memory.add_user_message(msg.content)
        else:
            memory.chat_memory.add_ai_message(msg.content)

    # ── Step 2: Build conversation with system prefix ─────────────────────────
    system_prompt = _build_system_prompt(student, db)

    # ConversationChain uses a fixed prompt template; we prepend the system
    # context as the first human turn if history is empty, otherwise as a
    # prefix. Using a custom PromptTemplate is the cleanest approach.
    conversation_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=(
            f"{system_prompt}\n\n"
            "Current conversation:\n"
            "{history}\n"
            "Student: {input}\n"
            "EduAssist:"
        ),
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=conversation_prompt,
        verbose=False,
    )

    # ── Step 3: Get reply ─────────────────────────────────────────────────────
    reply: str = conversation.predict(input=message)
    reply = reply.strip()

    # ── Step 4: Extract suggested topics ─────────────────────────────────────
    suggested_topics = _extract_suggested_topics(reply, llm_low_temp)

    # ── Step 5: Persist messages ──────────────────────────────────────────────
    user_msg = ChatMessage(student_id=student_id, role="user", content=message)
    ai_msg = ChatMessage(student_id=student_id, role="assistant", content=reply)
    db.add_all([user_msg, ai_msg])
    db.commit()

    logger.info(
        "Chat: student=%d | topics=%s | reply_len=%d",
        student_id,
        suggested_topics,
        len(reply),
    )

    return {"reply": reply, "suggested_topics": suggested_topics}
