"""bot/chain.py — ConversationChain builder and chat runner."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

import models
from bot.context import StudentContext, build_context


SYSTEM_PROMPT_TEMPLATE = """You are EduAssist, a patient academic tutor
for undergraduate engineering students.
Student: {name}
Weak topics: {weak_topics}
Today's schedule: {todays_topics}
{struggle_note}
Rules:
- Explain with concrete examples and analogies
- For maths/engineering: always show step-by-step working
- End every reply with one comprehension-check question
- Keep responses under 200 words unless the student asks for more
- Never say you cannot help - always attempt an explanation
{struggle_instruction}"""

STRUGGLE_NOTE = "Note: This student scored below 50% on their last 3 tests."
STRUGGLE_INSTRUCTION = """Prioritise encouragement over content density.
Suggest breaking study into 25-minute Pomodoro sessions.
Celebrate small wins explicitly."""


def _build_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        temperature=temperature,
    )


def build_chain(ctx: StudentContext) -> ConversationChain:
    """
    - Format SYSTEM_PROMPT_TEMPLATE with ctx fields
    - struggle_note / struggle_instruction: include only if
      ctx.is_struggling, else empty string
    - ConversationBufferWindowMemory(k=10, return_messages=True)
    - ChatPromptTemplate: [SystemMessage, MessagesPlaceholder,
      HumanMessagePromptTemplate]
    - Return ConversationChain(llm, prompt, memory, verbose=False)
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        name=ctx.name,
        weak_topics=", ".join(ctx.weak_topics) if ctx.weak_topics else "None",
        todays_topics=", ".join(ctx.todays_topics) if ctx.todays_topics else "None",
        struggle_note=STRUGGLE_NOTE if ctx.is_struggling else "",
        struggle_instruction=STRUGGLE_INSTRUCTION if ctx.is_struggling else "",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    memory = ConversationBufferWindowMemory(k=10, return_messages=True)
    llm = _build_llm(temperature=0.7)

    return ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=False)


def extract_topic_suggestions(reply: str, llm) -> list[dict]:
    """
    Second LLM call with this exact prompt:
    'From this tutor reply: "{reply}"
     Extract 1-2 related academic topics the student should also review.
     Return ONLY valid JSON, nothing else:
     [{{"topic": "...", "subject": "..."}}]
     If none are relevant, return [].'
    Parse with json.loads(). Return [] on any exception.
    """
    prompt = (
        f'From this tutor reply: "{reply}"\n'
        "Extract 1-2 related academic topics the student should also review.\n"
        "Return ONLY valid JSON, nothing else:\n"
        '[{"topic": "...", "subject": "..."}]\n'
        "If none are relevant, return []."
    )

    try:
        raw = llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        return []


async def run_chat(student_id: int, message: str, db: Session) -> dict[str, Any]:
    """
    1. ctx = build_context(student_id, db)
    2. chain = build_chain(ctx)
    3. Load last 10 ChatMessage rows from DB (oldest first)
       Inject into memory:
         for msg in history:
           if msg.role == "user":
             memory.chat_memory.add_user_message(msg.content)
           else:
             memory.chat_memory.add_ai_message(msg.content)
    4. reply = chain.predict(input=message)
    5. suggestions = extract_topic_suggestions(reply, chain.llm)
    6. Save ChatMessage(role="user", content=message) to DB
       Save ChatMessage(role="assistant", content=reply) to DB
    7. Return {
         "reply": reply,
         "suggested_topics": suggestions,
         "message_id": saved_assistant_msg.id
       }
    """
    ctx = build_context(student_id, db)
    chain = build_chain(ctx)

    history = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.student_id == student_id)
        .order_by(models.ChatMessage.timestamp.asc())
        .limit(10)
        .all()
    )
    for msg in history:
        if msg.role == "user":
            chain.memory.chat_memory.add_user_message(msg.content)
        else:
            chain.memory.chat_memory.add_ai_message(msg.content)

    reply = chain.predict(input=message)
    suggestions = extract_topic_suggestions(reply, chain.llm)

    user_msg = models.ChatMessage(student_id=student_id, role="user", content=message)
    ai_msg = models.ChatMessage(student_id=student_id, role="assistant", content=reply)
    db.add_all([user_msg, ai_msg])
    db.commit()
    db.refresh(ai_msg)

    return {
        "reply": reply,
        "suggested_topics": suggestions,
        "message_id": ai_msg.id,
    }
