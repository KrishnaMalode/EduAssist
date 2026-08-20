"""
rag/generate.py — RAG-backed MCQ question generator and answer grader.

Public API
----------
generate_questions(subject, topic, difficulty, n, db) -> list[dict]
    Retrieve context from FAISS (if available) and generate n MCQs via Gemini.

grade_submission(questions, answers) -> dict
    Auto-grade a submitted answer dict against the stored correct answers.

The options format used here (dict with keys A/B/C/D) matches the prompt
template in this module.  Use grade_submission() alongside these questions.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from rag.vectorstore_manager import VectorStoreManager

logger = logging.getLogger(__name__)

# ─── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Return a Gemini 1.5 Flash LLM configured for MCQ generation."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        temperature=temperature,
    )


# ─── Prompt template ──────────────────────────────────────────────────────────

_MCQ_PROMPT = PromptTemplate(
    input_variables=["context", "n", "difficulty", "topic", "subject"],
    template="""You are an exam question generator for undergraduate engineering.
Context from syllabus:
{context}

Generate exactly {n} {difficulty} multiple choice questions on the topic: {topic} (subject: {subject}).

Rules:
- Every question must be factually correct and unambiguous.
- Options A, B, C, D must all be plausible; only one is correct.
- The explanation must clearly state WHY the correct answer is right.
- difficulty must be one of: easy, medium, hard.
- id must be sequential starting at 1.

Return ONLY a JSON array, no other text:
[{{
  "id": 1,
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "A",
  "explanation": "...",
  "topic": "{topic}",
  "difficulty": "{difficulty}"
}}]""",
)

_REPAIR_PROMPT = PromptTemplate(
    input_variables=["error", "raw", "n", "topic", "subject", "difficulty"],
    template="""The following JSON is malformed. Fix it so it is a valid JSON array of exactly {n}
MCQ objects for {subject}/{topic} ({difficulty}).

Parse error: {error}

Malformed output:
{raw}

Return ONLY the corrected JSON array, no markdown fences, no extra text.""",
)


# ─── Retrieval helper ─────────────────────────────────────────────────────────

def _retrieve_context(subject: str, topic: str, k: int = 5) -> str:
    """
    Query the FAISS index for *subject* and return the top-*k* chunks joined
    as a single string.

    Falls back to a generic instruction when no index is available.

    Args:
        subject: Subject name matching a VectorStoreManager entry.
        topic:   Query string for similarity search.
        k:       Number of chunks to retrieve.

    Returns:
        Context string to inject into the MCQ prompt.
    """
    manager = VectorStoreManager.instance()
    store = manager.get_store(subject)

    if store is None:
        logger.info(
            "No FAISS index for '%s'; falling back to LLM training knowledge.", subject
        )
        return "Generate from your training knowledge."

    docs = store.similarity_search(topic, k=k)
    if not docs:
        logger.warning("FAISS search returned 0 results for '%s' / '%s'.", subject, topic)
        return "Generate from your training knowledge."

    context = "\n\n".join(doc.page_content for doc in docs)
    logger.debug(
        "Retrieved %d chunks for '%s'/'%s' (total chars: %d).",
        len(docs), subject, topic, len(context),
    )
    return context


# ─── JSON parsing & validation ────────────────────────────────────────────────

_REQUIRED_MCQ_KEYS = {"id", "question", "options", "correct_answer", "explanation", "topic", "difficulty"}


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))
    return text.strip()


def _validate_questions(questions: list[dict], n: int) -> None:
    """
    Raise ValueError if the parsed question list does not meet requirements.

    Checks:
    - Exactly *n* items in the list.
    - Each item has all required keys.
    - ``options`` is a dict with exactly keys A, B, C, D.
    - ``correct_answer`` is one of A, B, C, D.
    """
    if len(questions) != n:
        raise ValueError(f"Expected {n} questions, got {len(questions)}.")

    for i, q in enumerate(questions, start=1):
        missing = _REQUIRED_MCQ_KEYS - set(q.keys())
        if missing:
            raise ValueError(f"Question {i} missing keys: {missing}.")

        opts = q.get("options", {})
        if not isinstance(opts, dict) or set(opts.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"Question {i} options must be dict with keys A, B, C, D.")

        if q.get("correct_answer") not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Question {i} correct_answer must be A/B/C/D, got: {q.get('correct_answer')!r}."
            )


def _parse_and_validate(raw: str, n: int) -> list[dict]:
    """
    Parse JSON from *raw*, strip fences, and validate the structure.

    Raises:
        json.JSONDecodeError: On invalid JSON.
        ValueError: On schema validation failure.
    """
    clean = _strip_fences(raw)
    questions: Any = json.loads(clean)
    if not isinstance(questions, list):
        raise ValueError(f"Expected JSON array at top level, got {type(questions).__name__}.")
    _validate_questions(questions, n)
    return questions


# ─── Public: generate_questions ───────────────────────────────────────────────

def generate_questions(
    subject: str,
    topic: str,
    difficulty: str,
    n: int,
    db: Session,
) -> list[dict[str, Any]]:
    """
    Generate *n* MCQ questions for *topic* in *subject* at *difficulty* level.

    Retrieval-augmented generation pipeline:
    1. Query the FAISS index (if one exists for *subject*) for the top-5
       most relevant syllabus chunks.
    2. Inject the retrieved context into the MCQ prompt.
    3. Call Gemini 1.5 Flash to generate exactly *n* questions.
    4. Parse and validate the JSON output.
    5. On JSONDecodeError or schema error, pass the failure back to the LLM
       for self-repair (one retry).

    Args:
        subject:    Subject name, e.g. ``"Physics"``.
        topic:      Specific topic, e.g. ``"Newton's Laws"``.
        difficulty: ``"easy"`` | ``"medium"`` | ``"hard"``.
        n:          Number of questions to generate (1–20).
        db:         Active SQLAlchemy session (reserved for future DB lookups).

    Returns:
        A list of *n* MCQ dicts with keys:
        ``id``, ``question``, ``options`` (A/B/C/D dict), ``correct_answer``,
        ``explanation``, ``topic``, ``difficulty``.

    Raises:
        ValueError: If the LLM returns unparseable/invalid output after two
                    attempts.
    """
    context = _retrieve_context(subject, topic)
    llm = _build_llm()
    chain = LLMChain(llm=llm, prompt=_MCQ_PROMPT)

    raw: str = chain.run(
        context=context,
        n=n,
        difficulty=difficulty,
        topic=topic,
        subject=subject,
    )
    logger.debug("LLM raw output (attempt 1) length=%d", len(raw))

    # ── Parse attempt 1 ───────────────────────────────────────────────────────
    try:
        return _parse_and_validate(raw, n)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("MCQ parse attempt 1 failed: %s", exc)

    # ── Retry: ask the LLM to fix its own output ─────────────────────────────
    repair_chain = LLMChain(llm=_build_llm(temperature=0.1), prompt=_REPAIR_PROMPT)
    repaired: str = repair_chain.run(
        error=str(exc),
        raw=raw,
        n=n,
        topic=topic,
        subject=subject,
        difficulty=difficulty,
    )
    logger.debug("LLM repair output length=%d", len(repaired))

    try:
        return _parse_and_validate(repaired, n)
    except (json.JSONDecodeError, ValueError) as exc2:
        logger.error("MCQ parse attempt 2 (repair) failed: %s", exc2)
        raise ValueError(
            f"Failed to generate valid MCQs for '{subject}/{topic}' after 2 attempts. "
            f"Last error: {exc2}"
        ) from exc2


# ─── Public: grade_submission ─────────────────────────────────────────────────

def grade_submission(
    questions: list[dict[str, Any]],
    answers: dict[str, str],
) -> dict[str, Any]:
    """
    Grade a student's submitted answers against the stored correct answers.

    Args:
        questions: The list of MCQ dicts as returned by ``generate_questions()``.
        answers:   Mapping of question id (as string) to chosen option letter,
                   e.g. ``{"1": "B", "2": "A", "3": "C"}``.

    Returns:
        A dict with keys:
        - ``score_pct``    (float, 0–100)
        - ``correct``      (int)
        - ``total``        (int)
        - ``per_question`` (list of per-question detail dicts)

        Each per-question entry contains:
        ``id``, ``correct`` (bool), ``chosen``, ``right_answer``, ``explanation``.
    """
    total = len(questions)
    if total == 0:
        logger.warning("grade_submission called with empty question list.")
        return {"score_pct": 0.0, "correct": 0, "total": 0, "per_question": []}

    correct_count = 0
    per_question: list[dict[str, Any]] = []

    for q in questions:
        qid = str(q["id"])
        right_answer: str = q["correct_answer"]
        chosen: str | None = answers.get(qid)
        is_correct = chosen == right_answer
        if is_correct:
            correct_count += 1

        per_question.append(
            {
                "id": qid,
                "correct": is_correct,
                "chosen": chosen,
                "right_answer": right_answer,
                "explanation": q.get("explanation", ""),
            }
        )

    score_pct = round((correct_count / total) * 100, 2)
    logger.info(
        "Graded submission: %d/%d correct (%.1f%%)", correct_count, total, score_pct
    )

    return {
        "score_pct": score_pct,
        "correct": correct_count,
        "total": total,
        "per_question": per_question,
    }
