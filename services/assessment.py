"""
services/assessment.py — AI-powered MCQ test generator.

Pipeline:
  1. Attempt to load a FAISS vector store for the requested subject.
  2. If the index exists, retrieve the top-5 most relevant chunks (RAG).
  3. Build a prompt combining context + task requirements.
  4. Call Gemini 1.5 Flash → receive exactly n_questions MCQs as JSON.
  5. Validate output; retry once with a stricter prompt on failure.
  6. Persist the Test record and return the ORM object.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from models import Test

logger = logging.getLogger(__name__)

# Root path for per-subject FAISS indexes
VECTORSTORE_ROOT = Path(os.getenv("VECTORSTORE_ROOT", "data/vectorstore"))

# ─── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini 1.5 Flash instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        temperature=0.3,
    )


# ─── Embeddings (shared, lazy-loaded) ─────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFace sentence-transformers embedder."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# ─── FAISS helper ─────────────────────────────────────────────────────────────

def _load_vectorstore(subject: str) -> FAISS | None:
    """
    Try to load a FAISS index from disk for the given subject.

    Returns None if the index directory does not exist.
    """
    index_path = VECTORSTORE_ROOT / subject.lower().replace(" ", "_")
    if not index_path.exists():
        logger.info("No FAISS index found for subject '%s' at %s", subject, index_path)
        return None

    try:
        store = FAISS.load_local(
            str(index_path),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index for subject '%s'", subject)
        return store
    except Exception as exc:
        logger.warning("Failed to load FAISS index for '%s': %s", subject, exc)
        return None


def _retrieve_context(store: FAISS, query: str, k: int = 5) -> str:
    """
    Retrieve the top-k most relevant document chunks for the given query.

    Returns a single newline-delimited string of chunk contents.
    """
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


# ─── Prompt templates ─────────────────────────────────────────────────────────

_PROMPT_WITH_CONTEXT = PromptTemplate(
    input_variables=["subject", "topic", "difficulty", "n_questions", "context"],
    template="""You are an expert {subject} teacher creating a multiple-choice quiz.

Reference material:
\"\"\"
{context}
\"\"\"

Task:
Generate exactly {n_questions} {difficulty}-level MCQ questions about "{topic}" in {subject}.

Each question must:
- Be clearly worded and unambiguous.
- Have exactly 4 options labelled A, B, C, D (strings, not letters in the object).
- Have one correct answer indicated by its letter (A, B, C, or D).
- Include a concise explanation (1-2 sentences) of why the correct answer is right.

Respond with ONLY a valid JSON array. No markdown fences, no extra text.
Format:
[
  {{
    "id": 1,
    "question": "...",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_answer": "A",
    "explanation": "...",
    "topic": "{topic}",
    "difficulty": "{difficulty}"
  }},
  ...
]
""",
)

_PROMPT_WITHOUT_CONTEXT = PromptTemplate(
    input_variables=["subject", "topic", "difficulty", "n_questions"],
    template="""You are an expert {subject} teacher creating a multiple-choice quiz.

Generate exactly {n_questions} {difficulty}-level MCQ questions about "{topic}" in {subject}.

Each question must:
- Be clearly worded and unambiguous.
- Have exactly 4 options labelled A, B, C, D (strings, not letters in the object).
- Have one correct answer indicated by its letter (A, B, C, or D).
- Include a concise explanation (1-2 sentences) of why the correct answer is right.

Respond with ONLY a valid JSON array. No markdown fences, no extra text.
Format:
[
  {{
    "id": 1,
    "question": "...",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_answer": "A",
    "explanation": "...",
    "topic": "{topic}",
    "difficulty": "{difficulty}"
  }},
  ...
]
""",
)

_REPAIR_PROMPT = PromptTemplate(
    input_variables=["raw", "n_questions", "subject", "topic", "difficulty"],
    template="""The following text should be a JSON array of {n_questions} MCQ questions
for {subject} / {topic} ({difficulty}) but it is malformed or incomplete.

Fix it and return ONLY a valid JSON array matching this schema exactly:
[{{"id": int, "question": str, "options": [str,str,str,str], "correct_answer": str (A/B/C/D), "explanation": str, "topic": str, "difficulty": str}}]

Malformed text:
{raw}
""",
)


# ─── JSON validation ──────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"id", "question", "options", "correct_answer", "explanation", "topic", "difficulty"}


def _parse_questions(raw: str, n_questions: int) -> list[dict[str, Any]]:
    """
    Parse and validate the LLM's MCQ JSON output.

    Raises:
        ValueError: on JSON parse failure or schema mismatch.
    """
    text = raw.strip()
    # Strip optional markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    questions: Any = json.loads(text)

    if not isinstance(questions, list):
        raise ValueError(f"Expected JSON array, got {type(questions).__name__}")

    if len(questions) != n_questions:
        raise ValueError(
            f"Expected {n_questions} questions, got {len(questions)}"
        )

    for i, q in enumerate(questions):
        missing = _REQUIRED_KEYS - q.keys()
        if missing:
            raise ValueError(f"Question {i + 1} missing keys: {missing}")
        if not isinstance(q.get("options"), list) or len(q["options"]) != 4:
            raise ValueError(f"Question {i + 1} must have exactly 4 options")
        if q.get("correct_answer") not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Question {i + 1} correct_answer must be A/B/C/D, got: {q.get('correct_answer')}"
            )

    return questions


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_test(
    subject: str,
    topic: str,
    difficulty: str,
    n_questions: int,
    student_id: int,
    db: Session,
) -> Test:
    """
    Generate an MCQ test and persist it to the database.

    Args:
        subject:     Subject name, e.g. "Physics".
        topic:       Specific topic, e.g. "Newton's Laws".
        difficulty:  "easy" | "medium" | "hard".
        n_questions: Number of MCQs to generate (1-20).
        student_id:  The requesting student's ID.
        db:          Active SQLAlchemy session.

    Returns:
        The newly created Test ORM object (committed).

    Raises:
        ValueError: If JSON parsing fails after two attempts.
    """
    llm = _build_llm()

    # ── Step 1: RAG retrieval (optional) ──────────────────────────────────────
    vectorstore = _load_vectorstore(subject)
    if vectorstore:
        context = _retrieve_context(vectorstore, f"{topic} {difficulty}")
        chain = LLMChain(llm=llm, prompt=_PROMPT_WITH_CONTEXT)
        raw_output: str = chain.run(
            subject=subject,
            topic=topic,
            difficulty=difficulty,
            n_questions=n_questions,
            context=context,
        )
    else:
        chain = LLMChain(llm=llm, prompt=_PROMPT_WITHOUT_CONTEXT)
        raw_output = chain.run(
            subject=subject,
            topic=topic,
            difficulty=difficulty,
            n_questions=n_questions,
        )

    # ── Step 2: Parse with one retry ─────────────────────────────────────────
    questions: list[dict[str, Any]] | None = None

    for attempt in range(1, 3):
        try:
            questions = _parse_questions(raw_output, n_questions)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("MCQ parse attempt %d failed: %s", attempt, exc)
            if attempt == 2:
                raise ValueError(
                    f"LLM returned invalid MCQ JSON after 2 attempts. Last error: {exc}"
                ) from exc
            # Retry: ask LLM to repair the output
            repair_chain = LLMChain(llm=llm, prompt=_REPAIR_PROMPT)
            raw_output = repair_chain.run(
                raw=raw_output,
                n_questions=n_questions,
                subject=subject,
                topic=topic,
                difficulty=difficulty,
            )

    # ── Step 3: Persist ───────────────────────────────────────────────────────
    test = Test(
        student_id=student_id,
        topic=topic,
        subject=subject,
        difficulty=difficulty,
        questions_json=questions,
    )
    db.add(test)
    db.commit()
    db.refresh(test)

    logger.info(
        "Generated test %d for student %d: %s / %s (%s) — %d questions",
        test.id,
        student_id,
        subject,
        topic,
        difficulty,
        n_questions,
    )
    return test
