"""
rag/ingest.py — Document ingestion pipeline for EduAssist.

Converts PDFs or raw text into FAISS vector stores so they can be used
for retrieval-augmented question generation.

Public API
----------
ingest_pdf(subject, file_path)           Load PDF → chunk → embed → save/merge
ingest_text(subject, text, source_label) Wrap raw text → same pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from rag.vectorstore_manager import VectorStoreManager, _get_embeddings

logger = logging.getLogger(__name__)

# ─── Splitter (shared, stateless) ─────────────────────────────────────────────

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len,
    add_start_index=True,
)


# ─── Internal: build or merge a FAISS store ───────────────────────────────────

def _build_or_merge(
    subject: str,
    chunks: list[Document],
    manager: VectorStoreManager,
) -> tuple[FAISS, int, int]:
    """
    Add *chunks* to the existing FAISS index for *subject*, or create a new
    one if no index exists yet.

    Args:
        subject:  Subject name.
        chunks:   List of split Document objects to embed and store.
        manager:  VectorStoreManager instance.

    Returns:
        Tuple of (updated_store, chunks_added, total_chunks_after).

    Raises:
        ValueError: If *chunks* is empty.
    """
    if not chunks:
        raise ValueError("No chunks to ingest — document may be empty or unreadable.")

    embeddings = _get_embeddings()
    existing: FAISS | None = manager.get_store(subject)
    chunks_added = len(chunks)

    if existing is not None:
        logger.info(
            "Merging %d new chunks into existing '%s' index (%d chunks).",
            chunks_added, subject, existing.index.ntotal,
        )
        existing.add_documents(chunks)
        store = existing
    else:
        logger.info("Creating new FAISS index for '%s' with %d chunks.", subject, chunks_added)
        store = FAISS.from_documents(chunks, embeddings)

    total = store.index.ntotal
    manager.save_store(subject, store)
    return store, chunks_added, total


# ─── Public: ingest_pdf ───────────────────────────────────────────────────────

def ingest_pdf(subject: str, file_path: str) -> dict[str, Any]:
    """
    Load a PDF, split it into chunks, embed, and save/merge a FAISS index.

    Processing steps:
    1. Load all pages from *file_path* using ``PyPDFLoader``.
    2. Split pages into ≤500-character chunks with 75-char overlap via
       ``RecursiveCharacterTextSplitter``.
    3. If an index already exists for *subject*: call ``add_documents()``
       to merge (preserving previous content).  Otherwise create a new store.
    4. Save the updated store via ``VectorStoreManager``.

    Args:
        subject:   Subject name, e.g. ``"Physics"``.
                   Used as the directory name (lowercased, spaces → underscores).
        file_path: Absolute or relative path to a PDF file.

    Returns:
        Dict with keys:
        - ``subject``      (str)
        - ``chunks_added`` (int) — number of new chunks added this call
        - ``total_chunks`` (int) — total chunks in the index after merge
        - ``status``       (str) — ``"created"`` or ``"merged"``

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the PDF yields no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Ingesting PDF '%s' for subject '%s' …", path.name, subject)

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    logger.debug("Loaded %d pages from '%s'.", len(pages), path.name)

    chunks = _SPLITTER.split_documents(pages)
    logger.debug("Split into %d chunks.", len(chunks))

    manager = VectorStoreManager.instance()
    existed_before = manager.get_store(subject) is not None
    _store, chunks_added, total = _build_or_merge(subject, chunks, manager)

    status = "merged" if existed_before else "created"
    result: dict[str, Any] = {
        "subject": subject,
        "chunks_added": chunks_added,
        "total_chunks": total,
        "status": status,
    }
    logger.info("ingest_pdf complete: %s", result)
    return result


# ─── Public: ingest_text ──────────────────────────────────────────────────────

def ingest_text(subject: str, text: str, source_label: str = "manual") -> dict[str, Any]:
    """
    Ingest raw text content into the FAISS index for *subject*.

    Equivalent to ``ingest_pdf()`` but accepts a plain string instead of a
    file path.  Useful for ingesting lecture notes, web-scraped content, or
    structured syllabus text supplied by an admin.

    Args:
        subject:      Subject name.
        text:         The raw text content to embed.
        source_label: A human-readable label stored in the document metadata
                      (e.g. ``"Chapter 3 notes"``).  Defaults to ``"manual"``.

    Returns:
        Dict with keys:
        - ``subject``
        - ``chunks_added``
        - ``total_chunks``
        - ``status`` — ``"created"`` or ``"merged"``

    Raises:
        ValueError: If *text* is empty or whitespace-only.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot ingest empty text.")

    logger.info(
        "Ingesting text (%d chars) for subject '%s' (source: '%s') …",
        len(text), subject, source_label,
    )

    doc = Document(
        page_content=text,
        metadata={"source": source_label, "subject": subject},
    )
    chunks = _SPLITTER.split_documents([doc])
    logger.debug("Split into %d chunks.", len(chunks))

    manager = VectorStoreManager.instance()
    existed_before = manager.get_store(subject) is not None
    _store, chunks_added, total = _build_or_merge(subject, chunks, manager)

    status = "merged" if existed_before else "created"
    result: dict[str, Any] = {
        "subject": subject,
        "chunks_added": chunks_added,
        "total_chunks": total,
        "status": status,
    }
    logger.info("ingest_text complete: %s", result)
    return result
