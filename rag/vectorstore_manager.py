"""
rag/vectorstore_manager.py — Singleton FAISS index manager.

Responsible for all disk I/O relating to per-subject vector stores.
Each subject's index lives at ``{base_path}/{subject_slug}/``.

Singleton behaviour: each index is loaded from disk at most once per
process lifetime; subsequent calls to ``get_store()`` return the cached
in-memory FAISS object.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ─── Embeddings (module-level singleton) ──────────────────────────────────────

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings: HuggingFaceEmbeddings | None = None
_embed_lock = Lock()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the shared HuggingFace sentence-transformers embedder.

    Lazy-loaded and thread-safe.  Re-used across all manager instances so
    the model is only loaded into memory once per process.
    """
    global _embeddings
    if _embeddings is None:
        with _embed_lock:
            if _embeddings is None:   # double-checked locking
                logger.info("Loading embedding model '%s' …", _EMBED_MODEL)
                _embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
                logger.info("Embedding model ready.")
    return _embeddings


# ─── VectorStoreManager ───────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Per-process singleton that manages FAISS vector stores for each subject.

    Usage::

        manager = VectorStoreManager.instance()
        store   = manager.get_store("Physics")
        manager.save_store("Physics", updated_store)

    Thread safety: a per-instance lock guards both the in-memory cache and
    disk writes so concurrent FastAPI requests do not corrupt an index.
    """

    _singleton: VectorStoreManager | None = None
    _singleton_lock: Lock = Lock()

    def __init__(self, base_path: str = "data/vectorstore") -> None:
        """
        Initialise the manager with a root path for all FAISS indexes.

        Args:
            base_path: Directory that will contain one sub-folder per subject.
                       Created automatically if it does not exist.
        """
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, FAISS] = {}
        self._lock = Lock()
        logger.info("VectorStoreManager initialised at '%s'", self._base)

    # ── Singleton factory ──────────────────────────────────────────────────────

    @classmethod
    def instance(cls, base_path: str = "data/vectorstore") -> "VectorStoreManager":
        """
        Return (and create on first call) the process-wide singleton.

        Args:
            base_path: Passed to ``__init__`` on first construction only.

        Returns:
            The shared VectorStoreManager instance.
        """
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls(base_path=base_path)
        return cls._singleton

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _slug(self, subject: str) -> str:
        """
        Convert a subject name to a safe directory name.

        Examples: "Physics" → "physics", "Computer Science" → "computer_science"
        """
        return subject.strip().lower().replace(" ", "_")

    def _index_path(self, subject: str) -> Path:
        """Return the full path to a subject's FAISS index directory."""
        return self._base / self._slug(subject)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_store(self, subject: str) -> FAISS | None:
        """
        Load and return the FAISS index for *subject*.

        On the first call the index is loaded from disk and cached in memory.
        Subsequent calls for the same subject are served from the cache
        (O(1), no disk I/O).

        Args:
            subject: Human-readable subject name, e.g. "Physics".

        Returns:
            The FAISS store, or ``None`` if no index exists for this subject.
        """
        slug = self._slug(subject)

        # Fast path: already cached
        if slug in self._cache:
            return self._cache[slug]

        index_dir = self._index_path(subject)
        if not index_dir.exists():
            logger.debug("No FAISS index found for subject '%s' at %s", subject, index_dir)
            return None

        with self._lock:
            # Re-check inside lock (another thread may have loaded it)
            if slug in self._cache:
                return self._cache[slug]
            try:
                store = FAISS.load_local(
                    str(index_dir),
                    _get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                self._cache[slug] = store
                logger.info(
                    "Loaded FAISS index for '%s' (%d chunks).",
                    subject, store.index.ntotal,
                )
                return store
            except FileNotFoundError:
                logger.warning("Index directory for '%s' exists but is incomplete.", subject)
                return None
            except Exception as exc:   # noqa: BLE001
                logger.error("Failed to load FAISS index for '%s': %s", subject, exc)
                return None

    def save_store(self, subject: str, store: FAISS) -> None:
        """
        Persist *store* to disk and update the in-memory cache.

        Args:
            subject: Subject name (used to derive the directory name).
            store:   FAISS instance to persist.

        Raises:
            OSError: If the index directory cannot be created or written to.
        """
        index_dir = self._index_path(subject)
        index_dir.mkdir(parents=True, exist_ok=True)
        slug = self._slug(subject)

        with self._lock:
            store.save_local(str(index_dir))
            self._cache[slug] = store
            logger.info(
                "Saved FAISS index for '%s' to %s (%d chunks).",
                subject, index_dir, store.index.ntotal,
            )

    def list_subjects(self) -> list[str]:
        """
        Scan *base_path* for existing subject indexes.

        Returns:
            A list of subject slugs (directory names) that contain a valid
            ``index.faiss`` file.  Returns an empty list if *base_path*
            does not exist or is empty.
        """
        if not self._base.exists():
            return []

        subjects: list[str] = []
        try:
            for entry in sorted(self._base.iterdir()):
                if entry.is_dir() and (entry / "index.faiss").exists():
                    subjects.append(entry.name)
        except FileNotFoundError:
            logger.warning("base_path '%s' disappeared during listing.", self._base)

        return subjects

    def chunk_count(self, subject: str) -> int:
        """
        Return the number of embedded chunks for *subject*.

        Loads the index if not already cached.  Returns 0 if no index exists.
        """
        store = self.get_store(subject)
        return store.index.ntotal if store else 0

    def evict(self, subject: str) -> None:
        """
        Remove *subject*'s FAISS store from the in-memory cache.

        The index remains on disk.  Useful after a large ingest to free RAM.
        """
        slug = self._slug(subject)
        with self._lock:
            dropped = self._cache.pop(slug, None)
        if dropped:
            logger.info("Evicted '%s' from VectorStoreManager cache.", subject)
