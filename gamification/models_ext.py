from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from database import Base


class TopicMastery(Base):
    __tablename__ = "topic_mastery"

    id: int = Column(Integer, primary_key=True, index=True)
    student_id: int = Column(Integer, ForeignKey("students.id"), index=True, nullable=False)
    subject: str = Column(String(120), nullable=False)
    topic: str = Column(String(255), nullable=False)
    xp: int = Column(Integer, default=0)
    level: int = Column(Integer, default=1)
    mastery_pct: float = Column(Float, default=0.0)
    last_practiced: datetime | None = Column(DateTime, nullable=True, default=None)
    decay_rate: float = Column(Float, default=5.0)
    streak_days: int = Column(Integer, default=0)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow)


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id: int = Column(Integer, primary_key=True, index=True)
    student_id: int = Column(Integer, ForeignKey("students.id"), unique=True, index=True, nullable=False)
    total_xp: int = Column(Integer, default=0)
    global_level: int = Column(Integer, default=1)
    title: str = Column(String(100), default="Novice")
    badges: str = Column(String, default="[]")
    longest_streak: int = Column(Integer, default=0)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow)


class Badge(Base):
    __tablename__ = "badges"

    id: str = Column(String(120), primary_key=True, index=True)
    name: str = Column(String(255), nullable=False)
    description: str = Column(String(255), nullable=False)
    icon_key: str = Column(String(50), nullable=False)
    condition_json: str = Column(String, default="{}")


try:
    from models import DailyRevision as DailyRevision  # type: ignore
except Exception:  # pragma: no cover
    class DailyRevision(Base):
        __tablename__ = "daily_revisions"

        id: int = Column(Integer, primary_key=True, index=True)
        student_id: int = Column(Integer, ForeignKey("students.id"), index=True, nullable=False)
        date: str = Column(String(20), index=True, nullable=False)
        topics_json: str = Column(String, nullable=False)
        delivered_at: datetime = Column(DateTime, default=datetime.utcnow)
