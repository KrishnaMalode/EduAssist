from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

import models
from gamification.models_ext import Badge, StudentProfile, TopicMastery

XP_RULES = {
    "test_correct_answer": 10,
    "test_perfect_score": 50,
    "test_above_80": 25,
    "daily_revision_done": 30,
    "chat_message_sent": 2,
    "streak_bonus": 20,
    "mock_test_complete": 75,
    "ncert_topic_first": 40,
}

LEVEL_THRESHOLDS = [0, 100, 200, 350, 550, 800, 1000, 1250, 1550, 1900]

GLOBAL_LEVELS = [
    0, 100, 250, 500, 900, 1400, 2000, 2700, 3500, 4500,
    5700, 7000, 8500, 10200, 12100, 14200, 16500, 19000,
    21700, 24700, 28000, 31500, 35300, 39400, 43800,
    48500, 53500, 58800, 64400, 70300, 76500, 83000,
    89800, 96900, 104300, 112000, 120000, 128300, 136900,
    145800, 155000, 164500, 174300, 184400, 194800,
    205500, 216500, 227800, 239400, 251300,
]

TITLES = {
    range(1, 6): "Novice", range(6, 11): "Learner",
    range(11, 16): "Explorer", range(16, 21): "Scholar",
    range(21, 26): "Analyst", range(26, 31): "Specialist",
    range(31, 36): "Expert", range(36, 41): "Master",
    range(41, 46): "Grandmaster", range(46, 51): "Legendary",
}

BADGE_SEEDS = [
    {"id": "first_test", "name": "First Step", "description": "Complete your first test", "icon_key": "🎯"},
    {"id": "five_streak", "name": "On Fire", "description": "5-day study streak", "icon_key": "🔥"},
    {"id": "ten_streak", "name": "Unstoppable", "description": "10-day study streak", "icon_key": "⚡"},
    {"id": "perfect_score", "name": "Perfectionist", "description": "Score 100% on any test", "icon_key": "💯"},
    {"id": "comeback_kid", "name": "Comeback Kid", "description": "Score 80%+ after three <50%", "icon_key": "💪"},
    {"id": "speed_demon", "name": "Speed Demon", "description": "Finish a 10Q test in <5 mins", "icon_key": "⚡"},
    {"id": "ncert_explorer", "name": "NCERT Explorer", "description": "Practice 10 different NCERT topics", "icon_key": "📚"},
    {"id": "mock_master", "name": "Mock Master", "description": "Complete 5 mock tests", "icon_key": "📝"},
    {"id": "chat_100", "name": "Chatterbox", "description": "Send 100 chat messages", "icon_key": "💬"},
    {"id": "all_subjects", "name": "All-Rounder", "description": "Practice all 4 subjects", "icon_key": "🌟"},
    {"id": "level_5_any", "name": "Rising Star", "description": "Reach Level 5 in any topic", "icon_key": "⭐"},
    {"id": "level_10_any", "name": "Topic Master", "description": "Reach Level 10 in any topic", "icon_key": "👑"},
    {"id": "top_scorer", "name": "Top Scorer", "description": "Achieve avg score > 85%", "icon_key": "🏆"},
    {"id": "consistent_week", "name": "Consistent", "description": "Study every day for 7 days", "icon_key": "📅"},
    {"id": "full_mock_complete", "name": "Exam Ready", "description": "Complete a full/JEE/NEET mock", "icon_key": "🎓"},
]


def compute_level(xp: int, thresholds: list[int]) -> int:
    """Return level 1-10 based on xp vs LEVEL_THRESHOLDS."""
    level = 1
    for idx, thresh in enumerate(thresholds):
        if xp >= thresh:
            level = idx + 1
    return min(level, 10)


def compute_global_level(total_xp: int) -> int:
    """Return level 1-50 based on total_xp vs GLOBAL_LEVELS."""
    level = 1
    for idx, thresh in enumerate(GLOBAL_LEVELS):
        if total_xp >= thresh:
            level = idx + 1
    return min(level, 50)


def get_title(level: int) -> str:
    """Look up TITLES for given global level."""
    for level_range, title in TITLES.items():
        if level in level_range:
            return title
    return "Novice"


def _get_or_create_topic_mastery(student_id: int, subject: str, topic: str, db: Session) -> TopicMastery:
    mastery = (
        db.query(TopicMastery)
        .filter(
            TopicMastery.student_id == student_id,
            TopicMastery.subject == subject,
            TopicMastery.topic == topic,
        )
        .first()
    )
    if mastery:
        return mastery

    mastery = TopicMastery(student_id=student_id, subject=subject, topic=topic)
    db.add(mastery)
    db.flush()
    return mastery


def _get_or_create_profile(student_id: int, db: Session) -> StudentProfile:
    profile = db.query(StudentProfile).filter(StudentProfile.student_id == student_id).first()
    if profile:
        return profile

    profile = StudentProfile(student_id=student_id)
    db.add(profile)
    db.flush()
    return profile


def award_xp(student_id: int, subject: str, topic: str, action_key: str, db: Session) -> dict:
    """
    Award XP for a given action and update topic mastery + student profile.
    """
    xp_gain = XP_RULES.get(action_key, 0)
    now = datetime.utcnow()

    mastery = _get_or_create_topic_mastery(student_id, subject, topic, db)
    old_level = mastery.level

    mastery.xp += xp_gain
    mastery.level = compute_level(mastery.xp, LEVEL_THRESHOLDS)
    mastery.mastery_pct = min(100.0, (mastery.xp / LEVEL_THRESHOLDS[-1]) * 100.0)

    if mastery.last_practiced:
        last_date = mastery.last_practiced.date()
        if last_date == (now.date() - timedelta(days=1)):
            mastery.streak_days += 1
        elif last_date == now.date():
            mastery.streak_days = mastery.streak_days
        else:
            mastery.streak_days = 1
    else:
        mastery.streak_days = 1

    mastery.last_practiced = now
    mastery.updated_at = now

    profile = _get_or_create_profile(student_id, db)
    profile.total_xp += xp_gain
    profile.global_level = compute_global_level(profile.total_xp)
    profile.title = get_title(profile.global_level)
    profile.updated_at = now
    if mastery.streak_days > profile.longest_streak:
        profile.longest_streak = mastery.streak_days

    newly_earned = check_badges(student_id, db)

    db.commit()

    return {
        "xp_gained": xp_gain,
        "topic_xp_new": mastery.xp,
        "topic_level_new": mastery.level,
        "global_xp_new": profile.total_xp,
        "global_level_new": profile.global_level,
        "title": profile.title,
        "level_up": mastery.level != old_level,
        "badges_earned": newly_earned,
    }


def apply_decay(student_id: int, db: Session) -> dict:
    """
    Apply XP decay to all TopicMastery entries for a student.
    """
    now = datetime.utcnow()
    topics = db.query(TopicMastery).filter(TopicMastery.student_id == student_id).all()
    decayed: list[dict[str, Any]] = []

    for mastery in topics:
        if not mastery.last_practiced:
            continue

        days = (now - mastery.last_practiced).days
        if days == 0:
            continue
        if 1 <= days < 3:
            xp_loss = 5 * days
        elif 3 <= days < 7:
            xp_loss = 15 * days
        else:
            xp_loss = 30 * days

        new_xp = max(0, mastery.xp - xp_loss)
        mastery.xp = new_xp
        mastery.level = compute_level(new_xp, LEVEL_THRESHOLDS)
        mastery.mastery_pct = min(100.0, (mastery.xp / LEVEL_THRESHOLDS[-1]) * 100.0)
        mastery.updated_at = now

        decayed.append(
            {
                "topic": mastery.topic,
                "xp_lost": xp_loss,
                "new_xp": mastery.xp,
                "new_level": mastery.level,
            }
        )

    db.commit()
    return {"topics_decayed": decayed}


def _load_badges(profile: StudentProfile) -> list[str]:
    try:
        badges = json.loads(profile.badges or "[]")
        if isinstance(badges, list):
            return badges
    except json.JSONDecodeError:
        pass
    return []


def _save_badges(profile: StudentProfile, badge_ids: list[str]) -> None:
    profile.badges = json.dumps(sorted(set(badge_ids)))


def check_badges(student_id: int, db: Session) -> list[str]:
    """
    Evaluate badge conditions and return newly earned badge IDs.
    """
    profile = _get_or_create_profile(student_id, db)
    owned = set(_load_badges(profile))
    newly: list[str] = []

    results = (
        db.query(models.TestResult)
        .filter(models.TestResult.student_id == student_id)
        .order_by(models.TestResult.taken_at.asc())
        .all()
    )

    if "first_test" not in owned and results:
        newly.append("first_test")

    mastery_rows = db.query(TopicMastery).filter(TopicMastery.student_id == student_id).all()
    if "five_streak" not in owned and any(m.streak_days >= 5 for m in mastery_rows):
        newly.append("five_streak")
    if "ten_streak" not in owned and any(m.streak_days >= 10 for m in mastery_rows):
        newly.append("ten_streak")

    if "perfect_score" not in owned and any(r.score == 100 for r in results):
        newly.append("perfect_score")

    if "comeback_kid" not in owned:
        low_streak = 0
        for r in results:
            if r.score < 50:
                low_streak += 1
            else:
                if low_streak >= 3 and r.score >= 80:
                    newly.append("comeback_kid")
                    break
                low_streak = 0

    if "speed_demon" not in owned:
        for r in results:
            test = db.get(models.Test, r.test_id)
            if not test:
                continue
            total_q = len(test.questions_json or [])
            meta = r.answers_json if isinstance(r.answers_json, dict) else {}
            time_secs = meta.get("time_taken_secs")
            if total_q >= 10 and isinstance(time_secs, int) and time_secs < 300:
                newly.append("speed_demon")
                break

    if "ncert_explorer" not in owned:
        topics = (
            db.query(models.Test.topic)
            .join(models.TestResult, models.Test.id == models.TestResult.test_id)
            .filter(models.TestResult.student_id == student_id)
            .distinct()
            .all()
        )
        if len(topics) >= 10:
            newly.append("ncert_explorer")

    if "mock_master" not in owned:
        mock_tests = (
            db.query(models.Test)
            .filter(models.Test.student_id == student_id)
            .all()
        )
        mock_count = 0
        for t in mock_tests:
            if isinstance(t.questions_json, dict) and t.questions_json.get("meta", {}).get("mock_mode"):
                mock_count += 1
        if mock_count >= 5:
            newly.append("mock_master")

    if "chat_100" not in owned:
        chat_count = (
            db.query(models.ChatMessage)
            .filter(models.ChatMessage.student_id == student_id)
            .count()
        )
        if chat_count >= 100:
            newly.append("chat_100")

    if "all_subjects" not in owned:
        subjects = {m.subject for m in mastery_rows}
        if len(subjects) >= 4:
            newly.append("all_subjects")

    if "level_5_any" not in owned and any(m.level >= 5 for m in mastery_rows):
        newly.append("level_5_any")
    if "level_10_any" not in owned and any(m.level >= 10 for m in mastery_rows):
        newly.append("level_10_any")

    if "top_scorer" not in owned and results:
        avg_score = sum(r.score for r in results) / len(results)
        if avg_score >= 85:
            newly.append("top_scorer")

    if "consistent_week" not in owned and profile.longest_streak >= 7:
        newly.append("consistent_week")

    if "full_mock_complete" not in owned:
        for t in (
            db.query(models.Test)
            .filter(models.Test.student_id == student_id)
            .all()
        ):
            if isinstance(t.questions_json, dict):
                mode = t.questions_json.get("meta", {}).get("mock_mode")
                if mode in {"full", "jee", "neet"}:
                    newly.append("full_mock_complete")
                    break

    if newly:
        owned.update(newly)
        _save_badges(profile, list(owned))

    return newly


def get_topic_mastery_map(student_id: int, db: Session) -> list[dict]:
    """
    All TopicMastery rows for student. For each:
    days_since = (now - last_practiced).days
    decay_warning = days_since >= 2
    Return [{subject, topic, level, xp, mastery_pct,
             last_practiced, days_since, streak_days, decay_warning}]
    Grouped by subject in return (sort by subject, then level desc).
    """
    now = datetime.utcnow()
    rows = db.query(TopicMastery).filter(TopicMastery.student_id == student_id).all()
    items: list[dict[str, Any]] = []

    for r in rows:
        days_since = (now - r.last_practiced).days if r.last_practiced else None
        items.append(
            {
                "subject": r.subject,
                "topic": r.topic,
                "level": r.level,
                "xp": r.xp,
                "mastery_pct": r.mastery_pct,
                "last_practiced": r.last_practiced,
                "days_since": days_since,
                "streak_days": r.streak_days,
                "decay_warning": isinstance(days_since, int) and days_since >= 2,
            }
        )

    items.sort(key=lambda x: (x["subject"], -x["level"]))
    return items
