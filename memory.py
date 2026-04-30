
import sqlite3
import logging
from contextlib import contextmanager
from typing import Optional
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_HISTORY = 20  # last N messages rakhna memory mein


@contextmanager
def get_connection():
    """Context manager — connection automatically close hoga."""
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row  # dict-like rows
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads ke liye
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def init_db():
    """DB aur table initialize karo."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                message     TEXT    NOT NULL,
                created_at  DATETIME DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON chats(session_id, created_at)
        """)
    logger.info("✅ DB initialized")


def save(session_id: str, role: str, message: str) -> None:
    """Message save karo."""
    if not session_id or not message.strip():
        raise ValueError("session_id aur message empty nahi hone chahiye")

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chats (session_id, role, message) VALUES (?, ?, ?)",
            (session_id, role, message.strip())
        )
    logger.debug(f"Saved [{role}] for session {session_id[:8]}...")


def load(session_id: str, limit: int = MAX_HISTORY) -> list[dict]:
    """Last N messages load karo."""
    if not session_id:
        raise ValueError("session_id empty nahi hona chahiye")

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, message FROM (
                SELECT role, message, created_at
                FROM chats
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ) ORDER BY created_at ASC
        """, (session_id, limit)).fetchall()

    return [{"role": row["role"], "content": row["message"]} for row in rows]


def delete_session(session_id: str) -> None:
    """Poori session history delete karo."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
    logger.info(f"Session {session_id[:8]}... deleted")


def get_session_count(session_id: str) -> int:
    """Session mein kitne messages hain."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chats WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    return row["cnt"] if row else 0
