"""
core/database.py
================
SQLite persistence layer for vocabulary, lessons, and processing jobs.

Enhancements over original:
  - WAL mode for better concurrent read performance.
  - Full-text search on word / reading / meaning.
  - Lessons table to persist processed media.
  - Jobs table to track async processing state.
  - Export helpers (CSV, JSON).
  - Vocab statistics (total, known, frequency bands).
  - Thread-safe connection-per-call pattern (original was correct; kept).
  - Schema migrations via a lightweight version table.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.utils import get_logger

logger = get_logger(__name__)

# Current schema version – bump when adding new migrations
SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_factory(cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    """Return rows as dicts (sqlite3.Row alternative)."""
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# ImmersionDB
# ---------------------------------------------------------------------------

class ImmersionDB:
    """
    Single SQLite file, connection-per-call (thread-safe).

    Tables
    ------
    meta          – schema version
    vocabulary    – words encountered during immersion
    lessons       – processed media files
    segments      – transcript segments belonging to a lesson
    jobs          – async processing job state
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or settings.database.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_root_db()
        self._init_db()

    def _migrate_legacy_root_db(self) -> None:
        """
        Move old root-level immersion.db into storage/db on first run.
        Keeps existing user vocab/history while cleaning project root.
        """
        legacy = Path('immersion.db')
        if self.path.exists() or not legacy.exists():
            return
        try:
            shutil.move(str(legacy), str(self.path))
            for suffix in ('-shm', '-wal'):
                legacy_sidecar = Path(f'immersion.db{suffix}')
                if legacy_sidecar.exists():
                    shutil.move(str(legacy_sidecar), str(self.path) + suffix)
            logger.info(f'Migrated legacy DB to {self.path}')
        except Exception as e:
            logger.warning(f'Legacy DB migration skipped: {e}')

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = _row_factory
        if settings.database.wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Schema init & migrations
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            # Meta table first so we can read the current version
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

            current_version = self._get_version(conn)
            if current_version < 1:
                self._migrate_v1(conn)
            if current_version < 2:
                self._migrate_v2(conn)
            if current_version < 3:
                self._migrate_v3(conn)

            self._set_version(conn, SCHEMA_VERSION)

    def _get_version(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row:
            try:
                return int(row["value"])
            except (ValueError, KeyError):
                return 0
        return 0

    def _set_version(self, conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )

    def _migrate_v1(self, conn: sqlite3.Connection) -> None:
        """Initial schema."""
        logger.info("DB migration → v1: creating vocabulary table")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vocabulary (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                word      TEXT NOT NULL UNIQUE,
                reading   TEXT NOT NULL DEFAULT '',
                meaning   TEXT NOT NULL DEFAULT '—',
                count     INTEGER NOT NULL DEFAULT 1,
                known     INTEGER NOT NULL DEFAULT 0,
                notes     TEXT NOT NULL DEFAULT '',
                added_at  TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vocab_count ON vocabulary(count DESC)")

    def _migrate_v2(self, conn: sqlite3.Connection) -> None:
        """Add lessons and segments tables."""
        logger.info("DB migration → v2: creating lessons / segments tables")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lessons (
                id         TEXT PRIMARY KEY,
                filename   TEXT NOT NULL,
                media_type TEXT NOT NULL DEFAULT 'audio',
                duration   REAL NOT NULL DEFAULT 0,
                status     TEXT NOT NULL DEFAULT 'done',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_id  TEXT NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
                start_sec  REAL NOT NULL,
                end_sec    REAL NOT NULL,
                text       TEXT NOT NULL,
                tokens_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seg_lesson ON segments(lesson_id)"
        )

    def _migrate_v3(self, conn: sqlite3.Connection) -> None:
        """Add jobs table."""
        logger.info("DB migration → v3: creating jobs table")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id         TEXT PRIMARY KEY,
                filename   TEXT NOT NULL DEFAULT '',
                status     TEXT NOT NULL DEFAULT 'pending',
                progress   REAL NOT NULL DEFAULT 0,
                error      TEXT,
                started_at TEXT,
                ended_at   TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    # ------------------------------------------------------------------
    # Vocabulary CRUD
    # ------------------------------------------------------------------

    def add_word(self, word: str, reading: str = "", meaning: str = "—") -> None:
        """Insert word or increment its encounter count."""
        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO vocabulary (word, reading, meaning) VALUES (?,?,?)",
                    (word, reading, meaning),
                )
            except sqlite3.IntegrityError:
                conn.execute(
                    "UPDATE vocabulary SET count=count+1, last_seen=datetime('now') "
                    "WHERE word=?",
                    (word,),
                )

    def get_word(self, word: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT * FROM vocabulary WHERE word=?", (word,)
            ).fetchone()

    def update_word(
        self,
        word:    str,
        known:   Optional[bool] = None,
        notes:   Optional[str]  = None,
        meaning: Optional[str]  = None,
    ) -> bool:
        """Patch a vocabulary entry. Returns True if a row was updated."""
        fields: List[str] = []
        values: List[Any] = []
        if known is not None:
            fields.append("known=?");  values.append(int(known))
        if notes is not None:
            fields.append("notes=?");  values.append(notes)
        if meaning is not None:
            fields.append("meaning=?"); values.append(meaning)
        if not fields:
            return False
        values.append(word)
        with self._get_conn() as conn:
            cur = conn.execute(
                f"UPDATE vocabulary SET {', '.join(fields)} WHERE word=?", values
            )
            return cur.rowcount > 0

    def delete_word(self, word: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM vocabulary WHERE word=?", (word,))
            return cur.rowcount > 0

    def get_all(
        self,
        limit:  int = 500,
        offset: int = 0,
        known:  Optional[bool] = None,
        order:  str = "count DESC",
    ) -> List[Dict[str, Any]]:
        """Return all vocab entries with optional filtering."""
        where = ""
        params: List[Any] = []
        if known is not None:
            where = "WHERE known=?"
            params.append(int(known))
        sql = f"SELECT * FROM vocabulary {where} ORDER BY {order} LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._get_conn() as conn:
            return conn.execute(sql, params).fetchall()

    def search_vocab(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Full-text search across word, reading, and meaning columns."""
        like = f"%{query}%"
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT * FROM vocabulary "
                "WHERE word LIKE ? OR reading LIKE ? OR meaning LIKE ? "
                "ORDER BY count DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()

    # ------------------------------------------------------------------
    # Vocabulary statistics
    # ------------------------------------------------------------------

    def vocab_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total  = conn.execute("SELECT COUNT(*) as n FROM vocabulary").fetchone()["n"]
            known  = conn.execute("SELECT COUNT(*) as n FROM vocabulary WHERE known=1").fetchone()["n"]
            freq_5 = conn.execute("SELECT COUNT(*) as n FROM vocabulary WHERE count>=5").fetchone()["n"]
            top_10 = conn.execute(
                "SELECT word, count FROM vocabulary ORDER BY count DESC LIMIT 10"
            ).fetchall()
            return {
                "total":          total,
                "known":          known,
                "unknown":        total - known,
                "high_freq_gte5": freq_5,
                "top_10":         top_10,
            }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self) -> str:
        """Return all vocabulary as a CSV string."""
        rows = self.get_all(limit=100_000)
        buf  = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return buf.getvalue()

    def export_json(self) -> str:
        rows = self.get_all(limit=100_000)
        return json.dumps(rows, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def save_lesson(
        self,
        lesson_id:  str,
        filename:   str,
        segments:   List[Dict[str, Any]],
        media_type: str  = "audio",
        duration:   float = 0.0,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lessons (id, filename, media_type, duration) "
                "VALUES (?,?,?,?)",
                (lesson_id, filename, media_type, duration),
            )
            conn.execute("DELETE FROM segments WHERE lesson_id=?", (lesson_id,))
            conn.executemany(
                "INSERT INTO segments (lesson_id, start_sec, end_sec, text, tokens_json) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        lesson_id,
                        seg["start"],
                        seg["end"],
                        seg["text"],
                        json.dumps(seg.get("tokens", []), ensure_ascii=False),
                    )
                    for seg in segments
                ],
            )

    def get_lesson(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            lesson = conn.execute(
                "SELECT * FROM lessons WHERE id=?", (lesson_id,)
            ).fetchone()
            if not lesson:
                return None
            segs = conn.execute(
                "SELECT * FROM segments WHERE lesson_id=? ORDER BY start_sec",
                (lesson_id,),
            ).fetchall()
            for seg in segs:
                seg["tokens"] = json.loads(seg.get("tokens_json") or "[]")
                # Normalize column names to match the rest of the codebase
                seg.setdefault("start", seg.pop("start_sec", 0.0))
                seg.setdefault("end",   seg.pop("end_sec",   0.0))
            lesson["segments"] = segs
            return lesson

    def list_lessons(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT id, filename, media_type, duration, status, created_at "
                "FROM lessons ORDER BY created_at DESC"
            ).fetchall()

    def delete_lesson(self, lesson_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM lessons WHERE id=?", (lesson_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(self, job_id: str, filename: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, status) VALUES (?,?,?)",
                (job_id, filename, "pending"),
            )

    def update_job(
        self,
        job_id:   str,
        status:   Optional[str]   = None,
        progress: Optional[float] = None,
        error:    Optional[str]   = None,
    ) -> None:
        fields: List[str] = []
        values: List[Any] = []
        if status is not None:
            fields.append("status=?"); values.append(status)
            if status == "running":
                fields.append("started_at=datetime('now')")
            elif status in ("done", "failed"):
                fields.append("ended_at=datetime('now')")
        if progress is not None:
            fields.append("progress=?"); values.append(progress)
        if error is not None:
            fields.append("error=?"); values.append(error)
        if not fields:
            return
        values.append(job_id)
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", values
            )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
