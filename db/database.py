# db/database.py
from __future__ import annotations

from datetime import datetime
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import json as _json

from config import DB_FILE, DIFFICULTY_SCALE


_DB_PATH = DB_FILE if isinstance(DB_FILE, Path) else Path(DB_FILE)
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- Migrations ----------------
def _ensure_questions_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to questions table if missing (safe to run repeatedly)."""
    c = conn.cursor()
    cols = {row["name"] for row in c.execute("PRAGMA table_info(questions)").fetchall()}

    migrations = [
        ("question", "TEXT"),
        ("options_json", "TEXT"),
        ("answer", "TEXT"),
        ("explanation", "TEXT"),
        ("chapter_id", "TEXT"),
        ("chapter_title", "TEXT"),
        ("source_name", "TEXT"),
    ]

    for col, coltype in migrations:
        if col not in cols:
            c.execute(f"ALTER TABLE questions ADD COLUMN {col} {coltype}")
    conn.commit()


def _ensure_labs_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to labs table if missing (safe to run repeatedly)."""
    c = conn.cursor()
    cols = {row["name"] for row in c.execute("PRAGMA table_info(labs)").fetchall()}

    migrations = [
        ("status", "TEXT"),
        ("notes", "TEXT"),
        ("steps_md", "TEXT"),
        ("last_opened_at", "TEXT"),
    ]

    for col, coltype in migrations:
        if col not in cols:
            c.execute(f"ALTER TABLE labs ADD COLUMN {col} {coltype}")
    conn.commit()


def init_db() -> None:
    with get_conn() as conn:
        c = conn.cursor()

        # Questions
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                domain TEXT,
                blueprint TEXT,
                difficulty TEXT,

                question TEXT,
                options_json TEXT,
                answer TEXT,
                explanation TEXT,

                chapter_id TEXT,
                chapter_title TEXT,
                source_name TEXT,

                seen INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0
            )
            """
        )
        _ensure_questions_columns(conn)

        # History (readiness over time)
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                score REAL,
                readiness REAL,
                confidence TEXT
            )
            """
        )

        # Coverage
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage (
                blueprint TEXT PRIMARY KEY,
                seen INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0
            )
            """
        )

        # Books
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                title TEXT,
                added_at TEXT,
                pages INTEGER DEFAULT 0
            )
            """
        )

        # Labs
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS labs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                title TEXT,
                domain TEXT,
                blueprint TEXT,
                difficulty TEXT,
                added_at TEXT,
                status TEXT,
                notes TEXT,
                steps_md TEXT,
                last_opened_at TEXT
            )
            """
        )
        _ensure_labs_columns(conn)

        # Flashcards
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                domain TEXT,
                chapter_id TEXT,
                source_name TEXT,
                created_at TEXT,
                seen INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0,
                UNIQUE(front, back, domain, chapter_id, source_name)
            )
            """
        )

        conn.commit()


# ---------------- Questions (Fix A: bulk upsert) ----------------
def _normalize_question_row(q: Dict[str, Any]) -> Optional[tuple]:
    """
    Normalize a question dict into a DB row tuple matching the questions INSERT statement.
    Returns None if it can't be inserted (missing id).
    """
    qid = q.get("id")
    if not qid:
        return None

    difficulty = q.get("difficulty")
    if difficulty not in DIFFICULTY_SCALE:
        difficulty = DIFFICULTY_SCALE[0] if DIFFICULTY_SCALE else "Easy"

    opts = q.get("options", [])
    if isinstance(opts, str):
        try:
            opts = _json.loads(opts)
        except Exception:
            opts = [opts]
    if not isinstance(opts, list):
        opts = []
    options_json = _json.dumps(opts, ensure_ascii=False)

    return (
        qid,
        q.get("domain"),
        q.get("blueprint"),
        difficulty,
        q.get("question"),
        options_json,
        q.get("answer"),
        q.get("explanation"),
        q.get("chapter_id"),
        q.get("chapter_title"),
        q.get("source_name"),
    )


def bulk_upsert_questions(qs: List[Dict[str, Any]]) -> int:
    """
    Bulk upsert questions in ONE transaction for speed.
    Returns number of rows attempted (invalid rows skipped).
    """
    if not qs:
        return 0

    rows: List[tuple] = []
    for q in qs:
        row = _normalize_question_row(q)
        if row is not None:
            rows.append(row)

    if not rows:
        return 0

    with get_conn() as conn:
        c = conn.cursor()
        c.executemany(
            """
            INSERT INTO questions (
                id, domain, blueprint, difficulty,
                question, options_json, answer, explanation,
                chapter_id, chapter_title, source_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                domain=excluded.domain,
                blueprint=excluded.blueprint,
                difficulty=excluded.difficulty,
                question=excluded.question,
                options_json=excluded.options_json,
                answer=excluded.answer,
                explanation=excluded.explanation,
                chapter_id=excluded.chapter_id,
                chapter_title=excluded.chapter_title,
                source_name=excluded.source_name
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def upsert_question(q: Dict[str, Any]) -> None:
    """Compatibility wrapper (bulk_upsert_questions is preferred)."""
    row = _normalize_question_row(q)
    if row is None:
        raise ValueError("Question must include a non-empty 'id'.")

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO questions (
                id, domain, blueprint, difficulty,
                question, options_json, answer, explanation,
                chapter_id, chapter_title, source_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                domain=excluded.domain,
                blueprint=excluded.blueprint,
                difficulty=excluded.difficulty,
                question=excluded.question,
                options_json=excluded.options_json,
                answer=excluded.answer,
                explanation=excluded.explanation,
                chapter_id=excluded.chapter_id,
                chapter_title=excluded.chapter_title,
                source_name=excluded.source_name
            """,
            row,
        )
        conn.commit()


def update_question_progress(qid: str, correct: bool) -> None:
    if not qid:
        return

    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT seen, correct FROM questions WHERE id=?", (qid,))
        row = c.fetchone()
        if not row:
            return

        seen = int(row["seen"] or 0) + 1
        cor = int(row["correct"] or 0) + (1 if correct else 0)
        c.execute("UPDATE questions SET seen=?, correct=? WHERE id=?", (seen, cor, qid))
        conn.commit()

    adjust_question_difficulty(qid)


def adjust_question_difficulty(qid: str) -> None:
    if not qid or not DIFFICULTY_SCALE:
        return

    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT difficulty, seen, correct FROM questions WHERE id=?", (qid,))
        row = c.fetchone()
        if not row:
            return

        difficulty = row["difficulty"]
        seen = int(row["seen"] or 0)
        correct = int(row["correct"] or 0)

        if seen <= 0:
            return

        if difficulty not in DIFFICULTY_SCALE:
            difficulty = DIFFICULTY_SCALE[0]

        acc = correct / seen
        idx = DIFFICULTY_SCALE.index(difficulty)

        if acc > 0.85 and idx < len(DIFFICULTY_SCALE) - 1:
            idx += 1
        elif acc < 0.55 and idx > 0:
            idx -= 1

        new_diff = DIFFICULTY_SCALE[idx]
        if new_diff != difficulty:
            c.execute("UPDATE questions SET difficulty=? WHERE id=?", (new_diff, qid))
            conn.commit()


def get_questions() -> List[Dict[str, Any]]:
    """Return questions as dicts, decoding options_json back into 'options' list."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM questions")
        rows = c.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        opts_raw = d.pop("options_json", None)
        try:
            d["options"] = _json.loads(opts_raw) if opts_raw else []
        except Exception:
            d["options"] = []
        out.append(d)
    return out


# ---------------- Coverage ----------------
def update_coverage(blueprint: str, correct: bool) -> None:
    if not blueprint:
        return

    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT seen, correct FROM coverage WHERE blueprint=?", (blueprint,))
        row = c.fetchone()

        if row:
            seen = int(row["seen"] or 0) + 1
            cor = int(row["correct"] or 0) + (1 if correct else 0)
            c.execute("UPDATE coverage SET seen=?, correct=? WHERE blueprint=?", (seen, cor, blueprint))
        else:
            c.execute(
                "INSERT INTO coverage (blueprint, seen, correct) VALUES (?,?,?)",
                (blueprint, 1, 1 if correct else 0),
            )

        conn.commit()


def get_coverage() -> Dict[str, Dict[str, Any]]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM coverage")
        rows = c.fetchall()
        return {r["blueprint"]: dict(r) for r in rows}


# ---------------- History ----------------
def add_history(date_str: str, score: float, readiness: float, confidence: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (date, score, readiness, confidence) VALUES (?,?,?,?)",
            (date_str, float(score), float(readiness), confidence),
        )
        conn.commit()


def get_history() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM history ORDER BY date")
        rows = c.fetchall()
        return [dict(r) for r in rows]


def reset_history() -> None:
    """Delete all readiness/history rows (Readiness Over Time chart)."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()


# ---------------- Books Library ----------------
def add_book_record(filename: str, title: str, added_at: str, pages: int) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO books (filename, title, added_at, pages)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title=excluded.title,
                added_at=excluded.added_at,
                pages=excluded.pages
            """,
            (filename, title, added_at, int(pages)),
        )
        conn.commit()


def list_books() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM books ORDER BY added_at DESC")
        rows = c.fetchall()
        return [dict(r) for r in rows]


def delete_book_record(filename: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM books WHERE filename=?", (filename,))
        conn.commit()


# ---------------- Labs Library ----------------
def add_lab_record(
    filename: str,
    title: str,
    domain: str,
    blueprint: str,
    difficulty: str,
    added_at: str,
) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO labs (
                filename, title, domain, blueprint, difficulty,
                added_at, status, notes, steps_md, last_opened_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title=excluded.title,
                domain=excluded.domain,
                blueprint=excluded.blueprint,
                difficulty=excluded.difficulty,
                added_at=excluded.added_at
            """,
            (filename, title, domain, blueprint, difficulty, added_at, "In Progress", "", "", None),
        )
        conn.commit()


def list_labs() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM labs ORDER BY added_at DESC")
        rows = c.fetchall()
        return [dict(r) for r in rows]


def update_lab_progress(filename: str, status: str, notes: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE labs SET status=?, notes=? WHERE filename=?", (status, notes, filename))
        conn.commit()


def update_lab_steps(filename: str, steps_md: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE labs SET steps_md=? WHERE filename=?", (steps_md, filename))
        conn.commit()


def set_lab_last_opened(filename: str, last_opened_at: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE labs SET last_opened_at=? WHERE filename=?", (last_opened_at, filename))
        conn.commit()


def delete_lab_record(filename: str) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM labs WHERE filename=?", (filename,))
        conn.commit()


# ---------------- Flashcards ----------------
def add_flashcard(
    front: str,
    back: str,
    domain: str = "Network Fundamentals",
    chapter_id: Optional[str] = None,
    source_name: Optional[str] = None,
    created_at: Optional[str] = None,
) -> None:
    if not front or not back:
        return
    if not created_at:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT OR IGNORE INTO flashcards (front, back, domain, chapter_id, source_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (front, back, domain, chapter_id, source_name, created_at),
        )
        conn.commit()


def bulk_add_flashcards(cards: List[Dict[str, Any]]) -> int:
    if not cards:
        return 0

    rows: List[tuple] = []
    for card in cards:
        front = (card.get("front") or "").strip()
        back = (card.get("back") or "").strip()
        if not front or not back:
            continue

        domain = (card.get("domain") or "Network Fundamentals").strip()
        chapter_id = card.get("chapter_id")
        source_name = card.get("source_name")
        created_at = card.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows.append((front, back, domain, chapter_id, source_name, created_at))

    if not rows:
        return 0

    with get_conn() as conn:
        c = conn.cursor()
        c.executemany(
            """
            INSERT OR IGNORE INTO flashcards (front, back, domain, chapter_id, source_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def list_flashcards(domain: Optional[str] = None, domains_in: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM flashcards"
    params: List[Any] = []
    where: List[str] = []

    if domain:
        where.append("domain = ?")
        params.append(domain)

    if domains_in:
        placeholders = ",".join(["?"] * len(domains_in))
        where.append(f"domain IN ({placeholders})")
        params.extend(domains_in)

    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY created_at DESC, id DESC"

    with get_conn() as conn:
        c = conn.cursor()
        rows = c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def update_flashcard_progress(card_id: int, correct: bool) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        row = c.execute("SELECT seen, correct FROM flashcards WHERE id=?", (card_id,)).fetchone()
        if not row:
            return

        seen = int(row["seen"] or 0) + 1
        cor = int(row["correct"] or 0) + (1 if correct else 0)

        c.execute("UPDATE flashcards SET seen=?, correct=? WHERE id=?", (seen, cor, card_id))
        conn.commit()


def delete_flashcard(card_id: int) -> None:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM flashcards WHERE id=?", (card_id,))
        conn.commit()
