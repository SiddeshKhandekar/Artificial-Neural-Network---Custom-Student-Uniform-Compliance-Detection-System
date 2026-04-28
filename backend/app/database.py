"""
============================================================
Database Module -- SQLite ORM
============================================================
Defines the SQLite database schema and provides helper functions
for CRUD operations on `students` and `violations` tables.

SQLite is chosen for this project because:
  - Zero configuration (no server required)
  - Single-file database (easy to share/demo)
  - Perfect for localhost college projects
============================================================
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, date
from pathlib import Path
from contextlib import contextmanager

from app.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")    # Better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")     # Enforce FK constraints
    return conn


@contextmanager
def get_db():
    """Context manager for database connections -- ensures proper cleanup."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """
    Create the database tables if they don't exist.

    Schema:
    ┌─────────────────────────────────────────────────────────┐
    │ students                                                │
    ├─────────────┬────────────┬──────────────────────────────┤
    │ id          │ INTEGER PK │ Auto-increment               │
    │ student_id  │ TEXT UNIQUE│ e.g. "STU001"                │
    │ name        │ TEXT       │ Full name                    │
    │ department  │ TEXT       │ e.g. "Computer Science"      │
    │ year        │ INTEGER    │ 1-4                          │
    │ embedding   │ TEXT       │ JSON-serialized face vector  │
    │ photo_path  │ TEXT       │ Path to enrolled face photo  │
    │ created_at  │ DATETIME   │ Auto-set on creation         │
    └─────────────┴────────────┴──────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │ violations                                              │
    ├────────────────┬────────────┬───────────────────────────┤
    │ id             │ INTEGER PK │ Auto-increment            │
    │ student_id     │ TEXT FK    │ References students       │
    │ date           │ DATE       │ Date of violation         │
    │ violation_type │ TEXT       │ e.g. "Missing ID Card"    │
    │ confidence     │ REAL       │ Detection confidence      │
    │ image_path     │ TEXT       │ Snapshot of violation     │
    │ details        │ TEXT       │ JSON extra info           │
    │ created_at     │ DATETIME   │ Auto-set on creation      │
    └────────────────┴────────────┴───────────────────────────┘
    """
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT    UNIQUE NOT NULL,
                name        TEXT    NOT NULL,
                department  TEXT    DEFAULT '',
                year        INTEGER DEFAULT 1,
                embedding   TEXT    DEFAULT '[]',
                photo_path  TEXT    DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS violations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id      TEXT    NOT NULL,
                date            DATE    NOT NULL,
                violation_type  TEXT    NOT NULL,
                confidence      REAL    DEFAULT 0.0,
                image_path      TEXT    DEFAULT '',
                details         TEXT    DEFAULT '{}',
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS analysis_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id      TEXT    NOT NULL,
                total_score     REAL    DEFAULT 0.0,
                issues_found    TEXT    DEFAULT '[]',
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_violations_student
                ON violations(student_id);
            CREATE INDEX IF NOT EXISTS idx_violations_date
                ON violations(date);
            CREATE INDEX IF NOT EXISTS idx_history_student
                ON analysis_history(student_id);
                
            INSERT OR IGNORE INTO students (student_id, name, department, year, embedding)
            VALUES ('UNKNOWN', 'Unknown Person', 'Visitor/Unrecognized', 0, '[]');
        """)


# ── Student CRUD Operations ────────────────────────────────

def create_student(student_id: str, name: str, department: str = "",
                   year: int = 1, embedding: list = None,
                   photo_path: str = "") -> dict:
    """Insert a new student record."""
    emb_json = json.dumps(embedding if embedding else [])
    with get_db() as conn:
        conn.execute(
            """INSERT INTO students (student_id, name, department, year, embedding, photo_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (student_id, name, department, year, emb_json, photo_path)
        )
    return get_student(student_id)


def get_student(student_id: str) -> dict | None:
    """Retrieve a student by their student_id."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
    if row:
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else []
        return d
    return None


def get_all_students() -> list[dict]:
    """Retrieve all student records."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM students ORDER BY student_id").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else []
        result.append(d)
    return result


def update_student(student_id: str, **kwargs) -> dict | None:
    """Update fields of an existing student. Pass only fields to update."""
    allowed_fields = {"name", "department", "year", "embedding", "photo_path"}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return get_student(student_id)

    if "embedding" in updates:
        updates["embedding"] = json.dumps(updates["embedding"])

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [student_id]

    with get_db() as conn:
        conn.execute(
            f"UPDATE students SET {set_clause} WHERE student_id = ?",
            values
        )
    return get_student(student_id)


def delete_student(student_id: str) -> bool:
    """Delete a student and cascade-delete their violations."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM students WHERE student_id = ?", (student_id,)
        )
    return cursor.rowcount > 0


# ── Violation Operations ────────────────────────────────────

def log_violation(student_id: str, violation_type: str,
                  confidence: float = 0.0, image_path: str = "",
                  details: dict = None) -> dict:
    """Log a new violation record."""
    today = date.today().isoformat()
    details_json = json.dumps(details if details else {})
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO violations
               (student_id, date, violation_type, confidence, image_path, details)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (student_id, today, violation_type, confidence, image_path, details_json)
        )
        row = conn.execute(
            "SELECT * FROM violations WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return dict(row) if row else {}


def get_violations(student_id: str = None, start_date: str = None,
                   end_date: str = None, limit: int = 50) -> list[dict]:
    """Retrieve violations with optional filters."""
    query = "SELECT * FROM violations WHERE 1=1"
    params = []

    if student_id:
        query += " AND student_id = ?"
        params.append(student_id)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_violation_stats() -> dict:
    """Get summary statistics for violations."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM violations").fetchone()["c"]
        by_type = conn.execute(
            """SELECT violation_type, COUNT(*) as count
               FROM violations GROUP BY violation_type
               ORDER BY count DESC"""
        ).fetchall()
        by_student = conn.execute(
            """SELECT v.student_id, s.name, COUNT(*) as count
               FROM violations v
               LEFT JOIN students s ON v.student_id = s.student_id
               GROUP BY v.student_id
               ORDER BY count DESC"""
        ).fetchall()

    return {
        "total_violations": total,
        "by_type": [dict(r) for r in by_type],
        "by_student": [dict(r) for r in by_student],
    }


# ── Analysis History Operations ──────────────────────────────

def log_analysis(student_id: str, total_score: float, issues_found: list) -> dict:
    """Log a complete analysis event."""
    issues_json = json.dumps(issues_found if issues_found else [])
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO analysis_history
               (student_id, total_score, issues_found)
               VALUES (?, ?, ?)""",
            (student_id, total_score, issues_json)
        )
        row = conn.execute(
            "SELECT * FROM analysis_history WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return dict(row) if row else {}


def get_history(limit: int = 100) -> list[dict]:
    """Retrieve full analysis history."""
    query = """
        SELECT h.*, s.name, s.department 
        FROM analysis_history h
        LEFT JOIN students s ON h.student_id = s.student_id
        ORDER BY h.created_at DESC LIMIT ?
    """
    
    with get_db() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        
    result = []
    for row in rows:
        d = dict(row)
        try:
            d["issues_found"] = json.loads(d["issues_found"])
        except:
            d["issues_found"] = []
        result.append(d)
        
    return result
