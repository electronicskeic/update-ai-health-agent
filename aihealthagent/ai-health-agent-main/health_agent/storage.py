from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bcrypt


@dataclass(frozen=True)
class User:
    id: int
    username: str


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password_hash BLOB NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS profiles (
              user_id INTEGER PRIMARY KEY,
              profile_json TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS checkins (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              date TEXT NOT NULL,
              weight_kg REAL NOT NULL,
              note TEXT,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('user','assistant')),
              content TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def _hash_password(password: str) -> bytes:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def _verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bool(bcrypt.checkpw(password.encode("utf-8"), password_hash))
    except Exception:
        return False


def create_user(db_path: Path, *, username: str, password: str) -> User:
    u = (username or "").strip()
    if not u:
        raise ValueError("Username is required.")
    pw_hash = _hash_password(password)
    with _connect(db_path) as conn:
        cur = conn.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (u, pw_hash))
        user_id = int(cur.lastrowid)
    return User(id=user_id, username=u)


def authenticate(db_path: Path, *, username: str, password: str) -> User | None:
    u = (username or "").strip()
    if not u or not password:
        return None
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (u,)).fetchone()
        if not row:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return User(id=int(row["id"]), username=str(row["username"]))


def upsert_profile(db_path: Path, *, user_id: int, profile: dict[str, Any]) -> None:
    payload = json.dumps(profile, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO profiles(user_id, profile_json, updated_at)
            VALUES(?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
              profile_json = excluded.profile_json,
              updated_at = datetime('now')
            """,
            (int(user_id), payload),
        )


def load_profile(db_path: Path, *, user_id: int) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT profile_json FROM profiles WHERE user_id = ?", (int(user_id),)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["profile_json"])
    except Exception:
        return None


def add_checkin(db_path: Path, *, user_id: int, date: str, weight_kg: float, note: str | None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO checkins(user_id, date, weight_kg, note) VALUES (?, ?, ?, ?)",
            (int(user_id), str(date), float(weight_kg), note),
        )


def list_checkins(db_path: Path, *, user_id: int, limit: int = 200) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT date, weight_kg, note FROM checkins WHERE user_id = ? ORDER BY date DESC, id DESC LIMIT ?",
            (int(user_id), int(limit)),
        ).fetchall()
    return [{"date": r["date"], "weight_kg": float(r["weight_kg"]), "note": r["note"]} for r in rows]


def add_chat_message(db_path: Path, *, user_id: int, role: str, content: str) -> None:
    role = str(role)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO chat_messages(user_id, role, content) VALUES (?, ?, ?)",
            (int(user_id), role, str(content)),
        )


def list_chat_messages(db_path: Path, *, user_id: int, limit: int = 30) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    rows = list(reversed(rows))
    return [{"role": str(r["role"]), "content": str(r["content"]), "created_at": str(r["created_at"])} for r in rows]

