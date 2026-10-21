import os
import sqlite3
import time

DB_PATH = os.getenv("BOT_DB_PATH", "data/bot.db")

_conn = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "user_id INTEGER PRIMARY KEY,"
            "first_name TEXT,"
            "gateway_passed INTEGER NOT NULL DEFAULT 0,"
            "passed_at REAL,"
            "packs INTEGER NOT NULL DEFAULT 0,"
            "last_pack TEXT)"
        )
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS user_packs ("
            "user_id INTEGER NOT NULL,"
            "url TEXT NOT NULL,"
            "created_at REAL NOT NULL DEFAULT 0,"
            "PRIMARY KEY (user_id, url))"
        )
        _conn.execute(
            "INSERT OR IGNORE INTO user_packs (user_id, url, created_at) "
            "SELECT user_id, last_pack, 0 FROM users "
            "WHERE last_pack IS NOT NULL AND last_pack != ''"
        )
        _conn.commit()
    return _conn


def user_packs(user_id: int) -> list[str]:
    rows = _get_conn().execute(
        "SELECT url FROM user_packs WHERE user_id = ? "
        "ORDER BY created_at ASC, url ASC",
        (user_id,),
    ).fetchall()
    return [r[0] for r in rows]


def has_packs(user_id: int) -> bool:
    row = _get_conn().execute(
        "SELECT 1 FROM user_packs WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()
    return bool(row)


def get_user(user_id: int) -> dict | None:
    row = _get_conn().execute(
        "SELECT user_id, first_name, gateway_passed, passed_at, packs, last_pack "
        "FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "first_name": row[1],
        "gateway_passed": bool(row[2]),
        "passed_at": row[3],
        "packs": row[4],
        "last_pack": row[5],
    }


def remember(user_id: int, first_name: str | None = None) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (user_id, first_name, gateway_passed, passed_at) "
        "VALUES (?, ?, 1, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "gateway_passed = 1, "
        "passed_at = CASE WHEN users.gateway_passed = 1 THEN users.passed_at ELSE excluded.passed_at END, "
        "first_name = COALESCE(NULLIF(?, ''), users.first_name)",
        (user_id, first_name or "", time.time(), first_name or ""),
    )
    conn.commit()


def mark_pack_created(user_id: int, pack_url: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (user_id, packs, last_pack) VALUES (?, 1, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "packs = users.packs + 1, last_pack = excluded.last_pack",
        (user_id, pack_url),
    )
    conn.execute(
        "INSERT OR IGNORE INTO user_packs (user_id, url, created_at) VALUES (?, ?, ?)",
        (user_id, pack_url, time.time()),
    )
    conn.commit()


def started(user_id: int) -> None:
    conn = _get_conn()
    conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()


def stats() -> dict:
    conn = _get_conn()
    users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    gated = conn.execute(
        "SELECT COUNT(*) FROM users WHERE gateway_passed = 1"
    ).fetchone()[0]
    packs = conn.execute("SELECT COALESCE(SUM(packs), 0) FROM users").fetchone()[0]
    return {"users": users, "gated": gated, "packs": packs}
