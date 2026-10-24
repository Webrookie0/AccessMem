"""Persistent local identities with Argon2 password hashing."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from ..config import get_settings
from .models import User

_hasher = PasswordHasher()
_DEMO_HASH = "$argon2id$v=19$m=65536,t=3,p=4$sbZ2TskZQyjlHKMUwhiD0A$snno7uZjZ/3skALMrTVVSfNuxttxGJPfwdwA35U2Xhk"
_DEMO_USERS = (
    ("alice", "junior", "acme"),
    ("bob", "manager", "acme"),
    ("carol", "csuite", "acme"),
    ("dave", "csuite", "globex"),
)


@contextmanager
def _connection():
    path = Path(get_settings().user_database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=10)) as connection, connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                role TEXT NOT NULL CHECK(role IN ('junior', 'manager', 'csuite', 'admin')),
                tenant_id TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_demo INTEGER NOT NULL DEFAULT 0 CHECK(is_demo IN (0, 1)),
                disabled INTEGER NOT NULL DEFAULT 0 CHECK(disabled IN (0, 1))
            )"""
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        if "is_demo" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                "UPDATE users SET is_demo = 1 WHERE password_hash = ? AND username IN ('alice', 'bob', 'carol', 'dave')",
                (_DEMO_HASH,),
            )
        yield connection


def initialize_user_store() -> None:
    with _connection() as connection:
        if get_settings().demo_users_enabled:
            connection.executemany(
                "INSERT OR IGNORE INTO users(username, role, tenant_id, password_hash, is_demo) VALUES (?, ?, ?, ?, 1)",
                [(*user, _DEMO_HASH) for user in _DEMO_USERS],
            )


def get_user(username: str) -> User | None:
    initialize_user_store()
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? AND disabled = 0 AND (? = 1 OR is_demo = 0)",
            (username.lower(), int(get_settings().demo_users_enabled)),
        ).fetchone()
    return None if row is None else User(**dict(row))


def _verify(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def authenticate(username: str, password: str) -> User | None:
    if not password or len(password) > 1024:
        return None
    user = get_user(username)
    password_hash = user.password_hash if user else _DEMO_HASH
    if not _verify(password, password_hash) or user is None:
        return None
    if _hasher.check_needs_rehash(password_hash):
        with _connection() as connection:
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (_hasher.hash(password), user.username),
            )
    return user


def create_user(username: str, role: str, tenant_id: str, password: str) -> User:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", username):
        raise ValueError(
            "Username must contain only letters, numbers, dots, hyphens and underscores"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", tenant_id):
        raise ValueError("Tenant must contain only letters, numbers, dots, hyphens and underscores")
    if not 12 <= len(password) <= 1024:
        raise ValueError("Passwords must contain between 12 and 1024 characters")
    user = User(
        username=username.lower(),
        role=role,
        tenant_id=tenant_id,
        password_hash=_hasher.hash(password),
    )
    initialize_user_store()
    try:
        with _connection() as connection:
            connection.execute(
                "INSERT INTO users(username, role, tenant_id, password_hash) VALUES (?, ?, ?, ?)",
                (user.username, user.role, user.tenant_id, user.password_hash),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists") from exc
    return user


def disable_user(username: str) -> None:
    with _connection() as connection:
        result = connection.execute(
            "UPDATE users SET disabled = 1 WHERE username = ?", (username.lower(),)
        )
        if not result.rowcount:
            raise ValueError("User does not exist")


def list_users() -> list[dict]:
    initialize_user_store()
    with _connection() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT username, role, tenant_id, disabled FROM users ORDER BY username"
            )
        ]


def all_usernames() -> list[str]:
    return [user["username"] for user in list_users() if not user["disabled"]]
