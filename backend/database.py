import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "data" / "jobs"))
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT / DATA_DIR
DB_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "data" / "prompttovideo.sqlite3"))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize() -> None:
    with connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                prompt TEXT NOT NULL,
                duration INTEGER NOT NULL,
                language TEXT NOT NULL,
                style TEXT NOT NULL,
                voice TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                heartbeat_at TEXT,
                stage_started_at TEXT,
                output_file TEXT,
                assets TEXT NOT NULL DEFAULT '{}',
                fast_preview INTEGER NOT NULL DEFAULT 0,
                warning TEXT,
                stage_status TEXT NOT NULL DEFAULT 'pending'
            )
        """)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
        if "heartbeat_at" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT")
        if "stage_started_at" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN stage_started_at TEXT")
        if "fast_preview" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN fast_preview INTEGER NOT NULL DEFAULT 0")
        if "warning" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN warning TEXT")
        if "stage_status" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN stage_status TEXT NOT NULL DEFAULT 'pending'")
        connection.execute("UPDATE jobs SET heartbeat_at = updated_at WHERE heartbeat_at IS NULL")
        connection.execute("UPDATE jobs SET stage_started_at = updated_at WHERE stage_started_at IS NULL")


def serialize(row: sqlite3.Row) -> dict:
    result = dict(row)
    result["assets"] = json.loads(result.get("assets") or "{}")
    result["fast_preview"] = bool(result.get("fast_preview", 0))
    result["video_url"] = f"/media/{result['id']}/{Path(result['output_file']).name}" if result.get("output_file") else None
    return result


def get_job(job_id: str) -> dict | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return serialize(row) if row else None


def list_jobs() -> list[dict]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [serialize(row) for row in rows]


def create_job(job: dict) -> None:
    now = utc_now()
    with connect() as connection:
        connection.execute("""
            INSERT INTO jobs (id, mode, prompt, duration, language, style, voice, status, stage,
                              progress, total, created_at, updated_at, heartbeat_at, stage_started_at, fast_preview, warning, stage_status)
            VALUES (:id, :mode, :prompt, :duration, :language, :style, :voice, 'queued', 'Queued', 0, 0, :created_at, :updated_at, NULL, :created_at, :fast_preview, NULL, 'pending')
        """, {**job, "fast_preview": int(bool(job.get("fast_preview"))), "created_at": now, "updated_at": now})


def recover_interrupted_jobs(timeout_seconds: int = 900) -> list[str]:
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    cutoff = (now - timedelta(seconds=timeout_seconds)).isoformat()
    with connect() as connection:
        connection.execute(
            "UPDATE jobs SET status = 'failed', stage = 'Render timed out', "
            "error = 'Render worker heartbeat expired before server restart; cached work is preserved.', "
            "stage_status = 'failed', updated_at = ?, heartbeat_at = NULL WHERE status = 'running' AND COALESCE(heartbeat_at, updated_at) < ?",
            (now_text, cutoff),
        )
        connection.execute(
            "UPDATE jobs SET status = 'queued', stage = 'Recovered after server restart', error = NULL, "
            "stage_status = 'pending', warning = NULL, updated_at = ?, heartbeat_at = NULL, stage_started_at = ? WHERE status = 'running'",
            (now_text, now_text),
        )
        rows = connection.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at").fetchall()
    return [row["id"] for row in rows]


def claim_next_queued_job() -> str | None:
    now = utc_now()
    with connect() as connection:
        row = connection.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY fast_preview DESC, created_at LIMIT 1").fetchone()
        if row is None:
            return None
        job_id = row["id"]
        claimed = connection.execute(
            "UPDATE jobs SET status = 'running', stage = 'Starting render', stage_status = 'running', error = NULL, "
            "updated_at = ?, heartbeat_at = ?, stage_started_at = ? WHERE id = ? AND status = 'queued'",
            (now, now, now, job_id),
        ).rowcount
    return job_id if claimed else None


def heartbeat_job(job_id: str) -> None:
    now = utc_now()
    with connect() as connection:
        connection.execute(
            "UPDATE jobs SET heartbeat_at = ?, updated_at = ? WHERE id = ? AND status = 'running'",
            (now, now, job_id),
        )


def requeue_job(job_id: str, stage: str = "Queued for retry") -> None:
    now = utc_now()
    with connect() as connection:
        connection.execute(
            "UPDATE jobs SET status = 'queued', stage = ?, stage_status = 'pending', error = NULL, warning = NULL, updated_at = ?, heartbeat_at = NULL, stage_started_at = ? WHERE id = ?",
            (stage, now, now, job_id),
        )


def queue_state() -> dict[str, int]:
    with connect() as connection:
        rows = connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
    return {row["status"]: row["count"] for row in rows}


def update_job(job_id: str, **fields) -> None:
    now = utc_now()
    with connect() as connection:
        if "stage" in fields:
            row = connection.execute("SELECT stage FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row and row["stage"] != fields["stage"]:
                fields.setdefault("stage_started_at", now)
                fields.setdefault("stage_status", "running")
        fields["updated_at"] = now
        assignments = ", ".join(f"{key} = :{key}" for key in fields)
        connection.execute(f"UPDATE jobs SET {assignments} WHERE id = :id", {**fields, "id": job_id})


def resumable_job_ids() -> list[str]:
    with connect() as connection:
        rows = connection.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at").fetchall()
    return [row["id"] for row in rows]