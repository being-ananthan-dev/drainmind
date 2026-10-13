"""Database initialization and connection management for DrainMind."""

import sqlite3
import threading
from pathlib import Path
from typing import Optional
from src.utils.logging_config import get_logger

logger = get_logger("database.db")

# Thread-local storage for connections
_local = threading.local()

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS locations (
    location_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inspections (
    inspection_id     TEXT PRIMARY KEY,
    location_id       TEXT NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    timestamp         TEXT NOT NULL,
    source_type       TEXT NOT NULL CHECK(source_type IN ('image','uploaded_video','webcam')),
    notes             TEXT DEFAULT '',
    model_name        TEXT DEFAULT '',
    model_version     TEXT DEFAULT '',
    processing_status TEXT NOT NULL DEFAULT 'pending'
                      CHECK(processing_status IN ('pending','processing','complete','failed')),
    assessment_quality TEXT DEFAULT 'unknown'
                      CHECK(assessment_quality IN ('good','acceptable','low','unknown'))
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id  TEXT PRIMARY KEY,
    inspection_id   TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    debris_coverage REAL,
    water_coverage  REAL,
    priority_score  REAL,
    priority_category TEXT,
    trend_score     REAL,
    alignment_quality TEXT DEFAULT 'not_applicable'
                      CHECK(alignment_quality IN ('good','acceptable','poor','not_applicable'))
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id  TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    class_name    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    x1            REAL NOT NULL,
    y1            REAL NOT NULL,
    x2            REAL NOT NULL,
    y2            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id   TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL CHECK(artifact_type IN ('original','annotated','report','video_summary')),
    file_path     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inspections_location ON inspections(location_id);
CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_inspection ON observations(inspection_id);
CREATE INDEX IF NOT EXISTS idx_detections_inspection ON detections(inspection_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_inspection ON artifacts(inspection_id);
"""


def get_db_path() -> str:
    from src.config import get_config
    return get_config().db_path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection."""
    if db_path is None:
        db_path = get_db_path()

    if not hasattr(_local, "connections"):
        _local.connections = {}

    if db_path not in _local.connections:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        _local.connections[db_path] = conn
        logger.debug(f"Opened new DB connection: {db_path}")

    return _local.connections[db_path]


def initialize_database(db_path: Optional[str] = None):
    """Create all tables if they do not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def check_database_health(db_path: Optional[str] = None) -> dict:
    """Return diagnostic information about the database."""
    result = {"status": "unknown", "path": db_path or get_db_path(), "error": None}
    try:
        conn = get_connection(db_path)
        cur = conn.execute("SELECT COUNT(*) FROM inspections")
        count = cur.fetchone()[0]
        result["status"] = "ok"
        result["inspection_count"] = count
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result
