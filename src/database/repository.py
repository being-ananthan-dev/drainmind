"""Database repository functions for DrainMind."""

import uuid
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from src.database.db import get_connection
from src.utils.logging_config import get_logger

logger = get_logger("database.repository")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uid() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────
# LOCATIONS
# ──────────────────────────────────────────────────────────────

def create_location(name: str, description: str = "") -> str:
    """Create a new location. Returns location_id."""
    conn = get_connection()
    lid = _uid()
    conn.execute(
        "INSERT INTO locations (location_id, name, description, created_at) VALUES (?,?,?,?)",
        (lid, name.strip(), description.strip(), _now_iso())
    )
    conn.commit()
    logger.info(f"Created location '{name}' id={lid}")
    return lid


def get_or_create_location(name: str, description: str = "") -> str:
    """Return existing location_id or create if absent."""
    conn = get_connection()
    row = conn.execute(
        "SELECT location_id FROM locations WHERE name = ?", (name.strip(),)
    ).fetchone()
    if row:
        return row["location_id"]
    return create_location(name, description)


def list_locations() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT location_id, name, description, created_at FROM locations ORDER BY name"
    ).fetchall()
    return [dict(r) for r in rows]


def get_location(location_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM locations WHERE location_id = ?", (location_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_location(location_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM locations WHERE location_id = ?", (location_id,))
    conn.commit()


# ──────────────────────────────────────────────────────────────
# INSPECTIONS
# ──────────────────────────────────────────────────────────────

def create_inspection(
    location_id: str,
    source_type: str,
    notes: str = "",
    model_name: str = "",
    model_version: str = "",
    processing_status: str = "complete",
    assessment_quality: str = "unknown",
    timestamp: Optional[str] = None
) -> str:
    conn = get_connection()
    iid = _uid()
    ts = timestamp or _now_iso()

    # Normalize fields to satisfy SQLite CHECK constraints
    st_norm = str(source_type or "").lower()
    if "video" in st_norm:
        st_val = "uploaded_video"
    elif "webcam" in st_norm or "camera" in st_norm:
        st_val = "webcam"
    else:
        st_val = "image"

    ps_norm = str(processing_status or "").lower()
    ps_val = ps_norm if ps_norm in ('pending', 'processing', 'complete', 'failed') else 'complete'

    aq_norm = str(assessment_quality or "").lower()
    aq_val = aq_norm if aq_norm in ('good', 'acceptable', 'low', 'unknown') else 'good'

    conn.execute(
        """INSERT INTO inspections
           (inspection_id, location_id, timestamp, source_type, notes,
            model_name, model_version, processing_status, assessment_quality)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (iid, location_id, ts, st_val, notes,
         model_name, model_version, ps_val, aq_val)
    )
    conn.commit()
    logger.info(f"Created inspection id={iid} location={location_id}")
    return iid


def update_inspection_status(inspection_id: str, status: str, quality: str = None):
    conn = get_connection()
    if quality:
        conn.execute(
            "UPDATE inspections SET processing_status=?, assessment_quality=? WHERE inspection_id=?",
            (status, quality, inspection_id)
        )
    else:
        conn.execute(
            "UPDATE inspections SET processing_status=? WHERE inspection_id=?",
            (status, inspection_id)
        )
    conn.commit()


def get_inspection(inspection_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        """SELECT i.*, l.name as location_name
           FROM inspections i JOIN locations l ON i.location_id=l.location_id
           WHERE i.inspection_id=?""",
        (inspection_id,)
    ).fetchone()
    return dict(row) if row else None


def list_inspections(
    location_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = 200,
    offset: int = 0,
    order_by: str = "timestamp DESC"
) -> List[Dict[str, Any]]:
    conn = get_connection()
    sql = """
        SELECT i.inspection_id, i.location_id, l.name as location_name,
               i.timestamp, i.source_type, i.notes, i.model_name, i.model_version,
               i.processing_status, i.assessment_quality,
               o.debris_coverage, o.water_coverage, o.priority_score,
               o.priority_category, o.trend_score, o.alignment_quality
        FROM inspections i
        JOIN locations l ON i.location_id = l.location_id
        LEFT JOIN observations o ON o.inspection_id = i.inspection_id
        WHERE 1=1
    """
    params = []

    if location_id:
        sql += " AND i.location_id = ?"
        params.append(location_id)
    if date_from:
        sql += " AND i.timestamp >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND i.timestamp <= ?"
        params.append(date_to + "T23:59:59Z")
    if status:
        sql += " AND i.processing_status = ?"
        params.append(status)
    if min_score is not None:
        sql += " AND o.priority_score >= ?"
        params.append(min_score)
    if max_score is not None:
        sql += " AND o.priority_score <= ?"
        params.append(max_score)

    valid_orders = {
        "timestamp DESC": "i.timestamp DESC",
        "timestamp ASC": "i.timestamp ASC",
        "priority_score DESC": "o.priority_score DESC",
        "priority_score ASC": "o.priority_score ASC",
    }
    order_clause = valid_orders.get(order_by, "i.timestamp DESC")
    sql += f" ORDER BY {order_clause} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_inspections(**kwargs) -> int:
    rows = list_inspections(**{**kwargs, "limit": 100000, "offset": 0})
    return len(rows)


def delete_inspection(inspection_id: str):
    """Delete inspection and all cascaded records."""
    conn = get_connection()
    # Get artifacts so we can optionally delete files
    artifacts = conn.execute(
        "SELECT file_path FROM artifacts WHERE inspection_id=?", (inspection_id,)
    ).fetchall()
    conn.execute("DELETE FROM inspections WHERE inspection_id=?", (inspection_id,))
    conn.commit()
    logger.info(f"Deleted inspection {inspection_id}")
    return [r["file_path"] for r in artifacts]


# ──────────────────────────────────────────────────────────────
# OBSERVATIONS
# ──────────────────────────────────────────────────────────────

def save_observation(
    inspection_id: str,
    debris_coverage: Optional[float] = None,
    water_coverage: Optional[float] = None,
    priority_score: Optional[float] = None,
    priority_category: Optional[str] = None,
    trend_score: Optional[float] = None,
    alignment_quality: str = "not_applicable"
) -> str:
    conn = get_connection()
    oid = _uid()
    aq_norm = str(alignment_quality or "not_applicable").lower()
    aq_val = aq_norm if aq_norm in ('good', 'acceptable', 'poor', 'not_applicable') else 'not_applicable'

    conn.execute(
        """INSERT INTO observations
           (observation_id, inspection_id, debris_coverage, water_coverage,
            priority_score, priority_category, trend_score, alignment_quality)
           VALUES (?,?,?,?,?,?,?,?)""",
        (oid, inspection_id, debris_coverage, water_coverage,
         priority_score, priority_category, trend_score, aq_val)
    )
    conn.commit()
    return oid


def get_observation(inspection_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM observations WHERE inspection_id=?", (inspection_id,)
    ).fetchone()
    return dict(row) if row else None


# ──────────────────────────────────────────────────────────────
# DETECTIONS
# ──────────────────────────────────────────────────────────────

def save_detections(inspection_id: str, detections: List[Dict[str, Any]]):
    conn = get_connection()
    for d in detections:
        conn.execute(
            """INSERT INTO detections
               (detection_id, inspection_id, class_name, confidence, x1, y1, x2, y2)
               VALUES (?,?,?,?,?,?,?,?)""",
            (_uid(), inspection_id, d["class_name"], d["confidence"],
             d["x1"], d["y1"], d["x2"], d["y2"])
        )
    conn.commit()


def get_detections(inspection_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE inspection_id=? ORDER BY confidence DESC",
        (inspection_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# ARTIFACTS
# ──────────────────────────────────────────────────────────────

def save_artifact(inspection_id: str, artifact_type: str, file_path: str) -> str:
    conn = get_connection()
    aid = _uid()
    conn.execute(
        "INSERT INTO artifacts (artifact_id, inspection_id, artifact_type, file_path) VALUES (?,?,?,?)",
        (aid, inspection_id, artifact_type, file_path)
    )
    conn.commit()
    return aid


def get_artifacts(inspection_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM artifacts WHERE inspection_id=?", (inspection_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# DASHBOARD STATS
# ──────────────────────────────────────────────────────────────

def get_dashboard_stats() -> Dict[str, Any]:
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    locations = conn.execute("SELECT COUNT(DISTINCT location_id) FROM inspections").fetchone()[0]
    if locations == 0:
        locations = conn.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
    needs_inspection = conn.execute(
        "SELECT COUNT(DISTINCT i.location_id) FROM inspections i "
        "JOIN observations o ON i.inspection_id=o.inspection_id "
        "WHERE o.priority_score >= 50"
    ).fetchone()[0]
    worsening = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE trend_score > 5"
    ).fetchone()[0]

    return {
        "total_inspections": total,
        "locations_monitored": locations,
        "needs_inspection": needs_inspection,
        "recent_worsening": worsening,
    }


def seed_demo_data():
    """Populate realistic sample drainage inspection records for immediate testing & demo."""
    conn = get_connection()
    import uuid
    from datetime import datetime, timedelta, timezone

    sample_locations = [
        ("LOC-NORTH-CULVERT", "North Ring Road Box Culvert", "Main storm drain at Sector 4"),
        ("LOC-EAST-CANAL", "East Industrial Canal Inflow", "Open trapezoidal drainage trench"),
        ("LOC-CENTRAL-PIPE", "Central Commercial Underground Pipe", "900mm reinforced concrete stormwater pipe"),
        ("LOC-SOUTH-OUTFALL", "South Basin Retention Outfall", "Floodwater retention spillway channel"),
        ("LOC-WEST-JUNCTION", "West Highway Culvert Junction", "High-velocity runoff bypass drain"),
    ]

    loc_ids = []
    for code, name, desc in sample_locations:
        lid = get_or_create_location(name, desc)
        loc_ids.append((lid, name))

    # Add historical and current inspections across the last 14 days
    now = datetime.now(timezone.utc)
    demo_records = [
        # loc_idx, days_ago, source, deb_cov, wat_cov, score, cat, trend, notes
        (0, 12, "image", 28.5, 12.0, 31.0, "Monitor", 0.0, "Routine pre-monsoon survey"),
        (0, 1, "image", 78.4, 45.2, 74.2, "High inspection priority", 49.9, "Severe silt buildup & plastic accumulation post-downpour"),
        (1, 10, "video", 55.0, 30.0, 52.5, "Inspection recommended", 0.0, "Pipe crawler automated video inspection"),
        (1, 2, "video", 82.5, 65.0, 81.0, "High inspection priority", 27.5, "Heavy vegetative blockage & silt banking detected"),
        (2, 7, "image", 15.0, 10.0, 18.0, "Lower inspection priority", 0.0, "Clear channel flow; routine check"),
        (3, 8, "image", 42.0, 50.0, 48.0, "Monitor", 0.0, "Standing water detected near outfall weir"),
        (3, 0, "image", 64.0, 72.0, 68.5, "Inspection recommended", 22.0, "Submerged debris barrier forming; cleaning dispatch recommended"),
        (4, 5, "video", 22.0, 8.0, 21.0, "Lower inspection priority", 0.0, "Bypass junction clean and operational"),
    ]

    for loc_idx, days_ago, stype, deb, wat, sc, cat, tr, note in demo_records:
        lid, lname = loc_ids[loc_idx]
        ts = (now - timedelta(days=days_ago, hours=loc_idx*2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        iid = create_inspection(
            location_id=lid,
            source_type=stype,
            notes=note,
            model_name="drain_vision_ensemble",
            model_version="v2.0",
            processing_status="complete",
            assessment_quality="high_precision",
            timestamp=ts
        )
        save_observation(
            inspection_id=iid,
            debris_coverage=deb,
            water_coverage=wat,
            priority_score=sc,
            priority_category=cat,
            trend_score=tr,
        )


def get_inspection_activity(days: int = 30) -> List[Dict[str, Any]]:
    """Inspection count per day for the last N days."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT substr(timestamp, 1, 10) as date, COUNT(*) as count
           FROM inspections
           WHERE timestamp >= date('now', ?)
           GROUP BY date ORDER BY date""",
        (f"-{days} days",)
    ).fetchall()
    return [dict(r) for r in rows]


def get_score_distribution() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT priority_category, COUNT(*) as count
           FROM observations
           WHERE priority_category IS NOT NULL
           GROUP BY priority_category"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_location_coverage_history(location_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT i.timestamp, o.debris_coverage, o.water_coverage, o.priority_score
           FROM inspections i JOIN observations o ON i.inspection_id=o.inspection_id
           WHERE i.location_id=?
           ORDER BY i.timestamp""",
        (location_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_previous_coverage(location_id: str) -> Optional[float]:
    """Retrieve the most recent debris_coverage for a location."""
    conn = get_connection()
    row = conn.execute(
        """SELECT o.debris_coverage
           FROM inspections i
           JOIN observations o ON i.inspection_id = o.inspection_id
           WHERE i.location_id = ? AND o.debris_coverage IS NOT NULL
           ORDER BY i.timestamp DESC LIMIT 1""",
        (location_id,)
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None
