"""CSV export for DrainMind."""

from __future__ import annotations

import io
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("reporting.csv_export")


def inspections_to_dataframe(inspections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of inspection dicts to a tidy DataFrame."""
    if not inspections:
        return pd.DataFrame()

    rows = []
    for r in inspections:
        rows.append({
            "inspection_id": r.get("inspection_id", ""),
            "location": r.get("location_name", r.get("location_id", "")),
            "timestamp": r.get("timestamp", ""),
            "source_type": r.get("source_type", ""),
            "model": r.get("model_name", ""),
            "processing_status": r.get("processing_status", ""),
            "assessment_quality": r.get("assessment_quality", ""),
            "debris_coverage_%": r.get("debris_coverage"),
            "water_coverage_%": r.get("water_coverage"),
            "priority_score": r.get("priority_score"),
            "priority_category": r.get("priority_category", ""),
            "trend_score": r.get("trend_score"),
            "notes": r.get("notes", ""),
        })

    return pd.DataFrame(rows)


def export_csv(
    inspections: List[Dict[str, Any]],
    include_headers: bool = True,
) -> bytes:
    """
    Export inspection records as UTF-8 CSV bytes.
    """
    df = inspections_to_dataframe(inspections)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def export_csv_file(
    inspections: List[Dict[str, Any]],
    output_path: str,
) -> bool:
    """Write CSV to file. Returns True on success."""
    try:
        df = inspections_to_dataframe(inspections)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"CSV exported to {output_path}")
        return True
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return False
