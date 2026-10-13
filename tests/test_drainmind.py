"""
DrainMind — Unit Tests
Tests for validation, coverage, priority scoring, change detection, and database.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import shutil
import sqlite3
import numpy as np
import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Return a fresh temporary SQLite database path."""
    db_path = str(tmp_path / "test_drainmind.db")
    from src.database.db import initialize_database
    initialize_database(db_path)
    return db_path


@pytest.fixture
def blank_image():
    """Return a 200×200 black BGR image."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def colourful_image():
    """Return a 200×200 BGR image with a green region."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [0, 200, 0]
    return img


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def _make_upload(self, name, size_bytes=1024):
        """Create a mock file-like object."""
        import io
        buf = io.BytesIO(b"X" * size_bytes)
        buf.name = name
        buf.size = size_bytes
        buf.seek(0)
        return buf

    def test_valid_jpeg(self):
        from src.utils.validation import validate_image_file
        f = self._make_upload("drain.jpg", 100_000)
        ok, err = validate_image_file(f, max_size_mb=20)
        assert ok, err

    def test_invalid_extension(self):
        from src.utils.validation import validate_image_file
        f = self._make_upload("drain.gif", 100)
        ok, err = validate_image_file(f, max_size_mb=20)
        assert not ok
        assert "gif" in err.lower() or "unsupported" in err.lower()

    def test_too_large(self):
        from src.utils.validation import validate_image_file
        f = self._make_upload("drain.jpg", 30 * 1024 * 1024)  # 30 MB
        ok, err = validate_image_file(f, max_size_mb=20)
        assert not ok
        assert "exceed" in err.lower() or "size" in err.lower()

    def test_none_file(self):
        from src.utils.validation import validate_image_file
        ok, err = validate_image_file(None)
        assert not ok

    def test_valid_video_mp4(self):
        from src.utils.validation import validate_video_file
        f = self._make_upload("video.mp4", 10_000_000)
        ok, err = validate_video_file(f, max_size_mb=200)
        assert ok, err

    def test_invalid_video_extension(self):
        from src.utils.validation import validate_video_file
        f = self._make_upload("video.wmv", 100)
        ok, err = validate_video_file(f, max_size_mb=200)
        assert not ok

    def test_confidence_valid(self):
        from src.utils.validation import validate_confidence
        ok, _ = validate_confidence(0.5)
        assert ok

    def test_confidence_zero(self):
        from src.utils.validation import validate_confidence
        ok, err = validate_confidence(0.0)
        assert not ok

    def test_confidence_above_one(self):
        from src.utils.validation import validate_confidence
        ok, err = validate_confidence(1.5)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverage:
    def test_full_mask_coverage(self):
        from src.analysis.coverage import compute_coverage, define_central_roi
        img_shape = (100, 100)
        roi = define_central_roi(img_shape, fraction=0.5)
        mask = np.zeros((100, 100), dtype=np.uint8)
        x1, y1, x2, y2 = roi
        mask[y1:y2, x1:x2] = 255  # fill entire ROI
        cov = compute_coverage(mask, roi, img_shape)
        assert cov is not None
        assert abs(cov - 100.0) < 0.5

    def test_empty_mask_coverage(self):
        from src.analysis.coverage import compute_coverage, define_central_roi
        img_shape = (100, 100)
        roi = define_central_roi(img_shape)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cov = compute_coverage(mask, roi, img_shape)
        assert cov == 0.0 or cov is None or cov == pytest.approx(0.0)

    def test_none_mask_returns_none(self):
        from src.analysis.coverage import compute_coverage
        cov = compute_coverage(None, (10, 10, 90, 90), (100, 100))
        assert cov is None

    def test_zero_roi_area_returns_none(self):
        from src.analysis.coverage import compute_coverage
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        cov = compute_coverage(mask, (50, 50, 50, 50), (100, 100))  # zero area
        assert cov is None

    def test_partial_mask(self):
        from src.analysis.coverage import compute_coverage, define_central_roi
        img_shape = (100, 100)
        roi = (25, 25, 75, 75)  # 50×50 = 2500 px
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:50, 25:75] = 255  # fill top half of ROI
        cov = compute_coverage(mask, roi, img_shape)
        assert cov is not None
        assert 40 <= cov <= 60  # approximately 50%

    def test_coverage_clamp_to_100(self):
        from src.analysis.coverage import compute_coverage
        img_shape = (100, 100)
        roi = (10, 10, 90, 90)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        cov = compute_coverage(mask, roi, img_shape)
        assert cov is not None
        assert cov <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

class TestPriorityScore:
    def test_all_components_available(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score(debris_coverage=80, water_coverage=60, trend_score=70)
        assert r["score"] is not None
        assert 0 <= r["score"] <= 100
        assert r["is_complete"] is True
        assert len(r["missing"]) == 0

    def test_all_zero_gives_zero(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score(debris_coverage=0, water_coverage=0, trend_score=0)
        assert r["score"] == pytest.approx(0.0)

    def test_all_hundred_gives_hundred(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score(debris_coverage=100, water_coverage=100, trend_score=100)
        assert r["score"] == pytest.approx(100.0)

    def test_missing_water_renormalises(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score(debris_coverage=100, water_coverage=None, trend_score=100)
        assert r["is_complete"] is False
        assert "water_coverage" in r["missing"]
        assert r["score"] is not None
        assert 0 <= r["score"] <= 100

    def test_all_missing_returns_none(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score()
        assert r["score"] is None
        assert r["category"] is None

    def test_categories(self):
        from src.analysis.priority_score import compute_priority_score
        # ≥75 → High.  90×0.5 + 90×0.3 + 90×0.2 = 90
        assert compute_priority_score(debris_coverage=90, water_coverage=90, trend_score=90)["category"] == "High inspection priority"
        # ≥50 → Inspection recommended.  80×0.5 + 60×0.3 + 40×0.2 = 40+18+8 = 66
        assert compute_priority_score(debris_coverage=80, water_coverage=60, trend_score=40)["category"] == "Inspection recommended"
        # ≥25 → Monitor.  40×0.5 + 20×0.3 + 10×0.2 = 20+6+2 = 28
        assert compute_priority_score(debris_coverage=40, water_coverage=20, trend_score=10)["category"] == "Monitor"
        # <25 → Lower.  5×0.5 + 5×0.3 + 5×0.2 = 5
        assert compute_priority_score(debris_coverage=5, water_coverage=5, trend_score=5)["category"] == "Lower inspection priority"

    def test_score_clamps_at_100(self):
        from src.analysis.priority_score import compute_priority_score
        r = compute_priority_score(debris_coverage=200, water_coverage=200, trend_score=200)
        assert r["score"] <= 100.0

    def test_trend_score_computation(self):
        from src.analysis.priority_score import compute_trend_score
        ts = compute_trend_score(current_coverage=70, previous_coverage=50)
        assert ts is not None
        assert ts > 50  # worsening => above midpoint

    def test_trend_score_none_inputs(self):
        from src.analysis.priority_score import compute_trend_score
        assert compute_trend_score(None, 50) is None
        assert compute_trend_score(50, None) is None
        assert compute_trend_score(None, None) is None


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeDetection:
    def test_worsening_direction(self):
        from src.analysis.change_detection import compute_coverage_change
        r = compute_coverage_change(70, 40)
        assert r["direction"] == "worsening"
        assert r["change"] == pytest.approx(30.0)

    def test_improving_direction(self):
        from src.analysis.change_detection import compute_coverage_change
        r = compute_coverage_change(20, 50)
        assert r["direction"] == "improving"
        assert r["change"] == pytest.approx(-30.0)

    def test_stable_direction(self):
        from src.analysis.change_detection import compute_coverage_change
        r = compute_coverage_change(50.5, 50.0)
        assert r["direction"] == "stable"

    def test_none_inputs(self):
        from src.analysis.change_detection import compute_coverage_change
        r = compute_coverage_change(None, 40)
        assert r["reliable"] is False
        r2 = compute_coverage_change(40, None)
        assert r2["reliable"] is False

    def test_alignment_insufficient_keypoints(self):
        from src.analysis.change_detection import try_align_images
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        warped, info = try_align_images(blank, blank)
        assert warped is None
        assert info["aligned"] is False


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def _get_conn(self, db_path):
        from src.database.db import get_connection
        return get_connection(db_path)

    def test_initialize_creates_tables(self, tmp_db):
        conn = self._get_conn(tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "locations" in tables
        assert "inspections" in tables
        assert "observations" in tables
        assert "detections" in tables
        assert "artifacts" in tables

    def test_create_and_retrieve_location(self, tmp_db):
        # Override DB connection to use tmp_db
        from src.database.db import _local
        from src.database.db import get_connection, initialize_database
        # Force connection for this path
        conn = get_connection(tmp_db)
        conn.execute(
            "INSERT INTO locations (location_id, name, description, created_at) VALUES (?,?,?,?)",
            ("test-loc-1", "Test Drain", "A test drain", "2024-01-01T00:00:00Z")
        )
        conn.commit()
        row = conn.execute("SELECT * FROM locations WHERE location_id=?", ("test-loc-1",)).fetchone()
        assert row is not None
        assert row["name"] == "Test Drain"

    def test_foreign_key_cascade_delete(self, tmp_db):
        conn = self._get_conn(tmp_db)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO locations (location_id, name, description, created_at) VALUES (?,?,?,?)",
            ("loc-2", "Loc2", "", "2024-01-01T00:00:00Z")
        )
        conn.execute(
            "INSERT INTO inspections (inspection_id, location_id, timestamp, source_type, processing_status, assessment_quality)"
            " VALUES (?,?,?,?,?,?)",
            ("insp-2", "loc-2", "2024-01-01T00:00:00Z", "image", "complete", "good")
        )
        conn.execute(
            "INSERT INTO observations (observation_id, inspection_id, alignment_quality) VALUES (?,?,?)",
            ("obs-2", "insp-2", "not_applicable")
        )
        conn.commit()
        conn.execute("DELETE FROM inspections WHERE inspection_id=?", ("insp-2",))
        conn.commit()
        row = conn.execute("SELECT * FROM observations WHERE inspection_id=?", ("insp-2",)).fetchone()
        assert row is None

    def test_health_check_ok(self, tmp_db):
        from src.database.db import check_database_health
        result = check_database_health(tmp_db)
        assert result["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVExport:
    def test_export_empty(self):
        from src.reporting.csv_export import export_csv
        data = export_csv([])
        assert isinstance(data, bytes)

    def test_export_records(self):
        from src.reporting.csv_export import export_csv
        records = [
            {
                "inspection_id": "abc-123",
                "location_name": "Drain A",
                "timestamp": "2024-01-01T10:00:00Z",
                "source_type": "image",
                "model_name": "yolov8n",
                "processing_status": "complete",
                "assessment_quality": "good",
                "debris_coverage": 45.2,
                "water_coverage": 10.0,
                "priority_score": 62.5,
                "priority_category": "Inspection recommended",
                "trend_score": None,
                "notes": "Test note",
            }
        ]
        csv_bytes = export_csv(records)
        text = csv_bytes.decode("utf-8")
        assert "Drain A" in text
        assert "62.5" in text
        assert "inspection_id" in text


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_resize_large_image(self):
        from src.vision.preprocessing import resize_for_inference
        big = np.zeros((1200, 1600, 3), dtype=np.uint8)
        small = resize_for_inference(big, target_size=640)
        h, w = small.shape[:2]
        assert max(h, w) <= 640

    def test_resize_small_image_unchanged(self):
        from src.vision.preprocessing import resize_for_inference
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        out = resize_for_inference(img, target_size=640)
        assert out.shape == img.shape

    def test_bgr_rgb_roundtrip(self):
        from src.vision.preprocessing import bgr_to_rgb, rgb_to_bgr
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        assert np.array_equal(img, rgb_to_bgr(bgr_to_rgb(img)))

    def test_image_to_bytes(self):
        from src.vision.preprocessing import image_to_bytes
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b = image_to_bytes(img, ".png")
        assert b is not None and len(b) > 0

    def test_save_and_load(self, tmp_path):
        from src.vision.preprocessing import save_image, load_image_from_path
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[10:40, 10:40] = [255, 0, 0]
        path = str(tmp_path / "test.png")
        ok = save_image(img, path)
        assert ok
        loaded = load_image_from_path(path)
        assert loaded is not None
        assert loaded.shape == img.shape


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

class TestPaths:
    def test_safe_filename(self):
        from src.utils.paths import safe_filename
        assert "/" not in safe_filename("../../etc/passwd")
        assert "\\" not in safe_filename("C:\\Windows\\system32")

    def test_timestamped_filename_format(self):
        from src.utils.paths import timestamped_filename
        fn = timestamped_filename("test", "png")
        assert fn.startswith("test_")
        assert fn.endswith(".png")

    def test_path_traversal_prevented(self, tmp_path):
        """safe_output_path sanitizes filenames so traversal sequences become safe names."""
        from src.utils.paths import safe_output_path, is_within
        from pathlib import Path
        # The function sanitizes the filename (replaces / \ . with _), then checks is_within.
        # '../../etc/passwd' → safe name like '______etc_passwd' which IS within base_dir.
        result = safe_output_path(str(tmp_path), "../../etc/passwd")
        # The returned path must be inside tmp_path
        assert is_within(Path(result), tmp_path)
        # And must not contain traversal components
        assert ".." not in str(result)
