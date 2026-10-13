"""
DrainMind — Image Inspection Page
Upload or capture an image for computer vision-based drainage analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import pandas as pd

from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import (
    list_locations, get_or_create_location, create_inspection,
    save_observation, save_detections, save_artifact,
    get_previous_coverage
)
from src.vision.detector import get_detector
from src.vision.segmenter import compute_masks
from src.analysis.coverage import (
    compute_coverage, define_central_roi, estimate_coverage_from_detections
)
from src.analysis.priority_score import compute_priority_score
from src.analysis.condition_assessment import classify_detections, assess_quality
from src.vision.visualization import compose_annotated_image, score_color
from src.vision.preprocessing import (
    load_image_from_bytes, numpy_to_pil, image_to_bytes, save_image
)
from src.utils.validation import validate_uploaded_file, sanitize_filename
from src.utils.paths import ensure_dir, timestamped_filename

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Image Inspection — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Image Inspection</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Analyze drainage conditions from photos using computer vision & ML intelligence.</p>', unsafe_allow_html=True)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("❓ Onboarding Tour & Guide", key="side_tour_img", use_container_width=True):
        from src.utils.onboarding import show_onboarding_dialog
        show_onboarding_dialog()
    st.markdown("---")
    st.markdown("### ⚙ Vision Parameters")
    conf_val = st.slider(
        "Detection Confidence",
        min_value=0.05, max_value=0.90,
        value=float(cfg.confidence_threshold),
        step=0.05,
        help="Higher values reduce false positives; lower values detect subtle blockage."
    )
    st.markdown("---")
    st.markdown("### 📍 Location Registry")
    existing_locs = list_locations()
    loc_names = [l["name"] for l in existing_locs]
    loc_names.append("— Register New Location —")
    sel_loc = st.selectbox("Assign to Location", loc_names, index=0 if existing_locs else len(loc_names)-1)

    new_loc_name = ""
    new_loc_desc = ""
    if sel_loc == "— Register New Location —":
        new_loc_name = st.text_input("New Location Code/Name", placeholder="e.g. DRAIN-NORTH-04")
        new_loc_desc = st.text_input("Description / Landmark", placeholder="e.g. 5th Cross Road Culvert")

    notes = st.text_area("Field Inspector Notes", placeholder="e.g. Heavy leaf debris after storm.")

# ── Image Input Methods ───────────────────────────────────────────────────────
input_mode = st.radio("Select Image Source", ["📁 Upload Image File", "📷 Live Webcam Capture", "🖼 Test Sample Image"], horizontal=True)

img_bgr = None
file_source_name = "inspection.jpg"

if input_mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader(
        "Upload Drain Image",
        type=["jpg", "jpeg", "png", "webp"],
        help=f"Max size: {cfg.max_image_upload_mb} MB"
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        valid, err = validate_uploaded_file(file_bytes, uploaded_file.name, cfg.max_image_upload_mb)
        if not valid:
            st.error(f"Invalid file: {err}")
        else:
            img_bgr = load_image_from_bytes(file_bytes)
            file_source_name = uploaded_file.name

elif input_mode == "📷 Live Webcam Capture":
    camera_pic = st.camera_input("Capture live photo from camera")
    if camera_pic is not None:
        file_bytes = camera_pic.read()
        img_bgr = load_image_from_bytes(file_bytes)
        file_source_name = "webcam_capture.jpg"

else:
    sample_choice = st.selectbox(
        "Choose Synthetic / Demo Drain Scenario",
        [
            "Scenario A: Heavy Silt & Plastic Debris (Critical)",
            "Scenario B: Partial Obstruction & Stagnant Water (High)",
            "Scenario C: Minor Leaves & Clear Flow (Low / Normal)"
        ]
    )
    # Generate realistic synthetic test drain image
    h_s, w_s = 480, 640
    synth = np.full((h_s, w_s, 3), 110, dtype=np.uint8) # Concrete base
    # Drain channel walls
    cv2.fillPoly(synth, [np.array([[0, 0], [160, h_s], [0, h_s]])], (85, 85, 85))
    cv2.fillPoly(synth, [np.array([[w_s, 0], [w_s-160, h_s], [w_s, h_s]])], (85, 85, 85))
    
    if "Scenario A" in sample_choice:
        # Mud bank + plastic bottle + structural crack
        cv2.ellipse(synth, (320, 360), (190, 85), 0, 0, 360, (35, 65, 85), -1) # Silt
        cv2.rectangle(synth, (280, 300), (330, 350), (180, 150, 40), -1) # Plastic bottle
        cv2.line(synth, (120, 100), (190, 280), (25, 25, 25), 3) # Crack
    elif "Scenario B" in sample_choice:
        # Murky standing water pool + floating leaves
        cv2.ellipse(synth, (320, 380), (160, 60), 0, 0, 360, (50, 60, 50), -1) # Murky pool
        cv2.circle(synth, (290, 370), 25, (40, 110, 80), -1) # Foliage
    else:
        # Clear channel with small leaf speck
        cv2.circle(synth, (320, 380), 15, (40, 90, 60), -1)

    img_bgr = synth
    file_source_name = f"{sample_choice.split(':')[0].strip().replace(' ', '_').lower()}.jpg"

# ── Analysis Trigger ─────────────────────────────────────────────────────────
if img_bgr is not None:
    col_prev, col_act = st.columns([3, 1])
    with col_prev:
        h, w = img_bgr.shape[:2]
        st.markdown(f'<div class="mono" style="margin-bottom:8px">Ready for analysis • Resolution: {w}×{h}px</div>', unsafe_allow_html=True)
    with col_act:
        if st.button("▶ Run Inspection", type="primary", use_container_width=True, key="btn_run_analysis"):
            # Determine location
            if sel_loc == "— Register New Location —":
                if not new_loc_name or not new_loc_name.strip():
                    st.error("Please enter a location name before running analysis.")
                    st.stop()
                loc_id = get_or_create_location(new_loc_name.strip(), new_loc_desc or "")
                loc_label = new_loc_name.strip()
            else:
                loc_dict = {l["name"]: l["location_id"] for l in existing_locs}
                loc_id = loc_dict[sel_loc]
                loc_label = sel_loc

            with st.spinner("Processing drain intelligence pipeline…"):
                prog = st.progress(15, text="Detecting objects & blockages…")
                detector = get_detector()
                detections, backend = detector.detect(img_bgr, confidence=float(conf_val))

                prog.progress(45, text="Segmenting silt, debris, and water surfaces…")
                mask_results = compute_masks(img_bgr)
                debris_mask = mask_results.get("debris_mask")
                water_mask = mask_results.get("water_mask")
                crack_mask = mask_results.get("crack_mask")

                prog.progress(70, text="Calculating ROI coverage metrics…")
                roi = define_central_roi(img_bgr.shape)
                raw_deb_cov = compute_coverage(debris_mask, roi, img_bgr.shape) or 0.0
                raw_wat_cov = compute_coverage(water_mask, roi, img_bgr.shape) or 0.0

                # Combine segmented debris coverage with detection box area for high accuracy
                if detections:
                    box_cov_info = estimate_coverage_from_detections(detections, img_bgr.shape, roi)
                    box_debris = box_cov_info.get("debris_coverage") or 0.0
                    debris_cov = min(100.0, max(raw_deb_cov, box_debris))
                else:
                    debris_cov = raw_deb_cov

                water_cov = min(100.0, raw_wat_cov)
                prev_cov = get_previous_coverage(loc_id)
                trend_s = (debris_cov - prev_cov) if prev_cov is not None else 0.0

                prog.progress(85, text="Synthesizing multi-factor priority score…")
                score_result = compute_priority_score(
                    debris_coverage=debris_cov,
                    water_coverage=water_cov,
                    trend_score=max(0.0, trend_s),
                )

                prog.progress(95, text="Composing visual annotations…")
                ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                coverage_summary = {
                    "debris_coverage": debris_cov,
                    "water_coverage": water_cov,
                    "method": "drain_intelligence_v2",
                    "quality": "high_precision"
                }
                annotated = compose_annotated_image(
                    img_bgr, detections, coverage_summary,
                    score_result.get("score"), score_result.get("category"),
                    location_name=loc_label, timestamp=ts_str
                )
                prog.progress(100, text="Complete!")

                # Store result in session state so it survives Streamlit reruns
                st.session_state["active_inspection_analysis"] = {
                    "img_bgr": img_bgr,
                    "annotated": annotated,
                    "detections": detections,
                    "debris_cov": debris_cov,
                    "water_cov": water_cov,
                    "score_result": score_result,
                    "coverage_summary": coverage_summary,
                    "location_id": loc_id,
                    "location_name": loc_label,
                    "timestamp": ts_str,
                    "file_source_name": file_source_name,
                    "backend": backend,
                    "notes": notes,
                    "saved": False
                }

# ── Render Inspection Results ────────────────────────────────────────────────
if "active_inspection_analysis" in st.session_state and st.session_state["active_inspection_analysis"] is not None:
    res = st.session_state["active_inspection_analysis"]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">📊 Inspection Analysis & Telemetry</h2>', unsafe_allow_html=True)

    col_visual, col_telemetry = st.columns([1.25, 1.0], gap="large")

    score = res["score_result"].get("score", 0.0)
    cat = res["score_result"].get("category", "Unknown")
    color = score_color(score)

    with col_visual:
        st.image(numpy_to_pil(res["annotated"]), caption=f"Annotated Inspection • {res['location_name']}", use_container_width=True)
        
        # Action Buttons directly under the visual
        b_dl, b_save = st.columns(2)
        with b_dl:
            img_bytes = image_to_bytes(res["annotated"], ".png")
            if img_bytes:
                st.download_button(
                    "⬇ Download Annotated Image",
                    data=img_bytes,
                    file_name=timestamped_filename(f"drain_{res['location_name']}", "png"),
                    mime="image/png",
                    use_container_width=True,
                    key="btn_dl_img"
                )
        with b_save:
            if not res["saved"]:
                if st.button("💾 Save to Database", type="primary", use_container_width=True, key="btn_commit_save"):
                    try:
                        ann_dir = ensure_dir(cfg.annotated_dir)
                        ann_fn = timestamped_filename(f"annotated_{res['location_name']}", "png")
                        ann_path = str(ann_dir / ann_fn)
                        save_image(res["annotated"], ann_path)

                        raw_dir = ensure_dir(Path(cfg.annotated_dir).parent / "raw")
                        raw_fn = timestamped_filename(f"raw_{res['location_name']}", "png")
                        raw_path = str(raw_dir / raw_fn)
                        save_image(res["img_bgr"], raw_path)

                        iid = create_inspection(
                            location_id=res["location_id"],
                            source_type="image",
                            notes=notes or res["notes"] or "",
                            model_name=res["backend"],
                            model_version="v2.0-ensemble",
                            processing_status="complete",
                            assessment_quality="high_precision",
                            timestamp=res["timestamp"],
                        )
                        save_observation(
                            inspection_id=iid,
                            debris_coverage=res["debris_cov"],
                            water_coverage=res["water_cov"],
                            priority_score=res["score_result"].get("score"),
                            priority_category=res["score_result"].get("category"),
                            trend_score=0.0,
                            alignment_quality="not_applicable",
                        )
                        if res["detections"]:
                            save_detections(iid, res["detections"])
                        save_artifact(iid, "annotated", ann_path)
                        save_artifact(iid, "original", raw_path)

                        res["saved"] = True
                        st.session_state["active_inspection_analysis"] = res
                        st.success(f"✓ Saved to Database (ID: {iid[:8]}…)")
                    except Exception as ex:
                        st.error(f"Failed to save inspection: {ex}")
            else:
                st.success("✓ Saved in Database")

    with col_telemetry:
        # Priority Score Card
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {color}; margin-bottom: 14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="kpi-label">Composite Risk Assessment</span>
                    <span style="display:inline-block; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.75rem; background:{color}18; color:{color}; border:1px solid {color}33;">
                        {cat.upper()}
                    </span>
                </div>
                <div style="display:flex; align-items:baseline; gap:8px; margin: 8px 0 4px 0;">
                    <span style="font-size:2.8rem; font-weight:800; color:{color}; line-height:1;">{score:.1f}</span>
                    <span style="font-size:1rem; font-weight:600; color:var(--text-muted);">/ 100</span>
                </div>
                <div style="font-size:0.84rem; color:var(--text-muted); line-height:1.4;">{res["score_result"].get("note", "")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2x2 Telemetry Grid
        t_row1_c1, t_row1_c2 = st.columns(2)
        with t_row1_c1:
            st.metric("Blockage & Silt", f"{res['debris_cov']:.1f}%")
        with t_row1_c2:
            st.metric("Water Level", f"{res['water_cov']:.1f}%")

        t_row2_c1, t_row2_c2 = st.columns(2)
        with t_row2_c1:
            det_cnt = len(res["detections"])
            st.metric("Defects Detected", f"{det_cnt} items")
        with t_row2_c2:
            st.metric("AI Pipeline", "Ensemble")

        # Location & timestamp info
        st.markdown(
            f"""
            <div class="callout-box" style="margin-top:10px;">
                <strong>Asset Location:</strong> {res['location_name']}<br/>
                <span class="mono" style="font-size:0.75rem; color:var(--text-muted);">Timestamp: {res['timestamp']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detections table
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-heading">🎯 Identified Defect Objects & Bounding Boxes</h3>', unsafe_allow_html=True)
    if res["detections"]:
        df_d = pd.DataFrame([{
            "Defect / Item Class": d["class_name"],
            "Confidence": f"{d['confidence']*100:.1f}%",
            "Bounding Box (x1, y1, x2, y2)": f"({int(d['x1'])}, {int(d['y1'])}) → ({int(d['x2'])}, {int(d['y2'])})",
            "Source Detector": d.get("source", "vision_ensemble")
        } for d in res["detections"]])
        st.dataframe(df_d, use_container_width=True, hide_index=True)
    else:
        st.info("No discrete debris objects detected above threshold. Continuous silt/water segmentation metrics applied.")
else:
    if img_bgr is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📸</div>
                <div class="empty-title">Ready for Drain Inspection</div>
                <div class="empty-body">Upload an image, capture using your camera, or select a Test Sample Scenario above to begin.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
