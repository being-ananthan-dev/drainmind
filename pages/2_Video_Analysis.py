"""
DrainMind — Video Analysis Page
Processes continuous drainage video footage, CCTV feeds, or robot camera streams.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import (
    list_locations, get_or_create_location, create_inspection,
    save_observation, save_artifact
)
from src.vision.detector import get_detector
from src.video.stream_processor import process_video_file
from src.video.capture import check_webcam
from src.analysis.priority_score import compute_priority_score
from src.vision.preprocessing import numpy_to_pil, save_image
from src.utils.validation import validate_uploaded_video
from src.utils.paths import ensure_dir, timestamped_filename

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Video Analysis — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Video Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Continuous drainage inspection for CCTV, pipe crawler cameras, and aerial surveys.</p>', unsafe_allow_html=True)

tab_upload, tab_sample = st.tabs(["📹 Video File Upload", "🎬 Generate Test Drain Video"])

video_path_to_process = None
location_name = "Drain Site"
location_id = None

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("❓ Onboarding Tour & Guide", key="side_tour_vid", use_container_width=True):
        from src.utils.onboarding import show_onboarding_dialog
        show_onboarding_dialog()
    st.markdown("---")
    st.markdown("### ⚙ Video Settings")
    existing_locs = list_locations()
    loc_names = [l["name"] for l in existing_locs]
    loc_names.append("— Register New Location —")
    sel_loc = st.selectbox("Assign to Location", loc_names, key="vid_sel_loc")

    new_loc_name = ""
    if sel_loc == "— Register New Location —":
        new_loc_name = st.text_input("New Location Code/Name", placeholder="e.g. CRAWLER-LINE-02", key="vid_new_loc")

    sample_interval = st.slider(
        "Frame Sample Interval",
        min_value=2, max_value=30, value=cfg.frame_sample_interval, step=1,
        help="Process every Nth frame (lower is more granular, higher is faster)."
    )
    conf_thresh = st.slider("Detection Sensitivity", 0.05, 0.90, float(cfg.confidence_threshold), 0.05, key="vid_conf")

with tab_upload:
    uploaded_video = st.file_uploader(
        "Upload Video (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi"],
        help=f"Max size: {cfg.max_video_upload_mb} MB"
    )
    if uploaded_video is not None:
        valid, err = validate_uploaded_video(
            uploaded_video.read(), uploaded_video.name, cfg.max_video_upload_mb
        )
        uploaded_video.seek(0)
        if not valid:
            st.error(f"Invalid video file: {err}")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
            tfile.write(uploaded_video.read())
            tfile.flush()
            video_path_to_process = tfile.name

with tab_sample:
    st.markdown("Don't have drone or CCTV drain footage on hand? Click below to generate a realistic 60-frame simulated drain crawler video.")
    if st.button("Generate Simulated Drain Crawler Video", key="btn_gen_sim_vid"):
        with st.spinner("Synthesizing drain video frames…"):
            sim_path = str(Path(tempfile.gettempdir()) / "simulated_drain_crawler.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(sim_path, fourcc, 15, (640, 480))
            for i in range(60):
                f = np.full((480, 640, 3), 110, dtype=np.uint8)
                # Channel walls
                cv2.fillPoly(f, [np.array([[0, 0], [140, 480], [0, 480]])], (80, 80, 80))
                cv2.fillPoly(f, [np.array([[640, 0], [500, 480], [640, 480]])], (80, 80, 80))
                # Dynamic silt & debris building up along video timeline
                silt_radius = int(80 + (i * 2.2))
                cv2.ellipse(f, (320, 380), (silt_radius, 60), 0, 0, 360, (30, 60, 80), -1)
                if i > 20:
                    # Floating plastic debris enters frame
                    pos_x = int(220 + (i - 20) * 4)
                    cv2.rectangle(f, (pos_x, 340), (pos_x + 50, 380), (180, 150, 40), -1)
                if i > 35:
                    # Water pooling
                    cv2.ellipse(f, (320, 400), (int(silt_radius * 0.7), 35), 0, 0, 360, (50, 60, 50), -1)
                vw.write(f)
            vw.release()
            st.session_state["sim_video_ready"] = sim_path
            st.success("✓ Simulated drain video created!")

    if "sim_video_ready" in st.session_state:
        video_path_to_process = st.session_state["sim_video_ready"]
        st.info("Using simulated video for analysis.")

# ── Processing Execution ──────────────────────────────────────────────────────
if video_path_to_process is not None:
    if st.button("▶ Run Video Analysis", type="primary", key="btn_run_vid"):
        if sel_loc == "— Register New Location —":
            if not new_loc_name or not new_loc_name.strip():
                st.error("Please specify a location name in the sidebar.")
                st.stop()
            location_id = get_or_create_location(new_loc_name.strip(), "Video Inspection")
            location_name = new_loc_name.strip()
        else:
            loc_dict = {l["name"]: l["location_id"] for l in existing_locs}
            location_id = loc_dict[sel_loc]
            location_name = sel_loc

        detector = get_detector()
        prog_bar = st.progress(0, text="Initializing video decoder…")

        def on_progress(current, total, info):
            pct = min(100, int((current / max(1, total)) * 100))
            prog_bar.progress(pct, text=f"Analyzing frame {current}/{total} • Speed: {info.get('fps',0)} fps • Coverage: {info.get('current_coverage',0)}%")

        with st.spinner("Processing video feed…"):
            res = process_video_file(
                video_path_to_process,
                detector=detector,
                sample_interval=int(sample_interval),
                progress_callback=on_progress
            )

        if not res.get("success"):
            st.error(f"Video analysis error: {res.get('error')}")
        else:
            # Compute composite priority score from video statistics
            avg_cov = res.get("avg_coverage_estimate", 0.0)
            avg_wat = res.get("avg_water_estimate", 0.0)
            score_res = compute_priority_score(
                debris_coverage=avg_cov,
                water_coverage=avg_wat,
                trend_score=0.0
            )
            res["score_result"] = score_res
            res["location_id"] = location_id
            res["location_name"] = location_name
            res["saved"] = False
            st.session_state["active_video_analysis"] = res

# ── Render Video Analysis Results ────────────────────────────────────────────
if "active_video_analysis" in st.session_state and st.session_state["active_video_analysis"] is not None:
    vr = st.session_state["active_video_analysis"]
    score_info = vr.get("score_result", {})
    score_val = score_info.get("score", 0.0)
    score_cat = score_info.get("category", "Unknown")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-heading">📊 Continuous Video Inspection Telemetry</h2>', unsafe_allow_html=True)

    from src.vision.visualization import score_color
    color = score_color(score_val)

    # Risk Score & Metrics Grid
    c_score, c_m1, c_m2, c_m3, c_m4 = st.columns([1.3, 1, 1, 1, 1])
    with c_score:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {color}; padding: 14px 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="kpi-label">Risk Rating</span>
                    <span style="padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem; background:{color}18; color:{color}; border:1px solid {color}33;">
                        {score_cat.upper()}
                    </span>
                </div>
                <div style="font-size:2.2rem; font-weight:800; color:{color}; line-height:1.1; margin-top:4px;">
                    {score_val:.1f} <span style="font-size:0.9rem; color:var(--text-muted); font-weight:600;">/ 100</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_m1:
        st.metric("Frames Analyzed", f"{vr.get('frames_processed', 0)}")
    with c_m2:
        st.metric("Mean Blockage", f"{vr.get('avg_coverage_estimate', 0):.1f}%")
    with c_m3:
        st.metric("Peak Blockage", f"{vr.get('max_coverage_estimate', 0):.1f}%")
    with c_m4:
        st.metric("Water Level", f"{vr.get('avg_water_estimate', 0):.1f}%")

    # Timeline Chart
    if vr.get("coverage_history"):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-heading">📈 Progression Timeline: Blockage & Water Across Video Stream</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=vr["coverage_history"],
            mode="lines",
            name="Blockage Coverage %",
            line=dict(color="#DC2626", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.08)"
        ))
        if vr.get("water_history"):
            fig.add_trace(go.Scatter(
                y=vr["water_history"],
                mode="lines",
                name="Water Level %",
                line=dict(color="#0284C7", width=2, dash="dash")
            ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=240,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12, color="#475569"),
            xaxis=dict(title="Sampled Frame Index", showgrid=False, zeroline=False),
            yaxis=dict(title="Severity %", showgrid=True, gridcolor="#E2E8F0", zeroline=False, range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Keyframes Gallery
    st.markdown("#### 🖼 Extracted Critical Keyframes")
    k1, k2 = st.columns(2)
    with k1:
        if vr.get("max_blockage_frame") is not None:
            st.image(numpy_to_pil(vr["max_blockage_frame"]), caption=f"Peak Blockage Frame ({vr.get('max_coverage_estimate', 0)}% Coverage)", use_container_width=True)
    with k2:
        if vr.get("last_annotated_frame") is not None:
            st.image(numpy_to_pil(vr["last_annotated_frame"]), caption="Key Annotated Frame with Detected Defect Bounding Boxes", use_container_width=True)

    # Defect summary table
    if vr.get("class_summary"):
        st.markdown("#### 🎯 Detected Obstruction Frequencies")
        df_classes = pd.DataFrame([
            {"Defect / Obstruction Type": k, "Cumulative Count": v}
            for k, v in vr["class_summary"].items()
        ])
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

    # ── Save Button ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    save_col, _ = st.columns([1.5, 2.5])
    with save_col:
        if not vr.get("saved", False):
            if st.button("💾 Save Video Inspection to Database", type="primary", use_container_width=True, key="btn_save_vid_db"):
                try:
                    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    iid = create_inspection(
                        location_id=vr["location_id"],
                        source_type="video",
                        notes=f"Frames: {vr.get('frames_processed')}, Peak: {vr.get('max_coverage_estimate')}%, Avg: {vr.get('avg_coverage_estimate')}%",
                        model_name="video_drain_ensemble",
                        model_version="v2.0",
                        processing_status="complete",
                        assessment_quality="high_precision",
                        timestamp=ts_str,
                    )
                    save_observation(
                        inspection_id=iid,
                        debris_coverage=vr.get("avg_coverage_estimate"),
                        water_coverage=vr.get("avg_water_estimate"),
                        priority_score=score_val,
                        priority_category=score_cat,
                        trend_score=0.0,
                    )

                    # Save keyframe artifact
                    if vr.get("last_annotated_frame") is not None:
                        ann_dir = ensure_dir(cfg.annotated_dir)
                        fn = timestamped_filename(f"video_keyframe_{vr['location_name']}", "png")
                        fp = str(ann_dir / fn)
                        save_image(vr["last_annotated_frame"], fp)
                        save_artifact(iid, "video_keyframe", fp)

                    vr["saved"] = True
                    st.session_state["active_video_analysis"] = vr
                    st.success(f"✓ Video inspection summary committed to Database! (ID: {iid[:8]}…)")
                    st.info("Counts on Overview and Inspection History are now refreshed.")
                except Exception as ex:
                    st.error(f"Failed to save video inspection: {ex}")
        else:
            st.success("✓ This video inspection is saved in the database.")
