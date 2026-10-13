"""
DrainMind — Settings & Diagnostics Page
Configure vision models, scoring formula weights, system thresholds, and verify subsystem health.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st

from src.config import get_config, reload_config
from src.database.db import initialize_database, check_database_health
from src.database.repository import get_dashboard_stats
from src.vision.detector import get_detector, reset_detector
from src.analysis.priority_score import compute_priority_score

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Settings & Diagnostics — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Settings & System Diagnostics</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Configure AI model weights, priority scoring formulas, and inspect hardware telemetry.</p>', unsafe_allow_html=True)

tab_settings, tab_sim, tab_diag = st.tabs(["⚙ Model & Formula Settings", "🔬 Live Formula Simulator", "🩺 System Health Diagnostics"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_settings:
    st.markdown("### 🤖 Computer Vision Model Configuration")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        model_path = st.text_input(
            "Custom Model Weights Path (.pt)",
            value=cfg.model_path or "",
            placeholder="e.g. D:/DrainMind/models/custom_drain_yolo.pt",
            help="Leave blank to use default pretrained weights.",
            key="cfg_mpath"
        )
        model_name = st.selectbox(
            "Base YOLO Preset",
            ["yolov8n", "yolov8s", "yolov8m"],
            index=["yolov8n", "yolov8s", "yolov8m"].index(cfg.model_name) if cfg.model_name in ["yolov8n", "yolov8s", "yolov8m"] else 0,
            key="cfg_mname"
        )
        confidence = st.slider(
            "Default Confidence Threshold",
            min_value=0.05, max_value=0.90, value=float(cfg.confidence_threshold), step=0.05,
            key="cfg_conf"
        )

    with col_m2:
        input_size = st.select_slider(
            "Inference Frame Size (px)",
            options=[320, 416, 512, 640, 768, 1024],
            value=min([320, 416, 512, 640, 768, 1024], key=lambda x: abs(x - cfg.input_size)),
            key="cfg_isize"
        )
        frame_interval = st.number_input(
            "Video Frame Sample Interval",
            min_value=1, max_value=60, value=int(cfg.frame_sample_interval),
            key="cfg_fint"
        )
        device_choice = st.selectbox("Inference Hardware Device", ["cpu", "cuda:0"], index=0, key="cfg_dev")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚖ Priority Risk Score Formula Weights")
    st.markdown(
        """
        The composite risk score is governed by the linear equation:  
        $$\\text{Risk Score} = (W_D \\times \\text{Debris\\%}) + (W_W \\times \\text{Water\\%}) + (W_T \\times \\text{Trend\\%})$$
        """
    )

    ws = cfg.scoring_weights
    w1, w2, w3 = st.columns(3)
    with w1:
        w_debris = st.slider("Debris / Obstruction Weight (W_D)", 0.0, 1.0, float(ws.get("debris", 0.50)), 0.05, key="cfg_wd")
    with w2:
        w_water = st.slider("Waterlogging Weight (W_W)", 0.0, 1.0, float(ws.get("water", 0.30)), 0.05, key="cfg_ww")
    with w3:
        w_trend = st.slider("Deterioration Trend Weight (W_T)", 0.0, 1.0, float(ws.get("trend", 0.20)), 0.05, key="cfg_wt")

    sum_w = w_debris + w_water + w_trend
    if abs(sum_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {sum_w:.2f}. They will be automatically normalized to 1.0 during scoring.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button("💾 Save & Synchronize Settings", type="primary", key="btn_save_all_settings"):
        errors = []
        if model_path.strip() and not Path(model_path.strip()).is_file():
            errors.append(f"Model file does not exist: {model_path}")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            updates = {
                "model": {
                    "path": model_path.strip(),
                    "name": model_name,
                    "confidence_threshold": float(confidence),
                    "input_size": int(input_size),
                    "device": device_choice,
                },
                "video": {
                    "frame_sample_interval": int(frame_interval),
                },
                "scoring": {
                    "weights": {
                        "debris": float(w_debris),
                        "water": float(w_water),
                        "trend": float(w_trend)
                    }
                }
            }
            try:
                cfg.update_and_save(updates)
                reload_config()
                reset_detector()
                # Reload active detector
                get_detector(reload=True)
                st.success("✓ Settings saved and synced! Models and scoring weights reloaded globally across all pages.")
                st.rerun()
            except Exception as ex:
                st.error(f"Save failed: {ex}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: LIVE FORMULA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.markdown("### 🔬 Real-Time Formula Verification Test Bench")
    st.markdown("Test and verify how your active weights calculate priority scores under various drain condition scenarios.")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_deb = st.slider("Simulated Debris %", 0.0, 100.0, 65.0, 1.0, key="sim_deb")
    with s2:
        sim_wat = st.slider("Simulated Water Stagnation %", 0.0, 100.0, 40.0, 1.0, key="sim_wat")
    with s3:
        sim_tr = st.slider("Simulated Trend Delta %", 0.0, 100.0, 15.0, 1.0, key="sim_tr")

    sim_res = compute_priority_score(
        debris_coverage=sim_deb,
        water_coverage=sim_wat,
        trend_score=sim_tr
    )

    k_s1, k_s2 = st.columns(2, gap="large")
    with k_s1:
        from src.vision.visualization import score_color
        score_val = sim_res.get("score", 0.0)
        sim_col = score_color(score_val)
        cat_val = sim_res.get('category', 'Unknown')
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top:4px solid {sim_col}; min-height:140px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="kpi-label">Simulated Risk Rating</span>
                    <span style="padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem; background:{sim_col}18; color:{sim_col}; border:1px solid {sim_col}33;">
                        {cat_val.upper()}
                    </span>
                </div>
                <div style="font-size:2.8rem; font-weight:800; color:{sim_col}; line-height:1; margin: 10px 0 4px 0;">
                    {score_val:.1f} <span style="font-size:1rem; color:var(--text-muted); font-weight:600;">/ 100</span>
                </div>
                <div style="font-size:0.84rem; color:var(--text-muted); line-height:1.4;">
                    Computed via normalized multi-factor linear weighting.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k_s2:
        st.markdown("#### Weight Contributions Breakdown")
        w_used = sim_res.get("weights_used", {})
        comps = sim_res.get("components", {})
        df_sim = pd.DataFrame([
            {
                "Component": k.capitalize(),
                "Input Value": f"{comps.get(k, 0):.1f}%",
                "Configured Weight": f"{w_used.get(k, 0):.2f}",
                "Weighted Contribution": f"{(w_used.get(k, 0) * comps.get(k, 0)):.1f} pts"
            }
            for k in comps.keys()
        ])
        st.dataframe(df_sim, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_diag:
    st.markdown("### 🩺 Subsystem Diagnostics")

    # DB Health
    health = check_database_health()
    db_ok = health.get("status") == "ok"
    st.markdown(
        f"""
        - **Database Status:** {'🟢 OPERATIONAL (SQLite WAL)' if db_ok else '🔴 ERROR'}
        - **Database Path:** `{health.get('path')}`
        - **Database Size:** `{health.get('size_bytes', 0) / 1024:.1f} KB`
        """
    )

    # Detector benchmark
    st.markdown("---")
    st.markdown("### ⚡ Live Vision Inference Benchmark")
    if st.button("Run Diagnostic Benchmark Inference", key="btn_bench"):
        with st.spinner("Benchmarking detector inference latency…"):
            det = get_detector()
            dummy = np.full((480, 640, 3), 120, dtype=np.uint8)
            t0 = time.perf_counter()
            dets, backend = det.detect(dummy)
            latency_ms = (time.perf_counter() - t0) * 1000
            st.success(f"✓ Inference successful! Latency: {latency_ms:.1f} ms • Backend: {backend} • Model: {det.model_name}")
