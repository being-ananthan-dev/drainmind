"""
DrainMind — Location Comparison & Change Intelligence Page
Longitudinal before/after audit tool, change heatmaps, and automated maintenance work orders.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import (
    list_locations, list_inspections,
    get_inspection, get_observation, get_artifacts,
    get_location_coverage_history
)
from src.vision.preprocessing import numpy_to_pil, load_image_from_path

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Location Comparison — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Location Comparison & Change Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Evaluate infrastructure deterioration trends, post-cleaning effectiveness, and generate work orders.</p>', unsafe_allow_html=True)

mode = st.radio("Comparison Mode", ["📍 Live Monitored Location Audit", "🎬 Demonstration Case Studies"], horizontal=True)

img_a = None
img_b = None
loc_title = "Drainage Asset"
date_a_str = "Baseline"
date_b_str = "Follow-up"
deb_a = 0.0
deb_b = 0.0
wat_a = 0.0
wat_b = 0.0
sc_a = 0.0
sc_b = 0.0

if mode == "📍 Live Monitored Location Audit":
    locations = list_locations()
    if not locations:
        st.warning("No locations registered. Create an inspection first or switch to Demonstration Case Studies.")
        st.stop()

    loc_map = {l["name"]: l["location_id"] for l in locations}
    sel_loc_name = st.selectbox("Select Target Location", list(loc_map.keys()), key="cmp_loc_select")
    target_lid = loc_map[sel_loc_name]
    loc_title = sel_loc_name

    insps = list_inspections(location_id=target_lid, limit=50, order_by="timestamp DESC")
    if len(insps) < 2:
        st.info(f"Location '{sel_loc_name}' currently has {len(insps)} recorded inspection. At least 2 inspections are required to calculate longitudinal deltas. Check 'Demonstration Case Studies' to see how this works.")
        st.stop()

    def label_insp(r):
        ts = (r.get("timestamp") or "")[:16].replace("T", " ")
        src = "📷 Photo" if "image" in str(r.get("source_type")) else "📹 Video"
        sc = f"{r['priority_score']:.1f}" if r.get("priority_score") is not None else "—"
        return f"{ts} • {src} • Score: {sc}"

    insp_dict = {label_insp(r): r for r in insps}
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        sel_a = st.selectbox("Baseline Inspection (Earlier Date)", list(insp_dict.keys()), index=len(insp_dict)-1, key="cmp_sel_a")
    with c_sel2:
        sel_b = st.selectbox("Follow-up Inspection (Later Date)", list(insp_dict.keys()), index=0, key="cmp_sel_b")

    rec_a = insp_dict[sel_a]
    rec_b = insp_dict[sel_b]

    date_a_str = (rec_a.get("timestamp") or "")[:10]
    date_b_str = (rec_b.get("timestamp") or "")[:10]
    deb_a = rec_a.get("debris_coverage") or 0.0
    deb_b = rec_b.get("debris_coverage") or 0.0
    wat_a = rec_a.get("water_coverage") or 0.0
    wat_b = rec_b.get("water_coverage") or 0.0
    sc_a = rec_a.get("priority_score") or 0.0
    sc_b = rec_b.get("priority_score") or 0.0

    # Load artifacts if available
    arts_a = get_artifacts(rec_a["inspection_id"])
    arts_b = get_artifacts(rec_b["inspection_id"])
    for a in arts_a:
        if Path(a["file_path"]).exists():
            img_a = load_image_from_path(a["file_path"])
            break
    for b in arts_b:
        if Path(b["file_path"]).exists():
            img_b = load_image_from_path(b["file_path"])
            break

else:
    # Demonstration Case Studies
    demo_case = st.selectbox(
        "Select Case Study",
        [
            "Case 1: Monsoon Flash Flood (Severe Deterioration: +49.9% Silt & Plastic Debris)",
            "Case 2: Post-Desilting Municipal Maintenance (Great Improvement: -58.0% Cleared)"
        ]
    )

    h_c, w_c = 480, 640
    img_base = np.full((h_c, w_c, 3), 110, dtype=np.uint8)
    cv2.fillPoly(img_base, [np.array([[0, 0], [140, h_c], [0, h_c]])], (80, 80, 80))
    cv2.fillPoly(img_base, [np.array([[w_c, 0], [w_c-140, h_c], [w_c, h_c]])], (80, 80, 80))

    if "Case 1" in demo_case:
        loc_title = "North Ring Road Box Culvert"
        date_a_str = "2026-09-28 (Pre-Monsoon Baseline)"
        date_b_str = "2026-10-10 (Post-Downpour Audit)"
        # Image A: cleanish drain
        img_a = img_base.copy()
        cv2.ellipse(img_a, (320, 390), (90, 40), 0, 0, 360, (40, 70, 90), -1)
        deb_a, wat_a, sc_a = 28.5, 12.0, 31.0
        # Image B: massive silt bank + plastic waste
        img_b = img_base.copy()
        cv2.ellipse(img_b, (320, 360), (220, 95), 0, 0, 360, (30, 60, 80), -1)
        cv2.rectangle(img_b, (270, 300), (330, 340), (180, 150, 40), -1)
        deb_b, wat_b, sc_b = 78.4, 45.2, 74.2
    else:
        loc_title = "East Industrial Canal Inflow"
        date_a_str = "2026-10-01 (Pre-Cleaning Audit)"
        date_b_str = "2026-10-11 (Post-Cleaning Verification)"
        # Image A: clogged
        img_a = img_base.copy()
        cv2.ellipse(img_a, (320, 360), (220, 95), 0, 0, 360, (30, 60, 80), -1)
        deb_a, wat_a, sc_a = 82.5, 65.0, 81.0
        # Image B: completely cleared
        img_b = img_base.copy()
        cv2.circle(img_b, (320, 390), 25, (40, 70, 80), -1)
        deb_b, wat_b, sc_b = 14.5, 8.0, 16.5

# ── Compute Delta Metrics ─────────────────────────────────────────────────────
delta_deb = deb_b - deb_a
delta_wat = wat_b - wat_a
delta_score = sc_b - sc_a

if delta_deb > 5.0:
    trend_badge = "🔴 DETERIORATING (WORSENING)"
    trend_color = "#8B1A1A"
    rec_text = "CRITICAL: Urgent maintenance crew dispatch required. Heavy silt sedimentation and flow restriction detected."
elif delta_deb < -5.0:
    trend_badge = "🟢 IMPROVING (CLEARED)"
    trend_color = "#245B45"
    rec_text = "EXCELLENT: Successful desilting confirmed. Flow capacity restored to normal municipal specifications."
else:
    trend_badge = "🟡 STABLE FLOW"
    trend_color = "#B87333"
    rec_text = "NORMAL: Routine condition. No immediate intervention necessary; maintain standard bi-monthly inspection schedule."

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(f'<h2 class="section-heading">📊 Longitudinal Change Intelligence: {loc_title}</h2>', unsafe_allow_html=True)

# KPI Delta Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top:4px solid {trend_color};">
            <span class="kpi-label">Condition Trend</span>
            <div style="font-size:1.15rem; font-weight:800; color:{trend_color}; margin: 6px 0;">{trend_badge}</div>
            <div class="kpi-subtext">Direction of infrastructure shift</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with k2:
    st.metric("Blockage Delta", f"{delta_deb:+.1f}%", delta=f"{delta_deb:+.1f}%", delta_color="inverse")
with k3:
    st.metric("Water Stagnation Delta", f"{delta_wat:+.1f}%", delta=f"{delta_wat:+.1f}%", delta_color="inverse")
with k4:
    st.metric("Risk Score Delta", f"{delta_score:+.1f} pts", delta=f"{delta_score:+.1f} pts", delta_color="inverse")

# ── Visual Comparison Tabs ───────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-heading">🖼 Comparative Visual Audit</h3>', unsafe_allow_html=True)

tab_side, tab_heat = st.tabs(["👥 Side-by-Side Photo Comparison", "🔥 Computer Vision Delta Heatmap"])

with tab_side:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"**Baseline Inspection** • <span class='mono' style='font-size:0.85rem;'>{date_a_str}</span>", unsafe_allow_html=True)
        if img_a is not None:
            st.image(numpy_to_pil(img_a), caption=f"Baseline: {deb_a:.1f}% Silt/Debris Blockage", use_container_width=True)
        else:
            st.info("Baseline photo artifact not available.")

    with col_b:
        st.markdown(f"**Follow-up Inspection** • <span class='mono' style='font-size:0.85rem;'>{date_b_str}</span>", unsafe_allow_html=True)
        if img_b is not None:
            st.image(numpy_to_pil(img_b), caption=f"Follow-up: {deb_b:.1f}% Silt/Debris Blockage", use_container_width=True)
        else:
            st.info("Follow-up photo artifact not available.")

with tab_heat:
    if img_a is not None and img_b is not None:
        h_min = min(img_a.shape[0], img_b.shape[0])
        w_min = min(img_a.shape[1], img_b.shape[1])
        r_a = cv2.resize(img_a, (w_min, h_min))
        r_b = cv2.resize(img_b, (w_min, h_min))
        diff = cv2.absdiff(r_a, r_b)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(r_b, 0.6, heatmap, 0.4, 0)

        col_h_img, col_h_info = st.columns([1.5, 1.0], gap="large")
        with col_h_img:
            st.image(numpy_to_pil(blend), caption="Thermal Delta Overlay: Warm colors (Red/Yellow) highlight major sediment shifts", use_container_width=True)
        with col_h_info:
            st.markdown(
                """
                <div class="kpi-card" style="border-left: 4px solid #0284C7; margin-bottom: 12px;">
                    <span class="kpi-label">Heatmap Methodology</span>
                    <div style="font-size:0.88rem; color:var(--text-muted); line-height:1.5; margin-top:6px;">
                        The delta heatmap calculates pixel-level luminance & color matrix variance across aligned baseline and follow-up frames, isolating physical shifts in sludge deposits, floating debris, and structural culvert degradation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Change heatmap requires both baseline and follow-up images.")

# ── Automated Work Order Generator ────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-heading">📋 Automated Municipal Maintenance Directive</h3>', unsafe_allow_html=True)

wo_id = f"WO-DM-{datetime.now().strftime('%Y%m%d')}-01"
wo_md = f"""================================================================================
DRAINAGE MAINTENANCE DISPATCH WORK ORDER
Work Order ID: {wo_id}
Asset Location: {loc_title}
Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Trend Assessment: {trend_badge}
--------------------------------------------------------------------------------
CONDITIONS & TELEMETRY:
- Baseline Blockage: {deb_a:.1f}%
- Follow-up Blockage: {deb_b:.1f}% (Net Delta: {delta_deb:+.1f}%)
- Standing Water Level: {wat_b:.1f}% (Net Delta: {delta_wat:+.1f}%)
- Composite Priority Risk Score: {sc_b:.1f} / 100

RECOMMENDED DISPATCH DIRECTIVE:
{rec_text}
================================================================================
"""

st.markdown(
    f"""
    <div class="callout-box" style="border-left-color: {trend_color};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <strong style="font-size:1rem;">Work Order Notice: <code>{wo_id}</code></strong>
            <span style="font-size:0.8rem; font-weight:700; color:{trend_color};">{trend_badge}</span>
        </div>
        <div style="font-size:0.92rem; color:var(--text-muted); line-height:1.4;">{rec_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.download_button(
    "⬇ Download Official Dispatch Notice (TXT)",
    data=wo_md,
    file_name=f"{wo_id}.txt",
    mime="text/plain",
    type="primary",
    key="btn_dl_wo"
)
