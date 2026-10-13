"""
DrainMind — Inspection History Page
Searchable, filterable audit log of all recorded image and video drainage inspections.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from pathlib import Path
import json
import streamlit as st
import pandas as pd

from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import (
    list_locations, list_inspections,
    get_inspection, get_observation, get_detections, get_artifacts,
    delete_inspection
)
from src.reporting.csv_export import export_csv
from src.vision.preprocessing import numpy_to_pil, load_image_from_path

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Inspection History — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Inspection History & Audit Log</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Search, filter, review visual artifacts, and export drainage inspection records.</p>', unsafe_allow_html=True)

# ── Filter & Search Toolbar ───────────────────────────────────────────────────
f1, f2, f3, f4 = st.columns([2, 1.5, 1, 1])

with f1:
    search_query = st.text_input("🔍 Quick Search", placeholder="Filter by drain name, notes, or ID…", key="hist_q")

with f2:
    existing_locs = list_locations()
    loc_map = {"All Locations": None}
    for l in existing_locs:
        loc_map[l["name"]] = l["location_id"]
    sel_loc = st.selectbox("Location Filter", list(loc_map.keys()), key="hist_loc_filter")
    loc_filter_id = loc_map[sel_loc]

with f3:
    type_opts = {"All Sources": None, "📷 Photo": "image", "📹 Video": "uploaded_video"}
    sel_type = st.selectbox("Source Type", list(type_opts.keys()), key="hist_type_filter")
    source_filter = type_opts[sel_type]

with f4:
    sort_opts = {
        "Newest First": "timestamp DESC",
        "Oldest First": "timestamp ASC",
        "Highest Risk": "priority_score DESC",
        "Lowest Risk": "priority_score ASC",
    }
    sel_sort = st.selectbox("Sort Order", list(sort_opts.keys()), key="hist_sort")

# ── Fetch Records ─────────────────────────────────────────────────────────────
try:
    records = list_inspections(
        location_id=loc_filter_id,
        limit=500,
        order_by=sort_opts[sel_sort],
    )
    # Apply text filter if provided
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        records = [
            r for r in records
            if q in str(r.get("location_name", "")).lower()
            or q in str(r.get("notes", "")).lower()
            or q in str(r.get("inspection_id", "")).lower()
        ]
    if source_filter:
        records = [r for r in records if r.get("source_type") == source_filter]
except Exception as e:
    st.error(f"Failed to query database: {e}")
    records = []

# ── Toolbar Actions (Export & Metrics) ────────────────────────────────────────
top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    st.markdown(f'<div class="meta-text" style="font-size:0.95rem;">Found <strong>{len(records)}</strong> inspection records matching filters.</div>', unsafe_allow_html=True)
with top_c2:
    if records:
        csv_bytes = export_csv(records)
        st.download_button(
            "⬇ Export to CSV",
            data=csv_bytes,
            file_name=f"drainmind_inspections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="btn_export_hist_csv"
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if not records:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📜</div>
            <div class="empty-title">No inspection records found</div>
            <div class="empty-body">Try clearing your search query or run a new inspection.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # ── Interactive Table View ────────────────────────────────────────────────
    df_display = pd.DataFrame([{
        "ID": r["inspection_id"][:8] + "…",
        "Drain Location": r.get("location_name", "—"),
        "Date / Time": (r.get("timestamp") or "")[:16].replace("T", " "),
        "Source": "📷 Photo" if "image" in str(r.get("source_type")) else "📹 Video",
        "Blockage %": f"{r['debris_coverage']:.1f}%" if r.get("debris_coverage") is not None else "—",
        "Water %": f"{r['water_coverage']:.1f}%" if r.get("water_coverage") is not None else "—",
        "Risk Score": f"{r['priority_score']:.1f}" if r.get("priority_score") is not None else "—",
        "Severity Category": r.get("priority_category", "—"),
        "Field Notes": r.get("notes", "—"),
        "_full_id": r["inspection_id"],
    } for r in records])

    st.dataframe(df_display.drop(columns=["_full_id"]), use_container_width=True, hide_index=True)

    # ── Inspection Detail Inspector Drawer ────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Inspection Deep Dive & Artifacts")

    insp_opts = {
        f"{(r.get('timestamp') or '')[:16].replace('T', ' ')} — {r.get('location_name')} (Score: {r.get('priority_score', '—')})": r["inspection_id"]
        for r in records
    }
    sel_label = st.selectbox("Select Record to Inspect", list(insp_opts.keys()), key="hist_sel_record")
    active_iid = insp_opts[sel_label]

    insp = get_inspection(active_iid)
    obs = get_observation(active_iid)
    dets = get_detections(active_iid)
    arts = get_artifacts(active_iid)

    if insp:
        col_meta, col_obs = st.columns(2, gap="large")
        with col_meta:
            st.markdown('<h4 class="section-heading">📋 Inspection Metadata</h4>', unsafe_allow_html=True)
            st.markdown(
                f"""
                - **Inspection ID:** `{insp['inspection_id']}`
                - **Location:** **{insp.get('location_name')}** (`{insp.get('location_id')}`)
                - **Timestamp:** {insp.get('timestamp')}
                - **Source Type:** {insp.get('source_type')}
                - **AI Model Backend:** {insp.get('model_name')} ({insp.get('model_version')})
                - **Assessment Quality:** {insp.get('assessment_quality')}
                - **Inspector Notes:** *{insp.get('notes') or 'None'}*
                """
            )

        with col_obs:
            if obs:
                sc = obs.get("priority_score", 0.0) or 0.0
                st.markdown('<h4 class="section-heading">⚖ Condition Scoring</h4>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    - **Priority Risk Score:** **{sc:.1f} / 100**
                    - **Risk Category:** **{obs.get('priority_category')}**
                    - **Debris / Silt Coverage:** **{obs.get('debris_coverage', 0):.1f}%**
                    - **Standing Water Level:** **{obs.get('water_coverage', 0):.1f}%**
                    - **Trend Score Delta:** **{obs.get('trend_score', 0):+.1f}%**
                    """
                )

        # Artifacts and Visuals
        if arts:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h4 class="section-heading">🖼 Stored Visual Artifacts</h4>', unsafe_allow_html=True)
            num_cols = min(len(arts), 2)
            art_cols = st.columns(num_cols, gap="medium")
            for i, a in enumerate(arts):
                with art_cols[i % num_cols]:
                    fp = a.get("file_path", "")
                    atype = a.get("artifact_type", "image")
                    if Path(fp).exists():
                        img_loaded = load_image_from_path(fp)
                        if img_loaded is not None:
                            st.image(numpy_to_pil(img_loaded), caption=f"{atype.upper()}: {Path(fp).name}", use_container_width=True)
                        else:
                            st.caption(f"Preview unavailable: {Path(fp).name}")
                    else:
                        st.caption(f"File stored externally: {fp}")

        # Detections table
        if dets:
            st.markdown(f"#### 🎯 Detected Objects & Bounding Boxes ({len(dets)})")
            df_d = pd.DataFrame([{
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']*100:.1f}%",
                "Bounding Box": f"({int(d['x1'])}, {int(d['y1'])}) → ({int(d['x2'])}, {int(d['y2'])})",
            } for d in dets])
            st.dataframe(df_d, use_container_width=True, hide_index=True)

        # Actions: Delete record
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        del_confirm = st.checkbox(f"Confirm deletion of record {active_iid[:8]}…", key="chk_del_confirm")
        if st.button("🗑 Delete Record", type="secondary", disabled=not del_confirm, key="btn_del_rec"):
            try:
                delete_inspection(active_iid)
                st.success("Record deleted successfully.")
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to delete record: {ex}")
