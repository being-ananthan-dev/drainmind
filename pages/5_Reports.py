"""
DrainMind — Reports & Intelligence Export Page
Generate municipal compliance reports, executive summaries, CSV audit logs, and PDF dossiers.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import list_locations, list_inspections, seed_demo_data
from src.reporting.csv_export import export_csv
from src.reporting.pdf_export import generate_pdf_report, is_pdf_available
from src.utils.paths import timestamped_filename

cfg = get_config()
initialize_database()

st.set_page_config(page_title="Reports — DrainMind", layout="wide")

_css_path = Path(__file__).parent.parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Executive Reports & Data Dossiers</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Export official drainage condition audit reports as CSV logs and printable PDF dossiers.</p>', unsafe_allow_html=True)

# ── Report Filters ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Report Generation Filters")
f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1.2])

with f1:
    existing_locs = list_locations()
    loc_map = {"All Locations": None}
    for l in existing_locs:
        loc_map[l["name"]] = l["location_id"]
    sel_loc = st.selectbox("Drainage Location", list(loc_map.keys()), key="rep_loc_sel")
    loc_filter = loc_map[sel_loc]

with f2:
    status_opts = {"All Records": None, "Complete": "complete", "Failed": "failed"}
    sel_st = st.selectbox("Inspection Status", list(status_opts.keys()), key="rep_status_sel")
    status_filter = status_opts[sel_st]

with f3:
    date_from = st.date_input("From Date", value=None, key="rep_d_from")

with f4:
    date_to = st.date_input("To Date", value=None, key="rep_d_to")

# ── Fetch Records ─────────────────────────────────────────────────────────────
try:
    records = list_inspections(
        location_id=loc_filter,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        status=status_filter,
        limit=5000,
        order_by="timestamp DESC",
    )
except Exception as e:
    st.error(f"Failed to fetch records: {e}")
    records = []

if not records:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">No matching records found</div>
            <div class="empty-body">There are no inspection records matching your filters. You can adjust your filters or click below to populate sample field data.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("⚡ Populate Realistic Demo Field Data", type="primary", key="btn_rep_seed"):
        seed_demo_data()
        st.success("Sample field inspections loaded!")
        st.rerun()
    st.stop()

# ── Summary Metrics ────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(f'<h2 class="section-heading">📈 Summary Telemetry Audit ({len(records)} Inspections Selected)</h2>', unsafe_allow_html=True)

scores = [r["priority_score"] for r in records if r.get("priority_score") is not None]
debs = [r["debris_coverage"] for r in records if r.get("debris_coverage") is not None]
wats = [r["water_coverage"] for r in records if r.get("water_coverage") is not None]

mean_sc = f"{sum(scores)/len(scores):.1f}" if scores else "—"
peak_sc = f"{max(scores):.1f}" if scores else "—"
mean_deb = f"{sum(debs)/len(debs):.1f}%" if debs else "—"
mean_wat = f"{sum(wats)/len(wats):.1f}%" if wats else "—"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Inspections In Scope", len(records))
with k2:
    st.metric("Mean Priority Score", mean_sc)
with k3:
    st.metric("Peak Risk Recorded", peak_sc)
with k4:
    st.metric("Mean Blockage Level", mean_deb)

# ── Download Dossiers ─────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-heading">📥 Instant Report Generation & Document Downloads</h3>', unsafe_allow_html=True)

col_csv, col_pdf = st.columns(2, gap="large")

with col_csv:
    st.markdown(
        """
        <div class="kpi-card" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-label">Structured Data</span>
                <span style="padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem; background:#E0F2FE; color:#0284C7; border:1px solid #BAE6FD;">CSV SPREADSHEET</span>
            </div>
            <div style="font-size:1.05rem; font-weight:700; margin: 8px 0 4px 0; color:var(--text-main);">Raw Inspection Log Table</div>
            <div style="font-size:0.84rem; color:var(--text-muted); line-height:1.4;">
                Full machine-readable tabular export containing timestamps, geo-locations, bounding boxes, and debris percentages.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    csv_bytes = export_csv(records)
    csv_filename = timestamped_filename("drainmind_audit_report", "csv")
    st.download_button(
        "⬇ Download CSV Spreadsheet",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="btn_dl_csv_rep"
    )

with col_pdf:
    st.markdown(
        """
        <div class="kpi-card" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="kpi-label">Executive Document</span>
                <span style="padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem; background:#DCFCE7; color:#16A34A; border:1px solid #BBF7D0;">PRINTABLE PDF</span>
            </div>
            <div style="font-size:1.05rem; font-weight:700; margin: 8px 0 4px 0; color:var(--text-main);">Municipal Audit Dossier</div>
            <div style="font-size:0.84rem; color:var(--text-muted); line-height:1.4;">
                Formatted publication report with executive summaries, visual telemetry breakdowns, and formal compliance methodology.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if is_pdf_available():
        report_title = f"DrainMind Inspection Report — {sel_loc}"
        filter_summary = f"Location: {sel_loc}, Status: {sel_st}, From: {date_from or 'All'}, To: {date_to or 'All'}"
        pdf_bytes = generate_pdf_report(
            inspections=records,
            title=report_title,
            filters_applied=filter_summary,
            include_methodology=True
        )
        if pdf_bytes:
            pdf_filename = timestamped_filename("drainmind_executive_dossier", "pdf")
            st.download_button(
                "⬇ Download Executive PDF Dossier",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="btn_dl_pdf_rep"
            )
        else:
            st.error("PDF generation encountered an issue.")
    else:
        st.caption("PDF generation engine not detected.")

# ── Preview Table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📋 Record Breakdown Preview")

df_rep = pd.DataFrame([{
    "Drain Location": r.get("location_name", "—"),
    "Timestamp": (r.get("timestamp") or "")[:16].replace("T", " "),
    "Source": "📷 Photo" if "image" in str(r.get("source_type")) else "📹 Video",
    "Blockage %": f"{r['debris_coverage']:.1f}%" if r.get("debris_coverage") is not None else "—",
    "Water %": f"{r['water_coverage']:.1f}%" if r.get("water_coverage") is not None else "—",
    "Priority Score": f"{r['priority_score']:.1f}" if r.get("priority_score") is not None else "—",
    "Severity Urgency": r.get("priority_category", "—"),
    "Notes": r.get("notes", "—"),
} for r in records])

st.dataframe(df_rep, use_container_width=True, hide_index=True)
