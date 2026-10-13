"""
DrainMind — Computer Vision-Based Drainage Blockage & Waterlogging Risk Intelligence System.
Main Dashboard and Command Center.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.utils.logging_config import setup_logging, get_logger
from src.config import get_config
from src.database.db import initialize_database
from src.database.repository import (
    get_dashboard_stats, get_inspection_activity,
    get_score_distribution, list_inspections, list_locations,
    get_location_coverage_history, seed_demo_data
)
from src.vision.visualization import score_color
from src.utils.onboarding import (
    is_first_time_user, show_onboarding_dialog, reset_onboarding_state
)

# ── Bootstrap ────────────────────────────────────────────────────────────────
cfg = get_config()
setup_logging(log_file=cfg.get("logging", "file"), level=cfg.get("logging", "level", default="INFO"))
initialize_database()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrainMind — Drainage Intelligence Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DrainMind v2.0 — Municipal Drainage Inspection System"},
)

# ── Shared CSS ───────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; padding: 6px 0 14px 0;">
                <div style="font-size:28px; line-height:1;">🌊</div>
                <div>
                    <div style="font-size:1.3rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.4px;">DrainMind</div>
                    <div style="font-size:0.75rem; color:#94A3B8; font-weight:500;">Risk Intelligence v2.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚡ Workspaces")
        if st.button("📷 Photo Inspection", use_container_width=True):
            st.switch_page("pages/1_Image_Inspection.py")
        if st.button("📹 Video Analysis", use_container_width=True):
            st.switch_page("pages/2_Video_Analysis.py")
        if st.button("📜 Audit History", use_container_width=True):
            st.switch_page("pages/3_Inspection_History.py")
        if st.button("🔄 Location Comparison", use_container_width=True):
            st.switch_page("pages/4_Location_Comparison.py")
        if st.button("📊 Executive Reports", use_container_width=True):
            st.switch_page("pages/5_Reports.py")
        if st.button("⚙ System Settings", use_container_width=True):
            st.switch_page("pages/6_Settings.py")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🧭 User Guidance")
        if st.button("❓ Onboarding Tour & Guide", use_container_width=True, key="side_tour_btn"):
            show_onboarding_dialog()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🛠 Demo Controls")
        if st.button("⚡ Reset / Populate Demo Data", use_container_width=True, help="Inserts 8 realistic sample inspections across 5 drain sites"):
            seed_demo_data()
            st.success("Demo dataset loaded!")
            st.rerun()
        if st.button("🔄 Reset Tour (Show Popup Again)", use_container_width=True, help="Triggers the welcome popup again"):
            reset_onboarding_state()
            show_onboarding_dialog()


def render_overview():
    # ── Automatic Onboarding Pop-up for New Users ─────────────────────────────
    if is_first_time_user():
        show_onboarding_dialog()

    # ── Top Hero Banner (Clean, balanced layout) ──────────────────────────────
    col_title, col_btn1, col_btn2, col_btn3 = st.columns([2.0, 0.95, 0.95, 0.75], vertical_alignment="center")
    with col_title:
        st.markdown('<h1 class="page-title">Drainage Risk Command Center</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#475569; font-size:0.92rem; display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin-top:2px;">'
            f'<span>📅 <strong>{datetime.now().strftime("%A, %d %B %Y")}</strong></span>'
            f'<span style="background:#DCFCE7; color:#16A34A; font-size:0.75rem; font-weight:700; padding:3px 9px; border-radius:12px; border:1px solid rgba(22,163,74,0.2);">● SYSTEM ONLINE</span>'
            f'<span style="background:#E0F2FE; color:#0284C7; font-size:0.75rem; font-weight:700; padding:3px 9px; border-radius:12px; border:1px solid rgba(2,132,199,0.2);">YOLOv8 + CV ENSEMBLE</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_btn1:
        if st.button("📷 Photo Inspection", type="primary", use_container_width=True, key="top_new_photo"):
            st.switch_page("pages/1_Image_Inspection.py")
    with col_btn2:
        if st.button("📹 Video Analysis", type="secondary", use_container_width=True, key="top_new_video"):
            st.switch_page("pages/2_Video_Analysis.py")
    with col_btn3:
        if st.button("❓ User Tour", use_container_width=True, key="top_tour_guide"):
            show_onboarding_dialog()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Real-Time Metrics & KPIs (Equal-Height Flexbox Cards) ──────────────────
    try:
        stats = get_dashboard_stats()
    except Exception as e:
        st.error(f"Could not load statistics: {e}")
        stats = {"total_inspections": 0, "locations_monitored": 0, "needs_inspection": 0, "recent_worsening": 0}

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #0D5C3A;">
                <div>
                    <div class="kpi-label">TOTAL INSPECTIONS</div>
                    <div class="kpi-value" style="color:#0D5C3A;">{stats['total_inspections']}</div>
                </div>
                <div class="kpi-subtext">Recorded in Local SQLite DB</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid #0284C7;">
                <div>
                    <div class="kpi-label">LOCATIONS MONITORED</div>
                    <div class="kpi-value" style="color:#0284C7;">{stats['locations_monitored']}</div>
                </div>
                <div class="kpi-subtext">Unique Drainage Catchments</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        warn_color = "#DC2626" if stats['needs_inspection'] > 0 else "#16A34A"
        warn_status = "Action Required (Score ≥ 50)" if stats['needs_inspection'] > 0 else "All Monitored Assets Stable"
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {warn_color};">
                <div>
                    <div class="kpi-label">CRITICAL / HIGH PRIORITY</div>
                    <div class="kpi-value" style="color:{warn_color};">{stats['needs_inspection']}</div>
                </div>
                <div class="kpi-subtext" style="color:{warn_color}; font-weight:600;">{warn_status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        crit_color = "#D97706" if stats['recent_worsening'] > 0 else "#64748B"
        worsening_sub = f"{stats['recent_worsening']} Sites Deteriorating" if stats['recent_worsening'] > 0 else "No Negative Trends"
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {crit_color};">
                <div>
                    <div class="kpi-label">WORSENING TRENDS</div>
                    <div class="kpi-value" style="color:{crit_color};">{stats['recent_worsening']}</div>
                </div>
                <div class="kpi-subtext">{worsening_sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Interactive Charts Section ────────────────────────────────────────────
    if stats["total_inspections"] > 0:
        c1, c2 = st.columns([1.6, 1.2])

        with c1:
            st.markdown('<div class="section-heading">📈 Inspection Activity Volume (Last 30 Days)</div>', unsafe_allow_html=True)
            try:
                activity = get_inspection_activity(30)
                if activity:
                    df_act = pd.DataFrame(activity)
                    fig = px.bar(
                        df_act, x="date", y="count",
                        labels={"date": "Inspection Date", "count": "Inspections Count"},
                        color_discrete_sequence=["#0D5C3A"]
                    )
                    fig.update_layout(
                        margin=dict(l=8, r=8, t=10, b=10),
                        height=250,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#334155"),
                        xaxis=dict(showgrid=False, title=None),
                        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Count"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Inspections logged today — activity bar chart displays volume across consecutive dates.")
            except Exception as ex:
                st.caption(f"Activity chart notice: {ex}")

        with c2:
            st.markdown('<div class="section-heading">🎯 Risk Category Distribution</div>', unsafe_allow_html=True)
            try:
                dist = get_score_distribution()
                if dist:
                    df_dist = pd.DataFrame(dist)
                    color_map = {
                        "High inspection priority": "#DC2626",
                        "Inspection recommended": "#EA580C",
                        "Monitor": "#D97706",
                        "Lower inspection priority": "#16A34A",
                    }
                    colors = [color_map.get(c, "#64748B") for c in df_dist["priority_category"]]
                    fig2 = go.Figure(go.Pie(
                        labels=df_dist["priority_category"],
                        values=df_dist["count"],
                        hole=0.52,
                        marker=dict(colors=colors),
                        textinfo="label+value",
                        textposition="inside"
                    ))
                    fig2.update_layout(
                        margin=dict(l=8, r=8, t=10, b=10),
                        height=250,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#334155"),
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.caption("Distribution available after scoring.")
            except Exception as ex:
                st.caption(f"Distribution chart notice: {ex}")

        # ── Recent Inspections Feed ──────────────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🕒 Recent Field Inspections Feed</div>', unsafe_allow_html=True)
        recent = list_inspections(limit=10, order_by="timestamp DESC")
        if recent:
            df_rec = pd.DataFrame([{
                "ID": r["inspection_id"][:8] + "…",
                "Drain Location": r.get("location_name", "—"),
                "Date / Time": (r.get("timestamp") or "")[:16].replace("T", " "),
                "Source": "📷 Photo" if "image" in str(r.get("source_type")) else "📹 Video",
                "Blockage %": f"{r['debris_coverage']:.1f}%" if r.get("debris_coverage") is not None else "—",
                "Water Level %": f"{r['water_coverage']:.1f}%" if r.get("water_coverage") is not None else "—",
                "Risk Score": f"{r['priority_score']:.1f}" if r.get("priority_score") is not None else "—",
                "Severity Urgency": r.get("priority_category", "—"),
            } for r in recent])
            st.dataframe(df_rec, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🌊</div>
                <div class="empty-title">Welcome to DrainMind</div>
                <div class="empty-body">
                    Get started by taking a photo in <strong>Photo Inspection</strong>, analyzing a video in <strong>Video Analysis</strong>,
                    or click the button below to pre-load sample field inspections.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("⚡ Pre-load Realistic Demo Field Inspections", type="primary", key="btn_init_demo"):
            seed_demo_data()
            st.success("Sample field data loaded!")
            st.rerun()


# ── Run ───────────────────────────────────────────────────────────────────────
render_sidebar()
render_overview()
