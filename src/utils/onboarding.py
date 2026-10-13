"""
DrainMind — User Guidance & Onboarding Tour System
Provides an interactive onboarding dialog pop-up for new users,
explaining what the software is, what each tab does, and how to get started.
"""

from pathlib import Path
import streamlit as st

ONBOARDING_FLAG_FILE = Path(__file__).parent.parent.parent / "data" / ".onboarding_seen"


def is_first_time_user() -> bool:
    """Return True if the user has not yet completed the onboarding tour in this session or on this machine."""
    if st.session_state.get("onboarding_seen", False):
        return False
    return not ONBOARDING_FLAG_FILE.exists()


def mark_onboarding_completed():
    """Mark onboarding tour as completed in session state and persist locally."""
    st.session_state["onboarding_seen"] = True
    try:
        ONBOARDING_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_FLAG_FILE.write_text("seen", encoding="utf-8")
    except Exception:
        pass


def reset_onboarding_state():
    """Reset onboarding flag to allow re-triggering the tour."""
    st.session_state["onboarding_seen"] = False
    if ONBOARDING_FLAG_FILE.exists():
        try:
            ONBOARDING_FLAG_FILE.unlink()
        except Exception:
            pass


@st.dialog("🌊 Welcome to DrainMind — User Onboarding & Feature Guide", width="large")
def show_onboarding_dialog():
    """Interactive pop-up modal dialog welcoming new users and guiding them through the software."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(36,91,69,0.12), rgba(74,124,142,0.12)); border: 1px solid #DDE1DA; border-radius: 8px; padding: 14px 18px; margin-bottom: 15px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #245B45;">Intelligent Drainage Blockage & Flood Prevention Cockpit</div>
            <div style="font-size: 0.9rem; color: #525952; margin-top: 4px;">
                DrainMind combines deep learning (YOLOv8) and multi-spectral computer vision to detect silt, debris, standing water, and structural defects in drainage channels.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_about, tab_features, tab_quickstart = st.tabs([
        "📖 1. What is DrainMind?",
        "🗺 2. Tabs & Workspaces Guide",
        "🚀 3. Quick Start in 60s"
    ])

    with tab_about:
        st.markdown(
            """
            ### 🎯 Core Purpose & Capabilities
            DrainMind is an end-to-end local computer vision system built for municipal authorities, stormwater engineers, and field maintenance crews.

            #### What does it do?
            - **Automated Debris & Defect Detection:** Detects silt deposits, plastic litter, organic blockage, standing water pools, and structural concrete cracks.
            - **Quantified Blockage Metrics:** Calculates precise percentage coverage of obstruction inside the drainage channel flow bed.
            - **Composite Risk Priority Scoring (0–100):** Normalizes conditions into actionable categories:
              - 🟢 **Lower Priority (0–29.9)**: Normal flow; routine maintenance.
              - 🟡 **Monitor (30–49.9)**: Moderate silt buildup; monitor closely.
              - 🟠 **Inspection Recommended (50–69.9)**: Significant blockage; dispatch cleaning crew.
              - 🔴 **High Priority / Critical (70–100)**: Severe blockage & imminent flood hazard!
            - **Longitudinal Trend Tracking:** Compares baseline and follow-up inspections to detect worsening sedimentation.
            - **Automated Work Orders:** Generates official municipal maintenance dispatch notices with one click.
            """
        )

    with tab_features:
        st.markdown("### 🧭 Complete Navigation & Feature Map")
        st.markdown(
            """
            | Workspace / Tab | Primary Function | Key Features |
            | :--- | :--- | :--- |
            | **🏠 Overview Dashboard** (`app.py`) | Situational command center | Live KPI cards, 30-day inspection volume, risk distribution pie chart, recent field activity feed. |
            | **📷 Image Inspection** (`Tab 1`) | Single photo analysis | Upload or snap photos; detects silt, debris, cracks, and calculates the 0–100 priority score. |
            | **📹 Video Analysis** (`Tab 2`) | Continuous video survey | Analyzes CCTV and pipe crawler footage; generates blockage timeline graphs and extracts peak keyframes. |
            | **📜 Inspection History** (`Tab 3`) | Historical audit repository | Search by drain name or ID, view original vs. annotated images side-by-side, and batch export to CSV. |
            | **🔄 Location Comparison** (`Tab 4`) | Longitudinal change detection | Before/after audits, thermal difference heatmaps (red = new debris, green = cleared), and work orders. |
            | **📊 Executive Reports** (`Tab 5`) | Municipal compliance export | One-click Executive PDF Dossiers, raw CSV logs, and aggregated network statistics. |
            | **⚙ Settings & Diagnostics** (`Tab 6`) | AI calibration & test bench | Custom model weights (.pt), scoring formula weight sliders, live formula simulator, and DB diagnostics. |
            """
        )

    with tab_quickstart:
        st.markdown(
            """
            ### ⚡ 3 Steps to Test DrainMind Right Now:
            1. **Preload Sample Field Data:**
               - Click the **"⚡ Reset / Populate Demo Data"** button on the Overview dashboard or sidebar.
               - This immediately loads 8 realistic inspections across 5 urban drainage assets.
            2. **Test Image Inspection:**
               - Open **Image Inspection**, choose **"🖼 Test Sample Image"**, and click **"▶ Run Inspection"**.
               - Notice how the computer vision engine maps bounding boxes for plastic waste, silt, and cracks!
            3. **Generate an Official Work Order:**
               - Open **Location Comparison**, switch to **"Demonstration Case Studies"**, and see the visual thermal difference heatmap and downloadable dispatch order!
            """
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        if st.button("🚀 Got it! Start Exploring DrainMind", type="primary", use_container_width=True, key="btn_onboarding_close"):
            mark_onboarding_completed()
            st.rerun()
    with c2:
        if st.button("📷 Jump Straight to Image Inspection", use_container_width=True, key="btn_onboarding_jump"):
            mark_onboarding_completed()
            st.switch_page("pages/1_Image_Inspection.py")
