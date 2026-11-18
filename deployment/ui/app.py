"""
Macadamia Quality Grading AI — Streamlit Dashboard
Full-featured inspection dashboard: upload or capture → preprocess → detect → classify → XAI → report.
"""

import sys
import os
import io
import time
import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import MacadamiaGradingPipeline
from src.reporting.report_generator import generate_pdf_report, generate_csv_report

logging.basicConfig(level=logging.WARNING)

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macadamia Quality Grading AI",
    page_icon="🌰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0F1117; color: #E8ECF0; }
[data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #30363D; }

/* Cards */
.grade-card {
    background: linear-gradient(135deg, #1A2535 0%, #1E2D40 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 10px;
}
.grade-card h2 { margin: 0; font-size: 2.4rem; font-weight: 700; }
.grade-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #8B949E; letter-spacing: 0.08em; text-transform: uppercase; }

/* Quality badge */
.badge-g1   { background: #0D3321; border: 1px solid #2DA44E; border-radius: 8px; padding: 10px 18px; }
.badge-g2   { background: #2D1E00; border: 1px solid #E3A234; border-radius: 8px; padding: 10px 18px; }
.badge-rej  { background: #3B1016; border: 1px solid #CF222E; border-radius: 8px; padding: 10px 18px; }

/* Metric pill */
.defect-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.pill-healthy  { background:#0D3321; color:#3FB950; }
.pill-cracked  { background:#1E2F4D; color:#58A6FF; }
.pill-discolor { background:#2D2008; color:#D29922; }
.pill-moldy    { background:#3B1016; color:#F85149; }

/* Section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #58A6FF;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
    margin: 18px 0 12px;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline() -> MacadamiaGradingPipeline:
    cfg = os.path.join(PROJECT_ROOT, "config.yaml")
    return MacadamiaGradingPipeline(config_path=cfg)


# ── Helpers ────────────────────────────────────────────────────────────────────
def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pill_html(label: str) -> str:
    css = {
        "Healthy / Premium":       "pill-healthy",
        "Cracked / Broken":        "pill-cracked",
        "Discolored / Brown Center": "pill-discolor",
        "Moldy / Insect Damaged":  "pill-moldy",
    }
    cls = css.get(label, "pill-healthy")
    short = label.split("/")[0].strip()
    return f'<span class="defect-pill {cls}">{short}</span>'


def grade_badge_html(grade: str) -> str:
    if "Grade 1" in grade:
        cls, icon, color = "badge-g1", "✅", "#3FB950"
    elif "Grade 2" in grade:
        cls, icon, color = "badge-g2", "⚠️", "#E3A234"
    else:
        cls, icon, color = "badge-rej", "❌", "#F85149"
    return f'<div class="{cls}"><span style="font-size:1.6rem;">{icon}</span><br><b style="color:{color};font-size:1.1rem;">{grade}</b></div>'


def _size_color(grade: str) -> str:
    return {"Small": "#58A6FF", "Medium": "#3FB950", "Large": "#E3A234"}.get(grade, "#8B949E")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Macadamia_nut.jpg/320px-Macadamia_nut.jpg",
             use_container_width=True, caption="Macadamia Quality AI")
    st.markdown("## ⚙️ Inspection Settings")

    input_mode = st.radio("Input Source", ["📂 Upload Image", "📷 Camera Capture"], key="input_mode")

    st.markdown("**Background Mat**")
    mat_type = st.selectbox(
        "Mat Type",
        ["auto", "blue_mat", "green_mat", "dark_mat", "light_mat"],
        format_func=lambda x: {
            "auto": "🤖 Auto-detect",
            "blue_mat": "🔵 Blue Rubber Mat",
            "green_mat": "🟢 Green Mat",
            "dark_mat": "⚫ Dark / Black Mat",
            "light_mat": "⚪ Light / White Mat",
        }.get(x, x),
        key="mat_type",
    )

    st.markdown("**Metric Calibration**")
    use_custom_cal = st.toggle("Override Calibration", value=False, key="use_custom_cal")
    px_per_mm = None
    if use_custom_cal:
        px_per_mm = st.slider("Pixels per mm", 3.0, 25.0, 6.5, 0.5, key="px_per_mm")
        st.caption(f"≈ {1.0 / px_per_mm * 10:.1f} mm per 10 px")

    st.markdown("**Display Options**")
    show_xai_gallery = st.toggle("Show XAI Defect Gallery", value=True, key="show_xai")
    show_per_nut_table = st.toggle("Show Per-Nut Data Table", value=True, key="show_table")
    show_preprocessing = st.toggle("Show Preprocessing Steps", value=False, key="show_pre")

    st.markdown("---")
    st.markdown("**About**")
    st.caption("Macadamia Quality Grading AI v1.0  \nPowered by OpenCV · ONNX · Streamlit  \nOptical feature engine + XAI heatmaps")


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("# 🌰 Macadamia Quality Grading AI")
st.caption("Upload or capture a photo of a macadamia batch → get instant AI-powered sizing, defect classification, and an exportable audit report.")

# ── Image Ingestion Area ────────────────────────────────────────────────────────
image_data: Optional[bytes] = None
uploaded_filename = "batch"

ingest_col, preview_col = st.columns([1, 1], gap="large")
with ingest_col:
    if "Upload" in input_mode:
        uploaded = st.file_uploader(
            "Drop a batch photo here",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            key="file_up",
            help="JPEG or PNG recommended. Resolution ≥ 400×400 px.",
        )
        if uploaded:
            image_data = uploaded.read()
            uploaded_filename = os.path.splitext(uploaded.name)[0]
    else:
        cam_img = st.camera_input("Point camera at the macadamia tray and click 📷", key="cam_in")
        if cam_img:
            image_data = cam_img.read()
            uploaded_filename = f"camera_{int(time.time())}"

with preview_col:
    if image_data:
        pil_raw = Image.open(io.BytesIO(image_data))
        st.image(pil_raw, caption="📥 Uploaded / Captured Image", use_container_width=True)
    else:
        st.info("👈 Upload a batch photo or use the camera to begin grading.")

# ── Run Analysis ───────────────────────────────────────────────────────────────
st.markdown("---")

run_button = st.button(
    "🔬 Run Grading Analysis",
    type="primary",
    disabled=(image_data is None),
    use_container_width=True,
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run_button and image_data:
    with st.spinner("🔄 Preprocessing → Segmenting → Detecting → Classifying → Generating XAI…"):
        try:
            pipeline = load_pipeline()
            result = pipeline.process_image(
                image_data,
                pixels_per_mm_override=px_per_mm,
                mat_type_override=mat_type,
            )
            st.session_state.last_result = result
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.stop()

result = st.session_state.last_result

if result:
    summary = result["summary"]
    analytics = result["analytics"]
    nuts = result["nuts"]

    # ── KPI Banner ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Batch Grade Summary</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="grade-card"><h2>{summary['total_count']}</h2><p>Total Nuts</p></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="grade-card"><h2 style="color:#3FB950">{summary['premium_rate_pct']}%</h2><p>Premium Rate</p></div>""", unsafe_allow_html=True)
    with k3:
        defect_color = "#F85149" if summary['defect_rate_pct'] > 10 else ("#E3A234" if summary['defect_rate_pct'] > 5 else "#3FB950")
        st.markdown(f"""<div class="grade-card"><h2 style="color:{defect_color}">{summary['defect_rate_pct']}%</h2><p>Defect Rate</p></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="grade-card"><h2 style="color:#D29922">{summary['avg_diameter_mm']} mm</h2><p>Avg Diameter</p></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="grade-card"><h2 style="color:#8B949E">{summary['processing_time_sec']}s</h2><p>Inference Time</p></div>""", unsafe_allow_html=True)

    # Overall grade badge
    st.markdown(grade_badge_html(summary["quality_grade"]), unsafe_allow_html=True)
    st.markdown(f"**Batch ID:** `{summary['batch_id']}` &nbsp;|&nbsp; **Timestamp:** {summary['timestamp']}", unsafe_allow_html=True)

    # ── Annotated Output Image ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖼️ Annotated Batch View</div>', unsafe_allow_html=True)
    ann_pil = bgr_to_pil(result["images"]["annotated"])
    st.image(ann_pil, caption="AI-annotated inspection: Bounding boxes (green=healthy, blue=cracked, amber=discolored, red=moldy)", use_container_width=True)

    # ── Preprocessing Steps ─────────────────────────────────────────────────────
    if show_preprocessing:
        st.markdown('<div class="section-header">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.image(bgr_to_pil(result["images"]["preprocessed"]), caption="1. Color-normalized", use_container_width=True)
        with pc2:
            mask_rgb = cv2.cvtColor(result["images"]["binary_mask"], cv2.COLOR_GRAY2BGR)
            st.image(bgr_to_pil(mask_rgb), caption="2. Segmentation Mask", use_container_width=True)
        with pc3:
            st.image(bgr_to_pil(result["images"]["foreground"]), caption="3. Foreground Extracted", use_container_width=True)

    # ── Analytics Charts ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Distribution Analytics</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Size Distribution**")
        size_df = pd.DataFrame(
            {"Size Grade": list(analytics["size_distribution"].keys()),
             "Count":      list(analytics["size_distribution"].values())}
        ).set_index("Size Grade")
        st.bar_chart(size_df, color=["#58A6FF"])

    with chart_col2:
        st.markdown("**Defect Classification Breakdown**")
        defect_df = pd.DataFrame(
            {"Category": list(analytics["defect_distribution"].keys()),
             "Count":    list(analytics["defect_distribution"].values())}
        ).set_index("Category")
        st.bar_chart(defect_df, color=["#3FB950"])

    # Detailed percentages
    pct_cols = st.columns(4)
    for i, (label, pct) in enumerate(analytics["defect_percentages"].items()):
        with pct_cols[i]:
            st.metric(label.split("/")[0].strip(), f"{pct}%")

    # ── XAI Defect Gallery ──────────────────────────────────────────────────────
    if show_xai_gallery and result["xai_defects"]:
        st.markdown('<div class="section-header">🔬 XAI Defect Spotlight Gallery</div>', unsafe_allow_html=True)
        st.caption("Each panel shows: **[1] Raw crop** | **[2] Grad-CAM heatmap** | **[3] Defect contours**. Red = high anomaly intensity.")

        xai_nuts = result["xai_defects"]
        cols_per_row = 2
        for row_start in range(0, min(len(xai_nuts), 8), cols_per_row):
            row_nuts = xai_nuts[row_start: row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for col, nut in zip(row_cols, row_nuts):
                with col:
                    xai_pil = bgr_to_pil(nut["xai_side_by_side"])
                    st.image(xai_pil, use_container_width=True)
                    st.markdown(
                        f"{pill_html(nut['class_label'])} "
                        f"**#{nut['nut_id']}** · {nut['size_grade']} · {nut['diameter_mm']} mm · "
                        f"Conf: {nut['confidence']:.0%} · Surface affected: {nut['defect_severity_pct']}%",
                        unsafe_allow_html=True,
                    )
                    st.caption(nut.get("diagnostic_notes", ""))
    elif show_xai_gallery and not result["xai_defects"]:
        st.success("✅ No defective nuts detected — batch passes visual XAI audit.")

    # ── Per-Nut Data Table ──────────────────────────────────────────────────────
    if show_per_nut_table and nuts:
        st.markdown('<div class="section-header">📋 Per-Nut Inspection Record</div>', unsafe_allow_html=True)
        table_rows = []
        for n in nuts:
            table_rows.append({
                "ID": f"#{n['nut_id']}",
                "Size Grade": n["size_grade"],
                "Diameter (mm)": n["diameter_mm"],
                "Quality Class": n["class_label"],
                "Confidence": f"{n['confidence']:.0%}",
                "Is Defective": "⚠️ YES" if n["is_defective"] else "✅ NO",
                "Affected Area": f"{n['defect_severity_pct']}%",
                "Diagnostic Finding": n["diagnostic_notes"][:72] + ("…" if len(n["diagnostic_notes"]) > 72 else ""),
            })
        df = pd.DataFrame(table_rows)

        # Color rows by defect status
        def row_style(row):
            if "YES" in str(row.get("Is Defective", "")):
                return ["background-color: #2D1E00"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(row_style, axis=1), use_container_width=True, height=320)

    # ── Commercial Standards Summary ────────────────────────────────────────────
    with st.expander("📜 Commercial Grading Standards Reference", expanded=False):
        st.markdown("""
| Grade | Total Defects | Mold/Insect | Application |
|---|---|---|---|
| **Grade 1 (Premium Export)** | ≤ 5% | ≤ 1% | Retail whole kernels, premium export |
| **Grade 2 (Commercial)** | ≤ 15% | ≤ 3% | Confectionery, oil extraction, paste |
| **Reject (Substandard)** | > 15% | — | Rejected / rework required |
        """)
        st.markdown("*Size categories: Small < 18 mm · Medium 18–22 mm · Large > 22 mm (equivalent circular diameter)*")

    # ── Download Reports ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬇️ Download Audit Reports</div>', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        try:
            pdf_bytes = generate_pdf_report(result)
            st.download_button(
                label="📄 Download PDF Certificate",
                data=pdf_bytes,
                file_name=f"macadamia_quality_{summary['batch_id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation error: {e}")

    with dl_col2:
        try:
            csv_text = generate_csv_report(result)
            st.download_button(
                label="📊 Download CSV Data Export",
                data=csv_text,
                file_name=f"macadamia_quality_{summary['batch_id']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"CSV generation error: {e}")

    # ── Run History ─────────────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    # Keep latest 10
    if run_button:
        st.session_state.history.insert(0, {
            "batch_id": summary["batch_id"],
            "timestamp": summary["timestamp"],
            "nuts": summary["total_count"],
            "grade": summary["quality_grade"],
            "defect_pct": summary["defect_rate_pct"],
        })
        st.session_state.history = st.session_state.history[:10]

    if len(st.session_state.history) > 1:
        with st.expander("🕒 Inspection History", expanded=False):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
