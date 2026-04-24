"""
app.py: Streamlit front-end for Diabetic Retinopathy Detection.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import torch
from PIL import Image
from pathlib import Path

from src.model import load_model
from src.inference import CLASS_NAMES, compute_gradcam, predict, preprocess

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DRD · Retinopathy Screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_COLORS = ["#3fb950", "#7ee787", "#d29922", "#f0883e", "#f85149"]
SEVERITY_DESCS = [
    "No signs of diabetic retinopathy detected.",
    "Microaneurysms only — early changes present.",
    "More than just microaneurysms — moderate non-proliferative DR.",
    "Severe non-proliferative DR — high risk of progression.",
    "Proliferative DR — advanced, vision-threatening stage.",
]
DEFAULT_CHECKPOINT = str(Path("models") / "best_drd_model.pt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights...")
def _load(checkpoint: str) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cpu")
    model = load_model(checkpoint, device)
    return model, device

def _confidence_chart(probs: np.ndarray):
    """Horizontal bar chart for probability distribution."""
    df = pd.DataFrame({
        "Grade": CLASS_NAMES,
        "Confidence": (probs * 100).round(2),
        "Color": SEVERITY_COLORS
    })

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 100]), title="Confidence (%)"),
        y=alt.Y("Grade:N", sort=CLASS_NAMES, title=None),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Grade", "Confidence"]
    ).properties(height=200)
    
    return chart

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    checkpoint_path = st.text_input(
        "Model checkpoint path",
        value=DEFAULT_CHECKPOINT,
        help="Path relative to the project root.",
    )
    st.divider()
    st.markdown("""
    **Architecture:** EfficientNet-B3 + Attention  
    **Dataset:** APTOS 2019 (5-class grading)  
    **Input:** 384x384, Ben Graham preprocessing  
    **Explainability:** Grad-CAM on last conv block  
    """)
    st.caption("CPU inference only. No data is stored or transmitted.")

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.title("🔬 Retinopathy Screener")
st.caption("Diabetic Retinopathy Detection | AI-Assisted Grading")
st.divider()

# ---------------------------------------------------------------------------
# Upload & Initialization
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a fundus photograph (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a fundus photograph to begin screening.")
    st.stop()

if not Path(checkpoint_path).exists():
    st.error(f"Checkpoint not found at **{checkpoint_path}**. Verify the path in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
model, device = _load(checkpoint_path)
pil_img = Image.open(uploaded).convert("RGB")

try:
    with st.spinner("Running inference..."):
        tensor, rgb_float = preprocess(pil_img)
        pred_class, predict_probs = predict(model, tensor, device)

    with st.spinner("Computing Grad-CAM saliency map..."):
        cam_overlay = compute_gradcam(model, tensor, rgb_float, device, target_class=pred_class)
except Exception as exc:
    st.error(f"Inference failed: {exc}\n\nCheck that the model checkpoint matches the expected architecture.")
    st.stop()

# ---------------------------------------------------------------------------
# Results: Overview Metrics
# ---------------------------------------------------------------------------
st.subheader("Diagnosis Overview")

conf_pct = predict_probs[pred_class] * 100
entropy = float(-np.sum(predict_probs * np.log(predict_probs + 1e-9)))
max_entropy = float(np.log(len(predict_probs)))
uncertainty_pct = (entropy / max_entropy) * 100

sorted_idx = np.argsort(predict_probs)[::-1]
runner_idx = sorted_idx[1]

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Grade", CLASS_NAMES[pred_class])
col2.metric("Model Confidence", f"{conf_pct:.1f}%", f"Uncertainty: {uncertainty_pct:.0f}%", delta_color="inverse")
col3.metric("Runner-up", CLASS_NAMES[runner_idx], f"{predict_probs[runner_idx]*100:.1f}% prob", delta_color="off")

# Display the severity description nicely
st.success(f"**Clinical Note:** {SEVERITY_DESCS[pred_class]}")

# ---------------------------------------------------------------------------
# Results: Imaging Panels
# ---------------------------------------------------------------------------
st.subheader("Imaging & Explainability")
col_l, col_r = st.columns(2)

with col_l:
    st.image(pil_img, caption="Original Fundus Image (Input)", width="stretch")

with col_r:
    st.image(cam_overlay, caption=f"Grad-CAM Saliency — Grade {pred_class}", width="stretch")

# ---------------------------------------------------------------------------
# Results: Probability Chart
# ---------------------------------------------------------------------------
st.subheader("Class Probability Distribution")
st.altair_chart(_confidence_chart(predict_probs), width="stretch")

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.warning(
    "**Research use only.** This tool is not a certified medical device and must not be used for "
    "clinical diagnosis or patient management. Predictions carry inherent model uncertainty. "
    "All findings must be confirmed by a qualified ophthalmologist."
)