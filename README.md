# Diabetic Retinopathy Detection Web App

EfficientNet-B3 + Hybrid Attention · APTOS 2019 · Grad-CAM · Streamlit

---

## Project Structure

```
project-root/
├── models/
│   └── best_drd_model.pt     ← place your checkpoint here
├── src/
│   ├── __init__.py
│   ├── model.py              ← DRDModel architecture
│   └── inference.py          ← preprocessing, prediction, Grad-CAM
├── .gitignore
├── app.py                    ← Streamlit UI (run from project root)
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install PyTorch (CPU-only, ~230 MB)

```bash
pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your checkpoint

```
models/best_drd_model.pt
```

### 5. Launch the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## Usage

1. Upload a fundus photograph (JPG or PNG).
2. The app runs Ben Graham sharpening → Albumentations normalisation → EfficientNet-B3 inference.
3. Results panel shows:
   - Predicted DR grade & severity description
   - Softmax confidence score
   - Per-class probability bar chart
   - Grad-CAM heatmap blended over the original image

---

## Architecture Summary

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNet-B3 (timm, `pretrained=False`) |
| Attention | Channel Attention (ratio 16) + Spatial Attention (7×7 conv) |
| Pooling | Concat of AdaptiveAvgPool + AdaptiveMaxPool → 2×feat_dim |
| Head | LayerNorm → Dropout(0.3) → Linear(feat_dim×2, 5) |
| Grad-CAM target | `model.cnn.blocks[-1][-1].bn2` |
| Image size | 384 × 384 |
| Classes | No DR / Mild / Moderate / Severe / PDR |

---

## Disclaimer

This tool is for **research purposes only** and is not a certified medical
device. All outputs must be reviewed by a qualified ophthalmologist before
any clinical decisions are made.