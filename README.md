# 🌰 Macadamia Quality Grading AI System

> **Automated optical inspection, metric sizing, defect classification & explainable AI reporting for commercial macadamia processing.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-red)](https://streamlit.io)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.32-orange)](https://onnxruntime.ai)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.1-blueviolet)](https://opencv.org)
[![Tests](https://img.shields.io/badge/Tests-67%2F67%20passing-brightgreen)](#)

---

## ✨ Features

| Capability | Detail |
|---|---|
| 📷 **Multi-source Ingestion** | File upload, camera capture, base64, byte stream |
| 🧹 **Preprocessing** | Gray World / White Patch WB, CLAHE illumination, letterbox resize |
| 🎯 **Background Segmentation** | Adaptive border-LAB + Watershed nut separation; supports blue, green, dark, light & auto mats |
| 🔍 **Nut Detection & Sizing** | Contour + NMS localization → diameter in mm, size grade (S/M/L) |
| 🩺 **Defect Classification** | 4-class optical feature engine + ONNX CNN ensemble |
| 🧠 **XAI Heatmaps** | Grad-CAM saliency maps + defect contour overlays per nut |
| 📊 **Batch Analytics** | Distribution charts, premium %, defect rate, commercial grade |
| 📄 **Reports** | PDF certificate + CSV data export, downloadable from UI & API |
| 🚀 **Deployment** | FastAPI REST API + Streamlit dashboard; Docker-ready |

---

## 📁 Repository Structure

```
macadamia-grading/
├── config.yaml                       # Hyperparameters, class labels, grading standards
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Production container build
├── docker-entrypoint.sh              # Launches API + UI together
├── run.py                            # Local dev launcher (--api | --ui | --demo | --test)
│
├── data/
│   ├── raw/                          # Input batch photos
│   ├── processed/                    # Annotated images, PDF/CSV outputs
│   ├── annotations/                  # COCO format ground-truth annotations
│   └── generate_sample_data.py       # Synthetic dataset generator
│
├── src/
│   ├── pipeline.py                   # 🏗️  End-to-end orchestrator
│   ├── ingestion/
│   │   └── handler.py                # File / bytes / base64 / camera ingest
│   ├── preprocessing/
│   │   ├── normalization.py          # WB, CLAHE, letterbox
│   │   ├── segmentation.py           # Background subtraction & Watershed
│   │   └── calibration.py           # Pixel ↔ mm metric conversion
│   ├── models/
│   │   ├── detector.py               # Nut localization + metric sizing
│   │   ├── classifier.py             # 4-class defect classifier (optical + ONNX)
│   │   ├── model_builder.py          # Build baseline ONNX model weights
│   │   ├── train_yolo.py             # YOLOv8 fine-tuning script (GPU env)
│   │   ├── train_classifier.py       # MobileNetV2 transfer learning (GPU env)
│   │   └── weights/
│   │       ├── defect_classifier.onnx
│   │       └── nut_detector.onnx
│   ├── xai/
│   │   └── gradcam.py                # Grad-CAM saliency + defect audit cards
│   └── reporting/
│       └── report_generator.py       # PDF certificate + CSV export
│
├── deployment/
│   ├── api/
│   │   └── main.py                   # FastAPI REST endpoints
│   └── ui/
│       └── app.py                    # Streamlit inspection dashboard
│
└── tests/
    └── test_pipeline.py              # 67 unit + integration tests
```

---

## 🚀 Quick Start

### Option 1 — Local (Python only)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd macadamia-grading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build baseline ONNX model weights
python src/models/model_builder.py

# 4. Generate sample batch images
python data/generate_sample_data.py

# 5a. Start Streamlit dashboard ONLY
python run.py --ui

# 5b. Start FastAPI API ONLY
python run.py --api

# 5c. Start BOTH (API in background thread, UI foreground)
python run.py

# 5d. Run demo inference on all sample batches → data/processed/
python run.py --demo

# 5e. Run test suite
python run.py --test
```

### Option 2 — Docker

```bash
# Build container
docker build -t macadamia-grading:1.0 .

# Run with both ports exposed
docker run -p 8000:8000 -p 8501:8501 macadamia-grading:1.0

# Mount your own batch images
docker run -p 8000:8000 -p 8501:8501 \
  -v /path/to/your/photos:/app/data/raw \
  macadamia-grading:1.0
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI REST API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## 🌐 API Reference

### `POST /api/v1/grade` — Upload & grade a batch
```bash
curl -X POST http://localhost:8000/api/v1/grade \
  -F "file=@data/raw/batch_01_premium.jpg" \
  -F "mat_type=blue_mat"
```

### `POST /api/v1/grade/base64` — Grade from base64 string
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4...",
  "pixels_per_mm": 6.5,
  "mat_type": "auto"
}
```

### `POST /api/v1/report/pdf` — Download PDF certificate
```bash
curl -X POST http://localhost:8000/api/v1/report/pdf \
  -F "file=@data/raw/batch_01_premium.jpg" \
  -o certificate.pdf
```

### `POST /api/v1/report/csv` — Download CSV data export
```bash
curl -X POST http://localhost:8000/api/v1/report/csv \
  -F "file=@data/raw/batch_01_premium.jpg" \
  -o report.csv
```

---

## 🩺 Defect Classes

| Class ID | Label | Description | Grade 1 Tolerance |
|---|---|---|---|
| 0 | **Healthy / Premium** | Smooth ivory kernel, no fissures or mold | — |
| 1 | **Cracked / Broken** | Visible fracture, fissure lines | ≤ 2% |
| 2 | **Discolored / Brown Center** | Internal enzymatic browning | ≤ 2% |
| 3 | **Moldy / Insect Damaged** | Fungal mold, borehole, rot | ≤ 1% |

## 📐 Size Grades

| Grade | Diameter Range | Commercial Application |
|---|---|---|
| Small | < 18 mm | Confectionery / diced kernels |
| Medium | 18 – 22 mm | Standard retail snack packs |
| Large | > 22 mm | Jumbo wholes / premium export |

---

## 🎯 Training Your Own Model

### YOLOv8 Detector (requires GPU environment + `ultralytics`)
```bash
pip install ultralytics
python src/models/train_yolo.py \
  --data_dir data \
  --epochs 50 \
  --img_size 640 \
  --batch_size 16
```

### MobileNetV2 Defect Classifier (requires GPU + `torch torchvision`)
```bash
pip install torch torchvision
python src/models/train_classifier.py \
  --data_dir data/processed/crops \
  --epochs 25 \
  --batch_size 32
```

> Dataset layout expected by the classifier trainer:
> ```
> data/processed/crops/
>     train/
>         healthy/        ← class 0 crops
>         cracked/        ← class 1 crops
>         discolored/     ← class 2 crops
>         moldy/          ← class 3 crops
>     val/
>         healthy/ cracked/ discolored/ moldy/
> ```

Annotate images with [Roboflow](https://roboflow.com) or [Label Studio](https://labelstud.io) and export in COCO / YOLO format.

---

## 🧪 Test Suite

```bash
python -m pytest tests/ -v
# 67 passed in ~4s
```

Coverage includes: ingestion, normalization, segmentation, calibration, detection, classification, XAI engine, end-to-end pipeline, PDF/CSV reporting.

---

## 🗺️ Roadmap

- [ ] Fine-tuned YOLOv8n weights on real macadamia imagery
- [ ] Mobile deployment (Flutter + TFLite / CoreML export)
- [ ] Multi-batch session history & trend dashboard
- [ ] Integration with Roboflow dataset management API
- [ ] Real-time conveyor belt video stream mode
- [ ] AWS S3 / Google Cloud Storage report archival

---

## 📄 License

MIT License — open for research, agricultural R&D, and commercial deployment.
