# DrainMind

**Computer Vision-Based Drainage Blockage and Waterlogging Risk Intelligence System**

DrainMind analyses images and videos of drainage infrastructure to identify visible debris, potential drainage obstruction, and standing water. It produces explainable inspection-priority scores and maintains a searchable local inspection history.

> **Version:** 1.0.0 · **Licence:** For academic/research/demonstration use · **Runtime:** CPU-only laptop

---

## Architecture

```
DrainMind/
├── app.py                      ← Streamlit entry point + Overview Dashboard
├── requirements.txt
├── configs/
│   ├── settings.yaml           ← Default configuration
│   └── user_settings.yaml      ← User overrides (git-ignored)
├── src/
│   ├── config.py               ← Config loader with deep-merge
│   ├── vision/
│   │   ├── detector.py         ← YOLOv8 detector with fallback to stub
│   │   ├── segmenter.py        ← Segmentation + OpenCV heuristic fallback
│   │   ├── preprocessing.py    ← Image load/save/resize utilities
│   │   └── visualization.py    ← Bounding boxes, overlays, info bars
│   ├── analysis/
│   │   ├── coverage.py         ← ROI-based mask intersection coverage
│   │   ├── condition_assessment.py ← Detection classification + quality
│   │   ├── priority_score.py   ← Transparent R = 0.50D + 0.30W + 0.20T
│   │   └── change_detection.py ← ORB alignment + temporal comparison
│   ├── video/
│   │   ├── capture.py          ← VideoCapture with clean resource release
│   │   └── stream_processor.py ← Frame sampling + aggregation
│   ├── database/
│   │   ├── db.py               ← SQLite schema + thread-local connections
│   │   └── repository.py       ← CRUD + dashboard stats queries
│   ├── reporting/
│   │   ├── csv_export.py       ← Pandas CSV export
│   │   └── pdf_export.py       ← fpdf2 PDF with graceful degradation
│   └── utils/
│       ├── validation.py       ← File/input validation
│       ├── paths.py            ← Safe filenames + path traversal prevention
│       └── logging_config.py   ← Rotating file + console logging
├── pages/
│   ├── 1_Image_Inspection.py
│   ├── 2_Video_Analysis.py
│   ├── 3_Inspection_History.py
│   ├── 4_Location_Comparison.py
│   ├── 5_Reports.py
│   └── 6_Settings.py
├── assets/
│   └── style.css               ← Full CSS design system
├── models/
│   └── README.md               ← Model placement + training guide
├── data/                       ← SQLite DB + logs + raw images
├── outputs/                    ← Annotated images + PDF reports
└── tests/
    ├── conftest.py
    └── test_drainmind.py       ← 30+ unit tests
```

---

## Quick Start

### 1. Create and activate a virtual environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

YOLOv8 (Ultralytics) will be installed automatically. Model weights (`yolov8n.pt`, ≈6 MB) are downloaded on first inference if you have network access.

For PDF export support:
```bash
pip install fpdf2
```

### 3. Run the application

```bash
streamlit run app.py
```

The application opens at [http://localhost:8501](http://localhost:8501).

### 4. Run tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Pages

| Page | Description |
|---|---|
| Overview | Dashboard with KPIs, activity charts, recent inspections |
| Image Inspection | Upload + analyse a single image, save results |
| Video Analysis | Process uploaded video or webcam footage |
| Inspection History | Searchable filterable table with detail view |
| Location Comparison | Side-by-side comparison with image alignment |
| Reports | CSV and PDF export with summary statistics |
| Settings | Model config, scoring weights, diagnostics |

---

## Model Configuration

### No custom weights (default)

DrainMind downloads `yolov8n.pt` automatically on first use (requires internet once).
This model is trained on COCO and **does not specifically recognise drainage objects**.
Coverage estimates are heuristic only.

### Custom drainage-specific weights

1. Train or obtain a YOLOv8 model with drainage-specific classes (see `models/README.md`).
2. Place the `.pt` file in the `models/` directory.
3. Navigate to **Settings → Custom model weights path** and enter the absolute path.
4. Click **Save Settings**.

See [`models/README.md`](models/README.md) for full training guidance.

---

## Inspection Priority Score

```
R = 0.50 × D + 0.30 × W + 0.20 × T
```

| Component | Meaning |
|---|---|
| D | Visible obstruction coverage [0–100] |
| W | Standing water indicator [0–100] |
| T | Worsening trend score [0–100] |

**This is an inspection priority indicator, NOT a flood risk coefficient.**
Weights and thresholds are configurable in Settings.

| Score | Category |
|---|---|
| 0–24 | Lower inspection priority |
| 25–49 | Monitor |
| 50–74 | Inspection recommended |
| 75–100 | High inspection priority |

---

## Running Offline

After the first successful launch (which downloads `yolov8n.pt`), the application runs
completely offline. No external API calls are made during normal operation.

To run fully offline from the start, download `yolov8n.pt` manually from
[https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)
and set the model path in Settings.

---

## Troubleshooting

### Application won't start
- Verify `streamlit` is installed: `pip install streamlit`
- Run from the project root: `streamlit run app.py`

### Webcam not accessible
- Try camera index 0, 1, or 2 in the Video Analysis page.
- On Windows: ensure no other application is using the camera.
- The application **does not** require a webcam — prerecorded video works without one.

### Model download fails (offline environment)
- Download `yolov8n.pt` manually and configure the path in Settings.
- The application still runs with heuristic-only analysis if no model is loaded.

### PDF export not available
- Install fpdf2: `pip install fpdf2`

### Database issues
- The database is created automatically at `data/drainmind.db`.
- Check write permissions on the `data/` directory.

---

## Limitations and Validation

- **Generic model**: Without drainage-specific training, object classes reflect COCO (80 classes, none drainage-specific).
- **Coverage measurement**: Pixel-level, not hydraulic. Does not represent drain capacity.
- **Priority score**: Configurable composite, not a validated engineering risk metric.
- **Image alignment**: ORB-based; fails on low-texture or heavily different viewpoints.
- **Heuristic water detection**: False positives on dark pavement are expected.
- **Single-user local deployment**: No authentication, not suitable for multi-user production.
- **CPU-only**: Processing is slower than GPU; video analysis uses frame sampling to manage load.

---

## Privacy and Security

- All data stored locally. No external telemetry or API calls.
- File-type and size validation on all uploads.
- Path traversal prevention on all file writes.
- No shell command injection vectors.
- Deletion requires explicit checkbox confirmation.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection framework (AGPL-3.0)
- [OpenCV](https://opencv.org/) — image processing
- [Streamlit](https://streamlit.io/) — application framework
- [fpdf2](https://github.com/py-pdf/fpdf2) — PDF generation
