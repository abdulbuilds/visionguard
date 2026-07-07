---
title: TrafficSight AI
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.2"
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Traffic Sign Detection System — v2.0

> Real-Time Traffic Sign Recognition using **YOLOv8** · **Streamlit** · **pyttsx3**

---

## 🗂️ Project Structure

```
Road Sign Detection/
│
├── app.py                      ← Main Streamlit application (v2.0)
├── requirements.txt
│
├── models/
│   └── best.pt                 ← ⚠️ Place your YOLOv8 model here
│
├── outputs/
│   ├── detection_log.csv       ← Auto-created detection log
│   └── screenshots/            ← Saved annotated frames
│
├── assets/                     ← Static assets
│
└── utils/
    ├── __init__.py
    ├── detection.py            ← YOLOv8 inference helpers
    ├── voice_alert.py          ← pyttsx3 TTS engine (thread-safe)
    └── logger.py               ← CSV detection logger
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Copy `best.pt` into the `models/` folder.

### 3. Run the app
```bash
streamlit run app.py
```

---

## ✨ Features (v2.0)

| Feature | Description |
|---|---|
| 🎯 **Image Detection** | Upload JPG/PNG, run YOLOv8, see bounding boxes |
| 🔊 **Voice Alerts** | pyttsx3 announces detected sign names (3-sec debounce) |
| 📹 **Live Webcam** | Real-time detection with FPS counter & frame skipping |
| ✂️ **Sign Cropping** | Auto-crops each detected sign for side-panel inspection |
| 📸 **Screenshot** | Save annotated frames to `outputs/screenshots/` |
| 📊 **Analytics** | Histogram, bar, pie, radar, session timeline charts |
| 📈 **Session Stats** | Aggregated class counts, avg confidence, run history |
| 📜 **Detection Log** | Persistent CSV log with filter & export |
| 🖥️ **Dashboard Widgets** | Model status, class count, FPS, total detections |
| 💾 **Exports** | Annotated image PNG · CSV · JSON |

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Confidence Threshold | 0.25 | Ignore detections below this |
| IoU Threshold | 0.45 | NMS overlap threshold |
| Voice Alerts | ON | Toggle TTS announcements |
| Frame Skip | 1 | Webcam: process every N+1 frames |
| Camera Index | 0 | Webcam device index |

---

## 📋 Detection Log Format

`outputs/detection_log.csv` columns:

| Column | Description |
|---|---|
| Timestamp | Date & time of detection |
| Class Name | Detected sign class |
| Confidence (%) | Detection confidence |
| Source | `Upload` or `Webcam` |
