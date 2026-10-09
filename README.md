# DRIVERGUARD AI
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://driverguard-ai.streamlit.app)
[![Streamlit App](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://driverguard-ai.streamlit.app)

> 🚀 **Live Interactive Demo**: Try DriverGuard AI directly in your browser: **[driverguard-ai.streamlit.app](https://driverguard-ai.streamlit.app)**

### Real-Time Driver Drowsiness & Fatigue Detection System
> **Production-grade Computer Vision & Generative AI edge assistance system for commercial and consumer vehicular safety.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Gemini_1.5-Flash-8E75C2?style=for-the-badge&logo=google-gemini&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary

**DriverGuard AI** is a real-time computer vision and multimodal AI monitoring system engineered to combat driver fatigue and prevent roadway collisions caused by microsleeps.

Combining **MediaPipe Face Mesh (468 3D landmarks)**, **Contrast-Enhanced Dual-Eye Preprocessing (CLAHE)**, a custom **Deep Convolutional Neural Network (CNN)**, and a **Temporal Finite State Machine**, the system accurately distinguishes natural involuntary blinks from hazardous prolonged eyelid closures. 

A rolling multi-signal mathematical engine computes an **AI-Derived Fatigue Indicator (0–100)** incorporating **PERCLOS (Percentage of Eye Closure)** and blink frequency anomalies. Sessions are persisted in an **SQLite** time-series database and synthesized into structured executive safety reports using **Google Gemini 1.5** orchestrated via **LangChain**.

---

## 🏗️ System Architecture

```
                                  DRIVERGUARD AI PIPELINE
                                  
  ┌─────────────────┐        ┌───────────────────────┐        ┌─────────────────────────┐
  │  Webcam Video / │ ────▶  │  MediaPipe Face Mesh  │ ────▶  │ Dual-Eye Region BBoxes  │
  │ Demo Stream MP4 │        │   (468 3D Landmarks)  │        │   & Geometric EAR Crop  │
  └─────────────────┘        └───────────────────────┘        └─────────────────────────┘
                                                                           │
                                                                           ▼
  ┌─────────────────────────┐        ┌───────────────────────┐        ┌─────────────────────────┐
  │ Temporal State Machine  │ ◀────  │ Batched CNN Inference │ ◀────  │  CLAHE Normalization    │
  │ AWAKE ▶ BLINK ▶ DROWSY  │        │  (64x64x3 Dual Eye)   │        │     & RGB Resizing      │
  └─────────────────────────┘        └───────────────────────┘        └─────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Rolling AI-Derived Fatigue Engine (0–100)               │
  │ • PERCLOS (60s window)  • Prolonged Closure Penalty     │
  │ • Blink Rate Deviation  • Critical Event Multipliers   │
  └─────────────────────────────────────────────────────────┘
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
  ┌───────────────────────────┐    ┌───────────────────────────┐
  │ Multi-Modal Alert HUD     │    │ SQLite Session Database   │
  │ • Visual Warning Banners  │    │ • Telemetry Time-Series   │
  │ • Async Audio Alerts      │    │ • Automated CSV Export    │
  └───────────────────────────┘    └───────────────────────────┘
                                                │
                                                ▼
                                   ┌───────────────────────────┐
                                   │  Gemini 1.5 + LangChain   │
                                   │  • Structured JSON Report │
                                   │  • Safe AI Copilot Chat   │
                                   └───────────────────────────┘
```

---

## ✨ Key Features

1. **Precision Facial Landmark Tracking**
   - Tracks 468 facial mesh coordinates using Google MediaPipe.
   - Automatically handles face acquisition, lost frames, center-driver isolation, and multi-face edge cases.
2. **Dual-Eye Region Extraction & Preprocessing**
   - Landmark-driven dynamic eye bounding boxes with configurable padding margins.
   - Contrast-Limited Adaptive Histogram Equalization (**CLAHE**) in LAB color space to stabilize detection under variable cockpit lighting.
3. **Deep CNN Eye State Classifier**
   - Custom 3-block convolutional neural network with Batch Normalization, Dropout, and Global Average Pooling.
   - Predicts independent probabilities for both left and right eye states (`OPEN` vs. `CLOSED`).
   - Batched dual-eye tensor inference (`2, 64, 64, 3`) for minimal CPU/GPU latency.
4. **Temporal Drowsiness State Machine**
   - Distinguishes brief physiological blinks (80–380 ms) from prolonged microsleeps (> 1200 ms).
   - Multi-tier state transitions: `AWAKE` ➔ `BLINKING` ➔ `DROWSY (Warning)` ➔ `CRITICAL ALERT`.
   - Smooth recovery back to `AWAKE` requiring consecutive verified open frames.
5. **AI-Derived Fatigue Score (0–100)**
   - Computes rolling **PERCLOS** over a 60-second sliding window.
   - Factors in blink frequency deviations (hypo-blinking vs. rapid compensatory blinking).
   - Severity bands: **Low (0–30)**, **Moderate (31–60)**, **High (61–80)**, and **Critical (81–100)**.
6. **Executive Safety Reporting (Gemini 1.5 + LangChain)**
   - Generates structured, schema-validated JSON reports summarizing driving vigilance.
   - In-cabin conversational safety assistant to answer driver questions based strictly on recorded metrics.
   - **Deterministic Fallback**: If no Gemini API key is configured, the system provides offline rule-based reporting without interruption.
7. **Production Streamlit Dashboard**
   - Dark glassmorphism dashboard aesthetic with real-time KPI cards.
   - Interactive Plotly time-series charts for fatigue curves and eye open probabilities.
   - Single-click CSV export of raw session telemetry.
8. **Recruiter & Demo Mode**
   - Built-in video simulator for instant review without requiring an attached physical webcam.

---

## 📂 Project Structure

```
driverguard-ai/
│
├── app.py                      # Production Streamlit Dashboard
├── README.md                   # System Architecture & Documentation
├── requirements.txt            # Pinned dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Standard Python & dataset exclusions
├── LICENSE                     # MIT License
│
├── config/
│   └── config.yaml             # Central configuration (thresholds, weights, cameras)
│
├── models/
│   ├── eye_classifier.keras    # Exported trained Keras model
│   ├── model_metadata.json     # Hyperparameters and evaluation metadata
│   ├── confusion_matrix.png    # Test set confusion matrix
│   ├── training_curves.png     # Loss & accuracy curves
│   └── README.md               # Model architecture & dataset setup notes
│
├── data/
│   ├── raw/                    # Raw MRL Eye dataset directory
│   ├── processed/              # Normalized train/val/test splits
│   ├── sample/                 # Sample evaluation images
│   └── sessions.db             # Persistent SQLite telemetry database
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA & class distribution
│   ├── 02_eye_classifier_training.ipynb   # Augmentation & CNN training
│   └── 03_model_evaluation.ipynb         # Confusion matrix & latency profiling
│
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── face_detector.py    # MediaPipe Face Mesh wrapper
│   │   ├── landmarks.py        # 468 Mesh indices, EAR geometry & bounding boxes
│   │   ├── eye_detector.py     # Crop extraction & visual overlays
│   │   └── preprocessing.py    # CLAHE contrast, resizing & tensor normalization
│   ├── models/
│   │   ├── cnn_model.py        # CNN architecture, regularization & callbacks
│   │   ├── inference.py        # Batched inference, latency timer & temporal smoothing
│   │   └── evaluation.py       # Metrics calculator & unmeasured placeholders
│   ├── detection/
│   │   ├── blink_detector.py   # Blink duration, rate & count tracking
│   │   ├── drowsiness_detector.py # State machine (AWAKE, BLINK, DROWSY, CRITICAL)
│   │   └── fatigue_score.py    # 0-100 multi-factor fatigue scoring algorithm
│   ├── ai/
│   │   ├── gemini_service.py   # Google Gemini client wrapper
│   │   ├── prompts.py          # Structured JSON prompts & safety guidelines
│   │   └── report_generator.py # LangChain SessionReportChain & Safety Assistant
│   ├── analytics/
│   │   ├── session_logger.py   # SQLite telemetry storage & CSV export
│   │   ├── metrics.py          # Real-time session tracker & statistics
│   │   └── reports.py          # Plotly dark theme visualization charts
│   └── utils/
│       ├── logger.py           # Rotating file & console logging
│       ├── config.py           # YAML loader with environment resolution
│       └── helpers.py          # FPS meter, latency timer, sound alert dispatcher
│
├── tests/
│   ├── test_preprocessing.py   # Tensor resizing & normalization tests
│   ├── test_drowsiness.py      # State machine transition & recovery tests
│   ├── test_fatigue_score.py   # Score boundary & PERCLOS calculation tests
│   └── test_model.py           # Metric calculation & fallback tests
│
└── assets/
    ├── logo/                   # Visual brand assets
    └── demo/
        ├── sample_driver.mp4   # Demo mode test stream
        └── generate_demo_video.py # Synthetic driver video generator
```

---

## 🧠 ML Model Architecture & Training Pipeline

### CNN Architecture Specification (`driverguard_eye_cnn`)

The model receives a normalized RGB eye patch of shape `(64, 64, 3)`:

| Layer (Type) | Output Shape | Parameters | Details |
|---|---|---|---|
| **InputLayer** | `(None, 64, 64, 3)` | 0 | RGB patch normalized to `[0.0, 1.0]` |
| **Conv2D + BN + ReLU** | `(None, 64, 64, 32)` | 896 | 32 filters, 3x3 kernel, L2 Reg (`1e-4`) |
| **Conv2D + BN + ReLU** | `(None, 64, 64, 32)` | 9,248 | 32 filters, 3x3 kernel |
| **MaxPooling2D** | `(None, 32, 32, 32)` | 0 | 2x2 pool size |
| **Dropout (0.15)** | `(None, 32, 32, 32)` | 0 | Spatial regularization |
| **Conv2D + BN + ReLU** | `(None, 32, 32, 64)` | 18,496 | 64 filters, 3x3 kernel |
| **Conv2D + BN + ReLU** | `(None, 32, 32, 64)` | 36,928 | 64 filters, 3x3 kernel |
| **MaxPooling2D** | `(None, 16, 16, 64)` | 0 | 2x2 pool size |
| **Dropout (0.22)** | `(None, 16, 16, 64)` | 0 | Spatial regularization |
| **Conv2D + BN + ReLU** | `(None, 16, 16, 128)` | 73,856 | 128 filters, 3x3 kernel |
| **MaxPooling2D** | `(None, 8, 8, 128)` | 0 | 2x2 pool size |
| **Dropout (0.30)** | `(None, 8, 8, 128)` | 0 | Spatial regularization |
| **GlobalAveragePooling2D** | `(None, 128)` | 0 | Replaces dense flatten; minimizes overfitting |
| **Dense + BN + ReLU** | `(None, 128)` | 16,896 | Fully connected projection |
| **Dropout (0.40)** | `(None, 128)` | 0 | Regularization |
| **Dense (Sigmoid)** | `(None, 1)` | 129 | Binary Output: 0 = CLOSED, 1 = OPEN |

### Training Strategy & Callbacks
- **Data Augmentation**: Small rotations ($\pm 10^\circ$), horizontal shift ($\pm 8\%$), vertical shift ($\pm 8\%$), zoom ($\pm 10\%$), horizontal flipping.
- **EarlyStopping**: Monitored on `val_loss` with patience of 8 epochs (best weights automatically restored).
- **ReduceLROnPlateau**: Factor of 0.3 with patience of 3 epochs (minimum learning rate: `1e-6`).
- **ModelCheckpoint**: Saves the best model checkpoint to `models/eye_classifier.keras`.

---

## 📊 Benchmark & Measured Performance

The pipeline was benchmarked using `scripts/benchmark.py` and `scripts/export_tflite.py`.

### 1. End-to-End Pipeline Latency & Throughput

| Pipeline Stage | Mean Latency (ms) | P95 Latency (ms) | Complexity |
|---|---|---|---|
| **Face Landmarker (MediaPipe Tasks)** | 4.74 ms | 5.45 ms | $O(N)$ Landmark Regression |
| **Eye Region Extraction & CLAHE** | < 0.05 ms | 0.08 ms | Color Space Enhancement |
| **Hybrid Eye Inference (CNN + EAR)** | 0.94 ms | 1.30 ms | Quantized TFLite Engine |
| **Yawn Detection (Mouth Aspect Ratio)** | < 0.02 ms | 0.03 ms | Geometric Ratio Analysis |
| **3D Head Pose (`cv2.solvePnP`)** | 0.02 ms | 0.04 ms | Perspective-n-Point Euler |
| **Fatigue FSM & State Engine** | 0.12 ms | 0.17 ms | Temporal Sliding Buffer |
| **TOTAL FRAME PROCESSING TIME** | **~4.89 ms** | **~5.59 ms** | **204+ Effective FPS** |

### 2. Model Quantization: Native Keras vs. TensorFlow Lite

Using dynamic range quantization (`scripts/export_tflite.py`):

| Model Format | Mean Latency | Disk Size | Speedup / Reduction |
|---|---|---|---|
| **Keras Native (`.keras`)** | 39.27 ms | 717.4 KB | Baseline |
| **TFLite Quantized (`.tflite`)** | **0.94 ms** | **165.0 KB** | **41.9× Faster, 77.0% Smaller** |

---

## 🔬 Hybrid Decision: Deep CNN vs. Eye Aspect Ratio (EAR)

DriverGuard AI integrates a **hybrid decision engine** (`src/models/hybrid_classifier.py`) fusing deep learning feature representations with calibrated geometric facial landmarks:

$$\text{Decision Score} = w_{\text{CNN}} \cdot P_{\text{CNN}}(\text{Open}) + (1 - w_{\text{CNN}}) \cdot P_{\text{EAR}}(\text{Open})$$

### Comparison Matrix

| Attribute | Pure Geometric EAR | Standalone Deep CNN | DriverGuard Hybrid Fusion |
|---|---|---|---|
| **Sensitivity to Lighting** | Moderate (depends on mesh) | High (sensitive to glare/shadows) | **Robust** (CLAHE + dual signals) |
| **Facial Angle Tolerance** | Degrades at extreme yaw | Moderate | **High** (3D head pose compensated) |
| **Subtle Partial Closures** | Low (linear coordinate error) | High (receptive field texture) | **High** |
| **Inference Overhead** | 0.01 ms | 0.94 ms | **< 1.0 ms** |
| **Failure Recovery** | Fails on partial occlusions | Hallucinates on unseen artifacts | **Graceful fallback to EAR** |

---

## ⚡ Real-Time Drowsiness, Yawn & Head Pose Formulation

### 1. Mouth Aspect Ratio (MAR) & Yawn Detection
Using MediaPipe inner and outer lip landmarks (top: 13, bottom: 14, left: 78, right: 308, plus vertical pairs 81/178 and 311/402):
$$\text{MAR} = \frac{\|p_{13} - p_{14}\| + \|p_{81} - p_{178}\| + \|p_{311} - p_{402}\|}{3 \times \|p_{78} - p_{308}\|}$$
A yawn is registered when $\text{MAR} \ge \text{MAR}_{\text{threshold}}$ (default $0.62$, or baseline $+ 0.30$) for $\ge 1.2$ consecutive seconds.

### 2. 3D Head Pose & Nodding Estimation (`cv2.solvePnP`)
Euler rotation angles (Pitch, Yaw, Roll) are computed by projecting standard 3D anthropometric facial points against 2D camera coordinates:
- **Head Nodding (Microsleep)**: $\text{Pitch} \le -15^\circ$ for $\ge 0.8\text{s}$.
- **Road Inattention**: $|\text{Yaw}| \ge 24^\circ$ for $\ge 2.0\text{s}$.

### 3. Composite Fatigue Score
The composite **AI Fatigue Indicator** $F \in [0, 100]$ is computed over a 60-second sliding buffer:
$$F = w_1 \cdot S_{\text{PERCLOS}} + w_2 \cdot S_{\text{closure}} + w_3 \cdot S_{\text{blink}} + w_4 \cdot S_{\text{events}}$$
where event penalties factor in drowsy states, yawns, and nodding episodes.

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/driverguard-ai.git
cd driverguard-ai
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create your local `.env` file from the provided template:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If you do not have an API key, DriverGuard AI operates seamlessly with its built-in rule-based intelligence engine).*

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

DriverGuard AI includes an automated **pytest** suite validating the preprocessing pipeline, state machine transitions, fatigue scoring bounds, and model fallbacks:

```bash
pytest tests/ -v
```

---

## 🛡️ Privacy & Ethical Design

- **Edge Processing**: All video capture, facial landmark detection, and eye cropping run strictly locally in system memory. No video frames are recorded to disk or transmitted across the network by default.
- **Minimal Telemetry for Generative AI**: Google Gemini only receives aggregated statistical metadata (e.g. session duration, total blink count, event counts), never raw frames or facial biometric vectors.
- **Local SQLite Storage**: Driving telemetry is stored locally in `data/sessions.db` with full user export and deletion capabilities.

---

## ⚠️ Limitations & Real-World Challenges

1. **Illumination Variations**: Extreme backlighting or pitch darkness without infrared (IR) LEDs degrades facial landmark localization.
2. **Eyewear & Sunglasses**: Polarized sunglasses occlude iris and eyelid visibility, necessitating fallback to facial landmark EAR heuristics.
3. **Severe Head Rotation**: MediaPipe tracking requires driver head pitch/yaw within $\pm 35^\circ$ of the camera optical axis.
4. **Camera Positioning**: For optimal performance, mount the camera directly on the vehicle dashboard or steering column facing the driver.

---

## ⚖️ Automotive Safety Disclaimer

> **DRIVERGUARD AI IS AN EXPERIMENTAL COMPUTER-VISION RESEARCH & ASSISTANCE SYSTEM.**
> It is not a certified Automotive Safety Integrity Level (ASIL) automotive electronic control unit. It must never replace attentive driving, adequate rest, or vehicle manufacturer active safety equipment. Always pull over safely and rest when tired.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
