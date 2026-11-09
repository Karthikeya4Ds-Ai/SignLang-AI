<div align="center">

# 🤟 SignLang AI
### Real-Time Sign Language Translator (Video to Text + Speech)
**Deep Learning Spatiotemporal Gesture Recognition Pipeline powered by MediaPipe & Recurrent Neural Networks**

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-teal?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-99.15%25-brightgreen)](reports/)
[![Inference Latency](https://img.shields.io/badge/Latency-%3C20ms-success)](reports/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🌟 Executive Overview

**SignLang AI** is an end-to-end, real-time computer vision and deep learning assistive system that bridges the communication divide between deaf or hard-of-hearing individuals and hearing communities. 

Using standard consumer hardware (a basic laptop webcam and CPU), the system captures video, extracts 3D skeletal hand keypoints, neutralizes spatial distance through wrist-relative translation invariance, models temporal gesture sequences across 60-frame rolling buffers, constructs full English sentences via an intelligent fingerspelling debouncing engine, and vocalizes the result via a non-blocking **Text-to-Speech (TTS)** engine.

### 🎯 Key Highlights
* **47-Class Unified Vocabulary:** Recognizes 10 static digits (`0-9`), 26 static letters (`A-Z`), 10 dynamic gestures (`Hello`, `Goodbye`, `Thanks`, `Please`, `Yes`, `No`, `Food`, `Water`, `Help`, `Stop`), and a dedicated resting state (`_idle_`).
* **High Validation Accuracy (99.15%):** Trained on 2,350 custom recorded sequences with stratified validation and early stopping.
* **Smart NLP Sentence Builder:** Translates individual fingerspelled letters into words (`H-E-L-L-O` $\rightarrow$ `"HELLO"`), automatically appends spaces during conversational pauses, and supports backspacing/clearing.
* **Non-Blocking Multi-Threaded TTS:** Speech synthesis runs on an isolated background worker thread to ensure the camera loop never stutters or drops frames.
* **Dual User Interfaces:**
  1. **Cyberpunk Glassmorphism OpenCV HUD:** High-tech on-screen display with Top-3 probability meters, certainty percentage badges, FPS counter, and subtitle ribbon.
  2. **Streamlit Web Application:** Modern browser dashboard with live webcam analytics, parameter sliders, and downloadable transcript logs.

---

## 🏗️ Architecture & Pipeline

```
+------------------+      +-------------------+      +-------------------------+
| Live Camera Feed | ---> | MediaPipe Holistic| ---> | Wrist-Centering & Scale |
|  (30 - 60 FPS)   |      |  (42 Keypoints)   |      | Invariance Normalization|
+------------------+      +-------------------+      +-------------------------+
                                                                  |
                                                                  v
+------------------+      +-------------------+      +-------------------------+
| Output Synthesis | <--- | NLP Sentence Engine| <--- | Stacked LSTM / GRU /    |
| (HUD + Audio TTS)|      | (Debounce & Mode) |      | TFLite Inference Engine |
+------------------+      +-------------------+      +-------------------------+
```

1. **Perception Engine:** MediaPipe Holistic captures 21 landmarks per hand (42 landmarks total, 126 coordinate features per frame).
2. **Geometric Normalization:** Subtracts wrist coordinates $(x_0, y_0, z_0)$ and normalizes by maximum palm span, giving full translation and scale invariance.
3. **Temporal Processing:** Assembles landmark frames into a rolling 60-frame FIFO buffer ($T=60, D=126$).
4. **Sequence Classifier:** A stacked recurrent network (LSTM/GRU) with Batch Normalization and Dropout outputs a 47-class probability distribution.
5. **Debouncing & Word Assembler:** A 10-frame consensus buffer eliminates transition flickering. Spelled characters accumulate into words; idle gestures insert space delimiters.
6. **Vocalization:** An asynchronous thread sends speech to the OS audio system without locking OpenCV.

---

## 📊 Benchmark & Performance Comparison

Evaluated on consumer hardware (Intel Core i5-1350U, 16GB RAM, Windows 11, CPU-only):

| Model Architecture | Parameters | Val Accuracy | Val Loss | Mean Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stacked LSTM (Default)** | ~215,000 | **99.15%** | **0.0595** | 18.4 ms | 54.3 FPS |
| **Stacked GRU (Lightweight)**| ~162,000 | **99.02%** | **0.0612** | **13.1 ms** | **76.3 FPS** |
| **Bidirectional LSTM** | ~320,000 | **99.20%** | **0.0578** | 24.6 ms | 40.6 FPS |
| **TFLite Export (CPU Edge)** | ~162,000 | **99.02%** | - | **8.2 ms** | **120+ FPS** |

*Confusion Matrix and per-class precision/recall reports are automatically generated in `reports/`.*

---

## 📁 Repository Directory Structure

```text
SignLang_Project/
├── config/
│   └── config.yaml                 # Central project configuration & hyperparameters
├── src/
│   ├── __init__.py
│   ├── config_loader.py            # Safe YAML configuration & vocabulary loader
│   ├── normalizer.py               # Geometric wrist-relative & scale-invariant transforms
│   ├── landmark_extractor.py       # MediaPipe holistic wrapper & neon drawing styles
│   ├── nlp_engine.py               # Debounce buffer, fingerspelling assembler & sentence builder
│   ├── tts_engine.py               # Asynchronous, non-blocking Text-to-Speech audio worker
│   ├── model.py                    # Neural architectures (LSTM, GRU, BiLSTM) & TFLite export
│   ├── dataset_builder.py          # Data preparation, normalization & stratified splitting
│   ├── train.py                    # Training pipeline with early stopping & checkpoints
│   ├── evaluate.py                 # Confusion matrix, classification report & latency benchmarks
│   ├── data_collection.py          # Interactive dataset recording studio
│   └── realtime_detector.py        # Flagship OpenCV Cyberpunk HUD translator
├── app/
│   └── streamlit_app.py            # Interactive Streamlit browser application
├── models/
│   └── sign_classifier_model.keras # Trained deep sequence classifier
├── reports/
│   ├── confusion_matrix.png        # Publication-quality heatmap
│   └── classification_report.txt   # Precision, recall, and F1-score breakdown
├── docs/                           # Architecture docs, PPT, and report
├── .gitignore                      # Excludes large binaries & cache
├── LICENSE                         # MIT License
├── requirements.txt                # Exact pip dependencies
└── setup.py                        # Pip installable package metadata
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/karthikeya-parvathareddy/SignLang-AI.git
cd SignLang-AI

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Real-Time HUD Translator
Launch the flagship OpenCV live camera application:

```bash
python -m src.realtime_detector
```

**Keyboard Controls:**
* `[SPACE]` : Manually commit current word / insert space.
* `[B]` : Backspace last letter or word.
* `[C]` : Clear sentence transcript.
* `[M]` : Mute / Unmute Text-to-Speech audio.
* `[Q]` : Quit application.

### 3. Launch the Streamlit Web Dashboard
For an interactive browser experience with live metric charts:

```bash
streamlit run app/streamlit_app.py
```

### 4. Train or Benchmark Custom Architectures
To train a model from scratch with early stopping and automatic TFLite export:

```bash
# Train Gated Recurrent Unit (GRU)
python -m src.train --model_type gru --epochs 100 --batch_size 32

# Train Long Short-Term Memory (LSTM)
python -m src.train --model_type lstm --epochs 100 --batch_size 32
```

### 5. Evaluate and Generate Metrics
Generate the Confusion Matrix plot and classification report:

```bash
python -m src.evaluate
```

---

## 📖 Supported Vocabulary (47 Classes)

| Group | Classes Included |
| :--- | :--- |
| **Digits (10)** | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| **Alphabet (26)** | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` |
| **Dynamic Words (10)** | `Hello`, `Goodbye`, `Thanks`, `Please`, `Yes`, `No`, `Food`, `Water`, `Help`, `Stop` |
| **Control Token (1)** | `_idle_` (resting hand state / word boundary) |

---

## 📄 License
This project is open-source and distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 👤 Author & Contact
**Karthikeya Parvathareddy**  
* B.Tech in Data Science & Artificial Intelligence (2022–2026)  
* [LinkedIn Profile](https://linkedin.com) • [GitHub Portfolio](https://github.com/karthikeya-parvathareddy)
