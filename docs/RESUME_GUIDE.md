# SignLang AI - Resume Guide & Interview Talking Points

## 📝 High-Impact Resume Project Entries

You can replace or upgrade the existing SignLang AI entry on your resume with one of the following ATS-optimized variations:

### Option A: Machine Learning / Deep Learning Engineer Focus
```markdown
**SignLang AI — Real-Time Sign Language Translator (Video to Text + Speech)**
*Tech Stack: Python, TensorFlow/Keras, MediaPipe, OpenCV, Scikit-learn, pyttsx3, Streamlit*
* Engineered an end-to-end computer vision translation pipeline recognizing 47 static & dynamic ISL classes, achieving 99.15% validation accuracy across 2,350 custom sequence samples.
* Developed translation-invariant geometric normalization (wrist centering & scale scaling), neutralizing camera distance variance and boosting cross-environment robustness.
* Architected stacked LSTM and GRU temporal networks ($T=60, D=126$) and optimized inference pipeline with TFLite, achieving low-latency processing (<18ms per frame at 55+ FPS) on standard CPU hardware.
* Designed a fingerspelling-to-word NLP assembler with 10-frame consensus debouncing and integrated an asynchronous, non-blocking Text-to-Speech (TTS) audio vocalization thread.
* Shipped dual user interfaces: a Cyberpunk/Glassmorphic OpenCV HUD with real-time probability meters and an interactive Streamlit web dashboard.
```

### Option B: Data Science / AI Generalist Focus
```markdown
**SignLang AI — End-to-End Deep Learning Gesture Translator**
*Technologies: Python, TensorFlow, MediaPipe, OpenCV, Deep Learning, TTS, Streamlit*
* Built an assistive communication system translating real-time Indian Sign Language (ISL) gestures to text and synthesized speech on consumer-grade laptops.
* Extracted 42 3D hand landmarks via MediaPipe Holistic and designed spatio-temporal sequence models with Batch Normalization and Dropout regularization to prevent overfitting.
* Evaluated models via stratified 80/20 splits, generating publication-grade confusion matrices and per-class precision/recall metrics (99.1% overall accuracy).
* Created a multi-threaded speech synthesis pipeline and word reconstruction engine to assemble fingerspelled letters into full conversational sentences.
```

---

## 🎯 Technical Keywords for ATS Scanning
Ensure these keywords appear in your skills section:
* **Frameworks & Libraries:** TensorFlow, Keras, MediaPipe, OpenCV, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, pyttsx3
* **Core Concepts:** Computer Vision, Deep Learning, Recurrent Neural Networks (RNN/LSTM/GRU), Spatiotemporal Modeling, Feature Engineering, Landmark Extraction, Data Normalization, Model Optimization (TFLite), Asynchronous Programming, Text-to-Speech (TTS)

---

## 🗣️ Common Interview Questions & How to Answer Them

### Q1: "Why did you use MediaPipe landmarks instead of passing raw video pixels into a 3D-CNN?"
> *"I deliberately chose MediaPipe landmark extraction over full-image 3D-CNNs for two reasons: **computational efficiency** and **generalization**. Raw pixel 3D-CNNs require heavy GPU acceleration and frequently overfit to room lighting, wall colors, or user clothing. By extracting 42 skeletal hand landmarks, we condense hundreds of thousands of pixels into 126 coordinate features per frame. This lightweight representation enables 50+ FPS real-time sequence inference on consumer CPUs without any dedicated GPU."*

### Q2: "How did you solve the problem of users standing at different distances or positions from the webcam?"
> *"I implemented a geometric wrist-relative normalization algorithm. For every frame, the wrist landmark is designated as the origin $(0, 0, 0)$ and subtracted from all other 20 landmarks of that hand. Then, all coordinates are scaled by the maximum Euclidean distance from the wrist to any fingertip. This mathematical transform makes the feature vector completely invariant to translation and camera distance."*

### Q3: "How does your system turn raw letters into full sentences?"
> *"Because real-time frame predictions naturally flicker during motion transitions, I built a 10-frame rolling deque consensus filter requiring a 70% mode consensus before accepting a gesture. When single letters (A-Z) are recognized, a state machine concatenates them into an uncommitted word buffer. When a resting gesture (`_idle_`) is maintained for 1.8 seconds, the word is committed, a space delimiter is inserted, and an asynchronous worker thread calls the Text-to-Speech engine to vocalize the word without blocking the OpenCV camera feed."*

### Q4: "What were the trade-offs between LSTM and GRU in your benchmarks?"
> *"I benchmarked both architectures on our stratified 470-sample test set. The stacked LSTM achieved 99.15% validation accuracy with 18.4ms latency. The Gated Recurrent Unit (GRU) achieved comparable accuracy (99.02%) but trained ~20% faster and reduced CPU inference latency down to 13.1ms (~76 FPS) because it eliminates the separate cell state and uses fewer gating operations."*
