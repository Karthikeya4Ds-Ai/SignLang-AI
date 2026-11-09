"""
SignLang AI - Interactive Streamlit Web Application
A modern browser dashboard for real-time sign language recognition,
live probability distributions, sentence history logging, and transcript export.
"""

from pathlib import Path
import sys
import time

# Ensure project root is in python path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import streamlit as st
import numpy as np
import cv2

from src.config_loader import load_config, get_all_classes
from src.landmark_extractor import LandmarkExtractor
from src.nlp_engine import SignSentenceEngine
from src.tts_engine import AsyncTTSEngine
from src.inference_engine import NumpyLSTMInferenceEngine


st.set_page_config(
    page_title="SignLang AI - Real-Time Translation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #B0BEC5;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #374151;
        text-align: center;
    }
    .sentence-box {
        background-color: #111827;
        border-left: 4px solid #00E5FF;
        padding: 15px;
        border-radius: 6px;
        font-size: 1.3rem;
        color: #FFFFFF;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model():
    config = load_config()
    model_path = root / config.get("model", {}).get("saved_model_path", "models/sign_classifier_model.keras")
    if not model_path.is_file():
        model_path = root / "SignLang_Ai" / "Project_Files" / "sign_classifier_model.keras"
    engine = NumpyLSTMInferenceEngine(model_path)
    return engine, config, get_all_classes(config)


def main():
    st.markdown('<div class="main-header">🤟 SignLang AI: Neural Sign Translator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time deep learning spatiotemporal gesture recognition with MediaPipe & LSTM</div>', unsafe_allow_html=True)

    try:
        model, config, all_classes = load_cached_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ System Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.50, 0.95, 0.75, 0.05)
        debounce_val = st.slider("Debounce Buffer (Frames)", 5, 20, 10, 1)
        camera_idx = st.number_input("Webcam Device Index", min_value=0, max_value=3, value=0)

        st.markdown("---")
        st.header("🔊 Audio & TTS")
        tts_enabled = st.checkbox("Enable Text-to-Speech", value=True)
        speech_rate = st.slider("Speech Rate (WPM)", 100, 200, 155, 5)

        st.markdown("---")
        st.subheader("📚 47-Class Vocabulary")
        with st.expander("View Supported Signs"):
            st.markdown("**Numbers (0-9):** " + ", ".join([str(i) for i in range(10)]))
            st.markdown("**Alphabet (A-Z):** " + ", ".join([chr(c) for c in range(ord('A'), ord('Z')+1)]))
            st.markdown("**Dynamic Words:** Hello, Goodbye, Thanks, Please, Yes, No, Food, Water, Help, Stop")

    # Main Dashboard Columns
    col1, col2 = st.columns([7, 5])

    with col1:
        st.subheader("📹 Live Camera Feed")
        start_button = st.checkbox("▶️ Start Live Camera", value=False)
        frame_window = st.image([])

    with col2:
        st.subheader("📊 Live Recognition Analytics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            detected_metric = st.empty()
            detected_metric.metric("Current Sign", "Waiting...")
        with metric_col2:
            conf_metric = st.empty()
            conf_metric.metric("Certainty", "0.0%")

        st.markdown("#### Top Prediction Probabilities")
        prob_chart = st.empty()

        st.markdown("#### 📝 Live Sentence Builder")
        sentence_box = st.empty()
        sentence_box.markdown('<div class="sentence-box">_</div>', unsafe_allow_html=True)

        st.markdown("---")
        clear_btn = st.button("🗑️ Clear Sentence Transcript")

    if "sentence_history" not in st.session_state:
        st.session_state.sentence_history = []

    # Initialize Engine
    tts = AsyncTTSEngine(rate=speech_rate, enabled=tts_enabled)
    nlp = SignSentenceEngine(
        debounce_window=debounce_val,
        consensus_ratio=0.70,
        on_token_callback=lambda token, kind: tts.speak(token) if tts_enabled and kind in ("word", "word_completed") else None
    )

    if clear_btn:
        nlp.clear()
        sentence_box.markdown('<div class="sentence-box">_</div>', unsafe_allow_html=True)

    if start_button:
        cap = cv2.VideoCapture(int(camera_idx))
        if not cap.isOpened():
            st.error(f"Could not open webcam index {camera_idx}.")
            return

        seq_length = config.get("data", {}).get("sequence_length", 60)
        sequence_buffer = []

        with LandmarkExtractor() as extractor:
            while start_button:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame, results = extractor.process_frame(frame)
                extractor.draw_styled_landmarks(frame, results, modern_theme=True)

                keypoints = extractor.extract_raw_keypoints(results)
                sequence_buffer.append(keypoints)
                sequence_buffer = sequence_buffer[-seq_length:]

                if len(sequence_buffer) == seq_length:
                    seq_arr = np.array(sequence_buffer, dtype=np.float32)
                    probs = model.predict_single(seq_arr)

                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                    pred_class = all_classes[pred_idx]

                    if confidence >= conf_thresh:
                        detected_metric.metric("Current Sign", pred_class)
                        conf_metric.metric("Certainty", f"{confidence*100:.1f}%")
                        nlp.update(pred_class, confidence)
                    else:
                        detected_metric.metric("Current Sign", "_idle_")
                        conf_metric.metric("Certainty", f"{(1.0-confidence)*100:.1f}%")
                        nlp.update("_idle_", 1.0 - confidence)

                    # Update Top 5 Chart
                    top_5_idx = np.argsort(probs)[-5:][::-1]
                    chart_data = {all_classes[i]: probs[i] for i in top_5_idx}
                    prob_chart.bar_chart(chart_data)

                # Update Sentence display
                curr_sentence = nlp.get_full_display_text()
                sentence_box.markdown(f'<div class="sentence-box">{curr_sentence if curr_sentence else "_"}</div>', unsafe_allow_html=True)

                # Render video frame in RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            tts.stop()


if __name__ == "__main__":
    main()
