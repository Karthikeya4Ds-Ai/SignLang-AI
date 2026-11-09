"""
Real-Time Sign Language Translator (Video to Text + Speech)
Features:
- Modern Glassmorphic / Cyberpunk HUD overlay
- Top-3 candidate probability meters
- Real-time FPS & inference latency counter
- Fingerspelling-to-word assembly & dynamic sentence builder
- Asynchronous, non-blocking Text-to-Speech (TTS)
- Keyboard shortcuts for complete control
"""

from pathlib import Path
import sys
# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import cv2
import numpy as np

from src.config_loader import load_config, get_all_classes, get_project_root
from src.landmark_extractor import LandmarkExtractor
from src.nlp_engine import SignSentenceEngine
from src.tts_engine import AsyncTTSEngine
from src.inference_engine import NumpyLSTMInferenceEngine


def draw_hud(
    frame: np.ndarray,
    current_action: str,
    confidence: float,
    top_3: list,
    sentence_text: str,
    fps: float,
    latency_ms: float,
    tts_muted: bool,
    conf_threshold: float = 0.70
) -> np.ndarray:
    """
    Renders high-tech translucent HUD graphics on top of the webcam frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top Header Bar (Semi-transparent black)
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)

    # 2. Bottom Subtitle Ribbon (Sentence Builder)
    cv2.rectangle(overlay, (0, h - 75), (w, h), (20, 20, 25), -1)

    # 3. Right Sidebar (Top 3 Candidates)
    sidebar_w = 260
    cv2.rectangle(overlay, (w - sidebar_w, 65), (w - 10, 240), (25, 25, 30), -1)

    # Blend overlay with 0.75 alpha
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # --- Header Information ---
    # Title
    cv2.putText(frame, "SIGNLANG AI", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "v2.0 PRO", (185, 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # FPS & Latency
    metrics_str = f"FPS: {fps:.0f}  |  {latency_ms:.1f}ms"
    cv2.putText(frame, metrics_str, (w - 380, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # TTS Status Badge
    tts_color = (0, 0, 255) if tts_muted else (0, 220, 100)
    tts_text = "AUDIO: MUTED [M]" if tts_muted else "AUDIO: ACTIVE [M]"
    cv2.putText(frame, tts_text, (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 1, cv2.LINE_AA)

    # --- Primary Detected Gesture (Top Center Ribbon) ---
    if current_action and current_action != "_idle_":
        badge_color = (0, 220, 100) if confidence >= 0.85 else (0, 200, 255) if confidence >= conf_threshold else (180, 180, 180)
        action_text = f"{current_action} ({confidence*100:.0f}%)"
        (text_w, text_h), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        center_x = (w - text_w) // 2

        # Draw pill backdrop
        cv2.rectangle(frame, (center_x - 15, 65), (center_x + text_w + 15, 110), (30, 30, 35), -1)
        cv2.rectangle(frame, (center_x - 15, 65), (center_x + text_w + 15, 110), badge_color, 2)
        cv2.putText(frame, action_text, (center_x, 98), cv2.FONT_HERSHEY_DUPLEX, 1.0, badge_color, 2, cv2.LINE_AA)

    # --- Right Sidebar: Top-3 Probabilities ---
    cv2.putText(frame, "TOP PREDICTIONS", (w - sidebar_w + 15, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    for idx, (cls_name, prob) in enumerate(top_3):
        y_pos = 125 + (idx * 35)
        # Class name & percentage
        label = f"{cls_name[:9]}: {prob*100:.0f}%"
        cv2.putText(frame, label, (w - sidebar_w + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Progress bar
        bar_x = w - sidebar_w + 115
        bar_max_w = 110
        bar_fill = int(bar_max_w * prob)
        cv2.rectangle(frame, (bar_x, y_pos - 10), (bar_x + bar_max_w, y_pos), (50, 50, 55), -1)
        bar_color = (0, 220, 100) if idx == 0 and prob >= conf_threshold else (0, 160, 255)
        cv2.rectangle(frame, (bar_x, y_pos - 10), (bar_x + bar_fill, y_pos), bar_color, -1)

    # --- Bottom Sentence Subtitle Ribbon ---
    cv2.putText(frame, "SENTENCE:", (15, h - 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

    display_sentence = sentence_text if sentence_text else "Waiting for gestures... (Spell letters or sign dynamic words)"
    sentence_color = (255, 255, 255) if sentence_text else (130, 130, 130)
    cv2.putText(frame, display_sentence, (140, h - 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, sentence_color, 2, cv2.LINE_AA)

    # Bottom helper hotkeys
    helpers = "[SPACE]: Commit/Space  |  [B]: Backspace  |  [C]: Clear  |  [M]: Mute Audio  |  [Q]: Quit"
    cv2.putText(frame, helpers, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def run_realtime_app():
    """Main application loop."""
    root = get_project_root()
    config = load_config()
    all_classes = get_all_classes(config)

    seq_length = config.get("data", {}).get("sequence_length", 60)
    conf_threshold = config.get("inference", {}).get("confidence_threshold", 0.70)
    cam_index = config.get("inference", {}).get("camera_index", 0)

    # 1. Load Trained Model
    model_path = root / config.get("model", {}).get("saved_model_path", "models/sign_classifier_model.keras")
    if not model_path.is_file():
        # Fallback to legacy path
        model_path = root / "SignLang_Ai" / "Project_Files" / "sign_classifier_model.keras"

    print(f"Loading neural classifier from: {model_path}")
    engine = NumpyLSTMInferenceEngine(model_path)
    print("Neural Engine loaded successfully (vectorized pure NumPy)!")

    # 2. Initialize Asynchronous TTS Engine
    tts_cfg = config.get("tts", {})
    tts = AsyncTTSEngine(
        rate=tts_cfg.get("rate", 155),
        volume=tts_cfg.get("volume", 1.0),
        enabled=tts_cfg.get("enabled", True)
    )

    # 3. Setup NLP Sentence Builder with Audio Hooks
    def on_token_received(token: str, kind: str):
        # Speak completed words and dynamic signs
        if kind in ("word", "word_completed") and tts_cfg.get("speak_dynamic_words", True):
            tts.speak(token)

    def on_sentence_cleared(sentence: str):
        if tts_cfg.get("speak_completed_sentence", True):
            tts.speak(sentence)

    nlp = SignSentenceEngine(
        debounce_window=config.get("inference", {}).get("debounce_window", 10),
        consensus_ratio=config.get("inference", {}).get("consensus_ratio", 0.70),
        idle_duration_for_space=config.get("nlp", {}).get("idle_duration_for_space_seconds", 1.8),
        on_token_callback=on_token_received,
        on_sentence_callback=on_sentence_cleared
    )

    # 4. Initialize Camera & MediaPipe
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {cam_index}.")
        tts.stop()
        return

    # Try setting 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence_buffer = []
    current_action = ""
    current_confidence = 0.0
    top_3_candidates = []

    fps = 30.0
    latency_ms = 0.0
    prev_frame_time = time.time()

    print("\n--- SignLang AI Real-Time Translator Running ---")
    print("Press [Q] in the OpenCV window to exit.\n")

    with LandmarkExtractor(normalize_wrist=False) as extractor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Horizontal flip for intuitive selfie mirror feed
            frame = cv2.flip(frame, 1)

            # Compute FPS
            now = time.time()
            fps = 1.0 / max((now - prev_frame_time), 1e-4)
            prev_frame_time = now

            # Process MediaPipe
            frame, results = extractor.process_frame(frame)
            extractor.draw_styled_landmarks(frame, results, modern_theme=True)

            # Extract 126 features
            keypoints = extractor.extract_raw_keypoints(results)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-seq_length:]

            # Run Neural Inference when buffer is full
            if len(sequence_buffer) == seq_length:
                t0 = time.perf_counter()
                seq_arr = np.array(sequence_buffer, dtype=np.float32)

                # Pure vectorized NumPy inference (60+ FPS on CPU, zero TF latency)
                probs = engine.predict_single(seq_arr)
                latency_ms = (time.perf_counter() - t0) * 1000

                pred_idx = int(np.argmax(probs))
                current_confidence = float(probs[pred_idx])
                predicted_class = all_classes[pred_idx]

                # Update top 3 candidates
                top_3_indices = np.argsort(probs)[-3:][::-1]
                top_3_candidates = [(all_classes[i], float(probs[i])) for i in top_3_indices]

                # NLP State Engine Update
                if current_confidence >= conf_threshold:
                    current_action = predicted_class
                    nlp.update(predicted_class, current_confidence)
                else:
                    nlp.update("_idle_", 1.0 - current_confidence)

            # Render Glassmorphic Cyber HUD
            full_sentence = nlp.get_full_display_text()
            frame = draw_hud(
                frame=frame,
                current_action=current_action,
                confidence=current_confidence,
                top_3=top_3_candidates,
                sentence_text=full_sentence,
                fps=fps,
                latency_ms=latency_ms,
                tts_muted=tts.is_muted(),
                conf_threshold=conf_threshold
            )

            cv2.imshow("SignLang AI - Real-Time Translation", frame)

            # Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                nlp.add_space()
            elif key == ord('b') or key == 8:  # 'b' or Backspace
                nlp.backspace()
            elif key == ord('c'):  # 'c' Clear
                nlp.clear()
            elif key == ord('m'):  # 'm' Mute
                tts.toggle_mute()

    # Clean Exit
    print("Shutting down cleanly...")
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()


if __name__ == "__main__":
    run_realtime_app()
