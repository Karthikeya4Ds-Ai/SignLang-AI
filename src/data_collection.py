"""
Interactive Gesture Data Collection Engine
Records standardized 60-frame landmark sequences for custom sign classes with
automatic progress tracking, resume support, and visual on-screen countdowns.
"""

from pathlib import Path
import os
import time
import cv2
import numpy as np

from src.config_loader import load_config, get_all_classes, get_project_root
from src.landmark_extractor import LandmarkExtractor


def record_dataset():
    root = get_project_root()
    config = load_config()
    classes = get_all_classes(config)

    raw_dir_rel = config.get("data", {}).get("raw_data_dir", "SignLang_Ai/SignLang_Data_Dynamic")
    data_path = root / raw_dir_rel
    data_path.mkdir(parents=True, exist_ok=True)

    no_sequences = 50
    sequence_length = config.get("data", {}).get("sequence_length", 60)
    dynamic_classes = set(config.get("classes", {}).get("dynamic_gestures", []))

    print("\n==========================================")
    print("  SignLang AI: Dataset Collection Studio  ")
    print("==========================================")
    print(f"Destination: {data_path}")
    print(f"Total vocabulary: {len(classes)} classes")
    print(f"Sequences per class: {no_sequences} (each {sequence_length} frames)\n")

    first_incomplete_idx = 0
    found_incomplete = False

    for idx, action in enumerate(classes):
        action_dir = data_path / action
        action_dir.mkdir(parents=True, exist_ok=True)

        existing = [f for f in os.listdir(action_dir) if f.endswith('.npy') and f[:-4].isdigit()]
        count = len(existing)
        status = "COMPLETE" if count >= no_sequences else f"{count}/{no_sequences}"
        print(f"  [{idx:02d}] {action:12s} ({status})")

        if count < no_sequences and not found_incomplete:
            first_incomplete_idx = idx
            found_incomplete = True

    print(f"\nPress Enter to record from the first incomplete class ('{classes[first_incomplete_idx]}')")
    choice = input(f"Or enter an index (0-{len(classes)-1}): ").strip()

    start_idx = first_incomplete_idx
    if choice.isdigit() and 0 <= int(choice) < len(classes):
        start_idx = int(choice)

    actions_to_record = classes[start_idx:]
    print(f"\nStarting recording with: {classes[start_idx]}")
    time.sleep(1.5)

    cap = cv2.VideoCapture(config.get("inference", {}).get("camera_index", 0))
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    quit_app = False

    with LandmarkExtractor(normalize_wrist=False) as extractor:
        for action in actions_to_record:
            action_dir = data_path / action
            existing_nums = [int(f[:-4]) for f in os.listdir(action_dir) if f.endswith('.npy') and f[:-4].isdigit()]
            start_seq = max(existing_nums) + 1 if existing_nums else 0

            if start_seq >= no_sequences:
                continue

            for seq_num in range(start_seq, no_sequences):
                # 1. Wait for User Ready
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    # Ready banner
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (245, 117, 16), -1)
                    text = f"READY: '{action.upper()}' Sequence {seq_num+1}/{no_sequences}"
                    cv2.putText(frame, text, (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    instruction = "DO GESTURE MOTION" if action in dynamic_classes else "HOLD STATIC POSE"
                    cv2.putText(frame, instruction, (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "PRESS [SPACE] TO RECORD  |  [Q] TO QUIT", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow("SignLang AI Data Studio", frame)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord(' '):
                        break
                    if key == ord('q'):
                        quit_app = True
                        break

                if quit_app:
                    break

                # 2. Countdown
                for count in (2, 1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
                    cv2.putText(frame, f"STARTING IN {count}...", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("SignLang AI Data Studio", frame)
                    cv2.waitKey(1000)

                # 3. Capture Sequence
                frames_keypoints = []
                for frame_idx in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    frame, results = extractor.process_frame(frame)
                    extractor.draw_styled_landmarks(frame, results, modern_theme=True)

                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 200, 100), -1)
                    rec_text = f"RECORDING [{action.upper()}] Frame {frame_idx+1}/{sequence_length}"
                    cv2.putText(frame, rec_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                    cv2.imshow("SignLang AI Data Studio", frame)
                    cv2.waitKey(1)

                    keypoints = extractor.extract_raw_keypoints(results)
                    frames_keypoints.append(keypoints)

                # 4. Save
                save_file = action_dir / f"{seq_num}.npy"
                np.save(save_file, np.array(frames_keypoints))
                print(f"Saved: {save_file}")

            if quit_app:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection session closed.")


if __name__ == "__main__":
    record_dataset()
