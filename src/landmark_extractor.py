"""
Landmark Extractor Module
Handles MediaPipe holistic/hands detection, feature vector flattening,
and modern styled landmark visualization.
"""

from typing import Tuple, Optional, Any
import cv2
import numpy as np

# MediaPipe is imported safely with graceful fallback
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    mp = None
    mp_holistic = None
    mp_drawing = None

from src.normalizer import normalize_frame_keypoints


class LandmarkExtractor:
    """Extracts and visualizes 3D skeletal hand landmarks from RGB/BGR frames."""

    def __init__(
        self,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        normalize_wrist: bool = False,
        normalize_scale: bool = False
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.normalize_wrist = normalize_wrist
        self.normalize_scale = normalize_scale

        self._holistic: Optional[mp.solutions.holistic.Holistic] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Initializes the MediaPipe Holistic perception pipeline."""
        if mp_holistic is None:
            raise RuntimeError("MediaPipe is not installed. Please run pip install mediapipe.")
        self._holistic = mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def close(self):
        """Closes the MediaPipe graph cleanly."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def process_frame(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Converts BGR to RGB, runs holistic inference, and converts back.
        Returns:
            Tuple of (bgr_image, mediapipe_results)
        """
        if self._holistic is None:
            self.start()

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self._holistic.process(rgb_image)
        rgb_image.flags.writeable = True
        return bgr_image, results

    def extract_raw_keypoints(self, results) -> np.ndarray:
        """
        Extracts 21 left-hand and 21 right-hand 3D keypoints as a flat 126-D float vector.
        Missing hands are encoded as zeros.
        """
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        return np.concatenate([lh, rh])

    def extract_keypoints(self, results, apply_normalization: Optional[bool] = None) -> np.ndarray:
        """
        Extracts keypoints with optional geometric normalization.
        """
        keypoints = self.extract_raw_keypoints(results)

        should_norm = self.normalize_wrist if apply_normalization is None else apply_normalization
        if should_norm:
            keypoints = normalize_frame_keypoints(
                keypoints,
                center_wrist=True,
                scale_invariant=self.normalize_scale
            )

        return keypoints

    def draw_styled_landmarks(self, image: np.ndarray, results, modern_theme: bool = True):
        """
        Renders styled hand landmarks on the image.
        """
        if not mp_drawing or not mp_holistic:
            return

        if modern_theme:
            # Left Hand: Neon Cyan / Blue
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 220, 0), thickness=2, circle_radius=4),  # Points (Cyan BGR)
                    mp_drawing.DrawingSpec(color=(255, 120, 0), thickness=2, circle_radius=2)   # Connections (Blue BGR)
                )

            # Right Hand: Neon Violet / Magenta
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 230), thickness=2, circle_radius=4),  # Points (Magenta BGR)
                    mp_drawing.DrawingSpec(color=(180, 0, 255), thickness=2, circle_radius=2)   # Connections (Purple BGR)
                )
        else:
            # Classic default drawing style
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
