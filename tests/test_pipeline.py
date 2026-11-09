"""
End-to-End Pipeline Unit & Integration Tests
Validates configuration integrity, normalizer math, NLP state engine,
and neural model forward-pass inference.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.config_loader import load_config, get_all_classes, get_label_map
from src.normalizer import (
    normalize_hand_landmarks,
    normalize_frame_keypoints,
    normalize_sequence
)
from src.nlp_engine import SignSentenceEngine


class TestSignLangPipeline(unittest.TestCase):

    def test_01_config_and_vocabulary(self):
        """Verifies configuration loads and defines exactly 47 unique classes."""
        config = load_config()
        self.assertIsNotNone(config)

        classes = get_all_classes(config)
        self.assertEqual(len(classes), 47, f"Expected 47 classes, got {len(classes)}")
        self.assertEqual(len(set(classes)), 47, "Class names must be completely unique")

        label_map = get_label_map(classes)
        self.assertEqual(len(label_map), 47)
        self.assertEqual(label_map['0'], 0)
        self.assertEqual(label_map['A'], 10)
        self.assertEqual(label_map['_idle_'], 46)

    def test_02_wrist_relative_normalization(self):
        """Verifies that wrist-centering mathematically sets the wrist to (0,0,0)."""
        # Create synthetic hand with non-zero wrist
        hand = np.random.uniform(low=0.2, high=0.8, size=(21, 3)).astype(np.float32)
        wrist_orig = hand[0].copy()

        norm_hand = normalize_hand_landmarks(hand, center_wrist=True, scale_invariant=False)
        np.testing.assert_allclose(norm_hand[0], np.zeros(3), atol=1e-5)

        # Verify scale invariance scales coordinates
        scaled_hand = normalize_hand_landmarks(hand, center_wrist=True, scale_invariant=True)
        distances = np.linalg.norm(scaled_hand, axis=1)
        self.assertAlmostEqual(float(np.max(distances)), 1.0, places=4)

    def test_03_frame_and_sequence_normalization(self):
        """Verifies full 126-D frame and (60, 126) sequence transformations."""
        keypoints = np.random.uniform(0.1, 0.9, size=(126,)).astype(np.float32)
        norm_frame = normalize_frame_keypoints(keypoints, center_wrist=True, scale_invariant=True)
        self.assertEqual(norm_frame.shape, (126,))

        sequence = np.random.uniform(0.1, 0.9, size=(60, 126)).astype(np.float32)
        norm_seq = normalize_sequence(sequence, center_wrist=True, scale_invariant=True)
        self.assertEqual(norm_seq.shape, (60, 126))

    def test_04_nlp_fingerspelling_and_sentence_builder(self):
        """Tests word assembly from individual fingerspelled letters with debouncing."""
        spoken_tokens = []
        nlp = SignSentenceEngine(
            debounce_window=5,
            consensus_ratio=0.80,
            idle_duration_for_space=0.1,
            on_token_callback=lambda token, kind: spoken_tokens.append((token, kind))
        )

        # Feed 5 frames of 'H'
        for _ in range(5):
            nlp.update("H", 0.95)
        self.assertEqual(nlp._current_word, "H")

        # Feed 5 frames of 'I'
        for _ in range(5):
            nlp.update("I", 0.95)
        self.assertEqual(nlp._current_word, "HI")

        # Feed 5 frames of dynamic word 'Thanks'
        for _ in range(5):
            nlp.update("Thanks", 0.95)
        self.assertEqual(nlp.get_completed_sentence(), "HI Thanks")

        # Test backspace
        nlp.backspace()
        self.assertEqual(nlp.get_completed_sentence(), "HI")

        # Test clear
        nlp.clear()
        self.assertEqual(nlp.get_completed_sentence(), "")

    def test_05_numpy_neural_inference_engine(self):
        """Verifies weight extraction and vectorized inference forward-pass."""
        from src.inference_engine import NumpyLSTMInferenceEngine
        model_path = root / "models" / "sign_classifier_model.keras"
        engine = NumpyLSTMInferenceEngine(model_path)

        dummy = np.random.randn(60, 126).astype(np.float32)
        probs = engine.predict_single(dummy)

        self.assertEqual(probs.shape, (47,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertTrue(np.all(probs >= 0.0))


if __name__ == "__main__":
    unittest.main()
