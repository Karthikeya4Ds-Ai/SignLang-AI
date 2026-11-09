"""
Geometric Normalization Module
Provides wrist-relative translation invariance and bounding-scale invariance
for MediaPipe hand landmarks.
"""

from typing import Union
import numpy as np


def normalize_hand_landmarks(
    hand_points: np.ndarray,
    center_wrist: bool = True,
    scale_invariant: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Normalizes a single hand's 21 landmarks (shape: (21, 3)).

    Args:
        hand_points: Array of shape (21, 3) with (x, y, z) coordinates.
        center_wrist: If True, subtracts the wrist (point 0) coordinates from all points.
        scale_invariant: If True, scales coordinates by the max distance from the wrist.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized array of shape (21, 3). If hand is not detected (all zeros), returns zeros.
    """
    # If hand is missing (all zeros), return unchanged
    if not np.any(hand_points):
        return np.zeros_like(hand_points)

    normalized = hand_points.copy()

    if center_wrist:
        wrist = normalized[0:1, :]  # Shape: (1, 3)
        normalized = normalized - wrist

    if scale_invariant:
        # Compute Euclidean distance of each point from the wrist (origin)
        distances = np.linalg.norm(normalized, axis=1)  # Shape: (21,)
        max_dist = np.max(distances)
        if max_dist > eps:
            normalized = normalized / max_dist

    return normalized


def normalize_frame_keypoints(
    keypoints: np.ndarray,
    center_wrist: bool = True,
    scale_invariant: bool = True
) -> np.ndarray:
    """
    Normalizes a single frame of combined left and right hand keypoints.

    Args:
        keypoints: 1D array of length 126 (21*3 left hand + 21*3 right hand).
        center_wrist: Whether to center landmarks relative to respective wrist.
        scale_invariant: Whether to divide by the max palm span.

    Returns:
        Normalized 1D array of shape (126,).
    """
    if len(keypoints) != 126:
        raise ValueError(f"Expected 126 keypoints, got {len(keypoints)}")

    lh = keypoints[:63].reshape(21, 3)
    rh = keypoints[63:].reshape(21, 3)

    lh_norm = normalize_hand_landmarks(lh, center_wrist=center_wrist, scale_invariant=scale_invariant)
    rh_norm = normalize_hand_landmarks(rh, center_wrist=center_wrist, scale_invariant=scale_invariant)

    return np.concatenate([lh_norm.flatten(), rh_norm.flatten()])


def normalize_sequence(
    sequence: np.ndarray,
    center_wrist: bool = True,
    scale_invariant: bool = True
) -> np.ndarray:
    """
    Normalizes a full sequence of keypoints across time.

    Args:
        sequence: Array of shape (T, 126), e.g., (60, 126).
        center_wrist: Whether to center landmarks relative to respective wrist.
        scale_invariant: Whether to divide by the max palm span.

    Returns:
        Normalized sequence of shape (T, 126).
    """
    normalized_seq = np.zeros_like(sequence)
    for t in range(sequence.shape[0]):
        normalized_seq[t] = normalize_frame_keypoints(
            sequence[t],
            center_wrist=center_wrist,
            scale_invariant=scale_invariant
        )
    return normalized_seq
