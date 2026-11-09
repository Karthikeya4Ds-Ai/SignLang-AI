"""
Dataset Preparation & Normalization Pipeline
Loads raw keypoint sequences (.npy), applies geometric wrist-relative normalization,
encodes one-hot categorical targets, and creates stratified train/test splits.
"""

import os
from pathlib import Path
import sys
# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Tuple, Dict, Any, List
import numpy as np

from src.config_loader import load_config, get_all_classes, get_label_map, get_project_root
from src.normalizer import normalize_sequence


def to_one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts integer class indices into one-hot binary matrix."""
    return np.eye(num_classes, dtype=np.int32)[indices]


def stratified_split(
    X: np.ndarray,
    y_indices: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy stratified train/test split."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls_val in np.unique(y_indices):
        cls_indices = np.where(y_indices == cls_val)[0]
        rng.shuffle(cls_indices)
        split_point = int(len(cls_indices) * (1.0 - test_ratio))
        train_idx.extend(cls_indices[:split_point])
        test_idx.extend(cls_indices[split_point:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y_indices[train_idx], y_indices[test_idx]


def prepare_and_split_dataset(
    config: Dict[str, Any] = None,
    apply_normalization: bool = True,
    save_compressed: bool = True,
    output_dir: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scans raw sequence folders, applies optional wrist normalization,
    and returns (X_train, X_test, y_train, y_test).
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_rel = config.get("data", {}).get("raw_data_dir", "SignLang_Ai/SignLang_Data_Dynamic")
    raw_path = root / raw_rel

    if not raw_path.is_dir():
        # Fallback check for alternate path
        alt_path = root / "SignLang_Data_Dynamic"
        if alt_path.is_dir():
            raw_path = alt_path
        else:
            raise FileNotFoundError(f"Raw data directory not found at: {raw_path}")

    classes = get_all_classes(config)
    label_map = get_label_map(classes)
    total_classes = len(classes)
    seq_length = config.get("data", {}).get("sequence_length", 60)
    feat_dim = config.get("features", {}).get("feature_dim", 126)

    print(f"Loading data from: {raw_path}")
    print(f"Total vocabulary classes: {total_classes}")
    print(f"Expected sequence shape: ({seq_length}, {feat_dim})")
    print(f"Wrist Normalization: {'ENABLED' if apply_normalization else 'DISABLED'}")

    sequences = []
    labels = []
    missing_classes = []

    for action in classes:
        action_dir = raw_path / action
        if not action_dir.is_dir():
            missing_classes.append(action)
            continue

        npy_files = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
        loaded_count = 0

        for npy_file in npy_files:
            file_path = action_dir / npy_file
            try:
                seq = np.load(file_path)
            except Exception as e:
                print(f"  Warning: Could not read {file_path}: {e}")
                continue

            if seq.shape == (seq_length, feat_dim):
                if apply_normalization:
                    seq = normalize_sequence(seq, center_wrist=True, scale_invariant=True)
                sequences.append(seq)
                labels.append(label_map[action])
                loaded_count += 1
            else:
                print(f"  Warning: Skipping {npy_file}, unexpected shape: {seq.shape}")

        print(f"  [{action}] Loaded {loaded_count} sequences.")

    if missing_classes:
        print(f"\nNotice: The following classes had no data directory: {missing_classes}")

    total_loaded = len(sequences)
    print(f"\nData loading complete. Total sequences assembled: {total_loaded}")

    if total_loaded == 0:
        raise RuntimeError("No valid sequences were loaded. Verify the data path and file contents.")

    X = np.array(sequences, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int32)

    test_size = config.get("data", {}).get("test_split_ratio", 0.20)
    seed = config.get("system", {}).get("random_seed", 42)

    print(f"Splitting into {int((1-test_size)*100)}% train / {int(test_size*100)}% test (stratified)...")
    X_train, X_test, y_train_raw, y_test_raw = stratified_split(
        X, y_raw,
        test_ratio=test_size,
        seed=seed
    )

    y_train = to_one_hot(y_train_raw, num_classes=total_classes)
    y_test = to_one_hot(y_test_raw, num_classes=total_classes)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    if output_dir is None:
        output_dir = root / config.get("data", {}).get("processed_dir", "data/processed")
    os.makedirs(output_dir, exist_ok=True)

    if save_compressed:
        filename = "dataset_splits_normalized.npz" if apply_normalization else "dataset_splits.npz"
        npz_path = output_dir / filename
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        print(f"Saved compressed split archive to: {npz_path}")
        if not apply_normalization:
            # Also save raw copy
            np.savez_compressed(output_dir / "dataset_splits_raw.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    else:
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)
        print(f"Saved .npy split files to: {output_dir}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignLang AI Dataset Builder")
    parser.add_argument("--normalize", action="store_true", default=True, help="Apply wrist-relative normalization")
    parser.add_argument("--raw", action="store_true", help="Keep raw unnormalized coordinates")
    args = parser.parse_args()

    norm_flag = not args.raw
    prepare_and_split_dataset(apply_normalization=norm_flag)
