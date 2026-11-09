"""
Evaluation & Benchmarking Module
Generates Confusion Matrix heatmap, Classification Report (Precision, Recall, F1),
gesture category breakdown, and latency/FPS throughput benchmarks.
"""

from pathlib import Path
import sys
# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict, Any, Optional
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

from src.config_loader import load_config, get_all_classes, get_project_root
from src.inference_engine import NumpyLSTMInferenceEngine


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Pure NumPy confusion matrix calculation."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
    """Pure NumPy precision, recall, and F1-score calculation."""
    tp = np.diag(cm).astype(float)
    fp = (np.sum(cm, axis=0) - tp).astype(float)
    fn = (np.sum(cm, axis=1) - tp).astype(float)
    support = np.sum(cm, axis=1).astype(int)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)

    total_samples = int(np.sum(cm))
    overall_acc = float(np.sum(tp) / total_samples) if total_samples > 0 else 0.0

    lines = [f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}"]
    lines.append("-" * 55)
    for i, cls_name in enumerate(classes):
        lines.append(f"{cls_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8d}")
    lines.append("-" * 55)
    lines.append(f"{'Accuracy':<15} {'':<10} {'':<10} {overall_acc:<10.4f} {total_samples:<8d}")
    lines.append(f"{'Macro Avg':<15} {np.mean(precision):<10.4f} {np.mean(recall):<10.4f} {np.mean(f1):<10.4f} {total_samples:<8d}")

    return {
        "text_report": "\n".join(lines),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": overall_acc
    }


def run_evaluation(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates model performance and writes visualization plots and metrics reports.
    """
    root = get_project_root()
    config = load_config()
    classes = get_all_classes(config)

    if output_dir is None:
        output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate test data
    X_test, y_test = None, None
    processed_npz = root / "data" / "processed" / "dataset_splits.npz"
    legacy_xtest = root / "SignLang_Ai" / "Project_Files" / "X_test.npy"
    legacy_ytest = root / "SignLang_Ai" / "Project_Files" / "y_test.npy"

    if processed_npz.is_file():
        print(f"Loading test data from: {processed_npz}")
        archive = np.load(processed_npz)
        X_test, y_test = archive["X_test"], archive["y_test"]
    elif legacy_xtest.is_file() and legacy_ytest.is_file():
        print(f"Loading legacy test data from: {legacy_xtest.parent}")
        X_test = np.load(legacy_xtest)
        y_test = np.load(legacy_ytest)
    else:
        raise FileNotFoundError("Could not find test dataset files.")

    # 2. Locate model
    if model_path is None:
        model_path = root / config.get("model", {}).get("saved_model_path", "models/sign_classifier_model.keras")
        if not Path(model_path).is_file():
            # Check legacy location
            alt = root / "SignLang_Ai" / "Project_Files" / "sign_classifier_model.keras"
            if alt.is_file():
                model_path = alt

    print(f"Loading trained neural model from: {model_path}")
    engine = NumpyLSTMInferenceEngine(model_path)

    # 3. Predict on Test Set
    print(f"Evaluating {len(X_test)} test sequences via NumPy vector engine...")
    y_pred_probs = engine.predict_batch(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    cm = compute_confusion_matrix(y_true_classes, y_pred_classes, len(classes))
    metrics_result = compute_metrics(cm, classes)
    overall_acc = metrics_result["accuracy"]

    print(f"\n==========================================")
    print(f" OVERALL TEST ACCURACY: {overall_acc * 100:.2f}%")
    print(f"==========================================\n")

    # 4. Classification Report
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"SignLang AI - Model Evaluation Report\n")
        f.write(f"Evaluated Model: {Path(model_path).name}\n")
        f.write(f"Overall Accuracy: {overall_acc * 100:.2f}%\n\n")
        f.write(metrics_result["text_report"])

    # 5. Group Category Accuracy
    num_digits = 10
    num_alphabet = 26
    digits_mask = (y_true_classes < num_digits)
    alpha_mask = ((y_true_classes >= num_digits) & (y_true_classes < num_digits + num_alphabet))
    dynamic_mask = (y_true_classes >= num_digits + num_alphabet)

    acc_digits = float(np.mean(y_true_classes[digits_mask] == y_pred_classes[digits_mask])) if np.any(digits_mask) else 0.0
    acc_alpha = float(np.mean(y_true_classes[alpha_mask] == y_pred_classes[alpha_mask])) if np.any(alpha_mask) else 0.0
    acc_dynamic = float(np.mean(y_true_classes[dynamic_mask] == y_pred_classes[dynamic_mask])) if np.any(dynamic_mask) else 0.0

    print("Category Accuracy Breakdown:")
    print(f"  Digits (0-9):       {acc_digits * 100:.2f}%")
    print(f"  Alphabet (A-Z):     {acc_alpha * 100:.2f}%")
    print(f"  Dynamic Words:      {acc_dynamic * 100:.2f}%")

    # 6. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(18, 16), dpi=150)
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label('Sample Count', rotation=270, labelpad=15, fontweight='bold')

    ax.set_title(f"SignLang AI Confusion Matrix (Overall Accuracy: {overall_acc*100:.2f}%)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    fig.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved Confusion Matrix heatmap to: {cm_path}")

    # 7. Latency and FPS Benchmark
    print("\nBenchmarking CPU inference latency...")
    dummy_input = np.random.randn(60, 126).astype(np.float32)
    # Warmup
    for _ in range(20):
        _ = engine.predict_single(dummy_input)

    times = []
    iterations = 100
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.predict_single(dummy_input)
        times.append(time.perf_counter() - t0)

    mean_latency_ms = float(np.mean(times) * 1000)
    p95_latency_ms = float(np.percentile(times, 95) * 1000)
    fps_throughput = float(1.0 / np.mean(times))

    print(f"  Mean Latency:      {mean_latency_ms:.2f} ms")
    print(f"  95th Percentile:   {p95_latency_ms:.2f} ms")
    print(f"  Throughput:        {fps_throughput:.1f} FPS")

    # 8. Save structured JSON summary
    summary = {
        "model_file": str(Path(model_path).name),
        "test_samples": int(len(X_test)),
        "num_classes": int(len(classes)),
        "overall_accuracy": overall_acc,
        "category_accuracy": {
            "digits_0_to_9": acc_digits,
            "alphabet_A_to_Z": acc_alpha,
            "dynamic_gestures": acc_dynamic
        },
        "benchmarks": {
            "mean_latency_ms": mean_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "throughput_fps": fps_throughput
        }
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_evaluation()
