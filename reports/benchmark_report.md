# SignLang AI - Empirical Evaluation & Latency Benchmark Report

## 1. Executive Summary

This report provides the empirical evaluation metrics for the production **SignLang AI** system evaluated across a held-out stratified test set of 470 samples covering 47 vocabulary classes.

* **Evaluated Model:** Stacked LSTM Classifier (`models/sign_classifier_model.keras`)
* **Test Dataset Size:** 470 sequences ($20\%$ stratified test split from 2,350 total sequences)
* **Overall Test Accuracy:** **94.26%**
* **Inference Engine:** Vectorized Pure NumPy LSTM Engine
* **Hardware Environment:** Intel Core i5-1350U, 16GB RAM, Windows 11 (CPU-only)

---

## 2. Category Accuracy Breakdown

| Gesture Category | Class Range | Support (Samples) | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Static Alphabet (A–Z)** | Letters `A` through `Z` | 260 | **98.08%** |
| **Dynamic Words** | `Hello`, `Goodbye`, `Thanks`, `Please`, `Yes`, `No`, `Food`, `Water`, `Help`, `Stop`, `_idle_` | 110 | **93.64%** |
| **Static Digits (0–9)** | Numbers `0` through `9` | 100 | **85.00%** |
| **Combined Overall** | **All 47 Classes** | **470** | **94.26%** |

---

## 3. Real-Time Latency & Throughput Profile

Measured across 100 consecutive forward-pass iterations on consumer CPU:

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Inference Latency** | **17.74 ms** | $< 33.3 \text{ ms}$ (for 30 FPS) | **Pass (1.8x faster)** |
| **95th Percentile Latency** | **22.82 ms** | $< 40.0 \text{ ms}$ | **Pass** |
| **Throughput (FPS)** | **56.4 FPS** | $\ge 30 \text{ FPS}$ | **Pass (Real-time fluid)** |
| **Memory Footprint** | **~24 MB RAM** | $< 500 \text{ MB}$ | **Pass (95% reduction vs TF)** |

---

## 4. Visual Artifacts
* **Confusion Matrix Plot:** `reports/confusion_matrix.png`
* **Detailed Per-Class Report:** `reports/classification_report.txt`
* **Machine-Readable Metrics:** `reports/evaluation_summary.json`
