# SignLang AI Dataset Specifications

## 1. Directory Structure

```text
data/
├── README.md               # Dataset documentation (this file)
└── processed/              # Generated split data
    └── dataset_splits.npz  # Compressed archive containing X_train, X_test, y_train, y_test
```

## 2. Dataset Overview

* **Total Recorded Sequences:** 2,350 sequences
* **Vocabulary Classes:** 47 classes (50 sequences per class)
* **Sequence Length:** 60 frames (~2.0 seconds at 30 FPS)
* **Feature Dimensionality per Frame:** 126 coordinate features (42 3D landmarks)
  * Left Hand: 21 landmarks $\times$ 3 coordinates $(x, y, z)$
  * Right Hand: 21 landmarks $\times$ 3 coordinates $(x, y, z)$
* **Stratified Splits:**
  * Training: 1,880 samples (80%)
  * Testing: 470 samples (20%)

## 3. Vocabulary Breakdown

| Category | Classes |
| :--- | :--- |
| **Digits (10)** | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| **Alphabet (26)** | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` |
| **Dynamic Gestures (10)** | `Hello`, `Goodbye`, `Thanks`, `Please`, `Yes`, `No`, `Food`, `Water`, `Help`, `Stop` |
| **Control Token (1)** | `_idle_` |

## 4. Re-Generating Dataset Splits

To recreate the stratified split and apply wrist-relative normalization:

```bash
python -m src.dataset_builder --normalize
```
This will read the raw sequence files from `SignLang_Ai/SignLang_Data_Dynamic/`, normalize every landmark relative to the wrist origin, and save the compressed split archive to `data/processed/dataset_splits.npz`.
