# SignLang AI System Architecture & Technical Specifications

## 1. System Pipeline Overview

SignLang AI converts continuous video frames of human signing into syntactically valid sentences and spoken audio. The system operates as a five-stage processing pipeline:

```
[Camera Stream] 
      │
      ▼
1. Skeletal Landmark Extraction (MediaPipe Holistic, 42 Hand Points)
      │
      ▼
2. Geometric Normalization (Wrist-Centering & Max Palm Span Scaling)
      │
      ▼
3. Spatiotemporal Sequence Buffer (Rolling 60-Frame FIFO Buffer)
      │
      ▼
4. Deep Sequence Classifier (Stacked LSTM / GRU / TFLite Engine)
      │
      ▼
5. NLP Consensus Filter & Asynchronous TTS Speech Engine
```

---

## 2. Mathematical Formulation: Wrist-Relative Normalization

Raw MediaPipe keypoint coordinates are defined in normalized image frame space $[0.0, 1.0]$. This makes them sensitive to user camera positioning and scale variations.

To achieve **translation invariance** and **scale invariance**, we apply the following transformation for each hand $H \in \{\text{left}, \text{right}\}$ containing 21 landmark vectors $P_i = (x_i, y_i, z_i)$ for $i \in \{0, \dots, 20\}$:

### Step 1: Wrist-Relative Translation
Let $P_0 = (x_0, y_0, z_0)$ represent the wrist origin. Each landmark is translated:
$$P'_i = P_i - P_0 \quad \forall i \in \{0, \dots, 20\}$$
Consequently, the wrist landmark is centered at $(0, 0, 0)$.

### Step 2: Scale Normalization
Compute the maximum Euclidean distance from the wrist to any fingertip:
$$d_{\max} = \max_{i \in \{0, \dots, 20\}} \|P'_i\|_2$$
If $d_{\max} > \epsilon$, normalize all coordinates by $d_{\max}$:
$$P''_i = \frac{P'_i}{d_{\max}}$$

This ensures that hand gestures retain identical geometric representations regardless of whether the user sits $0.5\text{ m}$ or $2.0\text{ m}$ from the camera lens.

---

## 3. Deep Sequence Modeling: LSTM vs. GRU

The sequential input tensor has shape $(B, T, D)$ where:
* $B = \text{Batch size}$ (1 for live streaming; 32 for batch training)
* $T = 60 \text{ frames}$ ($\approx 2.0 \text{ seconds at } 30 \text{ FPS}$)
* $D = 126 \text{ features}$ ($21 \text{ landmarks} \times 3 \text{ coordinates} \times 2 \text{ hands}$)

### LSTM Architecture
```
Input(60, 126)
  │
  ▼
LSTM(64 units, return_sequences=True, activation='relu')
  ├── Dropout(0.20)
  └── BatchNormalization()
  │
  ▼
LSTM(128 units, return_sequences=True, activation='relu')
  ├── Dropout(0.20)
  └── BatchNormalization()
  │
  ▼
LSTM(64 units, return_sequences=False, activation='relu')
  └── BatchNormalization()
  │
  ▼
Dense(64 units, activation='relu')
  └── Dropout(0.20)
  │
  ▼
Dense(47 units, activation='softmax')
```

### GRU Architecture Comparison
The Gated Recurrent Unit (GRU) merges the cell state and hidden state, and couples the input and forget gates into a single update gate:
* **Parameter reduction:** ~25% fewer trainable parameters than LSTM (162K vs 215K).
* **Speedup:** Reduces per-sample forward-pass latency from 18.4ms to 13.1ms on CPU.

---

## 4. NLP Consensus & Debounce Logic

To prevent classification flickering between gesture transitions:
1. Predictions are appended to a 10-frame rolling deque $Q$.
2. The mode $C^* = \operatorname{mode}(Q)$ is calculated.
3. Consensus threshold: $\frac{\operatorname{count}(C^*)}{|Q|} \ge 0.70$.
4. Spelled characters ($A-Z$) are concatenated into a word buffer $W$.
5. A sustained `_idle_` state for $\ge 1.8\text{ seconds}$ finalizes $W$, appends a space delimiter, and triggers TTS pronunciation.
