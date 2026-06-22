# Distorted Visual CAPTCHA Recognition

## Problem Statement

Build a deep learning model that recognizes 6-character text sequences from visually distorted grayscale CAPTCHA images. Distortions include background noise, overlapping symbols, blur, occlusion, and irregular spacing.

**Evaluation Metric:** Character Error Rate (CER) — lower is better.

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| Train | 20,000 | ✓ |
| Test  | 5,000  | ✗ |

- Image size: **200 × 100 px**, grayscale
- Sequence length: **6 characters** (fixed)
- Vocabulary: **31 characters** — digits `2–9`, uppercase letters excluding `0, 1, I, L, O` (visually ambiguous)
- 2 anomalous rows dropped: `train-2184`, `train-6819` (Excel-mangled labels)

---

## Approach

### Architecture — Custom CRNN + CTC

```
Input (B, 1, 100, 200)
       ↓
CNN Backbone (5 conv blocks: Conv → BN → ReLU → Pool)
       ↓
Adaptive Pool → collapse height to 1
       ↓
(B, W', 256) — sequence of column feature vectors
       ↓
Bidirectional LSTM (2 layers, hidden=256)
       ↓
Linear projection → (T, B, 32)
       ↓
CTC Loss (train) / Greedy Decode (inference)
```

- **No pretrained weights** — fully custom architecture
- CNN collapses spatial height, treats image width as a time sequence
- BiLSTM reads character context in both directions
- CTC loss handles alignment automatically — no need to know exact character positions

### Why CRNN over Fixed-Head CNN

A 6-head CNN treats each position independently with no sequential context. CRNN reads the image left-to-right as a sequence of column strips, which matches how CAPTCHA text is laid out. This gives significantly lower CER.

---

## Results

| Model | Val CER |
|-------|---------|
| 6-head CNN baseline | ~0.965 (failed to converge) |
| Custom CRNN + CTC | ~0.026 |

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `NumPy` · `Pandas` · `Pillow` · `Matplotlib` · `Google Colab`
