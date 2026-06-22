# CAPTCHA Recognizer

Recognising distorted multi-character CAPTCHA sequences from noisy images end to end, using a CRNN with CTC loss and beam-search decoding.required.

---

## Problem

CAPTCHA images deliberately distort text — overlapping glyphs, warping, noise, and stray lines — to defeat naive OCR. Per-character segmentation fails because the characters touch and bleed into each other. The task is to read the full sequence directly from the raw image, without ever segmenting it into individual characters, and to do so robustly across the distortions present in the dataset.

---

## Approach

A three-notebook progression from data understanding to a baseline to the final sequence model.

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `01_eda.ipynb` | Anomaly detection, vocab verification, label/position frequency, image dimension & pixel statistics, mean image |
| 2 | `02_baseline.ipynb` | 6-head CNN baseline (one classification head per character position) + CER metric |
| 3 | `03_crnn.ipynb` | CRNN (CNN + BiLSTM + CTC) with beam-search decoding |

### Key design choices
- **Segmentation-free recognition:** CTC loss aligns the predicted frame sequence to the target string without per-character boxes.
- **Beam-search decoding:** CTC beam search (width 20) replaces greedy argmax decoding, collapsing repeats and blanks correctly and improving CER over the baseline.
- **Cleaned vocabulary:** ambiguous characters are excluded from the label space so visually identical glyphs aren't penalised as errors.
- **Anomaly filtering:** corrupt/mislabelled samples are dropped before training.

---

## Data

| Aspect | Detail |
|--------|--------|
| Task | Fixed-length distorted CAPTCHA strings on noisy grayscale images |
| Preprocessing | Grayscale → tensor; anomalous samples removed |
| Vocabulary | Digits + uppercase letters with ambiguous characters excluded |
| Split | Shuffled train/val with fixed seed (`random_state` set for reproducibility) |

---

## Model

`CRNN` — convolutional feature extractor feeding a bidirectional LSTM, trained with CTC loss.

| Component | Detail |
|-----------|--------|
| CNN | 5 conv blocks (64 → 128 → 256 → 256 → 256), BatchNorm + ReLU; asymmetric pooling `(2,1)` to preserve horizontal resolution |
| Bridge | `AdaptiveAvgPool2d((1, None))` collapses height, keeps width as the time axis |
| RNN | 2-layer BiLSTM, hidden 256, dropout 0.3 |
| Head | Linear → `log_softmax` over `vocab + 1` (CTC blank) classes |
| Loss | CTC loss |
| Decoding | CTC beam search, beam width 20 |

---

## Results

Character Error Rate (CER) on the validation split, as reported in the notebook outputs.

| Model | CER |
|-------|-----|
| 6-head CNN baseline | higher (reference) |
| **CRNN + CTC + beam search** | **~2.6%** |

The CRNN substantially outperforms the per-position CNN baseline; beam search over greedy decoding accounts for a further drop in CER.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep learning | `torch`, `torchvision` |
| Imaging | `pillow`, `numpy` |
| App / deploy | `streamlit`, `huggingface` |

---

## Notes & Limitations

- Tuned for the CIG dataset's distortion style and image size, out-of-distribution CAPTCHAs may need retraining.
- Research/educational project for the CIG AI/ML Open Project.