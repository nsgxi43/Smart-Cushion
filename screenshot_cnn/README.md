# screenshot_cnn — seven-class posture CNN trained on heat-map screenshots

> **Read this first — this is a side experiment, not the main pipeline.**
>
> The main classifier in this repository (`ml/`, `models/`, `esp32/`) is trained on the
> **real 24 × 21 / 504-sensor numerical pressure data** in `DATASET/`. See
> [`../MODEL_CARD.md`](../MODEL_CARD.md).
>
> This folder is a **separate, self-contained experiment** that trains a CNN on **PNG
> screenshots of rendered pressure heat maps** instead — a different input modality, a
> different class set (7 classes, including an empty-cushion `Unknown`), and a different
> and much weaker evaluation protocol. It does not feed the firmware and does not replace
> anything in `ml/`.
>
> **Do not compare its accuracy with the main pipeline's headline number.** See
> [Why the 0.84 here is not better than the 0.44 there](#why-the-084-here-is-not-better-than-the-044-there).

---

## What this is

A lightweight, genuine 2D CNN that classifies seated posture from heat-map screenshots,
trained in TensorFlow/Keras and exported as a fully INT8-quantized TFLite model.

Scope is dataset preparation, training and evaluation only — no firmware, deployment or
hardware integration.

**Input data:** 280 PNG screenshots (1702 × 1948), 40 per class, recorded in one
47-minute sitting by one person on 2026-11-15. The raw screenshots are **not committed**
(238 MB); see [Getting the data](#getting-the-data).

## Classes

Defined once in `models/classes.json`; every script derives its mapping from that file.

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Class | Normal | Backward | Forward | Right | Left | Slouch | Unknown |

`Unknown` = empty cushion, nobody seated. Note this set differs from the main pipeline's
six canonical classes (`C0_Normal … C5_Backward_Lean`, see `../CLAUDE.md`) — it adds
`Unknown` and uses different names. The two are **not interchangeable**.

---

## Model details

### Architecture

```
Input (72, 64, 1)  float32 in [0, 1]
  → Conv2D(8,  3×3, same) → BatchNorm → ReLU → MaxPool2D(2)   →  (36, 32, 8)
  → Conv2D(16, 3×3, same) → BatchNorm → ReLU → MaxPool2D(2)   →  (18, 16, 16)
  → Conv2D(32, 3×3, same) → BatchNorm → ReLU                  →  (18, 16, 32)
  → GlobalAveragePooling2D                                     →  (32,)
  → Dense(32, ReLU) → Dropout(0.3)
  → Dense(7, Softmax)
```

| Property | Value |
|---|---|
| Total parameters | **7,343** (7,231 trainable, 112 non-trainable) |
| float32 weights | 28.7 KB |
| `best_model.keras` on disk | 155.2 KB (includes optimizer state) |
| `best_model.tflite` (INT8) | **14.0 KB** |
| Input tensor | `(1, 72, 64, 1)` |
| Output tensor | `(1, 7)` softmax |

For comparison, the main pipeline's shipped model is a 16,790-parameter dense network
(`504-32-16-6`) — a different architecture for a different input shape.

### Architecture selection

`train.py` trains two variants and keeps the one with lower **validation** loss:

| Variant | Parameters | val_loss | val_accuracy |
|---|---|---|---|
| `base` — Conv/Pool stack only | 7,175 | 0.6411 | 0.8667 |
| **`bn` — adds BatchNorm (selected)** | **7,343** | **0.0381** | **1.0000** |

BatchNorm is the only departure from the plain stack. It costs 168 parameters, folds into
the preceding convolution at inference time, and the validation gap is unambiguous.

### Training configuration

Adam (lr 2e-3), batch 16, up to 400 epochs. EarlyStopping (patience 60, best weights
restored), ReduceLROnPlateau (×0.5, patience 20), ModelCheckpoint — all on validation
loss. Class weights compensate for uneven per-class counts left by the session-aware
split. The selected run stopped at epoch 251 with best validation loss at epoch 191.

Seeds are fixed (`--seed 1337`) and the `tf.data` augmentation map is sequential and
deterministic, so repeat runs of the same command reproduce the same numbers.

### Preprocessing

The screenshots are a filled-contour render using **eight evenly spaced samples of
matplotlib's `turbo` colormap**, plus pure black (area below the lowest filled band) and
pure white (contour strokes). 99.96 % of all pixels fall exactly on that ten-entry
palette, so decoding back to a band index is exact rather than approximate:

```
1702×1948 RGBA PNG
  → NEAREST downscale to 248×284 (4× the measured 62×71 native render grid)
  → nearest-palette lookup → band index 0..7, or black / white
  → black (below lowest band) → 0;  white (contour stroke) → neighbour-mean fill
  → area-average (BOX) to 64×72
  → ÷ 7  →  [0, 1]
```

Grayscale conversion would be wrong — `turbo` is not monotonic in luminance, so it
scrambles the pressure ordering. Raw RGB would force a 7 k-parameter network to relearn
the colormap from 155 images. The band index is monotonic and exactly recoverable.

Resolution 64 × 72 sits just above the measured native render grid of 62 × 71; anything
larger only resamples the same information.

### Augmentation (training split only)

- Small translations — ±4 px, symmetric padding (≈6 % of frame width)
- Intensity scaling — ±3 %, chosen by a validation-loss sweep
  (`evaluation/augmentation_sweep.json`)
- Gaussian noise — σ = 0.02
- **No horizontal flips** (would swap Left/Right), **no vertical flips** (would swap
  Forward/Backward)
- **No rotation** — a cushion mat has fixed orientation under the sitter, so rotating the
  pressure field is not physically plausible

### INT8 quantization

Full-integer quantization, representative dataset of 200 samples from the **training
split only**.

| Property | Value |
|---|---|
| Input | `int8`, `[1, 72, 64, 1]`, scale `0.00392157`, zero-point `-128` |
| Output | `int8`, `[1, 7]`, scale `0.00390625`, zero-point `-128` |
| File size | 14.0 KB (51.3 % smaller than float32 weights) |
| Float tensors remaining | **none** — inference is fully integer |

Quantization is **lossless** here: identical accuracy, identical confusion matrix,
identical per-class F1, **100 % prediction agreement** (61/61), mean absolute probability
difference 0.00139.

---

## Results

Held-out test set (61 images), which took no part in training or model selection:

| Metric | float32 Keras | INT8 TFLite |
|---|---|---|
| Accuracy | **0.8361** (51/61) | **0.8361** (51/61) |
| Macro F1 | 0.8095 | 0.8095 |
| Weighted F1 | 0.7814 | 0.7814 |
| Unknown recall | 1.0000 (6/6) | 1.0000 (6/6) |
| Test loss | 1.0365 | — |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 1.0000 | 1.0000 | 1.0000 | 5 |
| **Backward** | **0.0000** | **0.0000** | **0.0000** | 10 |
| Forward | 1.0000 | 1.0000 | 1.0000 | 10 |
| Right | 1.0000 | 1.0000 | 1.0000 | 10 |
| Left | 1.0000 | 1.0000 | 1.0000 | 10 |
| Slouch | 0.5000 | 1.0000 | 0.6667 | 10 |
| Unknown | 1.0000 | 1.0000 | 1.0000 | 6 |

Full output: [`evaluation/classification_report.txt`](evaluation/classification_report.txt).
Figures: [`evaluation/confusion_matrix.png`](evaluation/confusion_matrix.png),
[`evaluation/training_history.png`](evaluation/training_history.png).

### Why the 0.84 here is not better than the 0.44 there

The main pipeline reports **~0.42–0.44 across participants it has never seen**, and
**~0.87 within a single participant**. This experiment's 0.8361 is a *within-person,
within-day* number: all 280 screenshots come from one person in one 47-minute sitting.

So the honest comparison is **0.84 here vs ~0.87 there** — the same regime, and this
result is slightly worse despite an easier 7-class problem with a trivially separable
`Unknown`. It says nothing about cross-participant generalisation, which
`../reports/phase4_evaluation/domain_shift_diagnostic.md` identifies as the real
bottleneck. There is no cross-participant evidence in this folder at all.

### The one real weakness

Every error is the same error: **all 10 held-out Backward frames are predicted as
Slouch**. Normal↔Slouch, Forward↔Backward, Left↔Right and Normal↔Unknown are all zero.

Backward and Slouch are the two most saturated classes, separated by only ~0.03 in mean
normalised value, while Backward's *own* session-to-session drift spans 0.900–0.939 — a
wider range than the gap between the classes. The held-out Backward session is the
dimmest one and falls on the Slouch side of the learned boundary.

The errors are **confident, not borderline**: mean softmax confidence 0.966 on the wrong
class (9 of 10 above 0.97). A confidence threshold would not catch this — any threshold
low enough to pass correct Forward predictions (0.73–0.95) also passes nearly every wrong
Backward one.

This is consistent with the main pipeline's finding that this sensor saturates heavily and
that global level, not spatial structure, dominates the signal.

### What was tried

The hypothesis that ±10 % intensity augmentation was erasing the Backward/Slouch
distinction was tested with a sweep, selected **on validation loss only**:

| intensity | val_loss | val_accuracy |
|---|---|---|
| 0.00 | 0.0529 | 1.0000 |
| **0.03 (selected)** | **0.0381** | 1.0000 |
| 0.06 | 0.0458 | 1.0000 |
| 0.10 | 0.0710 | 1.0000 |

It improved validation loss but **did not fix the test failure** — test accuracy moved
0.8525 → 0.8361 (Backward 1/10 → 0/10), a difference of one image, not meaningful.

Validation accuracy is 1.0000 for every configuration tried, so validation can no longer
discriminate between candidate models. Development was stopped there deliberately;
further tuning would be fitting the test set.

### Other caveats

- **Unknown's 1.0000 recall is optimistic.** Unknown has a single 99-second recording
  session, so its train/val/test frames are not independent recordings.
- **Normal has only 5 test images**, so its per-class metrics are coarse.
- The model classifies *rendered heat maps*, not calibrated pressures — the screenshots
  carry no colorbar, so band values cannot be anchored to physical units.
- One person, one day. No cross-participant evidence.

---

## Dataset summary

| ID | Class | Images | Unique (MD5) | Sessions | Effective distinct¹ |
|---|---|---|---|---|---|
| 0 | Normal | 40 | 35 | 5 | 23 |
| 1 | Backward | 40 | 40 | 4 | 14 |
| 2 | Forward | 40 | 40 | 4 | 39 |
| 3 | Right | 40 | 40 | 4 | 39 |
| 4 | Left | 40 | 39 | 4 | 38 |
| 5 | Slouch | 40 | 36 | 3 | 13 |
| 6 | Unknown | 40 | 32 | **1** | 32 |

¹ Frames agreeing on ≥95 % of render cells, collapsed.

All 280 images are 1702 × 1948 RGBA PNG; **zero corrupt**. 17 exact-duplicate MD5 groups
covering 35 files. Full analysis: [`evaluation/dataset_report.md`](evaluation/dataset_report.md).

### Splitting

Recording sessions are recovered from the screenshot timestamps (gap > 20 s starts a new
session) and **kept whole** — no session spans two splits.

| Split | Images | Share | Per class (Normal…Unknown) |
|---|---|---|---|
| train | 155 | 56.2 % | 30, 20, 20, 20, 20, 20, 25 |
| val | 60 | 21.7 % | 5, 10, 10, 10, 10, 10, 5 |
| test | 61 | 22.1 % | 5, 10, 10, 10, 10, 10, 6 |

**The ratio is 56/22/22, not the usual 70/15/15.** With only 3–5 sessions per class,
70/15/15 is unreachable at session granularity — holding out one session each for
validation and test already consumes 25–50 % of a class. Leakage prevention was
prioritised over the ratio. **Unknown has one session only** and uses a contiguous
temporal split with a 2-frame guard band — a documented limitation.

Verified: no training frame is a >0.99 near-duplicate of any validation or test frame.

---

## How to run it

### 1. Environment

```bash
cd screenshot_cnn
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # Windows
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Dependencies: `numpy`, `pillow`, `tensorflow==2.20.0`, `scikit-learn`, `matplotlib`.
Verified on Python 3.12, Windows 11, CPU-only TensorFlow.

### 2. Getting the data

The 280 raw screenshots (238 MB) are **not committed**. To re-run the full pipeline,
place them as:

```
screenshot_cnn/dataset/
├── Normal/     *.png
├── Backward/   *.png
├── Forward/    *.png
├── Right/      *.png
├── Left/       *.png
├── Slouch/     *.png
└── Unknown/    *.png
```

Filenames must keep their original macOS screenshot timestamps
(`Screenshot 2026-11-15 at 3.08.55 PM.png`) — session detection and therefore the whole
split depend on them.

`--dataset` points elsewhere if you prefer:

```bash
.venv\Scripts\python training\prepare_dataset.py --dataset D:\path\to\Dataset
```

**You do not need the raw screenshots to retrain or re-evaluate.** The decoded inputs are
committed in `processed_dataset/` (`X.npy`, `y.npy`, `split.npy`, `splits.json`), so
steps 3 and 5 below work on a fresh clone.

### 3. Run the pipeline

```bash
.venv\Scripts\python training\inspect_dataset.py     # needs raw screenshots
.venv\Scripts\python training\prepare_dataset.py     # needs raw screenshots
.venv\Scripts\python training\train.py               # works from processed_dataset/
.venv\Scripts\python training\evaluate.py            # works from processed_dataset/
```

Run in order; each stage consumes the previous stage's output.

| Script | Reads | Writes |
|---|---|---|
| `inspect_dataset.py` | `dataset/` | `evaluation/dataset_summary.json`, `evaluation/figures/` |
| `prepare_dataset.py` | `dataset/` | `processed_dataset/X.npy`, `y.npy`, `split.npy`, `splits.json` |
| `train.py` | `processed_dataset/` | `models/best_model.keras`, `evaluation/training_history.{png,json}` |
| `evaluate.py` | `processed_dataset/`, `models/best_model.keras` | `models/best_model.tflite`, `evaluation/classification_report.txt`, `confusion_matrix.png`, `evaluation_summary.json` |

Training takes a few minutes on CPU; evaluation under a minute.

### 4. Useful flags

```bash
python training\train.py --arch base            # plain stack, no BatchNorm
python training\train.py --arch bn              # BatchNorm variant only
python training\train.py --intensity 0.10       # reproduce a sweep row
python training\train.py --epochs 400 --batch-size 16 --lr 2e-3 --seed 1337
python training\prepare_dataset.py --dataset /other/path
```

### 5. Using the trained model

```python
import numpy as np, json, tensorflow as tf

classes = [c["name"] for c in
           sorted(json.load(open("models/classes.json"))["classes"],
                  key=lambda c: c["id"])]

# --- float32 ---
model = tf.keras.models.load_model("models/best_model.keras")
x = np.load("processed_dataset/X.npy")[:1]        # (1, 72, 64, 1) float32 in [0,1]
print(classes[int(model.predict(x, verbose=0).argmax(1)[0])])

# --- INT8 TFLite ---
interp = tf.lite.Interpreter(model_path="models/best_model.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
s_in, z_in = inp["quantization"]
q = np.clip(np.round(x / s_in + z_in), -128, 127).astype(inp["dtype"])
interp.set_tensor(inp["index"], q); interp.invoke()
s_out, z_out = out["quantization"]
probs = (interp.get_tensor(out["index"]).astype(np.float32) - z_out) * s_out
print(classes[int(probs.argmax())], probs.max())
```

To classify a new screenshot end to end, use the same decoder the training pipeline uses
so preprocessing matches exactly:

```python
import sys; sys.path.insert(0, "training")
from common import decode_band_image
arr, stats = decode_band_image("path/to/Screenshot.png")   # (72, 64) float32 in [0,1]
x = arr[None, :, :, None]
```

---

## Files

```text
screenshot_cnn/
├── dataset/                       # put the 280 screenshots here (not committed)
├── processed_dataset/             # decoded inputs + split manifest (committed)
│   ├── X.npy  y.npy  split.npy
│   └── splits.json                # per-file split assignment, reproducible
├── training/
│   ├── common.py                  # class map, palette decode, session logic (shared)
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── classes.json               # 7-class mapping, single source of truth
│   ├── best_model.keras           # float32, 7,343 params
│   └── best_model.tflite          # fully INT8, 14.0 KB
├── evaluation/
│   ├── dataset_report.md          # full dataset analysis
│   ├── classification_report.txt  # float32 + INT8 + comparison
│   ├── confusion_matrix.png
│   ├── training_history.png / .json
│   ├── evaluation_summary.json
│   ├── augmentation_sweep.json
│   ├── dataset_summary.json
│   └── figures/
├── requirements.txt
└── README.md
```

## Status

| Stage | Status |
|---|---|
| Dataset inspection | ✅ run, reproducible |
| Preprocessing & session-aware split | ✅ run, leakage-verified |
| CNN training | ✅ two variants compared on validation |
| Test-set evaluation | ✅ measured, not estimated |
| INT8 conversion & comparison | ✅ measured on the same test set |
| Firmware / deployment / hardware | ⛔ out of scope — see `../esp32/` and `../ESP32_GUIDE.md` |
| Cross-participant validation | ⛔ **not possible with this data** (one person, one day) |

Every number above is produced by the scripts in `training/` and regenerable with the
commands given. Nothing is estimated.

## Recommended next step

If this line of work is continued, the useful move is **not** more hyperparameter search —
validation is already saturated at 100 %. It is more recording sessions for Backward, and
more than one participant. The main pipeline's domain-shift diagnostic already shows that
participant identity dominates these frames; this experiment has no evidence either way
because it only ever saw one person.
