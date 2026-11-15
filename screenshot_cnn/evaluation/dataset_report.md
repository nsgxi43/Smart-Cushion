# Dataset Report — Smart Cushion 7-Class Posture CNN

**Source:** `dataset/` (280 PNG heat-map screenshots, 7 classes) — opened read-only, never modified
**Reproduce:** `python training/inspect_dataset.py`
**Machine-readable:** `evaluation/dataset_summary.json`

---

## 1. Sample counts, dimensions, formats, integrity

| ID | Class | Images | Unique (MD5) | In duplicate groups | Sessions | Effective distinct¹ |
|---|---|---|---|---|---|---|
| 0 | Normal   | 40 | 35 | 10 | 5 | 23 |
| 1 | Backward | 40 | 40 | 0  | 4 | **14** |
| 2 | Forward  | 40 | 40 | 0  | 4 | 39 |
| 3 | Right    | 40 | 40 | 0  | 4 | 39 |
| 4 | Left     | 40 | 39 | 2  | 4 | 38 |
| 5 | Slouch   | 40 | 36 | 8  | 3 | **13** |
| 6 | Unknown  | 40 | 32 | 15 | **1** | 32 |
| | **Total** | **280** | **262** | **35** | **25** | **198** |

¹ Greedy clustering of frames agreeing on ≥95 % of render cells.

- **Dimensions:** all 280 images are exactly **1702 × 1948**. No resizing was needed for consistency.
- **Format / mode:** PNG, RGBA for all 280 (the alpha channel is constant 255 and unused).
- **Corrupt or unreadable:** **0**. Every file decodes cleanly.
- **Timestamps:** all 280 filenames carry a parsable macOS screenshot timestamp.

## 2. Class imbalance

**Nominally perfectly balanced** — 40 images in every class, imbalance ratio 1.00.

The *effective* balance is worse. After collapsing near-identical frames, Slouch contributes
13 and Backward 14 genuinely distinct frames versus 38–39 for Forward, Right and Left —
an effective imbalance ratio of **3.0**. Class weights are applied during training to
compensate for the residual count differences introduced by the session-aware split.

## 3. Duplicates and near-duplicates

**Exact duplicates:** 17 MD5 groups covering 35 files. Some are literal copies
(`… 3.16.11 PM copy.png`, `… 3.14.48 PM 1.png`); the rest are screenshots taken within the
same second. Unknown is worst affected (15 of 40 files).

**Near-duplicates**, measured on 62 × 71 band-index maps:

| Threshold | Pairs | Same session | Same class | Cross-class |
|---|---|---|---|---|
| ≥ 99 % identical cells | 34 | 34 | 34 | 0 |
| ≥ 95 % | 248 | 248 | 248 | 0 |
| ≥ 90 % | 608 | 565 | 608 | 0 |

**No near-duplicate pair crosses a class boundary** — labelling is internally consistent.
All redundancy is *within session*, which is why a random split would leak badly and a
session-aware split is used instead (§6).

## 4. Heat-map layout and colour-scale consistency

**Layout is fully consistent.** Identical dimensions, identical crop geometry, and an
identical render-cell period in every frame. Edge-energy analysis gives a column period of
**27.41 px** and a row period of **27.44 px**, i.e. a native render grid of about
**62 × 71** square cells, the same in all 280 images.

**The colour scale is consistent in palette.** 99.96 % of all pixels in the dataset fall
exactly on a ten-entry palette — eight evenly spaced samples of matplotlib's `turbo`
colormap plus pure black and pure white:

| Band | RGB | Meaning |
|---|---|---|
| L0 | `(44,20,60)` | lowest filled band |
| L1 | `(84,116,228)` | |
| L2 | `(100,204,212)` | |
| L3 | `(140,252,124)` | |
| L4 | `(212,232,92)` | |
| L5 | `(244,160,72)` | |
| L6 | `(200,68,36)` | |
| L7 | `(112,20,12)` | highest filled band |
| — | `(0,0,0)` | area **below** the lowest filled band |
| — | `(255,255,255)` | contour strokes |

This is a filled-contour render with eight levels, not a continuous heat map. Mean band
occupancy per class:

| Class | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | black | white |
|---|---|---|---|---|---|---|---|---|---|---|
| Normal | 6.4 | 7.9 | 4.8 | 2.4 | 2.5 | 3.8 | 12.5 | 58.5 | 0.3 | 0.9 |
| Backward | 0.4 | 0.6 | 0.9 | 1.2 | 2.0 | 4.8 | 23.9 | 65.6 | 0.0 | 0.5 |
| Forward | 20.5 | 10.9 | 2.5 | 1.5 | 1.6 | 2.9 | 14.5 | 43.1 | 1.2 | 1.2 |
| Right | 7.3 | 7.8 | 7.8 | 8.5 | 5.3 | 4.6 | 12.0 | 42.2 | 3.2 | 1.3 |
| Left | 1.7 | 4.6 | 10.0 | 11.8 | 8.4 | 11.1 | 21.2 | 30.6 | 0.2 | 0.5 |
| Slouch | 0.5 | 1.8 | 3.8 | 4.0 | 2.9 | 4.3 | 16.9 | 64.8 | 0.0 | 1.0 |
| Unknown | 30.3 | 20.8 | 8.9 | 6.3 | 3.7 | 1.9 | 0.7 | 0.1 | 20.9 | 6.3 |

### Caveat worth recording

Every frame in all six seated classes spans the full L0–L7 range (40/40 contain both
extremes); Unknown reaches L7 in only 21/40. That pattern is consistent with per-frame
auto-scaling of the colour limits, though it is not conclusive. There is **no colorbar in
any screenshot**, so the bands cannot be anchored to physical pressure values, and
identical colours in two different frames cannot be assumed to represent identical
pressures. This does not block image classification — the model is trained on the rendered
appearance — but it does mean the trained model maps *rendered heat maps*, not calibrated
pressures, to classes.

## 5. Preprocessing

Images are decoded from the renderer's palette back to a single-channel **band index**:

```
RGB screenshot
  → NEAREST downscale to 248 × 284 (4 × the native grid; palette colours survive exactly)
  → nearest-palette lookup → band index 0..7, or black / white
  → black (below lowest band) → 0          (it is the lowest pressure region)
  → white (contour stroke)   → mean of known neighbours, iteratively filled
  → area-average (BOX) to 64 × 72
  → divide by 7  →  [0, 1]
```

Final input tensor: **(72, 64, 1) float32 in [0, 1]**.

Why band index rather than RGB or grayscale:

- `turbo` is **not monotonic in luminance**, so a plain grayscale conversion scrambles the
  pressure ordering — low and high pressure can map to similar grey values.
- Raw RGB would force a 7 k-parameter network to relearn the colormap from 155 training
  images.
- The band index *is* monotonic in pressure and the decode is exact (99.96 % palette hit
  rate), so this preserves the pressure-distribution pattern rather than destroying it.

Input resolution 64 × 72 is chosen to sit just above the measured native grid of 62 × 71 —
higher resolution would only resample the same information — while being divisible by 4 for
the two pooling stages.

## 6. Dataset splitting

Recording sessions are recovered from the screenshot timestamps; a gap longer than 20 s
starts a new session. Within a burst the median inter-frame gap is 1–2 s, between bursts it
jumps to 44–240 s, so the boundaries are unambiguous.

| Class | Sessions | Timing |
|---|---|---|
| Normal   | 5 | 15:08:55–09:12 (5), 15:10:28–10:33 (5), 15:12:30–12:40 (10), 15:14:47–14:56 (10), 15:16:07–16:29 (10) |
| Forward  | 4 | 15:22:28–22:45, 15:23:56–24:17, 15:24:51–25:09, 15:28:21–28:39 (10 each) |
| Right    | 4 | 15:29:51–30:16, 15:31:17–31:41, 15:32:45–33:03, 15:33:35–33:59 (10 each) |
| Left     | 4 | 15:34:59–35:18, 15:36:02–36:17, 15:36:47–37:01, 15:37:45–38:02 (10 each) |
| Backward | 4 | 15:39:24–39:38, 15:42:08–42:25, 15:43:09–43:23, 15:47:23–47:35 (10 each) |
| Slouch   | 3 | 15:52:08–52:20 (10), 15:54:12–54:38 (20), 15:55:31–55:45 (10) |
| Unknown  | **1** | 15:19:38–21:17 (40) |

**Sessions are kept whole — no session appears in two splits.** Achieved sizes:

| Split | Images | Share | Per class (Normal…Unknown) |
|---|---|---|---|
| train | 155 | 56.2 % | 30, 20, 20, 20, 20, 20, 25 |
| val   | 60  | 21.7 % | 5, 10, 10, 10, 10, 10, 5 |
| test  | 61  | 22.1 % | 5, 10, 10, 10, 10, 10, 6 |
| dropped | 4 | — | Unknown guard band |

**Documented limitations:**

1. **The ratio is 56/22/22, not 70/15/15.** With only 3–5 sessions per class, 70/15/15 is
   unreachable at session granularity — holding out one session each for validation and
   test already consumes 25–50 % of a class. Leakage prevention was prioritised over
   hitting the target ratio.
2. **Unknown has a single session**, so it cannot be split by session. It uses a contiguous
   temporal split (first 25 frames train, 5 val, 6 test) with a **2-frame guard band**
   dropped at each boundary. Unknown's train/val/test are therefore *not* independent
   recordings, and its reported metrics are correspondingly optimistic.

**Cross-split near-duplicate verification:**

| Comparison | Max similarity | Pairs > 0.99 |
|---|---|---|
| train vs val | 0.9866 | 0 |
| train vs test | 0.9866 | 0 |
| val vs test | 0.9931 | 10 |

No training frame is a near-duplicate of any validation or test frame — the important
property. The ten val↔test pairs above 0.99 are all **Slouch**, and they come from
genuinely *different* recording sessions: Slouch is simply so static (13 effective distinct
frames out of 40) that separate recordings still look nearly identical. This is a property
of the data, not a split error.

## 7. Augmentation

Applied to the training split only:

- **Small translations** — ±4 px with symmetric padding (≈6 % of frame width).
- **Slight intensity scaling** — ±3 %, selected by a validation-loss sweep
  (`evaluation/augmentation_sweep.json`).
- **Mild sensor-style noise** — Gaussian, σ = 0.02.
- **No horizontal flips** (would swap Left/Right) and **no vertical flips** (would swap
  Forward/Backward).
- **No rotation.** A cushion mat has a fixed orientation under the sitter, so rotating the
  pressure field is not a physically plausible variation. Omitted deliberately, not
  overlooked.

## 8. Known risk carried into training

Class separation in this dataset is strongly correlated with a single global statistic —
mean band index: Unknown 1.20, Forward 4.41, Right 4.81, Left 4.94, Normal 5.45,
Slouch 6.15, Backward 6.44, with very small within-class spread. A model can score well by
keying on overall brightness rather than on spatial pressure structure.

This risk **materialised**: see `evaluation/classification_report.txt`. Backward and Slouch
are the two most saturated classes, separated by only ~0.03 in mean normalised value, while
session-to-session drift *within* Backward spans 0.900–0.939 — larger than the gap between
the two classes. The held-out Backward session falls on the Slouch side of the learned
boundary and all 10 of its frames are misclassified. Full analysis in `README.md`.

## 9. Figures

| File | Shows |
|---|---|
| `figures/class_montage.png` | one raw frame per class |
| `figures/band_index_montage.png` | the same frames as band indices (black = below lowest band, white = contour) |
| `figures/zoom_detail.png` | 1:1 crop showing the 27.4 px render cells and contour strokes |
| `figures/preprocessed_samples.png` | the final 64 × 72 CNN inputs, one per class |
