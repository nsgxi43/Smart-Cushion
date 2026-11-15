# Model card — Smart Cushion posture classifier

Everything below is read from artefacts under `reports/` and `models/`. Nothing
is estimated. Where a number could not be measured on this machine it says
`PENDING_HARDWARE`.

| | |
|---|---|
| Experiment ID | `SC-POSTURE-1.0.0` |
| Date trained | 2026-11-15 |
| Framework | TensorFlow / Keras 2.21, Python 3.12 |
| Task | 6-class sitting-posture classification from a 24 × 21 Velostat pressure matrix |
| Deployment target | ESP32 DevKit V1, inference fully on-device |
| Intended use | Posture feedback / research demonstrator |
| **Not** for | Clinical, diagnostic or safety use. See *Limitations*. |

---

## 1. Data

### Source

`DATASET/session_20261114_170217/` — unmodified. SHA-256 of all 69 files
recorded in `reports/checksums_phase1.csv` and re-verified in the Phase 9 audit.

| | |
|---|---|
| Real participants | 4 (P1, P2, P3, P4) |
| Postures | 6 |
| Repetitions | 5 per participant per posture |
| Recording takes | 120 |
| **Total real frames** | **1848** (P1 440, P2 471, P3 463, P4 474) |
| Frames per repetition | 10–21 (the 504-sensor scan runs at ~1.4 Hz) |
| Sensor geometry | 24 rows × 21 columns = 504, row-major `index = row*21 + col` |
| ADC | 12-bit (0–4095), 11 dB attenuation, 16 samples averaged |

Integrity: 0 NaNs, 0 out-of-range values, 0 non-integer values, 0 duplicate
frames, 0 timestamp parse failures, labels consistent in 24/24 recordings.

### Class mapping

The original folders use a position code `P1..P6` that collides visually with
the participant ids. Canonical names are used everywhere downstream. Single
source of truth: `ml/common.py::POSTURE_FOLDER_TO_CLASS`.

| Original folder | Class | Index |
|---|---|---|
| `P1_Normal` | `C0_Normal` | 0 |
| `P2_Forward_Lean` | `C1_Forward_Lean` | 1 |
| `P3_Slouching` | `C2_Slouching` | 2 |
| `P4_Left_Lean` | `C3_Left_Lean` | 3 |
| `P5_Right_Lean` | `C4_Right_Lean` | 4 |
| `P6_Backward_Lean` | `C5_Backward_Lean` | 5 |

### Known data defects

* **Column 20 is an intermittent hardware fault.** Every dead sensor in the
  dataset lies in column 20 (channel 4 of the second column mux on GPIO35):
  dead for all 24 rows of P4, 23 of 24 for P2, 2 for P1, none for P3.
  The shipped model **zeroes column 20** — that variant won the inner-validation
  selection.
* **The mat saturates.** 2–24 % of raw cells sit at ADC 0 and up to 42 % at
  4095. A typical frame has 415–457 of 504 cells active with a mean peak near
  3950. The maps are broad saturated regions with mux crosstalk striping, not
  localised contact patches.

---

## 2. Preprocessing

One pipeline, implemented twice and proven equivalent: `ml/preprocess.py` and
`esp32/preprocess.cpp`.

```
delta = abs(raw - baseline)        int32
delta = 0 if delta <= threshold    int32    threshold = max(3*sigma_cal, 20)
delta = min(delta, 4095)           int32
x     = delta * (1/1024)           float32
```

* `baseline` and `threshold` come from the **device's own** 20-frame startup
  calibration on an unloaded cushion — never baked into the model.
* `SCALE = 1024` and `CLIP_MAX = 4095` are fixed a priori from the 12-bit ADC
  contract. They are **not** fitted to data, so they cannot leak a held-out
  participant.
* **No per-frame normalisation.** Absolute pressure is kept.

**Verification (`reports/preprocess_equivalence.md`):** 300 recorded frames and
8 edge cases, Python vs C++ — **300/300 bit-identical, 0 ULP, max absolute
difference 0**. All 5 calibration scenarios agree, with identical baseline and
threshold arrays. The only float operation is a multiply by an exact power of
two, which is exact in binary floating point.

The archived `pressure_data.csv` files were verified to already equal stages
1–3 at **99.55 % of cells exactly** (`reports/dataset_report/pressure_reconstruction_check.csv`),
which is what makes training on them faithful to what the device computes.

---

## 3. Architecture

```
Input   504 float32          (row-major, column 20 zeroed)
  │
Dense   32   ReLU            16 160 params      L2 1e-4
Dropout 0.3
  │
Dense   16   ReLU               528 params      L2 1e-4
Dropout 0.3
  │
Dense    6   linear (logits)    102 params
  │
softmax — computed in float in esp32/inference.cpp, NOT in the quantised graph
```

**16 790 trainable parameters.**

Softmax is deliberately stripped from the exported graph. That keeps the
quantised model to plain `FULLY_CONNECTED` ops, which is what made a
dependency-free, bit-exact C++ kernel possible.

### Training configuration

| | |
|---|---|
| Optimiser | Adam, learning rate 1e-3 |
| Loss | sparse categorical cross-entropy (`from_logits=True`) |
| Epochs | 120 |
| Batch size | 64 |
| Regularisation | L2 1e-4 on every Dense kernel, dropout 0.3 |
| Seed | 20261115 (`keras.utils.set_random_seed`) |
| **Training set** | **1848 real frames only** — synthetic data was dropped, see §5 |
| Hardware | CPU (TensorFlow has no GPU support on native Windows ≥ 2.11) |

---

## 4. Evaluation protocol

`ml/evaluate.py`. **Nested leave-one-real-participant-out.**

* **Outer loop** — 4 folds, one real participant held out each time. The
  held-out participant contributes nothing to training, hyperparameter or
  architecture selection, the column-20 masking decision, or the preprocessing
  constants.
* **Inner loop** — leave-one-participant-out over the 3 remaining real
  participants. This is the *only* thing allowed to choose a model.
* **Splits are by recording take** (`participant|posture|repN`), so no frame
  from one 10-second recording can land on both sides of any split.
* Every synthetic participant **derived from** the held-out person is also
  removed from training (`ml/augment.py::SYNTHETIC_SOURCES` records derivation).

Verified by the Phase 9 audit: across all 8 train/test splits there are
**0 shared recording takes** and **0 training rows derived from the held-out
participant** (`reports/phase9_audit/leakage_check.csv`).

### Search space — 960 model fits, 43.1 minutes

30 candidates = 8 families × hyperparameters × column-20 masked/unmasked:

| Family | Variants |
|---|---|
| Logistic regression | C ∈ {0.01, 0.1, 1.0} |
| Random forest | 3 depth/leaf settings, 200–300 trees |
| SVM (RBF) | C ∈ {1, 10} |
| sklearn MLP | (64,32) and (32,16), 2 alphas |
| Keras | 504-64-32-6, 504-32-16-6, 504-24-6, 504-16-6 |

720 inner-loop fits + 240 outer refits = **960**. One row per fit in
`reports/phase4_evaluation/inner_selection_scores.csv` and
`per_candidate_per_fold.csv`.

### How the shipped model was chosen

By **inner-validation score only** — never on held-out data. Ties within one
standard error broken by fewest parameters.

| Architecture | mask col 20 | inner score | params |
|---|---|---|---|
| **504-32-16-6** | **yes** | **0.4226** | **16 790** |
| 504-32-16-6 | no | 0.4107 | 16 790 |
| 504-64-32-6 | no | 0.3697 | 34 598 |
| 504-24-6 | no | 0.3578 | 12 270 |
| 504-16-6 | yes | 0.3474 | 8 182 |

---

## 5. Was synthetic data used? No.

`AUGMENTED_DATASET/` contains 6 synthetic participants (P5_SYNTHETIC –
P10_SYNTHETIC, 2759 frames), each a transformed copy of **one** real
participant with smooth per-sensor gain/offset fields, localised drift blobs,
sensor noise, one ±1 cell shift and temporal jitter. Full provenance per frame;
height/weight/sex recorded as `SYNTHETIC/NA`. **They are not people.**

The spec's decision rule was applied verbatim — *keep synthetic only if
held-out real performance improves or stays equal*:

| Setting | Mean held-out accuracy (nested) |
|---|---|
| REAL_ONLY | **0.3383** |
| REAL_PLUS_SYNTHETIC | 0.3373 |

Augmentation did **not** help, so it was **dropped**. The shipped model is
trained on real data only. This is a negative result and is reported as one.

(The augmentation did do what it was designed to do — it preserved posture
structure: 0 frames identical to source, 100 % internal consistency with the
preprocessing identity, within-participant separability matching the source
participant. It simply did not transfer to new *people*, which is expected:
it models sensor variation, not new bodies.)

---

## 6. Results

### Headline — held-out real participants

Chance level is **16.7 %**.

| Held-out participant | Accuracy | Macro F1 |
|---|---|---|
| P1 | 0.4182 | 0.4134 |
| P2 | 0.3355 | 0.2197 |
| P3 | 0.3974 | 0.3695 |
| P4 | 0.6245 | 0.5832 |
| **Mean** | **0.4439** | **0.3965** |
| Min | 0.3355 | — |

Source: `models/phase4_decision.json`, `reports/phase4_evaluation/per_candidate_per_fold.csv`.

**Training-set accuracy was 0.7857.** It is stored in
`models/model_metadata.json` under the key
`train_accuracy_not_generalisation` precisely so that nobody quotes it as
performance. It is not a generalisation estimate.

### Run-to-run variance — read this before quoting 0.4439

Refitting the *same* architecture under the *same* protocol in
`ml/shipped_model_report.py` gave a pooled accuracy of **0.4188** (folds 0.414 /
0.329 / 0.395 / 0.536) rather than 0.4439. Keras is not fully deterministic
across runs even with a fixed seed.

**So the honest statement is ~0.42–0.44, not a precise 0.4439.** With 4 folds of
one person each, the run-to-run noise is comparable to the differences between
candidate architectures. Treat the mean as indicative, not as a measurement.

### Per class (shipped architecture, pooled over the 4 folds)

`reports/phase4_evaluation/shipped_model_per_class.md`

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| C0_Normal | 0.417 | 0.440 | 0.361 | 325 |
| **C1_Forward_Lean** | **0.117** | **0.132** | **0.119** | 304 |
| C2_Slouching | 0.247 | 0.394 | 0.300 | 297 |
| C3_Left_Lean | 0.568 | 0.700 | 0.616 | 300 |
| C4_Right_Lean | 0.677 | 0.643 | 0.659 | 311 |
| **C5_Backward_Lean** | **0.147** | **0.206** | **0.171** | 311 |

### Confusion matrix (rows = true, columns = predicted)

| | C0 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| **C0_Normal** | 143 | 1 | 181 | 0 | 0 | 0 |
| **C1_Forward_Lean** | 60 | 40 | 42 | 3 | 0 | 159 |
| **C2_Slouching** | 129 | 13 | 117 | 2 | 0 | 36 |
| **C3_Left_Lean** | 16 | 33 | 11 | 210 | 1 | 29 |
| **C4_Right_Lean** | 19 | 0 | 16 | 75 | 200 | 1 |
| **C5_Backward_Lean** | 67 | 141 | 6 | 16 | 17 | 64 |

**What this actually says:**

* **The lateral classes work.** C3_Left_Lean (0.70 recall) and C4_Right_Lean
  (0.64) are the only two that generalise across people. Leaning sideways moves
  the load to one side of the mat, and that survives differences in body mass.
* **The sagittal classes fail.** C1_Forward_Lean and C5_Backward_Lean are
  mutually confused — 159 forward-leans predicted as backward, 141 backward
  predicted as forward. They are close to unusable.
* **C0_Normal collapses into C2_Slouching** (181 of 325).

So the model is roughly a *left / right / upright-ish* classifier, not a
six-class posture classifier. Any demo or report must say so.

---

## 7. Why accuracy is not higher — it is the sensor, not the model

`reports/phase4_evaluation/domain_shift_diagnostic.md`

| | Accuracy |
|---|---|
| **Within** a participant (leave-one-repetition-out) | **~0.87** |
| **Across** participants (leave-one-participant-out) | **0.37 – 0.44** |

Every feature transform tested lands in the same narrow band — global scale
(the spec default), per-frame peak normalisation, per-frame mean normalisation,
binary occupancy, per-session z-scoring, row/column profiles, centre of
pressure. Every model family tested lands there too.

The decisive measurement: a logistic regression trained to predict **which
participant** a frame came from reaches **99.5 %** in-sample across 4 classes
(chance 25 %). Participant identity is written into these frames far more
strongly than posture is, and 4 participants are not enough variation to learn
a participant-invariant representation.

Best cross-participant transform found: **centre of pressure, 6 features,
0.44** — slightly better than all 504 raw values. Not shipped, because the spec
fixes the model input at 504 dimensions, but it is the strongest lead for
future work.

---

## 8. Quantisation and the embedded kernel

Full-integer INT8: weights, activations, input and output.

| Check | Result | Tolerance |
|---|---|---|
| INT8 vs float top-1 agreement | **0.9935** | ≥ 0.99 |
| INT8 vs float max probability difference | 0.0700 | documented |
| **Generated C++ vs TFLite INT8 — argmax** | **500 / 500** | identical on all frames |
| **Generated C++ vs TFLite INT8 — max logit difference** | **0** | ±1 quantisation step |
| Independent numpy replica vs TFLite | 500 / 500 | — |

`esp32/inference.cpp` reimplements TFLite's integer `FULLY_CONNECTED` kernel:
int32 accumulators, per-channel requantisation, **single-rounding** on an int64
intermediate (`TFLITE_SINGLE_ROUNDING`).

> Note for anyone porting this: gemmlowp's older double-rounding path
> (`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`) is **not**
> equivalent. It pushes values just below `.5` up — a true 71.4962 becomes 71.5
> after the first rounding, then 72 — which flipped a near-tie argmax on 1 frame
> in 500. Getting this right is what took the match from 499/500 to 500/500.

### Memory

`reports/phase5_export/memory_budget.md`

| | Bytes | |
|---|---|---|
| Flash (model parameters) | 17 563 | 17.2 kB |
| Static RAM (all buffers incl. calibration) | 32 832 | 32.1 kB |
| Inference workspace alone | 4 536 | 4.4 kB |
| Dynamic allocation in the inference loop | **none** | no `malloc`/`new`/`vector` |
| Stack | < 256 B | no recursion, no VLAs |

The ESP32 DevKit V1 has 520 kB SRAM and 4 MB flash, so this fits with very
large margin. **Compiler-reported flash/RAM: PENDING_HARDWARE** — no toolchain
is installed on the machine that produced this.

### Artefacts

| File | Size | What |
|---|---|---|
| `models/model_float.keras` | 231 857 B | float model (logits output) |
| `models/model_int8.tflite` | 20 496 B | quantised model |
| `esp32/model_data.h` | 77 555 B | generated C source: int8 weights, int32 biases, per-channel multipliers/shifts |
| `models/model_metadata.json` | — | every number above, machine-readable |

---

## 9. Runtime behaviour on device

| | |
|---|---|
| Frame rate | ~1.4 Hz (504 sensors × 16 samples × 300 µs settle) |
| Probability smoothing | moving average over the last **3** frames |
| Confidence threshold | **0.55** — below this the label reported is `UNKNOWN` |
| Calibration | 20 unloaded frames at boot; refuses a loaded or unstable baseline |
| **Inference time** | **PENDING_HARDWARE** — printed as the 4th field of every `PRED,` line |

---

## 10. Limitations

1. **Four real participants.** Every cross-participant number is a mean over
   four folds of one person each. Run-to-run variance (§6) is comparable to the
   between-architecture differences.
2. **It is not really a six-class classifier.** Left/right lean work; forward
   and backward lean are mutually confused and near-unusable; normal and
   slouching blur together (§6).
3. **Synthetic participants are not people.** They model sensor and session
   variation, not new body types, and they did not improve generalisation.
4. **The sensor saturates**, which caps what any model can achieve (§7).
   Hardware changes — series resistance, lower ADC attenuation, reduced
   crosstalk — will buy more than modelling work will.
5. **Column 20 is an intermittent hardware fault**; the model masks it, but the
   underlying connection should be fixed.
6. **Nothing has been measured on hardware.** Compilation, upload, on-device
   inference time and live accuracy are all PENDING_HARDWARE. See
   `ESP32_GUIDE.md`.
7. **All 4 participants are in the shipped model's training set.** Testing on
   P1–P4 measures nothing. Recruit new people.

---

## 11. Reproducing every number here

```powershell
.\venv\Scripts\Activate.ps1
python ml\inspect_dataset.py        # dataset report
python ml\build_dataset.py          # models/dataset.npz
python ml\verify_preprocess.py      # Python vs C++ equivalence
python ml\augment.py                # synthetic participants
python ml\validate_synthetic.py
python ml\build_dataset.py          # rebuild including synthetic
python ml\evaluate.py               # nested LOPO, ~43 min
python ml\diagnose_shift.py         # domain-shift diagnostic
python ml\decide_model.py           # decision rule + architecture choice
python ml\export_int8.py            # train production model + INT8 + model_data.h
python ml\verify_inference.py       # C++ vs TFLite + memory budget
python ml\shipped_model_report.py   # per-class for the shipped architecture
python ml\verify_firmware.py        # scan-preservation check
python ml\audit.py                  # 10 acceptance checks
python ml\experiment_summary.py
python -m pytest tests -q           # 69 tests
```
