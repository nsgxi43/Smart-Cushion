# CLAUDE.md — Smart Cushion posture classifier

Key decisions, file paths and commands. Read this first when resuming work.

## Where things live

| Path | Purpose |
|---|---|
| `DATASET/` | **Original data. Never modified.** SHA-256 recorded in `reports/checksums_phase1.csv`. |
| `AUGMENTED_DATASET/` | Synthetic participants P5_SYNTHETIC..P10_SYNTHETIC (Phase 3). |
| `ml/` | Python pipeline: inspection, preprocessing, dataset build, augmentation, training, export. |
| `models/` | `dataset.npz`, `preprocess_config.json`, trained models, quantised artefacts. |
| `esp32/` | Firmware. `backup_20261115/` holds the untouched original sketch. |
| `dashboard/` | Streamlit app + serial tester. |
| `reports/` | All evidence. Nothing is claimed here that is not backed by a file in `reports/`. |
| `tests/` | pytest suite. |
| `venv/` | Python 3.12 virtualenv (git-ignored). |

## Environment

```
py -3.12 -m venv venv
venv\Scripts\activate          # PowerShell:  .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PlatformIO and arduino-cli are **not installed** on this machine; firmware
compilation and upload are PENDING_HARDWARE (see README.md).

## Hard constraints (do not change)

* Matrix geometry: `ROWS=24`, `COLS=21`, `N=504`.
* Feature ordering: **row-major**, `index = row * 21 + col`. This is the order
  the firmware prints on the `FRAME,` line, the order `reshape(-1)` produced in
  `PLOT_6_POSITIONS.py`, and the order of the `sensor_*` CSV columns.
* Pins: `ROW_S0..S3 = 25,26,27,14`; `COL_S0..S3 = 16,17,18,19`;
  `ROW_EN_1/2 = 21,22`; `ROW_SIG = 13`; ADC `GPIO34` (cols 0–15),
  `GPIO35` (cols 16–20). `SAMPLES=16`, `SETTLE_US=300`, 12-bit, 11 dB, 115200.
* The scan loop in `esp32/` must stay byte-identical in behaviour to
  `esp32/backup_20261115/matrix_scanner_original.ino`.

## Posture label mapping (decided Phase 1)

The original folders use a position code `P1..P6` that visually collides with
the participant ids `P1..P4`. Canonical class names are used everywhere
downstream. Single source of truth: `ml/common.py::POSTURE_FOLDER_TO_CLASS`.

| Original folder | Canonical class | Index |
|---|---|---|
| `P1_Normal` | `C0_Normal` | 0 |
| `P2_Forward_Lean` | `C1_Forward_Lean` | 1 |
| `P3_Slouching` | `C2_Slouching` | 2 |
| `P4_Left_Lean` | `C3_Left_Lean` | 3 |
| `P5_Right_Lean` | `C4_Right_Lean` | 4 |
| `P6_Backward_Lean` | `C5_Backward_Lean` | 5 |

## Dataset facts established in Phase 1

* 4 real participants (P1–P4) × 6 postures × 5 repetitions = **120 recording
  takes**, **1848 frames** total. ~15 frames per 10-second repetition
  (the 504-sensor scan runs at only ~1.4 Hz).
* **Two CSV schemas.** P1 (collected by `PLOT_6_POSITIONS.py`) uses 7 metadata
  columns and 1-indexed `sensor_001..sensor_504`, and ships `metadata.csv` +
  `.npy`. P2–P4 (collected by `collect_p*.py`) use 9 metadata columns and
  0-indexed `sensor_000..sensor_503`, and ship `participant_profile.csv`.
  Both flatten row-major, so sorting sensor columns by numeric suffix recovers
  the identical physical order. Handled in `ml/common.py::sensor_columns`.
* P1 has **no** `participant_profile.csv` (height/weight/sex unknown).
* **Column 20 is an intermittent hardware fault.** Every dead sensor found
  (std = 0 across the participant's whole session) lies in column 20:
  P4 all 24 rows dead, P2 23 of 24, P1 2 of 24, P3 none. Column 20 is
  channel 4 of the second column multiplexer on `GPIO35`.
* **The mat is heavily saturated.** 2–24 % of raw cells sit at ADC 0 and up to
  42 % at ADC 4095, mean raw ≈ 2000–3700 of 4095. Pressure maps are broad
  saturated regions with mux crosstalk striping, not localised hotspots.

## Preprocessing contract (Phase 2)

`pressure_data.csv` was **verified** to equal, to 99.55 % of cells exactly,

```
delta = abs(raw - baseline)          # int32
delta = 0 if delta <= threshold      # int32, threshold = max(3*std_cal, 20)
delta = min(delta, 4095)             # int32
```

(evidence: `reports/dataset_report/pressure_reconstruction_check.csv`, produced
by recovering each recording's integer baseline by exact vote in
`ml/build_dataset.py::recover_baseline`). The remaining 0.45 % is explained by
the recovered threshold being a lower bound; residual errors are almost all
exactly 1 or 20 counts.

Training therefore uses the archived `pressure_data.csv` values and applies
only the final scaling stage, which is what the device computes at run time:

```
x = delta * (1/1024)                 # float32
```

`SCALE = 1024` and `CLIP_MAX = 4095` are fixed a priori from the 12-bit ADC
contract, never fitted to data, so they cannot leak a held-out participant.
There is **no per-frame normalisation** — absolute pressure carries class
information.

## Commands

```
venv\Scripts\python ml\inspect_dataset.py     # Phase 1 report
venv\Scripts\python ml\build_dataset.py       # -> models/dataset.npz
venv\Scripts\python ml\preprocess.py          # -> models/preprocess_config.json
```

(Later phases are appended as they are built.)

## Phase 3 — synthetic participants

`ml/augment.py`. Each synthetic derives from exactly ONE real participant
(`SYNTHETIC_SOURCES`: P5←P1, P6←P2, P7←P3, P8←P4, P9←P1, P10←P2) so a LOPO fold
can drop every frame derived from the held-out person. 2759 frames, 0 identical
to source, 100 % internally consistent with the Phase 2 identity.

## Phase 4 — evaluation protocol

`ml/evaluate.py`. **Nested** leave-one-real-participant-out. The inner LOPO over
the 3 training participants selects among all 30 candidates (4 sklearn families
+ 4 Keras architectures × column-20 masked/unmasked). The held-out participant
is used for nothing but the final score. The per-candidate outer table is
**diagnostic only** — picking its best row would be selection on the test set.

`ml/decide_model.py` makes exactly two decisions:
1. synthetic on/off, by the spec's rule (keep only if held-out real accuracy
   improves or stays equal);
2. architecture, by **inner** score only, smallest wins within 1 SE.

### The dominant finding

Cross-participant generalisation is the bottleneck, not the model.
`reports/phase4_evaluation/domain_shift_diagnostic.md`:

* within a participant (leave-one-repetition-out): **~0.87**
* across participants (LOPO): **0.37–0.44** for *every* feature transform and
  model family tested — global scale, per-frame norm, session-z, binary,
  row/column profiles, centre of pressure
* a logistic regression predicting **which participant** a frame came from
  reaches **99.5 %** in-sample (chance 25 %)

Participant identity is written into these frames far more strongly than
posture is. With 4 participants there is not enough variation to learn a
participant-invariant representation. Centre-of-pressure (6 features) was the
best cross-participant transform at 0.44 — reported as a recommendation, not
shipped, because the spec fixes the input at 504 dimensions.

## Phase 5 — INT8 export

`ml/export_int8.py` → `esp32/model_data.h`; `esp32/inference.cpp` is hand-written
and generic over the header. Three bugs worth remembering:

1. **XNNPACK** fuses FULLY_CONNECTED into a `DELEGATE` node, hiding per-layer
   tensors. Use `OpResolverType.BUILTIN_REF` (`export_int8._interpreter`).
2. Reading TFLite **intermediate** tensors needs
   `experimental_preserve_all_tensors=True`; otherwise the arena is reused and
   you read garbage.
3. **Requantisation must be single-rounding.** Modern TFLite defines
   `TFLITE_SINGLE_ROUNDING`: one rounding on an int64 intermediate. gemmlowp's
   double rounding (SaturatingRoundingDoublingHighMul + RoundingDivideByPOT)
   pushes values just below .5 up — a true 71.4962 becomes 71.5 then 72 — which
   flipped a near-tie argmax on 1 frame in 500. Both sides now round once.

Result: generated C++ vs TFLite INT8 = **500/500 argmax, max logit diff 0**.

## Commands (full pipeline)

```
venv\Scripts\python ml\inspect_dataset.py
venv\Scripts\python ml\build_dataset.py
venv\Scripts\python ml\preprocess.py
venv\Scripts\python ml\verify_preprocess.py
venv\Scripts\python ml\augment.py
venv\Scripts\python ml\validate_synthetic.py
venv\Scripts\python ml\build_dataset.py        # rebuild with synthetic
venv\Scripts\python ml\evaluate.py             # ~45 min
venv\Scripts\python ml\diagnose_shift.py
venv\Scripts\python ml\decide_model.py
venv\Scripts\python ml\export_int8.py
venv\Scripts\python ml\verify_inference.py
venv\Scripts\python ml\verify_firmware.py
venv\Scripts\python ml\audit.py
venv\Scripts\python ml\experiment_summary.py
venv\Scripts\python -m pytest tests -q
```

## Gotchas on this machine

* `g++` (WinLibs) mis-resolves its own include path because the toolchain lives
  under a path with spaces. `clang++` from the same package works;
  `ml/cpp_bridge.py` tries clang++ first.
* Large heredocs through the Bash tool truncate; write big files with the
  editor instead.
* PlatformIO and arduino-cli are not installed — all firmware build/upload/
  timing numbers are PENDING_HARDWARE.
