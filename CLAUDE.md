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
