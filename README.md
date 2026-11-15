# Smart Cushion — six-class sitting-posture classifier on an ESP32

24 × 21 Velostat pressure matrix (504 sensors, 4 × 74HC4067 multiplexers) read
by an ESP32 DevKit V1. The posture classifier runs **on the board** — no PC or
cloud at inference time. This repository contains the data pipeline, the
synthetic-participant augmentation, the participant-independent evaluation, the
INT8 export and generated C++ kernel, the firmware, a serial tester and a
Streamlit dashboard.

> **Read `reports/phase9_audit/acceptance_table.md` first.** It is the honest
> summary: what passed, what failed, and what cannot be measured without the
> board. Headline accuracy numbers live in
> `reports/phase4_evaluation/nested_results.csv`, and the limitations that
> matter are in `reports/phase4_evaluation/domain_shift_diagnostic.md`.

## Classes

| Original folder | Class | Index |
|---|---|---|
| `P1_Normal` | `C0_Normal` | 0 |
| `P2_Forward_Lean` | `C1_Forward_Lean` | 1 |
| `P3_Slouching` | `C2_Slouching` | 2 |
| `P4_Left_Lean` | `C3_Left_Lean` | 3 |
| `P5_Right_Lean` | `C4_Right_Lean` | 4 |
| `P6_Backward_Lean` | `C5_Backward_Lean` | 5 |

The original dataset names posture folders with a position code `P1..P6` that
collides visually with the participant ids `P1..P4`. Everything downstream uses
the `C0..C5` names. The mapping lives in one place:
`ml/common.py::POSTURE_FOLDER_TO_CLASS`.

## Layout

```
DATASET/            original recordings — never modified (SHA-256 in reports/checksums_phase1.csv)
AUGMENTED_DATASET/  synthetic participants P5_SYNTHETIC..P10_SYNTHETIC + provenance
ml/                 inspection, preprocessing, augmentation, evaluation, export, audit
models/             dataset.npz, preprocess_config.json, trained + quantised models
esp32/              firmware, preprocessing/inference C++, generated model_data.h,
                    backup_20261115/ (untouched original sketch)
dashboard/          Streamlit app, CLI tester, serial + simulated sources
reports/            every number claimed anywhere is backed by a file here
tests/              pytest suite
```

## Setup

```powershell
cd "C:\Users\Niranjana Gopinath\Projects\Smart-Cushion"
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used (Python 3.12, numpy 2.5,
pandas 3.1, scikit-learn 1.9, TensorFlow 2.21, Streamlit 1.67).

## Running the pipeline

Each step writes its evidence under `reports/`.

```powershell
# 1. inspect the original data (read-only)
python ml\inspect_dataset.py           # -> reports/dataset_report/

# 2. build the model-ready arrays
python ml\build_dataset.py             # -> models/dataset.npz
python ml\preprocess.py                # -> models/preprocess_config.json

# 3. Python/C++ preprocessing equivalence
python ml\verify_preprocess.py         # -> reports/preprocess_equivalence.md

# 4. synthetic participants
python ml\augment.py                   # -> AUGMENTED_DATASET/
python ml\validate_synthetic.py        # -> reports/synthetic_report/
python ml\build_dataset.py             # rebuild, now including synthetic

# 5. evaluation (slow: ~30-60 min)
python ml\evaluate.py                  # -> reports/phase4_evaluation/
python ml\diagnose_shift.py            # -> domain_shift_diagnostic.md
python ml\decide_model.py              # -> models/phase4_decision.json

# 6. export + verify
python ml\export_int8.py               # -> esp32/model_data.h, models/model_int8.tflite
python ml\verify_inference.py          # -> reports/phase5_export/

# 7. firmware preservation check + full audit
python ml\verify_firmware.py           # -> reports/firmware_scan_diff.md
python ml\audit.py                     # -> reports/phase9_audit/acceptance_table.md

# 8. tests
python -m pytest tests -q
```

## Dashboard

```powershell
streamlit run dashboard\app.py
```

Pick **Simulated (recorded CSV)** in the sidebar to drive the whole UI without
hardware, or **ESP32 serial** with the board connected. The dashboard always
states where a prediction came from:

* 🟢 **ESP32** — arrived as a `PRED,` line from the board. This is embedded inference.
* 🟡 **Python** — computed on this PC from the exported model. This is **not**
  embedded inference and is never presented as such.

Features: live 24 × 21 heatmap, large posture label, confidence, 60-second
history, time-in-posture chart, model info panel, start/stop test, start/stop
recording, CSV export, reset, device commands (`CAL`, `MODE RAW`,
`MODE PREDICT`, `STATUS`) and a manual-posture selector for measuring live
accuracy.

## CLI tester

```powershell
python dashboard\serial_tester.py --list-ports
python dashboard\serial_tester.py --simulate --participant P3 --map
python dashboard\serial_tester.py --port COM5
```

---

# PENDING_HARDWARE — what you need to run

Nothing below could be measured here: no ESP32 is connected, and neither
PlatformIO nor arduino-cli is installed on this machine. **No claim is made
anywhere in this repository that the firmware compiles, uploads or runs.**

## H1 — Install a toolchain

Either of these. arduino-cli is the lighter option.

```powershell
# arduino-cli
winget install ArduinoSA.CLI
arduino-cli config init
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
```

```powershell
# or PlatformIO
pip install platformio
```

## H2 — Compile, and record flash / RAM

The sketch lives in `esp32/smart_cushion/`. It needs `preprocess.{h,cpp}`,
`inference.{h,cpp}` and `model_data.h`, which sit one directory up — copy them
in first (the Arduino IDE only compiles files beside the `.ino`):

```powershell
cd esp32
copy preprocess.h  smart_cushion\
copy preprocess.cpp smart_cushion\
copy inference.h   smart_cushion\
copy inference.cpp smart_cushion\
copy model_data.h  smart_cushion\

arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 smart_cushion
```

**Record the "Sketch uses … bytes" and "Global variables use … bytes" lines**
and paste them into `reports/phase5_export/memory_budget.md` under a new
*Measured on hardware* heading. The predicted figures in that file are derived
from the generated header, not from a compiler.

With PlatformIO instead:

```powershell
pio run -e esp32doit-devkit-v1
```

## H3 — Flash

```powershell
arduino-cli board list                      # find the port, e.g. COM5
arduino-cli upload -p COM5 --fqbn esp32:esp32:esp32doit-devkit-v1 smart_cushion
```

## H4 — Measure on-device inference time

The firmware brackets `scPreprocessFrame()` + `scInfer()` with `micros()` and
prints the result as the 4th field of every `PRED,` line:

```
PRED,C2_Slouching,0.8123,4210
                         ^^^^ microseconds
```

```powershell
python dashboard\serial_tester.py --port COM5 --seconds 60
```

Take the median of the `µs` column over at least 50 frames and record it in
`models/model_metadata.json` (`esp32_inference_us`, currently
`"PENDING_HARDWARE"`) and in `reports/experiment_summary.md`.

## H5 — Physical posture test

1. Put the cushion on a chair with **nothing on it**.
2. Flash and open the dashboard: `streamlit run dashboard\app.py`, choose
   **ESP32 serial**, pick the port, press **Start test**.
3. Wait for `CAL,OK` and `STATE,READY`. If you see `CAL,PRESSURE_DETECTED` the
   cushion was loaded — clear it and press **Recalibrate (CAL)**.
4. Have the participant sit. For each of the six postures:
   * select the posture in the **Manual posture** dropdown,
   * hold it for ~15 seconds,
   * press **Log this comparison** about once per second (the scan runs at
     roughly 1.4 Hz, so you get ~20 samples per posture).
5. Repeat for every posture, then for every participant you can recruit.
6. Press **Download comparison CSV** and save it to
   `reports/hardware_validation/<date>_<participant>.csv`.
7. The dashboard shows running agreement; the CSV has one row per logged frame
   with the manual label, the predicted label, confidence and origin.

**Recruit people who are not P1–P4.** The whole point of the evaluation
protocol is performance on participants the model has never seen, and the four
in `DATASET/` are all in the training set of the shipped model.

---

# Read this before trusting any accuracy number

* **Four real participants.** Every cross-participant estimate here is a mean
  over four folds of one person each. The fold-to-fold spread is large, so
  treat the mean as a rough indication, not a measurement.
* **Synthetic participants are not new people.** P5_SYNTHETIC–P10_SYNTHETIC are
  transformed copies of real recordings with per-sensor gain/offset, drift,
  noise, a ±1 cell shift and temporal jitter. They add robustness to *sensor
  and session variation*. They cannot add body types, and their height, weight
  and sex are recorded as `SYNTHETIC/NA`.
* **The sensor saturates.** 2–24 % of raw cells sit at ADC 0 and up to 42 % at
  4095; roughly 415–457 of the 504 cells are active in a typical frame with a
  mean peak near 3950. The maps are broad saturated regions with multiplexer
  crosstalk striping, not localised contact patches.
* **Column 20 is an intermittent hardware fault.** Every dead sensor in the
  dataset lies there: dead for P2 and P4, alive for P3, partly dead for P1.
* **Training accuracy is never reported as test accuracy.** The only
  generalisation numbers in this repository come from the nested
  leave-one-real-participant-out protocol in `ml/evaluate.py`.
