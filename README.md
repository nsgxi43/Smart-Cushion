# Smart Cushion — six-class sitting-posture classifier on an ESP32

24 × 21 Velostat pressure matrix (504 sensors, 4 × 74HC4067 multiplexers) read
by an ESP32 DevKit V1. The posture classifier runs **on the board** — no PC or
cloud at inference time. This repository contains the data pipeline, the
synthetic-participant augmentation, the participant-independent evaluation, the
INT8 export and generated C++ kernel, the firmware, a serial tester and a
Streamlit dashboard.

## Start here

| Document | What it answers |
|---|---|
| **[MODEL_CARD.md](MODEL_CARD.md)** | What model, trained how, on what data, with what results and what limits. All the ML detail. |
| **[ESP32_GUIDE.md](ESP32_GUIDE.md)** | How to compile, flash and run it on the board, and how to measure inference time and live accuracy. |
| [reports/phase9_audit/acceptance_table.md](reports/phase9_audit/acceptance_table.md) | What passed, what failed, what is unmeasured (10 PASS, 0 FAIL, 5 PENDING_HARDWARE). |
| [reports/experiment_summary.md](reports/experiment_summary.md) | One-page experiment record with the git commit hash. |
| [reports/phase4_evaluation/domain_shift_diagnostic.md](reports/phase4_evaluation/domain_shift_diagnostic.md) | Why accuracy is what it is. |
| [CLAUDE.md](CLAUDE.md) | Decisions, gotchas and commands for whoever picks this up next. |

### The result in one paragraph

A 16 790-parameter Keras network (`504-32-16-6`, INT8) classifies six sitting
postures on the ESP32 itself. On participants it has never seen it scores
**~0.42–0.44 accuracy against a 0.167 chance level**. Left and right lean work
(~0.65–0.70 recall); forward and backward lean are mutually confused and close
to unusable. Within a single participant the same pipeline reaches ~0.87 — the
gap is cross-participant domain shift caused by a saturating sensor, not by the
model. Synthetic augmentation was generated, evaluated, and **dropped** because
it did not improve held-out performance. Training accuracy was 0.7857 and is
never reported as performance.

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

# Running it on the ESP32

Full step-by-step instructions — toolchain install, assembling the sketch
folder, compiling, flashing, calibration, the serial protocol, measuring
on-device inference time, and running a live accuracy test — are in
**[ESP32_GUIDE.md](ESP32_GUIDE.md)**.

Nothing in that guide has been executed: no board was connected and no
toolchain is installed on the machine that built this project. **No claim is
made anywhere in this repository that the firmware compiles, uploads or runs.**
Five audit items stay PENDING_HARDWARE until you run them:

| # | Item | Section |
|---|---|---|
| H1 | Firmware compiles | ESP32_GUIDE §3 |
| H2 | Firmware uploads | ESP32_GUIDE §4 |
| H3 | On-device inference time measured with `micros()` | ESP32_GUIDE §8 |
| H4 | Live accuracy on a real seated person | ESP32_GUIDE §9 |
| H5 | Compiler-reported flash / RAM | ESP32_GUIDE §3 |

The short version:

```powershell
winget install ArduinoSA.CLI
arduino-cli config init
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index ; arduino-cli core install esp32:esp32

cd esp32
copy preprocess.h smart_cushion\  ; copy preprocess.cpp smart_cushion\
copy inference.h  smart_cushion\  ; copy inference.cpp  smart_cushion\
copy model_data.h smart_cushion\
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 smart_cushion
arduino-cli board list
arduino-cli upload -p COM5 --fqbn esp32:esp32:esp32doit-devkit-v1 smart_cushion
cd ..

python dashboard\serial_tester.py --port COM5 --seconds 60
streamlit run dashboard\app.py
```

> **Re-copy `model_data.h` into `smart_cushion\` every time you retrain** — it
> is generated by `ml\export_int8.py`, and a stale copy silently flashes an old
> model.

> **The `FRAME,` prefix breaks the original `PLOT.PY` and `collect_p*.py`
> scripts.** Flash `esp32ackup_20261115\matrix_scanner_original.ino` if you
> need them, or teach them to strip the prefix. See `esp32/FIRMWARE_CHANGES.md`.

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
