# PROGRESS LOG — Smart Cushion posture classifier

Running log. Newest phase at the bottom. Every claim here must be backed by a
file under `reports/`. Anything not measurable without the board is labelled
`PENDING_HARDWARE`.

Project root: `C:\Users\Niranjana Gopinath\Projects\Smart-Cushion`
(extracted from `Downloads\Smart-Cushion-main.zip` on 2026-11-15)

---

## Phase 0 — Environment check — DONE

| Item | Result |
|---|---|
| System Python | 3.12.10 (also 3.11 available via `py -3.11`) |
| Chosen interpreter | Python 3.12 (TensorFlow 2.16+ supports 3.12) |
| Virtualenv | `venv/` created with `py -3.12 -m venv venv` |
| PlatformIO | **NOT INSTALLED** (`pio` not on PATH) |
| arduino-cli | **NOT INSTALLED** (`arduino-cli` not on PATH) |
| git | 2.56.0.windows.4 — repo initialised in project root |
| Board connected | none detected — all on-device numbers are PENDING_HARDWARE |

Pre-existing packages in system Python: boto3, pytest, colorama (no numpy /
pandas / sklearn / tensorflow). A clean venv was therefore created rather than
polluting the system interpreter.

---

## Phase 1 — Inspection — see `reports/dataset_report/`

Artifacts:
- `reports/checksums_phase1.csv` — SHA-256 of all 69 original DATASET files
  plus the 7 original root scripts (re-verified in Phase 9).
- `esp32/backup_20261115/matrix_scanner_original.ino` — untouched copy of the
  firmware supplied by the user
  (sha256 `c669202b66a7f8bad5d502b2a703e112f578e5112fe354819732139a9a92f5fd`).
- `reports/dataset_report/README.md` — full findings.

(Findings summary is appended below once the inspection run completes.)

### Phase 1 findings summary

| | |
|---|---|
| Participants | P1–P4 (4 real), 6 postures each, 5 repetitions = 120 recording takes |
| Frames | 1848 total (~77 per recording, ~15 per repetition) |
| Effective frame rate | ~1.4 Hz (the 504-sensor scan takes ~0.7 s) |
| Integrity | 0 NaN, 0 out-of-range, 0 non-integer, 0 duplicate frames, 0 bad timestamps |
| Label consistency | 24/24 recordings agree between folder name, label column and participant id |

Four findings that shaped the rest of the project:

1. **Two CSV schemas.** P1 (from `PLOT_6_POSITIONS.py`) has 7 metadata columns and
   `sensor_001..504`; P2–P4 (from `collect_p*.py`) have 9 and `sensor_000..503`.
   Both flatten row-major; sorting by numeric suffix recovers the identical
   physical order (cross-participant correlation of the mean C0 map 0.92–0.99).
   P1 has no `participant_profile.csv`, so its height/weight/sex are unknown.
2. **Column 20 is an intermittent hardware fault.** Every dead sensor in the
   dataset lies in column 20: P4 all 24 rows, P2 23 of 24, P1 2, P3 none.
   Column 20 is channel 4 of the second column mux on GPIO35.
3. **Severe ADC saturation.** 2–24 % of cells at 0 and up to 42 % at 4095;
   mean raw 2000–3700 of 4095; ~415–457 of 504 cells active per frame with mean
   peak ~3950. The maps are broad saturated regions with mux crosstalk striping,
   not localised ischial hotspots.
4. **Preprocessing contract verified, not assumed.** `pressure_data.csv` equals
   `clip(gate(|raw − baseline|, max(3σ,20)), 0, 4095)` at 99.55 % of cells
   exactly (`reports/dataset_report/pressure_reconstruction_check.csv`).

---

## Phase 2 — Preprocessing — DONE

Pipeline (int32 then one float multiply), identical in `ml/preprocess.py` and
`esp32/preprocess.cpp`:

```
delta = abs(raw - baseline)        int32
delta = 0 if delta <= threshold    int32    threshold = max(3*sigma_cal, 20)
delta = min(delta, 4095)           int32
x     = delta * (1/1024)           float32
```

`SCALE = 1024` and `CLIP_MAX = 4095` are fixed a priori from the 12-bit ADC
contract, never fitted to data, so they cannot leak a held-out participant.
No per-frame normalisation.

**Result: bit-exact.** 300 recorded frames + 8 edge cases identical to 0 ULP;
all 5 calibration scenarios agree between Python and C++, with identical
baseline and threshold arrays. Evidence: `reports/preprocess_equivalence.md`.

Calibration refuses a baseline that fails any of: `PRESSURE_DETECTED`,
`UNSTABLE_BASELINE`, `EXCESSIVE_NOISE`, `NO_SIGNAL`, `RAIL_SATURATED`,
`TOO_FEW_FRAMES`, and enters `CALIBRATION_FAILED`; serial command `CAL` retries.

---

## Phase 3 — Synthetic participants — DONE

P5_SYNTHETIC..P10_SYNTHETIC, 2759 frames, each derived from exactly one real
participant (P5←P1, P6←P2, P7←P3, P8←P4, P9←P1, P10←P2) so LOPO folds can drop
everything derived from the held-out person.

| check | result |
|---|---|
| Frames identical to their source | **0 of 2759** |
| Provenance complete | 36/36 recordings |
| Internal consistency with the Phase 2 identity | **100.000 %** of cells |
| Within-participant separability vs source | matches (e.g. P7 0.842 vs P3 0.851) |
| Centroid shift / between-posture gap | mean 0.61, max 0.81 (below 1 = posture identity survives) |

Height/weight/sex recorded as `SYNTHETIC/NA`. These are not real people.
Evidence: `reports/synthetic_report/`.

---

## Phase 4 — Models and evaluation — DONE

Nested leave-one-real-participant-out, 30 candidates, both data settings.
Full output: `reports/phase4_evaluation/`.

### Decision rule outcome

| setting | mean held-out accuracy |
|---|---|
| REAL_ONLY | **0.3383** |
| REAL_PLUS_SYNTHETIC | 0.3373 |

Synthetic augmentation did **not** improve held-out real performance, so by the
stated rule it was **dropped**. The shipped model is trained on real data only.
This is a negative result and is reported as one.

### Shipped architecture, held out per fold

Chosen by inner-validation score only (0.4226 inner; next best 0.4107):
`504-32-16-6`, column 20 masked, 16 790 parameters.

| held-out | accuracy | macro F1 |
|---|---|---|
| P1 | 0.4182 | 0.4134 |
| P2 | 0.3355 | 0.2197 |
| P3 | 0.3974 | 0.3695 |
| P4 | 0.6245 | 0.5832 |
| **mean** | **0.4439** | **0.3965** |

Chance is 0.1667. Training-set accuracy was 0.7857 — recorded only to show it is
**not** the generalisation number.

### Why it is not higher

`reports/phase4_evaluation/domain_shift_diagnostic.md`. Within a participant the
postures separate at ~0.87 (leave-one-repetition-out). Across participants no
feature transform or model family tested exceeds ~0.44: global scale, per-frame
norm, per-frame mean norm, binary occupancy, session-z, row/column profiles and
centre of pressure all land in 0.37–0.44. A logistic regression predicting
*which participant* a frame came from reaches 99.5 % in-sample (chance 25 %).
Participant identity dominates posture in this sensor's output, and 4
participants are not enough to learn a participant-invariant representation.

---

## Phase 5 — Export — DONE

| check | result | tolerance |
|---|---|---|
| INT8 vs float top-1 agreement | **0.9935** | ≥ 0.99 |
| INT8 vs float max probability difference | 0.0700 | documented |
| Generated C++ vs TFLite INT8 argmax | **500/500** | identical on all frames |
| Generated C++ vs TFLite INT8 max logit difference | **0** | ±1 step |
| Independent numpy replica | 500/500 | — |

Memory: flash 17 563 B (17.2 kB), static RAM 32 832 B (32.1 kB), inference
workspace 4 536 B, **no dynamic allocation**. Fits the ESP32's 520 kB SRAM /
4 MB flash with large margin. Compiler-reported figures: PENDING_HARDWARE.

---

## Phases 6–8 — Firmware, UI, tests — DONE

* Scan preserved: `reports/firmware_scan_diff.md` **PASS** (17 constants, both
  helper functions, the 18-operation scan sequence, 13 pinMode calls).
* Protocol, smoothing (3-frame window), confidence threshold 0.55, UNKNOWN
  state, `CAL`/`MODE`/`STATUS` commands, startup calibration with
  CALIBRATION_FAILED + retry.
* Dashboard verified serving (HTTP 200); simulated source exercises the whole
  UI without hardware; every prediction is tagged ESP32 or Python.
* **69 pytest tests pass.**

---

## Phase 9 — Audit — 10 PASS, 0 FAIL, 5 PENDING_HARDWARE

`reports/phase9_audit/acceptance_table.md`, `reports/experiment_summary.md`.
