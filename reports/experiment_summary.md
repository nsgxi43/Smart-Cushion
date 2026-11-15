# Experiment summary

| field | value |
|---|---|
| Experiment ID | `SC-POSTURE-1.0.0` |
| Date | 2026-11-15T12:30:14 |
| Git commit | `e1862b0cc987dad2c2ff17c63b186615ea8b4032` |
| Dataset version | `DATASET/session_20261114_170217` (1848 real frames, 24 recordings, SHA-256 in reports/checksums_phase1.csv) |
| Augmented dataset | `AUGMENTED_DATASET/session_synthetic_v1` (2759 synthetic frames) |
| Classes | C0_Normal, C1_Forward_Lean, C2_Slouching, C3_Left_Lean, C4_Right_Lean, C5_Backward_Lean |
| Model architecture | 504-32-16-6 (logits) |
| Parameters | 16790 |
| Column 20 masked | True |
| Training configuration | dropout 0.30, L2 0.0001, 120 epochs, batch 64, Adam 1e-3, seed 20261115 |
| Training set | REAL_ONLY |
| Evaluation split | nested leave-one-REAL-participant-out (4 outer folds); inner LOPO over the 3 remaining real participants selects the model; splits are by recording take |
| Quantisation | full-integer INT8 (weights, activations, input, output) |
| Model size | 17.0 kB INT8 weights+params in flash |
| Estimated static RAM | 32832 B (32.1 kB), inference workspace 4536 B |
| Flash (model parameters) | 17563 B (17.2 kB) |
| **ESP32 inference time** | **PENDING_HARDWARE** (no board connected; the firmware prints it as the 4th field of each `PRED,` line) |
| Chance level | 16.7% (6 classes) |

## Real held-out participant metrics

These are the only generalisation numbers in this project. Each fold trains on 3 real participants (plus, when enabled, the synthetic participants not derived from the held-out person) and tests on the 4th real participant's frames.

### Nested protocol (unbiased: the model is chosen by the inner loop)

| held-out | REAL_ONLY | REAL_PLUS_SYNTHETIC |
|---|---|---|
| P1 | 0.3091 | 0.3750 |
| P2 | 0.2017 | 0.0849 |
| P3 | 0.3974 | 0.4082 |
| P4 | 0.4451 | 0.4810 |
| **mean** | **0.3383** | **0.3373** |
| std | 0.1071 | 0.1740 |
| min | 0.2017 | 0.0849 |

### The shipped architecture, held out per fold

Architecture chosen on inner-validation score only; these held-out numbers are reported, not used to choose it.

| held-out participant | accuracy | macro F1 |
|---|---|---|
| P1 | 0.4182 | 0.4134 |
| P2 | 0.3355 | 0.2197 |
| P3 | 0.3974 | 0.3695 |
| P4 | 0.6245 | 0.5832 |
| **mean** | **0.4439** | **0.3965** |
| min | 0.3355 | |

### Per-class, REAL_ONLY, nested protocol (pooled over folds)

| class            |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| C0_Normal        |      0.5206 |   0.3108 | 0.3103 |       325 |
| C1_Forward_Lean  |      0.0181 |   0.0033 | 0.0056 |       304 |
| C2_Slouching     |      0.2154 |   0.3468 | 0.251  |       297 |
| C3_Left_Lean     |      0.6272 |   0.51   | 0.5137 |       300 |
| C4_Right_Lean    |      0.449  |   0.4373 | 0.442  |       311 |
| C5_Backward_Lean |      0.233  |   0.4244 | 0.3009 |       311 |

### Confusion matrix, REAL_ONLY (rows = true, cols = predicted)

|                  |   C0_Normal |   C1_Forward_Lean |   C2_Slouching |   C3_Left_Lean |   C4_Right_Lean |   C5_Backward_Lean |
|:-----------------|------------:|------------------:|---------------:|---------------:|----------------:|-------------------:|
| C0_Normal        |         101 |                 0 |            223 |              0 |               0 |                  1 |
| C1_Forward_Lean  |           2 |                 1 |            100 |             67 |               0 |                134 |
| C2_Slouching     |          74 |                80 |            103 |              8 |               0 |                 32 |
| C3_Left_Lean     |          13 |                12 |             11 |            153 |              60 |                 51 |
| C4_Right_Lean    |          13 |                74 |              0 |             10 |             136 |                 78 |
| C5_Backward_Lean |          38 |                85 |             17 |             29 |              10 |                132 |

## Synthetic participant metrics — reported separately

Synthetic participants are **not** people and their scores are **never** merged into the real held-out numbers above. They are reported only to show that augmentation preserved posture structure.

| participant   | kind      |   within_participant_nearest_centroid_acc |
|:--------------|:----------|------------------------------------------:|
| P1            | real      |                                    0.7273 |
| P2            | real      |                                    0.7686 |
| P3            | real      |                                    0.851  |
| P4            | real      |                                    0.7595 |
| P5_SYNTHETIC  | synthetic |                                    0.7273 |
| P6_SYNTHETIC  | synthetic |                                    0.7665 |
| P7_SYNTHETIC  | synthetic |                                    0.8423 |
| P8_SYNTHETIC  | synthetic |                                    0.7637 |
| P9_SYNTHETIC  | synthetic |                                    0.7227 |
| P10_SYNTHETIC | synthetic |                                    0.7665 |

Within-participant nearest-class-centroid accuracy. Each synthetic participant tracks its real source closely, which is the intended behaviour: the transforms changed the sensor response, not the posture.

| synthetic data check | value |
|---|---|
| recordings | 36 |
| frames | 2759 |
| frames identical to their source | 0 |
| provenance complete | True |
| internal preprocessing identity exact | 100.000% of cells |

## Verification

| check | result | evidence |
|---|---|---|
| Python vs C++ preprocessing | 300/300 frames bit-identical, 0 ULP | `reports/preprocess_equivalence.md` |
| INT8 vs float top-1 agreement | 0.9935 (tolerance >= 0.99) | `reports/phase5_export/export_report.json` |
| INT8 vs float max probability difference | 0.0700 | same |
| Generated C++ vs TFLite INT8 | argmax 500/500, max logit diff 0 (tolerance: identical argmax, +/-1 step) | `reports/phase5_export/cpp_vs_tflite.json` |
| Firmware scan preserved | PASS | `reports/firmware_scan_diff.md` |

## Limitations

- **Four real participants.** Every cross-participant number is a mean over four folds of one person each. The fold spread is large (min 0.202, max 0.445 for the shipped setting). Treat the mean as indicative, not as a measurement.
- **Synthetic data adds sensor-variation robustness, not new body types.** P5–P10 are transformed copies of P1–P4 with per-sensor gain/offset, drift blobs, noise, a +/-1 cell shift and temporal jitter. Height, weight and sex are recorded as `SYNTHETIC/NA`.
- **The sensor saturates**, which caps what any model can do. See `reports/phase4_evaluation/domain_shift_diagnostic.md`: within a participant the postures separate at ~0.87, across participants no feature transform or model family tested exceeds ~0.44.
- **Column 20 is an intermittent hardware fault** — dead for P2 and P4, alive for P3.
- **No on-device measurement exists.** Inference time, flash/RAM as reported by the compiler, and live accuracy are all PENDING_HARDWARE.

## Reproducing every number here

```
python ml\inspect_dataset.py
python ml\build_dataset.py
python ml\verify_preprocess.py
python ml\augment.py
python ml\validate_synthetic.py
python ml\build_dataset.py
python ml\evaluate.py
python ml\diagnose_shift.py
python ml\decide_model.py
python ml\export_int8.py
python ml\verify_inference.py
python ml\verify_firmware.py
python ml\audit.py
python ml\experiment_summary.py
python -m pytest tests -q
```
