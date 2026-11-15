# Phase 2 - Python / C++ preprocessing equivalence

Python: `ml/preprocess.py`  
C++: `esp32/preprocess.cpp` (driven through `esp32/host_harness.exe`)

## 1. Full pipeline on real recorded frames

| metric | value |
|---|---|
| frames compared | 300 |
| bit-identical frames | 300 / 300 |
| max absolute difference | 0 |
| max ULP difference | 0 |
| calibration used | `P1|C0_Normal` (recovered in Phase 1) |

## 2. Per-stage

| stage | frames | exact |
|---|---|---|
| 1-3 raw -> integer pressure | 100 | 100 |
| 4 pressure -> float features | 100 | 100 |

## 3. Edge cases

| case | bit-identical | max abs diff | non-zero features |
|---|---|---|---|
| all_zero | True | 0 | 83 |
| all_max | True | 0 | 504 |
| alternating | True | 0 | 293 |
| just_below_thr | True | 0 | 0 |
| just_above_thr | True | 0 | 504 |
| ramp | True | 0 | 400 |
| random_uniform | True | 0 | 488 |
| random_clustered | True | 0 | 504 |

## 4. Calibration state machine

| scenario | Python | C++ | agree | baseline exact | threshold exact |
|---|---|---|---|---|---|
| clean_unloaded | OK | OK | True | True | True |
| too_few_frames | TOO_FEW_FRAMES | TOO_FEW_FRAMES | True | - | - |
| pressure_detected | PRESSURE_DETECTED | PRESSURE_DETECTED | True | - | - |
| excessive_noise | UNSTABLE_BASELINE | UNSTABLE_BASELINE | True | - | - |
| unstable_baseline | UNSTABLE_BASELINE | UNSTABLE_BASELINE | True | - | - |

## Documented tolerance

**Exact.** Every compared frame was bit-identical in IEEE-754 float32 (0 ULP). The integer stages run in int32 on both sides and the only float operation is a multiply by an exact power of two (1/1024), which is exact in binary floating point.
