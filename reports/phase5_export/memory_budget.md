# Phase 5 - generated C++ vs TFLite INT8

| metric | value | tolerance | pass |
|---|---|---|---|
| frames compared | 500 | | |
| argmax agreement | 500 / 500 | identical on all frames | True |
| max logit difference | 0 | +/- 1 quantisation step | True |
| max probability difference | 2.15e-07 | | |
| independent numpy replica argmax | 500 / 500 | | True |

**Overall: PASS**

## INT8 vs float

| metric | value | tolerance |
|---|---|---|
| top-1 agreement | 0.9935 | >= 0.99 |
| max probability difference | 0.0700 | documented |

## Memory budget

### Flash (model parameters)

| item | bytes |
|---|---|
| int8 weights | 16736 |
| int32 biases | 216 |
| int32 requant multipliers | 216 |
| int32 requant shifts | 216 |
| ScLayer descriptors | 72 |
| class name strings | 107 |
| **total** | **17563 (17.2 kB)** |

### Static RAM

| item | bytes |
|---|---|
| gBufA (int32 x SC_MAX_UNITS) | 2016 |
| gBufB (int32 x SC_MAX_UNITS) | 2016 |
| quantised input (int8 x 504) | 504 |
| scBaseline (int32 x 504) | 2016 |
| scThreshold (int32 x 504) | 2016 |
| calibration gSum (double x 504) | 4032 |
| calibration gSumSq (double x 504) | 4032 |
| calibration mean/sd scratch (double x 504 x 2) | 8064 |
| medianOfMeans tmp (float x 504) | 2016 |
| firmware readings[24][21] (int) | 2016 |
| firmware gFrame (int32 x 504) | 2016 |
| firmware gFeatures (float x 504) | 2016 |
| firmware gProbHistory (float x 3 x 6) | 72 |
| **total** | **32832 (32.1 kB)** |

Inference workspace alone: **4536 B**. Dynamic allocation in the inference loop: **none**. Stack: < 256 B: runLayer uses a handful of int32 locals and no recursion or VLAs.

The ESP32 DevKit V1 has 520 kB SRAM and 4 MB flash, so this fits with very large margin. Compiler-reported figures are PENDING_HARDWARE (no toolchain installed).

## Timing

- Host round trip through the test harness: **3.521 ms/frame** — this is a PC measurement including stdio and is **not** an ESP32 number.
- **ESP32 inference time: PENDING_HARDWARE.** The firmware brackets `scPreprocessFrame` + `scInfer` with `micros()` and prints the result as the 4th field of every `PRED,` line.
