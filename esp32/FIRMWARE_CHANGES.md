# Firmware changes — annotated

| | |
|---|---|
| Original | `esp32/backup_20261115/matrix_scanner_original.ino` (sha256 `c669202b66a7f8bad5d502b2a703e112f578e5112fe354819732139a9a92f5fd`) |
| New | `esp32/smart_cushion/smart_cushion.ino` |
| Automated check | `ml/verify_firmware.py` → `reports/firmware_scan_diff.md` (**PASS**) |

The scan is untouched. The automated check confirms, mechanically:

* all 17 pin and timing constants identical and equal to the required values
* `setAddress()` and `readAverage()` byte-identical after comment/whitespace normalisation
* the 18-operation hardware sequence inside the scan identical and in the same order
* the same 13 `pinMode()` calls
* `analogReadResolution(12)`, 11 dB attenuation on both ADC pins, `Serial.begin(115200)` all present

## Change 1 — the scan body moved from `loop()` into `scanMatrix()`

**What:** the entire `for (int r = 0; r < ROWS; r++) { ... }` block was cut from
`loop()` and pasted, unmodified, into a new function `scanMatrix()`.

**Why:** the spec requires inference to run *after* the full 504-value frame has
been acquired. Keeping the scan in its own function makes it impossible for
later edits to interleave inference work with the acquisition timing, and it
lets the calibration routine reuse the identical scan.

**Risk:** none to timing. The function is called once per `loop()` iteration at
exactly the same point the inline block used to run. There is no extra work
between the first and last ADC read.

## Change 2 — frame output moved to `printFrame()` and prefixed with `FRAME,`

**Was:**
```cpp
Serial.print(readings[r][c]);
if (!(r == ROWS - 1 && c == COLS - 1)) Serial.print(",");
```
**Now:** the same 504 values, same row-major order, same comma separation,
emitted from `printFrame()` with a `FRAME,` prefix.

**Why:** the required protocol is `FRAME,<504 values>`, and a prefix lets the
host distinguish frames from `PRED`/`STATE`/`CAL`/`INFO` lines.

**Compatibility note:** this **breaks** the original `PLOT.PY` and the
`collect_p*.py` collectors, which parse a bare comma-separated line and reject
anything whose field count is not 504. `MODE RAW` does not restore the bare
format — it only suppresses `PRED`. If you want to re-run the original
collection scripts against this firmware, either flash the backup sketch or
change their `read_frame()` to strip a leading `FRAME,`. The project's own
tooling (`dashboard/`) speaks the new protocol.

## Change 3 — startup calibration

`setup()` now ends with `runCalibration()`, which scans 20 frames, feeds them to
`scCalibrationAddFrame()`, and calls `scCalibrationFinish()`. The baseline is
**refused** if any acceptance rule fails (`PRESSURE_DETECTED`,
`UNSTABLE_BASELINE`, `EXCESSIVE_NOISE`, `NO_SIGNAL`, `RAIL_SATURATED`,
`TOO_FEW_FRAMES`) and the device enters `CALIBRATION_FAILED`. In that state the
scan and `FRAME,` output continue, but every `PRED,` line reports `UNKNOWN` with
confidence 0. Sending `CAL` retries.

The baseline/threshold formulas are the same ones the original collection
scripts used (`baseline = mean of 20 unloaded frames`,
`threshold = max(3*std, 20)`), so the device reproduces the preprocessing the
training data was built with.

## Change 4 — inference after the frame, with smoothing and an UNKNOWN state

```cpp
scanMatrix();            // unchanged
flattenFrame();          // readings[r][c] -> gFrame[r*21 + c]
printFrame();
if (gMode == MODE_PREDICT && scCalState() == SC_CAL_READY) {
  uint32_t t0 = micros();
  scPreprocessFrame(gFrame, gFeatures);
  scInfer(gFeatures, &res);
  uint32_t dt = micros() - t0;
  ...
}
```

* `micros()` brackets preprocessing **and** inference, so the reported
  `inference_us` is the full feature-to-label cost, not just the matmul.
* probabilities are averaged over the last `SC_SMOOTH_WINDOW = 3` frames. The
  scan runs at roughly 1.4 Hz, so a longer window would make the label lag the
  sitter by several seconds.
* if the smoothed top probability is below `SC_CONFIDENCE_THRESHOLD = 0.55`,
  the label reported is `UNKNOWN`.

## Change 5 — serial commands

`pollSerial()` reads at most 64 characters per line and dispatches
`MODE RAW`, `MODE PREDICT`, `MODE BOTH`, `CAL`, `STATUS`, `HELP`. It is called
once per loop iteration, before the scan, so command handling never interrupts
an in-progress acquisition.

## Memory

No dynamic allocation anywhere. All buffers are file-scope statics:
`readings[24][21]` (int, 2016 B), `gFrame[504]` (int32, 2016 B),
`gFeatures[504]` (float, 2016 B), `gProbHistory[3][6]` (float, 72 B), plus the
preprocessing calibration arrays `scBaseline`/`scThreshold` (504 × int32 each,
4032 B) and the calibration accumulators `gSum`/`gSumSq` (504 × double each,
8064 B). Measured totals are in `reports/phase5_export/memory_budget.md`.
