"""Phase 2 - the one and only preprocessing pipeline.

Every stage here is reproducible bit-for-bit on the ESP32 using int32 and
single-precision float only. `esp32/preprocess.cpp` is a literal translation of
`preprocess_frame()`; `tests/test_preprocess_equivalence.py` pins them together.

Pipeline (applied to one 504-value raw ADC frame, row-major r*21+c)
-------------------------------------------------------------------
  1. delta[i] = abs(raw[i] - baseline[i])            int32
  2. if delta[i] <= threshold[i]: delta[i] = 0       int32   (noise gate)
  3. delta[i] = min(delta[i], CLIP_MAX)              int32   (saturation clamp)
  4. x[i]     = delta[i] * (1.0f / SCALE)            float32 (fixed global scale)

Deliberate choices
------------------
* No per-frame normalisation. Dividing by a per-frame max or mean would erase
  absolute pressure, which is the main cue separating e.g. C2_Slouching
  (heavy, concentrated) from C0_Normal (lighter, spread).
* SCALE and CLIP_MAX are fixed a priori from the hardware contract, not fitted
  to any participant's data, so they cannot leak the held-out participant.
  SCALE = 1024 is a power of two (a quarter of the 12-bit full scale) chosen so
  that ordinary seated pressures land in roughly [0, 1].
* baseline[] and threshold[] come from the *device's own* startup calibration,
  exactly as the original collection scripts computed them from 20 unloaded
  frames. They are per-sensor and per-session, never baked into the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS = 24
COLS = 21
N_FEATURES = ROWS * COLS          # 504
ADC_MIN = 0
ADC_MAX = 4095

# ---- fixed pipeline constants (mirrored verbatim in esp32/preprocess.cpp) ----
CLIP_MAX = 4095                   # 12-bit full scale; no valid reading is clipped
SCALE = 1024.0                    # fixed global divisor, power of two
INV_SCALE = 1.0 / SCALE

# ---- calibration constants (same formula as the original collection scripts) -
CALIBRATION_FRAMES = 20           # unloaded frames averaged for the baseline
NOISE_MULTIPLIER = 3              # threshold = max(3 * std, MIN_THRESHOLD)
MIN_THRESHOLD = 20

# ---- calibration acceptance rules (device startup, Phase 2 requirement) -----
# Reject a baseline that was captured with someone sitting on the cushion, or
# that is not stable across the calibration window.
CAL_MAX_MEAN_ABS_DEV = 25.0       # mean |frame - running mean| must stay below
CAL_MAX_LOADED_CELLS = 10         # cells deviating > CAL_LOAD_DELTA from median
CAL_LOAD_DELTA = 150.0            # "pressure detected" deviation, ADC counts
CAL_MAX_NOISE_STD = 120.0         # mean per-sensor std ceiling
CAL_MIN_BASELINE = 1.0            # a wholly zero baseline means a wiring fault
CAL_MAX_BASELINE = float(ADC_MAX - 50)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "models" / "preprocess_config.json"


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------
def preprocess_frame(raw, baseline, threshold):
    """One raw frame -> one 504-dim float32 feature vector.

    raw, baseline, threshold: array-like of length 504 (row-major r*21+c).
    Integer stages are done in int32 so the ESP32 translation is exact.
    """
    raw_i = np.asarray(raw, dtype=np.int32)
    base_i = np.asarray(baseline, dtype=np.int32)
    thr_i = np.asarray(threshold, dtype=np.int32)
    if raw_i.shape[-1] != N_FEATURES:
        raise ValueError("expected %d values, got %d" % (N_FEATURES, raw_i.shape[-1]))

    delta = np.abs(raw_i - base_i).astype(np.int32)          # stage 1
    delta = np.where(delta <= thr_i, np.int32(0), delta)     # stage 2
    delta = np.minimum(delta, np.int32(CLIP_MAX))            # stage 3
    return (delta.astype(np.float32) * np.float32(INV_SCALE))  # stage 4


def preprocess_batch(raw_frames, baseline, threshold):
    """(N,504) raw -> (N,504) float32 features."""
    raw_i = np.asarray(raw_frames, dtype=np.int32)
    base_i = np.asarray(baseline, dtype=np.int32).reshape(1, -1)
    thr_i = np.asarray(threshold, dtype=np.int32).reshape(1, -1)
    delta = np.abs(raw_i - base_i).astype(np.int32)
    delta = np.where(delta <= thr_i, np.int32(0), delta)
    delta = np.minimum(delta, np.int32(CLIP_MAX))
    return delta.astype(np.float32) * np.float32(INV_SCALE)


def pressure_to_features(pressure_frames):
    """Scale already-baseline-subtracted 'pressure' counts to model features.

    The recorded `pressure_data.csv` files already contain stages 1-3 (the
    collection scripts applied abs-difference, the identical noise gate and a
    0..4095 clamp), so only stage 4 remains. This is what makes training on the
    archived pressure CSVs faithful to what the device will compute at run time.
    """
    p = np.asarray(pressure_frames, dtype=np.int32)
    p = np.clip(p, 0, CLIP_MAX)
    return p.astype(np.float32) * np.float32(INV_SCALE)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class CalibrationError(Exception):
    """Raised when a candidate baseline must be rejected."""


def compute_calibration(frames):
    """20+ unloaded frames (N,504) -> (baseline int32, threshold int32).

    Mirrors `esp32/preprocess.cpp::calibrationFinish()`.
    """
    f = np.asarray(frames, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != N_FEATURES:
        raise ValueError("calibration frames must be (N,504)")
    if f.shape[0] < CALIBRATION_FRAMES:
        raise CalibrationError("need >= %d frames, got %d"
                               % (CALIBRATION_FRAMES, f.shape[0]))

    mean = f.mean(axis=0)
    std = f.std(axis=0)
    baseline = np.rint(mean).astype(np.int32)
    threshold = np.rint(np.maximum(NOISE_MULTIPLIER * std,
                                   float(MIN_THRESHOLD))).astype(np.int32)
    return baseline, threshold


def validate_calibration(frames):
    """Decide whether a calibration window is acceptable.

    Returns (ok: bool, reason: str, diagnostics: dict). The device refuses the
    baseline and enters CALIBRATION_FAILED when ok is False.
    """
    f = np.asarray(frames, dtype=np.float64)
    diag = {}
    if f.ndim != 2 or f.shape[1] != N_FEATURES:
        return False, "BAD_SHAPE", diag
    if f.shape[0] < CALIBRATION_FRAMES:
        return False, "TOO_FEW_FRAMES", {"frames": int(f.shape[0])}

    mean = f.mean(axis=0)
    std = f.std(axis=0)
    median_level = float(np.median(mean))

    diag["mean_abs_dev"] = float(np.abs(f - mean).mean())
    diag["mean_noise_std"] = float(std.mean())
    diag["max_noise_std"] = float(std.max())
    diag["loaded_cells"] = int((np.abs(mean - median_level) > CAL_LOAD_DELTA).sum())
    diag["baseline_min"] = float(mean.min())
    diag["baseline_max"] = float(mean.max())
    diag["baseline_median"] = median_level

    # a sustained local deviation means something is resting on the cushion
    if diag["loaded_cells"] > CAL_MAX_LOADED_CELLS:
        return False, "PRESSURE_DETECTED", diag
    # drifting / unstable window
    if diag["mean_abs_dev"] > CAL_MAX_MEAN_ABS_DEV:
        return False, "UNSTABLE_BASELINE", diag
    if diag["mean_noise_std"] > CAL_MAX_NOISE_STD:
        return False, "EXCESSIVE_NOISE", diag
    # wiring / ADC faults
    if diag["baseline_max"] < CAL_MIN_BASELINE:
        return False, "NO_SIGNAL", diag
    if diag["baseline_min"] > CAL_MAX_BASELINE:
        return False, "RAIL_SATURATED", diag
    return True, "OK", diag


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------
def build_config(model_input_spec=None, notes=None):
    cfg = {
        "config_version": 1,
        "geometry": {
            "rows": ROWS, "cols": COLS, "n_features": N_FEATURES,
            "feature_order": "row_major: feature_index = row * 21 + col",
            "firmware_transmission_order": "same as feature_order (FRAME line)",
        },
        "adc": {"bits": 12, "min": ADC_MIN, "max": ADC_MAX,
                "attenuation_db": 11, "samples_averaged": 16,
                "settle_us": 300, "baud": 115200},
        "pipeline": [
            {"stage": 1, "op": "delta = abs(raw - baseline)", "dtype": "int32"},
            {"stage": 2, "op": "if delta <= threshold: delta = 0", "dtype": "int32"},
            {"stage": 3, "op": "delta = min(delta, CLIP_MAX)", "dtype": "int32"},
            {"stage": 4, "op": "x = delta * INV_SCALE", "dtype": "float32"},
        ],
        "constants": {
            "CLIP_MAX": CLIP_MAX,
            "SCALE": SCALE,
            "INV_SCALE": INV_SCALE,
            "NOISE_MULTIPLIER": NOISE_MULTIPLIER,
            "MIN_THRESHOLD": MIN_THRESHOLD,
            "CALIBRATION_FRAMES": CALIBRATION_FRAMES,
        },
        "calibration_acceptance": {
            "CAL_MAX_MEAN_ABS_DEV": CAL_MAX_MEAN_ABS_DEV,
            "CAL_MAX_LOADED_CELLS": CAL_MAX_LOADED_CELLS,
            "CAL_LOAD_DELTA": CAL_LOAD_DELTA,
            "CAL_MAX_NOISE_STD": CAL_MAX_NOISE_STD,
            "CAL_MIN_BASELINE": CAL_MIN_BASELINE,
            "CAL_MAX_BASELINE": CAL_MAX_BASELINE,
            "failure_states": ["PRESSURE_DETECTED", "UNSTABLE_BASELINE",
                               "EXCESSIVE_NOISE", "NO_SIGNAL", "RAIL_SATURATED",
                               "TOO_FEW_FRAMES", "BAD_SHAPE"],
            "retry_command": "CAL",
        },
        "no_per_frame_normalisation": True,
        "rationale": (
            "Absolute pressure carries class information, so no per-frame max/mean "
            "normalisation is applied. SCALE and CLIP_MAX are hardware-derived "
            "constants, never fitted to data, so they cannot leak a held-out "
            "participant."
        ),
        "model_input_spec": model_input_spec or {},
        "notes": notes or [],
    }
    return cfg


def save_config(path=CONFIG_PATH, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_config(**kwargs)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    cfg = save_config()
    print("wrote " + str(CONFIG_PATH))
    print(json.dumps(cfg["constants"], indent=2))
