"""Firmware scan preservation, preprocess_config.json, and the C++/Python
constant agreement."""
import json
import re
from pathlib import Path

import pytest

import preprocess as pp

ROOT = Path(__file__).resolve().parent.parent
ESP32 = ROOT / "esp32"
ORIGINAL = ESP32 / "backup_20261115" / "matrix_scanner_original.ino"
SKETCH = ESP32 / "smart_cushion" / "smart_cushion.ino"

REQUIRED_PINS = {"ROWS": 24, "COLS": 21, "ROW_S0": 25, "ROW_S1": 26, "ROW_S2": 27,
                 "ROW_S3": 14, "COL_S0": 16, "COL_S1": 17, "COL_S2": 18,
                 "COL_S3": 19, "ROW_EN_1": 21, "ROW_EN_2": 22, "ROW_SIG": 13,
                 "COL_ADC_1": 34, "COL_ADC_2": 35, "SAMPLES": 16,
                 "SETTLE_US": 300}


def consts(src):
    return {n: int(m) for n, m in
            re.findall(r"const\s+int\s+(\w+)\s*=\s*(\d+)\s*;", src)}


def test_backup_of_original_firmware_exists():
    assert ORIGINAL.is_file(), "the original sketch must be preserved untouched"


def test_sketch_pins_match_requirements():
    c = consts(SKETCH.read_text(encoding="utf-8"))
    for name, want in REQUIRED_PINS.items():
        assert c.get(name) == want, "%s is %s, must be %s" % (name, c.get(name), want)


def test_sketch_pins_match_the_original():
    co = consts(ORIGINAL.read_text(encoding="utf-8"))
    cn = consts(SKETCH.read_text(encoding="utf-8"))
    for name in REQUIRED_PINS:
        assert co[name] == cn[name], "%s changed from the original" % name


def test_adc_configuration_preserved():
    s = SKETCH.read_text(encoding="utf-8")
    assert "analogReadResolution(12)" in s
    assert "analogSetPinAttenuation(COL_ADC_1, ADC_11db)" in s
    assert "analogSetPinAttenuation(COL_ADC_2, ADC_11db)" in s
    assert "Serial.begin(115200)" in s


def test_scan_preservation_check_passes():
    """Run the Phase 9 check 8 verifier and require a pass."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "ml" / "verify_firmware.py")],
                       capture_output=True, text=True)
    report = ROOT / "reports" / "firmware_scan_diff.json"
    assert report.is_file(), r.stdout[-2000:] + r.stderr[-2000:]
    j = json.loads(report.read_text())
    failed = [c["name"] for c in j["checks"] if not c.get("pass")]
    assert j["overall_pass"], "failed sub-checks: %s" % failed


def test_inference_runs_after_the_full_frame():
    """The sketch must scan, flatten, then infer - never interleave."""
    s = SKETCH.read_text(encoding="utf-8")
    body = s[s.index("void loop()"):]
    i_scan = body.index("scanMatrix()")
    i_flat = body.index("flattenFrame()")
    i_pre = body.index("scPreprocessFrame")
    i_inf = body.index("scInfer(")
    assert i_scan < i_flat < i_pre < i_inf


def test_flatten_is_row_major():
    s = SKETCH.read_text(encoding="utf-8")
    assert "gFrame[i++] = (int32_t)readings[r][c];" in s
    # the r loop must be outside the c loop
    blk = s[s.index("static void flattenFrame"):]
    assert blk.index("for (int r = 0") < blk.index("for (int c = 0")


def test_protocol_lines_present():
    s = SKETCH.read_text(encoding="utf-8")
    assert 'Serial.print("FRAME,")' in s
    assert 'Serial.print("PRED,")' in s
    for cmd in ('"CAL"', '"MODE RAW"', '"MODE PREDICT"', '"STATUS"'):
        assert cmd in s, "missing command %s" % cmd


def test_unknown_state_and_threshold_present():
    s = SKETCH.read_text(encoding="utf-8")
    assert "SC_CONFIDENCE_THRESHOLD" in s
    assert "STATE_UNKNOWN" in s
    assert "SC_SMOOTH_WINDOW" in s
    assert "smoothedPrediction" in s


def test_no_dynamic_allocation_in_inference_path():
    for f in ("inference.cpp", "preprocess.cpp"):
        src = (ESP32 / f).read_text(encoding="utf-8")
        for bad in ("malloc(", "calloc(", "realloc(", "new ", "std::vector"):
            assert bad not in src, "%s uses %s" % (f, bad)


# --------------------------------------------------------- config agreement
def test_preprocess_config_exists_and_matches_python():
    p = ROOT / "models" / "preprocess_config.json"
    assert p.is_file(), "run ml/preprocess.py"
    cfg = json.loads(p.read_text())
    c = cfg["constants"]
    assert c["CLIP_MAX"] == pp.CLIP_MAX
    assert c["SCALE"] == pp.SCALE
    assert c["INV_SCALE"] == pp.INV_SCALE
    assert c["NOISE_MULTIPLIER"] == pp.NOISE_MULTIPLIER
    assert c["MIN_THRESHOLD"] == pp.MIN_THRESHOLD
    assert c["CALIBRATION_FRAMES"] == pp.CALIBRATION_FRAMES
    g = cfg["geometry"]
    assert g["rows"] == 24 and g["cols"] == 21 and g["n_features"] == 504
    assert "row * 21 + col" in g["feature_order"]
    assert cfg["no_per_frame_normalisation"] is True


def test_cpp_constants_match_python():
    h = (ESP32 / "preprocess.h").read_text(encoding="utf-8")
    defines = dict(re.findall(r"#define\s+(SC_\w+)\s+([-\w.]+)", h))
    assert int(defines["SC_ROWS"]) == 24
    assert int(defines["SC_COLS"]) == 21
    # SC_N_FEATURES is defined as an expression, so check it textually
    assert "#define SC_N_FEATURES  (SC_ROWS * SC_COLS)" in h
    assert int(defines["SC_CLIP_MAX"]) == pp.CLIP_MAX
    assert float(defines["SC_SCALE"].rstrip("f")) == pp.SCALE
    assert int(defines["SC_CAL_FRAMES"]) == pp.CALIBRATION_FRAMES
    assert int(defines["SC_NOISE_MULTIPLIER"]) == pp.NOISE_MULTIPLIER
    assert int(defines["SC_MIN_THRESHOLD"]) == pp.MIN_THRESHOLD
    assert float(defines["SC_CAL_LOAD_DELTA"].rstrip("f")) == pp.CAL_LOAD_DELTA
    assert float(defines["SC_CAL_MAX_MEAN_ABS_DEV"].rstrip("f")) == pp.CAL_MAX_MEAN_ABS_DEV
    assert float(defines["SC_CAL_MAX_NOISE_STD"].rstrip("f")) == pp.CAL_MAX_NOISE_STD
    assert int(defines["SC_CAL_MAX_LOADED_CELLS"]) == pp.CAL_MAX_LOADED_CELLS


def test_original_firmware_backup_is_unmodified():
    """The backup must still hash to what Phase 1 recorded."""
    import hashlib
    expect = "c669202b66a7f8bad5d502b2a703e112f578e5112fe354819732139a9a92f5fd"
    got = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    assert got == expect, "the firmware backup was modified"
