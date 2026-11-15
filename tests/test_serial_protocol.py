"""Serial protocol parsing and the simulated source."""
import time

import numpy as np
import pytest

from serial_source import (N_SENSORS, SimulatedSource, parse_line)


def frame_line(values):
    return "FRAME," + ",".join(str(int(v)) for v in values)


def test_parse_valid_frame():
    vals = np.arange(N_SENSORS) % 4096
    ev = parse_line(frame_line(vals))
    assert ev["kind"] == "frame"
    assert ev["raw"].shape == (N_SENSORS,)
    assert np.array_equal(ev["raw"], vals)


def test_parse_frame_wrong_length_is_rejected():
    ev = parse_line("FRAME," + ",".join(["1"] * 503))
    assert ev["kind"] == "info" and "503" in ev["text"]


def test_parse_frame_non_integer_is_rejected():
    vals = ["1"] * N_SENSORS
    vals[7] = "12.5"
    ev = parse_line("FRAME," + ",".join(vals))
    assert ev["kind"] == "info" and "non-integer" in ev["text"]


def test_parse_frame_out_of_adc_range_is_rejected():
    vals = [1] * N_SENSORS
    vals[3] = 4096
    ev = parse_line(frame_line(vals))
    assert ev["kind"] == "info" and "range" in ev["text"]


def test_parse_pred():
    ev = parse_line("PRED,C2_Slouching,0.8123,4210")
    assert ev["kind"] == "pred"
    assert ev["label"] == "C2_Slouching"
    assert abs(ev["confidence"] - 0.8123) < 1e-9
    assert ev["inference_us"] == 4210
    assert ev["origin"] == "ESP32"


def test_parse_pred_unknown_state():
    ev = parse_line("PRED,UNKNOWN,0.000,0")
    assert ev["kind"] == "pred" and ev["label"] == "UNKNOWN"
    assert ev["confidence"] == 0.0


def test_parse_pred_malformed():
    assert parse_line("PRED,C0_Normal,0.9")["kind"] == "info"
    assert parse_line("PRED,C0_Normal,abc,10")["kind"] == "info"


@pytest.mark.parametrize("line,kind", [
    ("STATE,READY", "state"),
    ("STATE,CALIBRATION_FAILED", "state"),
    ("CAL,12/20", "cal"),
    ("CAL,PRESSURE_DETECTED", "cal"),
    ("INFO,mode=RAW", "info"),
])
def test_parse_status_lines(line, kind):
    ev = parse_line(line)
    assert ev["kind"] == kind
    assert ev["text"] == line.split(",", 1)[1]


def test_parse_blank_and_banner():
    assert parse_line("") is None
    assert parse_line("   ") is None
    ev = parse_line("ESP32 matrix scanner ready")
    assert ev["kind"] == "info"


def test_roundtrip_frame_formatting():
    """A frame formatted the way the firmware prints it must parse back exactly."""
    rng = np.random.default_rng(3)
    vals = rng.integers(0, 4096, N_SENSORS)
    assert np.array_equal(parse_line(frame_line(vals))["raw"], vals)


def test_simulated_source_emits_frames_and_predictions():
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 4096, (6, N_SENSORS)).astype(np.int32)

    def predictor(raw):
        return "C0_Normal", 0.42

    src = SimulatedSource(frames, labels=["C0_Normal"] * 6, fps=40.0,
                          loop=False, predictor=predictor).start()
    deadline = time.monotonic() + 8
    got = []
    while time.monotonic() < deadline and len(got) < 30:
        got.extend(src.drain())
        time.sleep(0.05)
        if not src.running and src.q.empty():
            break
    src.stop()

    kinds = [e["kind"] for e in got]
    assert "frame" in kinds
    preds = [e for e in got if e["kind"] == "pred"]
    assert preds, "simulator produced no predictions"
    # a simulated prediction is NEVER labelled as coming from the board
    assert all(p["origin"] == "Python" for p in preds)
    assert all(p["origin"] != "ESP32" for p in preds)


def test_simulated_source_announces_itself():
    src = SimulatedSource(np.zeros((2, N_SENSORS), dtype=np.int32),
                          fps=40.0, loop=False).start()
    time.sleep(0.6)
    evs = src.drain()
    src.stop()
    texts = " ".join(e.get("text", "") for e in evs)
    assert "SIMULATED" in texts.upper()


def test_queue_drops_oldest_rather_than_blocking():
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 4096, (50, N_SENSORS)).astype(np.int32)
    src = SimulatedSource(frames, fps=500.0, loop=True, maxsize=8).start()
    time.sleep(1.0)
    src.stop()
    assert src.q.qsize() <= 8, "bounded queue must not grow without limit"
