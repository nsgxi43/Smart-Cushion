"""Python and C++ preprocessing must agree, and calibration must refuse bad input."""
import numpy as np
import pytest

import preprocess as pp

RNG = np.random.default_rng(7)
N = pp.N_FEATURES


def _cal(baseline=2500, thr=20):
    return (np.full(N, baseline, dtype=np.int32),
            np.full(N, thr, dtype=np.int32))


# ---------------------------------------------------------------- python side
def test_feature_vector_shape_and_dtype():
    b, t = _cal()
    x = pp.preprocess_frame(RNG.integers(0, 4096, N), b, t)
    assert x.shape == (N,)
    assert x.dtype == np.float32


def test_wrong_length_rejected():
    b, t = _cal()
    with pytest.raises(ValueError):
        pp.preprocess_frame(np.zeros(503, dtype=np.int32), b, t)


def test_noise_gate_boundary_is_inclusive():
    """delta == threshold is gated; delta == threshold + 1 survives."""
    b, t = _cal(2500, 20)
    assert pp.preprocess_frame(np.full(N, 2520), b, t).max() == 0
    out = pp.preprocess_frame(np.full(N, 2521), b, t)
    assert np.allclose(out, 21 / 1024.0)


def test_absolute_difference_is_symmetric():
    b, t = _cal(2500, 20)
    up = pp.preprocess_frame(np.full(N, 2500 + 300), b, t)
    down = pp.preprocess_frame(np.full(N, 2500 - 300), b, t)
    assert np.array_equal(up, down)


def test_clipping_at_clip_max():
    b, t = _cal(0, 20)
    out = pp.preprocess_frame(np.full(N, 4095), b, t)
    assert np.allclose(out, pp.CLIP_MAX / 1024.0)


def test_scale_is_an_exact_power_of_two():
    assert pp.SCALE == 1024.0
    assert pp.INV_SCALE == 1.0 / 1024.0
    # exact in binary floating point: multiply then divide round-trips
    v = np.arange(0, 4096, dtype=np.float32) * np.float32(pp.INV_SCALE)
    assert np.array_equal(v * np.float32(1024.0), np.arange(0, 4096, dtype=np.float32))


def test_no_per_frame_normalisation():
    """Doubling the pressure must double the features, not leave them unchanged."""
    b, t = _cal(0, 20)
    a = pp.preprocess_frame(np.full(N, 100), b, t)
    d = pp.preprocess_frame(np.full(N, 200), b, t)
    assert np.allclose(d, 2 * a)


def test_pressure_to_features_matches_full_pipeline():
    b, t = _cal(2500, 20)
    raw = RNG.integers(0, 4096, N).astype(np.int32)
    delta = np.abs(raw.astype(np.int64) - b)
    press = np.clip(np.where(delta <= t, 0, delta), 0, pp.CLIP_MAX).astype(np.int32)
    assert np.array_equal(pp.pressure_to_features(press),
                          pp.preprocess_frame(raw, b, t))


def test_batch_matches_single():
    b, t = _cal()
    frames = RNG.integers(0, 4096, (17, N)).astype(np.int32)
    batch = pp.preprocess_batch(frames, b, t)
    for i in range(frames.shape[0]):
        assert np.array_equal(batch[i], pp.preprocess_frame(frames[i], b, t))


# -------------------------------------------------------------- calibration
def test_calibration_accepts_a_clean_unloaded_window():
    frames = np.rint(RNG.normal(2500, 6, (25, N))).astype(np.int32)
    ok, reason, _ = pp.validate_calibration(frames)
    assert ok and reason == "OK"
    b, t = pp.compute_calibration(frames)
    assert b.shape == (N,) and t.shape == (N,)
    assert (t >= pp.MIN_THRESHOLD).all()


def test_calibration_rejects_pressure_on_the_cushion():
    frames = np.rint(RNG.normal(2500, 6, (25, N))).astype(np.int32)
    frames[:, 200:280] += 900
    ok, reason, _ = pp.validate_calibration(frames)
    assert not ok and reason == "PRESSURE_DETECTED"


def test_calibration_rejects_an_unstable_window():
    frames = np.rint(RNG.normal(2500, 6, (25, N))
                     + np.linspace(0, 400, 25).reshape(-1, 1)).astype(np.int32)
    ok, reason, _ = pp.validate_calibration(frames)
    assert not ok and reason in ("UNSTABLE_BASELINE", "PRESSURE_DETECTED")


def test_calibration_rejects_too_few_frames():
    frames = np.rint(RNG.normal(2500, 6, (10, N))).astype(np.int32)
    ok, reason, _ = pp.validate_calibration(frames)
    assert not ok and reason == "TOO_FEW_FRAMES"
    with pytest.raises(pp.CalibrationError):
        pp.compute_calibration(frames)


def test_calibration_rejects_a_dead_array():
    frames = np.zeros((25, N), dtype=np.int32)
    ok, reason, _ = pp.validate_calibration(frames)
    assert not ok and reason == "NO_SIGNAL"


def test_calibration_threshold_formula():
    frames = np.rint(RNG.normal(2500, 40, (40, N))).astype(np.int32)
    b, t = pp.compute_calibration(frames)
    expect = np.rint(np.maximum(pp.NOISE_MULTIPLIER * frames.astype(np.float64).std(axis=0),
                                float(pp.MIN_THRESHOLD)))
    assert np.array_equal(t, expect.astype(np.int32))


# ------------------------------------------------------------------ C++ side
def test_cpp_matches_python_on_random_frames(harness):
    b, t = _cal(2500, 20)
    harness.set_cal(b, t)
    for _ in range(25):
        raw = RNG.integers(0, 4096, N).astype(np.int32)
        py = pp.preprocess_frame(raw, b, t)
        cpp = harness.preprocess(raw)
        assert np.array_equal(py.view(np.int32), cpp.view(np.int32))


def test_cpp_matches_python_on_recorded_frames(harness, dataset):
    b, t = _cal(2500, 20)
    harness.set_cal(b, t)
    X = dataset["X_raw"]
    idx = RNG.choice(X.shape[0], size=40, replace=False)
    for i in idx:
        raw = X[i].astype(np.int32)
        assert np.array_equal(pp.preprocess_frame(raw, b, t).view(np.int32),
                              harness.preprocess(raw).view(np.int32))


def test_cpp_matches_python_per_stage(harness):
    b, t = _cal(2500, 25)
    harness.set_cal(b, t)
    raw = RNG.integers(0, 4096, N).astype(np.int32)
    delta = np.abs(raw.astype(np.int64) - b)
    press = np.clip(np.where(delta <= t, 0, delta), 0, pp.CLIP_MAX).astype(np.int32)
    assert np.array_equal(press, harness.raw_to_pressure(raw))
    assert np.array_equal(pp.pressure_to_features(press).view(np.int32),
                          harness.pressure_to_features(press).view(np.int32))


def test_cpp_calibration_agrees_with_python(harness):
    cases = {
        "clean": np.rint(RNG.normal(2500, 6, (25, N))).astype(np.int32),
        "loaded": None,
        "few": np.rint(RNG.normal(2500, 6, (8, N))).astype(np.int32),
    }
    loaded = np.rint(RNG.normal(2500, 6, (25, N))).astype(np.int32)
    loaded[:, 100:180] += 800
    cases["loaded"] = loaded

    for name, frames in cases.items():
        ok, reason, _ = pp.validate_calibration(frames)
        harness.cal_begin()
        for f in frames:
            harness.cal_add(f)
        cres, cstate = harness.cal_finish()
        assert cres == ("OK" if ok else reason), (name, cres, reason)
        if ok:
            pb, pt = pp.compute_calibration(frames)
            cb, ct = harness.get_cal()
            assert np.array_equal(pb, cb) and np.array_equal(pt, ct)
            assert cstate == "READY"
        else:
            assert cstate == "CALIBRATION_FAILED"
