"""Model output shape, probability normalisation, INT8 compatibility,
and the generated C++ kernel."""
import json
from pathlib import Path

import numpy as np
import pytest

from common import CLASS_NAMES, MODELS_DIR, N_SENSORS
from preprocess import pressure_to_features

ESP32 = Path(__file__).resolve().parent.parent / "esp32"


def _need(path, what):
    if not Path(path).is_file():
        pytest.skip("%s missing; run %s" % (Path(path).name, what))


@pytest.fixture(scope="module")
def meta():
    p = MODELS_DIR / "model_metadata.json"
    _need(p, "ml/export_int8.py")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def tflite_bytes():
    p = MODELS_DIR / "model_int8.tflite"
    _need(p, "ml/export_int8.py")
    return p.read_bytes()


@pytest.fixture(scope="module")
def sample_features(dataset):
    X = pressure_to_features(dataset["X_pressure"])
    rng = np.random.default_rng(4)
    return X[rng.choice(X.shape[0], 40, replace=False)]


# ----------------------------------------------------------------- float model
def test_float_model_output_shape_and_probabilities(sample_features):
    p = MODELS_DIR / "model_float.keras"
    _need(p, "ml/export_int8.py")
    from tensorflow import keras
    m = keras.models.load_model(p)
    out = m.predict(sample_features, verbose=0)
    assert out.shape == (sample_features.shape[0], len(CLASS_NAMES))
    assert np.isfinite(out).all()

    # the exported graph ends in logits; softmax must normalise them
    z = out - out.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_model_input_dimension_is_504(meta):
    assert meta["class_names"] == CLASS_NAMES
    from tensorflow import keras
    m = keras.models.load_model(MODELS_DIR / "model_float.keras")
    assert m.input_shape[-1] == N_SENSORS == 504


# ------------------------------------------------------------------ INT8 model
def test_int8_model_is_fully_quantised(tflite_bytes):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert inp["dtype"] == np.int8, "input must be int8"
    assert out["dtype"] == np.int8, "output must be int8"
    s_in, z_in = inp["quantization"]
    assert s_in > 0, "input must carry a real quantisation scale"
    # no float kernels left in the graph
    for op in interp._get_ops_details():
        assert op["op_name"] in ("FULLY_CONNECTED", "QUANTIZE", "DEQUANTIZE",
                                 "RESHAPE"), op["op_name"]


def test_int8_output_shape_and_softmax(tflite_bytes, sample_features):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    for x in sample_features[:10]:
        q = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        logits = (interp.get_tensor(out["index"])[0].astype(np.float64) - z_out) * s_out
        assert logits.shape == (len(CLASS_NAMES),)
        z = logits - logits.max()
        e = np.exp(z)
        probs = e / e.sum()
        assert abs(probs.sum() - 1.0) < 1e-9
        assert 0 <= probs.min() and probs.max() <= 1


def test_int8_agrees_with_float_on_top1(meta):
    assert meta["int8_vs_float_top1_agreement"] >= 0.99, (
        "INT8 must agree with float on at least 99%% of frames, got %.4f"
        % meta["int8_vs_float_top1_agreement"])


# ------------------------------------------------------------- generated C++
def test_model_data_header_exists_and_is_consistent(meta):
    h = ESP32 / "model_data.h"
    _need(h, "ml/export_int8.py")
    text = h.read_text()
    assert "#define SC_N_CLASSES %d" % len(CLASS_NAMES) in text
    assert "#define SC_MODEL_INPUT_DIM %d" % N_SENSORS in text
    for c in CLASS_NAMES:
        assert '"%s"' % c in text
    assert "static const int8_t sc_l0_w" in text
    # every class name appears in the declared order
    pos = [text.index('"%s"' % c) for c in CLASS_NAMES]
    assert pos == sorted(pos), "class names must be emitted in label order"


def test_generated_cpp_matches_tflite(tflite_bytes, sample_features, meta):
    """Identical argmax on all frames; logits within +/-1 quantisation step."""
    import tensorflow as tf
    from cpp_bridge import Harness, build
    try:
        build(with_inference=True)
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip("C++ harness unavailable: %s" % exc)

    X = sample_features
    if meta.get("mask_col20"):
        from evaluate import COL20_IDX
        X = X.copy()
        X[:, COL20_IDX] = 0.0

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]

    with Harness(with_inference=True) as h:
        h.set_cal(np.zeros(N_SENSORS, dtype=np.int32),
                  np.zeros(N_SENSORS, dtype=np.int32))
        for x in X:
            q = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q.reshape(1, -1))
            interp.invoke()
            ref = interp.get_tensor(out["index"])[0].astype(np.int32)
            got = h.infer_features(x)
            assert got["argmax"] == int(ref.argmax()), "argmax mismatch"
            assert np.abs(got["logits"] - ref).max() <= 1, "logits beyond 1 step"
            assert abs(float(got["prob"].sum()) - 1.0) < 1e-4
            assert 0.0 <= got["confidence"] <= 1.0


def test_cpp_probabilities_sum_to_one_on_extremes():
    from cpp_bridge import Harness, build
    try:
        build(with_inference=True)
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip("C++ harness unavailable: %s" % exc)
    with Harness(with_inference=True) as h:
        h.set_cal(np.zeros(N_SENSORS, dtype=np.int32),
                  np.zeros(N_SENSORS, dtype=np.int32))
        for feats in (np.zeros(N_SENSORS, dtype=np.float32),
                      np.full(N_SENSORS, 4095 / 1024.0, dtype=np.float32)):
            r = h.infer_features(feats)
            assert abs(float(r["prob"].sum()) - 1.0) < 1e-4
            assert 0 <= r["argmax"] < len(CLASS_NAMES)


def test_decision_file_records_the_synthetic_rule():
    p = MODELS_DIR / "phase4_decision.json"
    _need(p, "ml/decide_model.py")
    d = json.loads(p.read_text())
    rule_keeps = d["nested_mean_real_plus_synthetic"] >= d["nested_mean_real_only"]
    assert bool(d["use_synthetic"]) == rule_keeps, (
        "shipped model must follow the stated decision rule")
    assert d["chosen_heldout_mean_acc"] <= 1.0
    assert "caveat" in d and "four real participants" in d["caveat"].lower()
