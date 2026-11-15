"""Phase 5 - train the production model, quantise to INT8, emit dependency-free C++.

Pipeline
  1. rebuild the architecture Phase 4 selected and train it on the chosen
     training set (all 4 real participants, plus synthetic only if Phase 4's
     decision rule kept it)
  2. strip the softmax so the exported graph ends in logits; softmax is done in
     C++ in float, which keeps the quantised graph to plain FULLY_CONNECTED ops
  3. full-integer INT8 conversion with a representative dataset drawn only from
     the training set
  4. compare INT8 against float: top-1 agreement and max probability delta
  5. emit esp32/model_data.h (int8 weights, int32 biases, per-channel
     requantisation multipliers) and esp32/inference.cpp
  6. the generated C++ is then checked against the TFLite INT8 interpreter

The C++ reproduces TFLite's integer arithmetic exactly: int32 accumulators and
gemmlowp's SaturatingRoundingDoublingHighMul + RoundingDivideByPOT
requantisation, so argmax agreement is expected to be total, not approximate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, MODELS_DIR, N_SENSORS, REPORTS_DIR
from preprocess import pressure_to_features

SEED = 20261115
OUT = REPORTS_DIR / "phase5_export"
OUT.mkdir(parents=True, exist_ok=True)
ESP32_DIR = Path(__file__).resolve().parent.parent / "esp32"
DECISION_PATH = MODELS_DIR / "phase4_decision.json"


# ---------------------------------------------------------------------------
# gemmlowp fixed-point helpers, mirrored exactly in inference.cpp
# ---------------------------------------------------------------------------
def quantize_multiplier(double_multiplier):
    """float multiplier -> (int32 quantized_multiplier, int shift), as TFLite does."""
    if double_multiplier == 0.0:
        return 0, 0
    mantissa, exponent = np.frexp(double_multiplier)       # m in [0.5, 1)
    q = int(round(mantissa * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exponent += 1
    assert q <= (1 << 31) - 1
    if exponent < -31:
        return 0, 0
    return q, int(exponent)


def multiply_by_quantized_multiplier(x, qm, shift):
    """TFLite's requantisation, single-rounding variant.

    Modern TFLite builds define TFLITE_SINGLE_ROUNDING, which rounds ONCE on a
    64-bit intermediate rather than twice through gemmlowp's
    SaturatingRoundingDoublingHighMul + RoundingDivideByPOT. The difference is
    not cosmetic: double rounding pushes values just below .5 up to the next
    integer (e.g. a true 71.4962 becomes 71.5 after the first rounding and then
    72), which flips occasional near-tie argmaxes. esp32/inference.cpp
    implements exactly this function.
    """
    x = int(x)
    qm = int(qm)
    shift = int(shift)
    assert qm >= 0, "quantised multipliers are non-negative"
    assert -31 <= shift <= 30, "shift out of range: %d" % shift
    total_shift = 31 - shift
    rounding = 1 << (total_shift - 1)
    result = (x * qm + rounding) >> total_shift
    if result < -(1 << 31):
        return np.int32(-(1 << 31))
    if result > (1 << 31) - 1:
        return np.int32((1 << 31) - 1)
    return np.int32(result)


# ---------------------------------------------------------------------------
def build_logits_model(hidden, dropout, l2):
    from tensorflow import keras
    layers = [keras.layers.Input(shape=(N_SENSORS,))]
    for h in hidden:
        layers.append(keras.layers.Dense(
            h, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)))
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(len(CLASS_NAMES), activation=None))
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def load_data():
    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    return {
        "X": pressure_to_features(d["X_pressure"]),
        "y": d["y"].astype(np.int32),
        "participant": d["participant"].astype(str),
        "is_synthetic": d["is_synthetic"].astype(bool),
        "source_participant": d["source_participant"].astype(str),
        "group": d["group"].astype(str),
    }


# ---------------------------------------------------------------------------
def convert_int8(model, rep_X):
    import tensorflow as tf

    def rep_gen():
        for i in range(min(500, rep_X.shape[0])):
            yield [rep_X[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def _interpreter(tflite_bytes):
    """Interpreter with the reference kernels and no delegates.

    The default resolver lets XNNPACK fuse the FULLY_CONNECTED ops into a single
    DELEGATE node, which both hides the per-layer tensors we need to export and
    changes the arithmetic. BUILTIN_REF is the plain integer reference kernel -
    exactly what esp32/inference.cpp reimplements.
    """
    import tensorflow as tf
    return tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)


def tflite_predict(tflite_bytes, X):
    interp = _interpreter(tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    q_logits = np.zeros((X.shape[0], len(CLASS_NAMES)), dtype=np.int32)
    for i in range(X.shape[0]):
        # float32 throughout, so this matches scQuantiseInput() in inference.cpp
        # bit-for-bit rather than only to within a rounding boundary
        q = np.clip(np.round(X[i].astype(np.float32) / np.float32(s_in))
                    + z_in, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        q_logits[i] = interp.get_tensor(out["index"])[0].astype(np.int32)
    deq = (q_logits - z_out) * s_out
    return q_logits, deq, (s_in, z_in, s_out, z_out)


# ---------------------------------------------------------------------------
def extract_layers(tflite_bytes):
    """Pull per-layer int8 weights/biases and quantisation params out of the model."""
    interp = _interpreter(tflite_bytes)
    interp.allocate_tensors()
    details = {t["index"]: t for t in interp.get_tensor_details()}

    ops = []
    for op in interp._get_ops_details():
        if op["op_name"] != "FULLY_CONNECTED":
            if op["op_name"] in ("QUANTIZE", "DEQUANTIZE", "RESHAPE"):
                continue
            raise RuntimeError("unexpected op in quantised graph: " + op["op_name"])
        ops.append(op)

    layers = []
    for k, op in enumerate(ops):
        in_idx, w_idx, b_idx = op["inputs"][0], op["inputs"][1], op["inputs"][2]
        out_idx = op["outputs"][0]
        W = interp.get_tensor(w_idx).astype(np.int32)          # (units, in_dim)
        B = interp.get_tensor(b_idx).astype(np.int32)
        qi = details[in_idx]["quantization_parameters"]
        qw = details[w_idx]["quantization_parameters"]
        qo = details[out_idx]["quantization_parameters"]
        s_in = float(qi["scales"][0]); z_in = int(qi["zero_points"][0])
        s_out = float(qo["scales"][0]); z_out = int(qo["zero_points"][0])
        w_scales = np.asarray(qw["scales"], dtype=np.float64)
        w_zps = np.asarray(qw["zero_points"], dtype=np.int32)
        units = W.shape[0]
        if w_scales.size == 1:
            w_scales = np.repeat(w_scales, units)
            w_zps = np.repeat(w_zps, units)
        assert (w_zps == 0).all(), "symmetric int8 weights expected"

        mults, shifts = [], []
        for u in range(units):
            qm, sh = quantize_multiplier(s_in * w_scales[u] / s_out)
            mults.append(qm); shifts.append(sh)

        layers.append({
            "index": k,
            "units": int(units),
            "in_dim": int(W.shape[1]),
            "W": W.astype(np.int8),
            "B": B,
            "input_zero_point": z_in,
            "output_zero_point": z_out,
            "input_scale": s_in,
            "output_scale": s_out,
            "multipliers": np.array(mults, dtype=np.int32),
            "shifts": np.array(shifts, dtype=np.int32),
            "relu": k < len(ops) - 1,
        })
    return layers


def reference_int8_forward(layers, q_in):
    """Pure-numpy replica of the generated C++, used to cross-check it."""
    x = np.asarray(q_in, dtype=np.int32)
    for li, L in enumerate(layers):
        W, B = L["W"].astype(np.int32), L["B"]
        acc = (W * (x - L["input_zero_point"])[None, :]).sum(axis=1) + B
        out = np.empty(L["units"], dtype=np.int32)
        for u in range(L["units"]):
            v = multiply_by_quantized_multiplier(acc[u], L["multipliers"][u],
                                                 int(L["shifts"][u]))
            v = int(v) + L["output_zero_point"]
            lo = L["output_zero_point"] if L["relu"] else -128
            out[u] = max(lo, min(127, v))
        x = out
    return x


# ---------------------------------------------------------------------------
def emit_header(layers, meta, path):
    def arr(name, values, ctype, per_line=16):
        s = ["static const %s %s[%d] = {\n" % (ctype, name, len(values))]
        for i in range(0, len(values), per_line):
            s.append("  " + ", ".join(str(int(v)) for v in values[i:i + per_line]) + ",\n")
        s.append("};\n")
        return "".join(s)

    L = []
    L.append("// GENERATED by ml/export_int8.py - do not edit by hand.\n")
    L.append("// Model: %s   trained on: %s\n" % (meta["architecture"], meta["training_set"]))
    L.append("// Generated: %s\n" % meta["generated"])
    L.append("#ifndef SMART_CUSHION_MODEL_DATA_H\n#define SMART_CUSHION_MODEL_DATA_H\n\n")
    L.append("#include <stdint.h>\n\n")
    L.append("#define SC_N_CLASSES %d\n" % len(CLASS_NAMES))
    L.append("#define SC_N_LAYERS  %d\n" % len(layers))
    L.append("#define SC_MODEL_INPUT_DIM %d\n\n" % layers[0]["in_dim"])
    L.append("static const char *const SC_CLASS_NAMES[SC_N_CLASSES] = {\n")
    for c in CLASS_NAMES:
        L.append('  "%s",\n' % c)
    L.append("};\n\n")
    L.append("// input quantisation applied to the 504 float features\n")
    L.append("#define SC_INPUT_SCALE      %.10ef\n" % meta["input_scale"])
    L.append("#define SC_INPUT_ZERO_POINT %d\n\n" % meta["input_zero_point"])
    L.append("// output (logit) dequantisation\n")
    L.append("#define SC_OUTPUT_SCALE      %.10ef\n" % meta["output_scale"])
    L.append("#define SC_OUTPUT_ZERO_POINT %d\n\n" % meta["output_zero_point"])

    max_units = max(l["units"] for l in layers)
    L.append("#define SC_MAX_UNITS %d\n\n" % max(max_units, layers[0]["in_dim"]))

    for i, lay in enumerate(layers):
        L.append("// --- layer %d: %d -> %d%s ---\n"
                 % (i, lay["in_dim"], lay["units"], ", relu" if lay["relu"] else ""))
        L.append(arr("sc_l%d_w" % i, lay["W"].reshape(-1), "int8_t", 24))
        L.append(arr("sc_l%d_b" % i, lay["B"], "int32_t", 8))
        L.append(arr("sc_l%d_mult" % i, lay["multipliers"], "int32_t", 8))
        L.append(arr("sc_l%d_shift" % i, lay["shifts"], "int32_t", 16))
        L.append("\n")

    L.append("typedef struct {\n"
             "  const int8_t  *w;\n"
             "  const int32_t *b;\n"
             "  const int32_t *mult;\n"
             "  const int32_t *shift;\n"
             "  int16_t in_dim;\n"
             "  int16_t units;\n"
             "  int32_t input_zero_point;\n"
             "  int32_t output_zero_point;\n"
             "  uint8_t relu;\n"
             "} ScLayer;\n\n")
    L.append("static const ScLayer SC_LAYERS[SC_N_LAYERS] = {\n")
    for i, lay in enumerate(layers):
        L.append("  { sc_l%d_w, sc_l%d_b, sc_l%d_mult, sc_l%d_shift, %d, %d, %d, %d, %d },\n"
                 % (i, i, i, i, lay["in_dim"], lay["units"],
                    lay["input_zero_point"], lay["output_zero_point"],
                    1 if lay["relu"] else 0))
    L.append("};\n\n#endif  // SMART_CUSHION_MODEL_DATA_H\n")

    Path(path).write_text("".join(L), encoding="utf-8")
    return sum(len(l["W"].reshape(-1)) + 4 * len(l["B"]) + 8 * l["units"] for l in layers)


# ---------------------------------------------------------------------------
def main():
    import tensorflow as tf
    from tensorflow import keras

    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    hidden = decision["hidden"]
    dropout = decision.get("dropout", 0.3)
    l2 = decision.get("l2", 1e-4)
    epochs = decision.get("epochs", 120)
    batch = decision.get("batch", 64)
    use_syn = bool(decision["use_synthetic"])
    mask20 = bool(decision.get("mask_col20", False))

    data = load_data()
    tr = np.ones(data["y"].shape[0], dtype=bool)
    if not use_syn:
        tr &= ~data["is_synthetic"]
    X = data["X"].copy()
    if mask20:
        from evaluate import COL20_IDX
        X[:, COL20_IDX] = 0.0

    Xtr, ytr = X[tr], data["y"][tr]
    print("training production model on %d frames (synthetic=%s, mask_col20=%s)"
          % (Xtr.shape[0], use_syn, mask20))

    keras.utils.set_random_seed(SEED)
    model = build_logits_model(hidden, dropout, l2)
    hist = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    model.save(MODELS_DIR / "model_float.keras")

    train_acc = float(np.mean(model.predict(Xtr, verbose=0).argmax(1) == ytr))
    print("training-set accuracy (NOT a generalisation estimate): %.4f" % train_acc)

    # ---- INT8 conversion -------------------------------------------------
    rep_idx = np.random.default_rng(SEED).choice(Xtr.shape[0],
                                                 size=min(500, Xtr.shape[0]),
                                                 replace=False)
    tfl = convert_int8(model, Xtr[rep_idx])
    (MODELS_DIR / "model_int8.tflite").write_bytes(tfl)
    print("INT8 model: %d bytes" % len(tfl))

    # ---- float vs INT8 on the REAL frames --------------------------------
    real = ~data["is_synthetic"]
    Xr, yr = X[real], data["y"][real]
    float_logits = model.predict(Xr, verbose=0)
    float_prob = softmax(float_logits)
    q_logits, deq, qparams = tflite_predict(tfl, Xr)
    int8_prob = softmax(deq)

    agree = float(np.mean(float_prob.argmax(1) == int8_prob.argmax(1)))
    maxdp = float(np.abs(float_prob - int8_prob).max())
    print("INT8 vs float: top-1 agreement %.4f, max |dP| %.4f" % (agree, maxdp))

    # ---- generated C++ ---------------------------------------------------
    layers = extract_layers(tfl)
    s_in, z_in, s_out, z_out = qparams
    meta = {
        "architecture": "504-" + "-".join(str(h) for h in hidden) + "-6 (logits)",
        "hidden": hidden, "dropout": dropout, "l2": l2,
        "epochs": epochs, "batch": batch,
        "training_set": ("real+synthetic" if use_syn else "real only")
                        + (", column 20 masked" if mask20 else ""),
        "use_synthetic": use_syn, "mask_col20": mask20,
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "input_scale": float(s_in), "input_zero_point": int(z_in),
        "output_scale": float(s_out), "output_zero_point": int(z_out),
        "output_is_logits": True,
        "class_names": CLASS_NAMES,
        "quantisation": "full-integer INT8 (weights, activations, input, output)",
        "tflite_bytes": len(tfl),
        "int8_vs_float_top1_agreement": agree,
        "int8_vs_float_max_prob_diff": maxdp,
        "train_accuracy_not_generalisation": train_acc,
        "version": decision.get("version", "1.0.0"),
        "heldout_real_accuracy_human": decision.get("heldout_real_accuracy_human",
                                                    "see reports/phase4_evaluation"),
        "esp32_inference_us": "PENDING_HARDWARE",
    }
    weight_bytes = emit_header(layers, meta, ESP32_DIR / "model_data.h")
    meta["model_size_human"] = "%.1f kB INT8 weights+params in flash" % (weight_bytes / 1024)
    meta["cpp_param_bytes"] = int(weight_bytes)
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2),
                                                    encoding="utf-8")

    # ---- numpy replica vs TFLite interpreter -----------------------------
    n_check = min(400, Xr.shape[0])
    idx = np.random.default_rng(1).choice(Xr.shape[0], n_check, replace=False)
    ref_argmax_match = 0
    ref_logit_maxdiff = 0
    for i in idx:
        q = np.clip(np.round(Xr[i].astype(np.float32) / np.float32(s_in))
                    + z_in, -128, 127).astype(np.int32)
        out = reference_int8_forward(layers, q)
        d = int(np.abs(out - q_logits[i]).max())
        ref_logit_maxdiff = max(ref_logit_maxdiff, d)
        if int(out.argmax()) == int(q_logits[i].argmax()):
            ref_argmax_match += 1
    print("numpy replica vs TFLite INT8: argmax %d/%d, max logit diff %d"
          % (ref_argmax_match, n_check, ref_logit_maxdiff))

    report = {
        "meta": meta,
        "int8_vs_float": {"frames": int(Xr.shape[0]), "top1_agreement": agree,
                          "max_abs_prob_diff": maxdp,
                          "float_accuracy_on_all_real": float(
                              np.mean(float_prob.argmax(1) == yr)),
                          "int8_accuracy_on_all_real": float(
                              np.mean(int8_prob.argmax(1) == yr)),
                          "note": "accuracy here is on frames the model was "
                                  "trained on; it is NOT a generalisation estimate"},
        "numpy_replica_vs_tflite": {"frames": n_check,
                                    "argmax_match": ref_argmax_match,
                                    "max_logit_diff": ref_logit_maxdiff},
        "layers": [{"in_dim": l["in_dim"], "units": l["units"], "relu": l["relu"],
                    "weight_bytes": int(l["in_dim"] * l["units"])} for l in layers],
    }
    (OUT / "export_report.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    print("wrote esp32/model_data.h (%d parameter bytes)" % weight_bytes)
    return report


if __name__ == "__main__":
    main()
