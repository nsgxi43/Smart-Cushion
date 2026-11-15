#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 - evaluate on the held-out test set, then convert to INT8 and compare.

The test split is touched only here. It took no part in training or in model
selection (the architecture variant and the early-stopping epoch were both
chosen on the validation split).

Produces:
    evaluation/classification_report.txt
    evaluation/confusion_matrix.png
    evaluation/evaluation_summary.json
    models/best_model.tflite          (fully INT8-quantized)

Usage:
    python training/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import CLASS_NAMES, NUM_CLASSES, ROOT, set_seeds

PROCESSED = os.path.join(ROOT, "processed_dataset")
MODELS = os.path.join(ROOT, "models")
EVAL = os.path.join(ROOT, "evaluation")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

FOCUS_PAIRS = [("Normal", "Slouch"), ("Forward", "Backward"),
               ("Left", "Right"), ("Normal", "Unknown")]


def load_splits():
    X = np.load(os.path.join(PROCESSED, "X.npy"))
    y = np.load(os.path.join(PROCESSED, "y.npy"))
    s = np.load(os.path.join(PROCESSED, "split.npy"))
    return {n: (X[s == n], y[s == n]) for n in ("train", "val", "test")}


# --------------------------------------------------------------------------
# INT8 conversion
# --------------------------------------------------------------------------


def convert_int8(keras_path: str, Xtrain: np.ndarray, out_path: str) -> dict:
    """Full-integer quantization. Representative data comes from TRAIN only."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)

    def representative_dataset():
        idx = np.arange(len(Xtrain))
        rng = np.random.default_rng(1337)
        rng.shuffle(idx)
        for i in idx[:min(200, len(idx))]:
            yield [Xtrain[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()

    with open(out_path, "wb") as fh:
        fh.write(tflite)

    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    info = dict(
        size_bytes=len(tflite),
        input_dtype=str(np.dtype(inp["dtype"])),
        input_shape=[int(v) for v in inp["shape"]],
        input_scale=float(inp["quantization"][0]),
        input_zero_point=int(inp["quantization"][1]),
        output_dtype=str(np.dtype(out["dtype"])),
        output_shape=[int(v) for v in out["shape"]],
        output_scale=float(out["quantization"][0]),
        output_zero_point=int(out["quantization"][1]),
    )

    # confirm nothing fell back to float
    float_tensors = []
    for d in interp.get_tensor_details():
        dt = np.dtype(d["dtype"])
        if dt in (np.float32, np.float64) and d["name"]:
            float_tensors.append(d["name"])
    info["float_tensors_remaining"] = float_tensors
    info["ops"] = sorted({d["op_name"] for d in
                          getattr(interp, "_get_ops_details", lambda: [])()}) or None
    return info


def tflite_predict(tflite_path: str, X: np.ndarray) -> np.ndarray:
    """Run the INT8 model, quantizing input and dequantizing output."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]

    probs = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
    for i in range(len(X)):
        q = np.round(X[i:i + 1] / s_in + z_in)
        q = np.clip(q, -128, 127).astype(inp["dtype"])
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out["index"]).astype(np.float32)
        probs[i] = (raw - z_out) * s_out
    return probs


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def metrics_block(y_true, y_pred, title: str) -> tuple[str, dict]:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    labels = list(range(NUM_CLASSES))
    rep = classification_report(y_true, y_pred, labels=labels,
                                target_names=CLASS_NAMES, digits=4,
                                zero_division=0)
    d = classification_report(y_true, y_pred, labels=labels,
                              target_names=CLASS_NAMES, output_dict=True,
                              zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

    lines = [f"=== {title} ===", "",
             f"accuracy      : {acc:.4f}  ({int((np.array(y_true)==np.array(y_pred)).sum())}/{len(y_true)})",
             f"macro F1      : {macro:.4f}",
             f"weighted F1   : {weighted:.4f}", "",
             rep, "confusion matrix (rows = true, cols = predicted)",
             "            " + " ".join(f"{c[:5]:>6}" for c in CLASS_NAMES)]
    for i, c in enumerate(CLASS_NAMES):
        lines.append(f"  {c:<10}" + " ".join(f"{v:6d}" for v in cm[i]))
    lines.append("")
    lines.append("focus pairs (confusions in both directions):")
    for a, b in FOCUS_PAIRS:
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        lines.append(f"  {a:<9} -> {b:<9} {cm[ia, ib]:3d}      "
                     f"{b:<9} -> {a:<9} {cm[ib, ia]:3d}")
    uk = CLASS_NAMES.index("Unknown")
    lines.append("")
    lines.append(f"Unknown recall   : {d['Unknown']['recall']:.4f}  "
                 f"({cm[uk, uk]}/{cm[uk].sum()})")
    lines.append(f"Unknown precision: {d['Unknown']['precision']:.4f}")
    summary = dict(accuracy=float(acc), macro_f1=float(macro),
                   weighted_f1=float(weighted), per_class=d,
                   confusion_matrix=cm.tolist())
    return "\n".join(lines), summary


def plot_confusion(cms: list[tuple[str, np.ndarray]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cms), figsize=(7.2 * len(cms), 6.4), squeeze=False)
    for ax, (title, cm) in zip(axes[0], cms):
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black",
                            fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.suptitle("Smart Cushion CNN - test-set confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    set_seeds(args.seed)

    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}\n")

    data = load_splits()
    Xtr, _ = data["train"]
    Xte, yte = data["test"]
    keras_path = os.path.join(MODELS, "best_model.keras")
    tflite_path = os.path.join(MODELS, "best_model.tflite")

    model = tf.keras.models.load_model(keras_path)
    print(f"model: {model.count_params():,} parameters")

    # ---- float evaluation ---------------------------------------------------
    test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
    pf = model.predict(Xte, verbose=0)
    yf = pf.argmax(1)
    print(f"float32  test loss {test_loss:.4f}  test accuracy {test_acc:.4f}")

    # ---- INT8 conversion ----------------------------------------------------
    print("\nconverting to fully INT8-quantized TFLite "
          "(representative data from the TRAIN split only) ...")
    info = convert_int8(keras_path, Xtr, tflite_path)
    print(f"  input  : {info['input_dtype']} {info['input_shape']}  "
          f"scale {info['input_scale']:.8f}  zero_point {info['input_zero_point']}")
    print(f"  output : {info['output_dtype']} {info['output_shape']}  "
          f"scale {info['output_scale']:.8f}  zero_point {info['output_zero_point']}")
    print(f"  size   : {info['size_bytes']/1024:.1f} KB")
    if info["float_tensors_remaining"]:
        print(f"  NOTE float tensors remaining: {info['float_tensors_remaining']}")
    else:
        print("  no float tensors remain - inference is fully integer")

    pq = tflite_predict(tflite_path, Xte)
    yq = pq.argmax(1)

    # ---- reports -------------------------------------------------------------
    txt_f, sum_f = metrics_block(yte, yf, "FLOAT32 Keras model - held-out test set")
    txt_q, sum_q = metrics_block(yte, yq, "INT8 TFLite model - held-out test set")

    agree = float((yf == yq).mean())
    keras_kb = os.path.getsize(keras_path) / 1024
    tfl_kb = info["size_bytes"] / 1024
    weights_kb = model.count_params() * 4 / 1024

    cmp_lines = [
        "", "=" * 68, "FLOAT32 vs INT8 COMPARISON", "=" * 68, "",
        f"{'metric':<22}{'float32':>14}{'INT8':>14}{'delta':>14}",
        f"{'accuracy':<22}{sum_f['accuracy']:>14.4f}{sum_q['accuracy']:>14.4f}"
        f"{sum_q['accuracy']-sum_f['accuracy']:>+14.4f}",
        f"{'macro F1':<22}{sum_f['macro_f1']:>14.4f}{sum_q['macro_f1']:>14.4f}"
        f"{sum_q['macro_f1']-sum_f['macro_f1']:>+14.4f}",
        f"{'weighted F1':<22}{sum_f['weighted_f1']:>14.4f}{sum_q['weighted_f1']:>14.4f}"
        f"{sum_q['weighted_f1']-sum_f['weighted_f1']:>+14.4f}",
        f"{'Unknown recall':<22}{sum_f['per_class']['Unknown']['recall']:>14.4f}"
        f"{sum_q['per_class']['Unknown']['recall']:>14.4f}"
        f"{sum_q['per_class']['Unknown']['recall']-sum_f['per_class']['Unknown']['recall']:>+14.4f}",
        "",
        f"prediction agreement : {agree:.4f} "
        f"({int((yf==yq).sum())}/{len(yte)} test images identical)",
        f"mean |prob| difference: {np.abs(pf-pq).mean():.5f}",
        "",
        f"{'per-class F1':<12}{'float32':>10}{'INT8':>10}",
    ]
    for c in CLASS_NAMES:
        cmp_lines.append(f"  {c:<10}{sum_f['per_class'][c]['f1-score']:>10.4f}"
                         f"{sum_q['per_class'][c]['f1-score']:>10.4f}")
    cmp_lines += [
        "",
        f"{'model size':<22}{'':>14}",
        f"  Keras .keras file    {keras_kb:8.1f} KB",
        f"  float32 weights      {weights_kb:8.1f} KB  ({model.count_params():,} parameters)",
        f"  INT8 .tflite file    {tfl_kb:8.1f} KB",
        f"  reduction vs float32 weights: {100*(1-tfl_kb/weights_kb):.1f} %",
        "",
    ]

    report = "\n".join([txt_f, "", "", txt_q, "\n".join(cmp_lines)])
    rp = os.path.join(EVAL, "classification_report.txt")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("\n" + report)
    print(f"saved {rp}")

    plot_confusion(
        [("float32 Keras", np.array(sum_f["confusion_matrix"])),
         ("INT8 TFLite", np.array(sum_q["confusion_matrix"]))],
        os.path.join(EVAL, "confusion_matrix.png"))
    print(f"saved {os.path.join(EVAL, 'confusion_matrix.png')}")

    with open(os.path.join(EVAL, "evaluation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(
            test_size=int(len(yte)),
            test_per_class=np.bincount(yte, minlength=NUM_CLASSES).tolist(),
            float_test_loss=float(test_loss),
            float32=sum_f, int8=sum_q,
            agreement=agree,
            quantization=info,
            sizes_kb=dict(keras_file=keras_kb, float32_weights=weights_kb, tflite_int8=tfl_kb),
        ), fh, indent=2)
    print(f"saved {os.path.join(EVAL, 'evaluation_summary.json')}")


if __name__ == "__main__":
    main()
