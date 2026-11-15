"""Phase 5 step 6 - the generated C++ must match the TFLite INT8 interpreter.

Tolerance (from the Phase 5 spec):
  * identical argmax on ALL test frames
  * logits within +/-1 quantisation step

Also measures the static memory the model and the inference workspace occupy,
and times the C++ kernel on this PC. The PC timing is NOT an ESP32 figure and
is labelled as such; on-device timing stays PENDING_HARDWARE until the board
reports it on a PRED line.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, MODELS_DIR, N_SENSORS, REPORTS_DIR
from cpp_bridge import Harness, build
from export_int8 import (extract_layers, reference_int8_forward, softmax,
                         tflite_predict)
from preprocess import pressure_to_features

OUT = REPORTS_DIR / "phase5_export"
OUT.mkdir(parents=True, exist_ok=True)
ESP32 = Path(__file__).resolve().parent.parent / "esp32"
N_FRAMES = 500


def static_memory_report(meta, layers):
    """Bytes of static storage, derived from the generated header and sources."""
    weight_bytes = sum(l["in_dim"] * l["units"] for l in layers)
    bias_bytes = sum(l["units"] * 4 for l in layers)
    mult_bytes = sum(l["units"] * 4 for l in layers)
    shift_bytes = sum(l["units"] * 4 for l in layers)
    max_units = max([l["units"] for l in layers] + [layers[0]["in_dim"]])

    flash = {
        "int8 weights": weight_bytes,
        "int32 biases": bias_bytes,
        "int32 requant multipliers": mult_bytes,
        "int32 requant shifts": shift_bytes,
        "ScLayer descriptors": 24 * len(layers),
        "class name strings": sum(len(c) + 1 for c in CLASS_NAMES) + 4 * len(CLASS_NAMES),
    }
    ram = {
        "gBufA (int32 x SC_MAX_UNITS)": max_units * 4,
        "gBufB (int32 x SC_MAX_UNITS)": max_units * 4,
        "quantised input (int8 x 504)": N_SENSORS,
        "scBaseline (int32 x 504)": N_SENSORS * 4,
        "scThreshold (int32 x 504)": N_SENSORS * 4,
        "calibration gSum (double x 504)": N_SENSORS * 8,
        "calibration gSumSq (double x 504)": N_SENSORS * 8,
        "calibration mean/sd scratch (double x 504 x 2)": N_SENSORS * 8 * 2,
        "medianOfMeans tmp (float x 504)": N_SENSORS * 4,
        "firmware readings[24][21] (int)": N_SENSORS * 4,
        "firmware gFrame (int32 x 504)": N_SENSORS * 4,
        "firmware gFeatures (float x 504)": N_SENSORS * 4,
        "firmware gProbHistory (float x 3 x 6)": 3 * 6 * 4,
    }
    # the calibration-only buffers are reclaimable in principle but are static
    # here, so they are reported in full rather than hidden.
    inference_only = (ram["gBufA (int32 x SC_MAX_UNITS)"]
                      + ram["gBufB (int32 x SC_MAX_UNITS)"]
                      + ram["quantised input (int8 x 504)"])
    return {
        "flash_breakdown_bytes": flash,
        "flash_total_bytes": sum(flash.values()),
        "static_ram_breakdown_bytes": ram,
        "static_ram_total_bytes": sum(ram.values()),
        "inference_workspace_bytes": inference_only,
        "dynamic_allocation_in_inference_loop": False,
        "stack_bytes_estimate": "< 256 B: runLayer uses a handful of int32 locals "
                                "and no recursion or VLAs",
    }


def main():
    tfl_path = MODELS_DIR / "model_int8.tflite"
    if not tfl_path.is_file():
        raise SystemExit("models/model_int8.tflite missing; run ml/export_int8.py")
    tfl = tfl_path.read_bytes()
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    layers = extract_layers(tfl)

    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    real = ~d["is_synthetic"].astype(bool)
    X = pressure_to_features(d["X_pressure"][real])
    if meta.get("mask_col20"):
        from evaluate import COL20_IDX
        X = X.copy()
        X[:, COL20_IDX] = 0.0
    y = d["y"][real].astype(np.int32)

    rng = np.random.default_rng(99)
    idx = rng.choice(X.shape[0], size=min(N_FRAMES, X.shape[0]), replace=False)
    Xs = X[idx]

    print("comparing %d frames" % Xs.shape[0])

    # --- TFLite reference ---
    q_logits, deq, (s_in, z_in, s_out, z_out) = tflite_predict(tfl, Xs)

    # --- generated C++ (compiled with inference.cpp) ---
    build(with_inference=True, force=True)
    argmax_match = 0
    logit_maxdiff = 0
    prob_maxdiff = 0.0
    t_total = 0.0
    tfl_prob = softmax(deq)

    with Harness(with_inference=True) as h:
        base = np.zeros(N_SENSORS, dtype=np.int32)
        thr = np.zeros(N_SENSORS, dtype=np.int32)
        h.set_cal(base, thr)                 # features are supplied directly
        t0 = time.perf_counter()
        for k in range(Xs.shape[0]):
            r = h.infer_features(Xs[k])
            d_log = int(np.abs(r["logits"] - q_logits[k]).max())
            logit_maxdiff = max(logit_maxdiff, d_log)
            prob_maxdiff = max(prob_maxdiff,
                               float(np.abs(r["prob"] - tfl_prob[k]).max()))
            if r["argmax"] == int(q_logits[k].argmax()):
                argmax_match += 1
        t_total = time.perf_counter() - t0

    # --- numpy replica as a third opinion ---
    rep_match = 0
    for k in range(Xs.shape[0]):
        q = np.clip(np.round(Xs[k].astype(np.float32) / np.float32(s_in))
                    + z_in, -128, 127).astype(np.int32)
        out = reference_int8_forward(layers, q)
        if int(out.argmax()) == int(q_logits[k].argmax()):
            rep_match += 1

    n = Xs.shape[0]
    result = {
        "frames": n,
        "argmax_match": argmax_match,
        "argmax_agreement": argmax_match / n,
        "max_logit_diff": logit_maxdiff,
        "max_prob_diff": prob_maxdiff,
        "numpy_replica_argmax_match": rep_match,
        "tolerance_argmax": "identical on all frames",
        "tolerance_logits": "+/-1 quantisation step",
        "pass": bool(argmax_match == n and logit_maxdiff <= 1),
        "host_roundtrip_ms_per_frame_NOT_esp32": 1000.0 * t_total / n,
        "host_note": "this timing includes the stdin/stdout harness round trip "
                     "on a PC and is NOT an ESP32 figure",
        "esp32_inference_us": "PENDING_HARDWARE",
    }
    mem = static_memory_report(meta, layers)
    result["memory"] = mem

    (OUT / "cpp_vs_tflite.json").write_text(json.dumps(result, indent=2),
                                            encoding="utf-8")

    md = ["# Phase 5 - generated C++ vs TFLite INT8\n\n"]
    md.append("| metric | value | tolerance | pass |\n|---|---|---|---|\n")
    md.append("| frames compared | %d | | |\n" % n)
    md.append("| argmax agreement | %d / %d | identical on all frames | %s |\n"
              % (argmax_match, n, argmax_match == n))
    md.append("| max logit difference | %d | +/- 1 quantisation step | %s |\n"
              % (logit_maxdiff, logit_maxdiff <= 1))
    md.append("| max probability difference | %.3g | | |\n" % prob_maxdiff)
    md.append("| independent numpy replica argmax | %d / %d | | %s |\n"
              % (rep_match, n, rep_match == n))
    md.append("\n**Overall: %s**\n\n" % ("PASS" if result["pass"] else "FAIL"))

    md.append("## INT8 vs float\n\n")
    md.append("| metric | value | tolerance |\n|---|---|---|\n")
    md.append("| top-1 agreement | %.4f | >= 0.99 |\n"
              % meta["int8_vs_float_top1_agreement"])
    md.append("| max probability difference | %.4f | documented |\n\n"
              % meta["int8_vs_float_max_prob_diff"])

    md.append("## Memory budget\n\n### Flash (model parameters)\n\n")
    md.append("| item | bytes |\n|---|---|\n")
    for k, v in mem["flash_breakdown_bytes"].items():
        md.append("| %s | %d |\n" % (k, v))
    md.append("| **total** | **%d (%.1f kB)** |\n\n"
              % (mem["flash_total_bytes"], mem["flash_total_bytes"] / 1024))

    md.append("### Static RAM\n\n| item | bytes |\n|---|---|\n")
    for k, v in mem["static_ram_breakdown_bytes"].items():
        md.append("| %s | %d |\n" % (k, v))
    md.append("| **total** | **%d (%.1f kB)** |\n\n"
              % (mem["static_ram_total_bytes"], mem["static_ram_total_bytes"] / 1024))
    md.append("Inference workspace alone: **%d B**. "
              "Dynamic allocation in the inference loop: **none**. "
              "Stack: %s.\n\n"
              % (mem["inference_workspace_bytes"], mem["stack_bytes_estimate"]))
    md.append("The ESP32 DevKit V1 has 520 kB SRAM and 4 MB flash, so this fits "
              "with very large margin. Compiler-reported figures are "
              "PENDING_HARDWARE (no toolchain installed).\n\n")

    md.append("## Timing\n\n")
    md.append("- Host round trip through the test harness: **%.3f ms/frame** — "
              "this is a PC measurement including stdio and is **not** an ESP32 "
              "number.\n" % result["host_roundtrip_ms_per_frame_NOT_esp32"])
    md.append("- **ESP32 inference time: PENDING_HARDWARE.** The firmware brackets "
              "`scPreprocessFrame` + `scInfer` with `micros()` and prints the "
              "result as the 4th field of every `PRED,` line.\n")

    (OUT / "memory_budget.md").write_text("".join(md), encoding="utf-8")

    print("argmax %d/%d  max logit diff %d  max |dP| %.3g"
          % (argmax_match, n, logit_maxdiff, prob_maxdiff))
    print("numpy replica argmax %d/%d" % (rep_match, n))
    print("flash %d B, static RAM %d B, inference workspace %d B"
          % (mem["flash_total_bytes"], mem["static_ram_total_bytes"],
             mem["inference_workspace_bytes"]))
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
