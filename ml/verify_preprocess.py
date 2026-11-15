"""Phase 2 verification: Python and C++ preprocessing must agree bit-for-bit.

Runs the identical frames through ml/preprocess.py and esp32/preprocess.cpp
(via esp32/host_harness.exe) and reports the numerical tolerance achieved.
Writes reports/preprocess_equivalence.md and .json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp
from common import MODELS_DIR, REPORTS_DIR
from cpp_bridge import Harness

RNG = np.random.default_rng(20261115)
N_REAL_FRAMES = 300


def edge_case_frames():
    """Frames that exercise every branch of the pipeline."""
    N = pp.N_FEATURES
    cases = {
        "all_zero":        np.zeros(N, dtype=np.int32),
        "all_max":         np.full(N, 4095, dtype=np.int32),
        "alternating":     np.tile([0, 4095], N // 2).astype(np.int32),
        "just_below_thr":  np.full(N, 0, dtype=np.int32),   # paired with baseline below
        "just_above_thr":  np.full(N, 0, dtype=np.int32),
        "ramp":            np.arange(N, dtype=np.int32) % 4096,
        "random_uniform":  RNG.integers(0, 4096, N).astype(np.int32),
        "random_clustered": np.clip(RNG.normal(3000, 400, N), 0, 4095).astype(np.int32),
    }
    return cases


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    X_raw = data["X_raw"]
    baselines = json.loads((MODELS_DIR / "recovered_baselines.json").read_text())

    results = {"checks": [], "max_abs_diff_overall": 0.0,
               "max_ulp_diff_overall": 0, "frames_compared": 0}

    with Harness(with_inference=False) as h:

        # ---------- 1. real frames, real recovered calibration ----------
        key = sorted(baselines.keys())[0]
        base = np.array(baselines[key]["baseline"], dtype=np.int32)
        thr = np.array(baselines[key]["threshold_lower_bound"], dtype=np.int32)
        h.set_cal(base, thr)
        got_b, got_t = h.get_cal()
        assert np.array_equal(got_b, base) and np.array_equal(got_t, thr)

        idx = RNG.choice(X_raw.shape[0], size=min(N_REAL_FRAMES, X_raw.shape[0]),
                         replace=False)
        max_abs = 0.0
        max_ulp = 0
        n_exact = 0
        for i in idx:
            raw = X_raw[i].astype(np.int32)
            py = pp.preprocess_frame(raw, base, thr)
            cpp = h.preprocess(raw)
            d = np.abs(py - cpp)
            max_abs = max(max_abs, float(d.max()))
            ulp = np.abs(py.view(np.int32).astype(np.int64)
                         - cpp.view(np.int32).astype(np.int64)).max()
            max_ulp = max(max_ulp, int(ulp))
            if np.array_equal(py.view(np.int32), cpp.view(np.int32)):
                n_exact += 1
        results["checks"].append({
            "name": "real_frames_full_pipeline",
            "frames": int(len(idx)),
            "bitwise_identical_frames": n_exact,
            "max_abs_diff": max_abs,
            "max_ulp_diff": max_ulp,
            "calibration_source": key,
        })
        results["frames_compared"] += int(len(idx))
        results["max_abs_diff_overall"] = max(results["max_abs_diff_overall"], max_abs)
        results["max_ulp_diff_overall"] = max(results["max_ulp_diff_overall"], max_ulp)

        # ---------- 2. stages 1-3 only: raw -> integer pressure ----------
        n_exact = 0
        for i in idx[:100]:
            raw = X_raw[i].astype(np.int32)
            d = np.abs(raw.astype(np.int64) - base.astype(np.int64))
            py_press = np.where(d <= thr, 0, d)
            py_press = np.clip(py_press, 0, pp.CLIP_MAX).astype(np.int32)
            cpp_press = h.raw_to_pressure(raw)
            if np.array_equal(py_press, cpp_press):
                n_exact += 1
        results["checks"].append({
            "name": "stages_1_3_raw_to_pressure",
            "frames": 100, "exact_frames": n_exact,
        })

        # ---------- 3. stage 4 only: pressure -> features ----------
        X_press = data["X_pressure"]
        n_exact = 0
        max_abs4 = 0.0
        for i in idx[:100]:
            p = X_press[i].astype(np.int32)
            py = pp.pressure_to_features(p)
            cpp = h.pressure_to_features(p)
            max_abs4 = max(max_abs4, float(np.abs(py - cpp).max()))
            if np.array_equal(py.view(np.int32), cpp.view(np.int32)):
                n_exact += 1
        results["checks"].append({
            "name": "stage_4_pressure_to_features",
            "frames": 100, "bitwise_identical_frames": n_exact,
            "max_abs_diff": max_abs4,
        })

        # ---------- 4. edge cases ----------
        edge_rows = []
        for name, raw in edge_case_frames().items():
            if name == "just_below_thr":
                b = np.full(pp.N_FEATURES, 1000, dtype=np.int32)
                t = np.full(pp.N_FEATURES, 20, dtype=np.int32)
                raw = b + 20                       # delta == threshold -> gated
            elif name == "just_above_thr":
                b = np.full(pp.N_FEATURES, 1000, dtype=np.int32)
                t = np.full(pp.N_FEATURES, 20, dtype=np.int32)
                raw = b + 21                       # delta == threshold+1 -> kept
            else:
                b, t = base, thr
            h.set_cal(b, t)
            py = pp.preprocess_frame(raw, b, t)
            cpp = h.preprocess(raw)
            edge_rows.append({
                "case": name,
                "bitwise_identical": bool(np.array_equal(py.view(np.int32),
                                                         cpp.view(np.int32))),
                "max_abs_diff": float(np.abs(py - cpp).max()),
                "py_nonzero": int((py > 0).sum()),
            })
        results["checks"].append({"name": "edge_cases", "cases": edge_rows})
        h.set_cal(base, thr)

        # ---------- 5. calibration equivalence ----------
        cal_rows = []
        scenarios = {
            "clean_unloaded": np.clip(
                RNG.normal(2500, 8, (25, pp.N_FEATURES)), 0, 4095),
            "too_few_frames": np.clip(
                RNG.normal(2500, 8, (10, pp.N_FEATURES)), 0, 4095),
            "pressure_detected": None,      # built below
            "excessive_noise": np.clip(
                RNG.normal(2500, 200, (25, pp.N_FEATURES)), 0, 4095),
            "unstable_baseline": None,      # built below
        }
        loaded = np.clip(RNG.normal(2500, 8, (25, pp.N_FEATURES)), 0, 4095)
        loaded[:, 200:260] += 900           # someone sitting on it
        scenarios["pressure_detected"] = loaded

        drift = np.clip(RNG.normal(2500, 8, (25, pp.N_FEATURES)), 0, 4095)
        drift += np.linspace(0, 400, 25).reshape(-1, 1)   # slow drift
        scenarios["unstable_baseline"] = np.clip(drift, 0, 4095)

        for name, frames in scenarios.items():
            frames = np.rint(frames).astype(np.int32)
            ok, reason, diag = pp.validate_calibration(frames)
            h.cal_begin()
            for f in frames:
                h.cal_add(f)
            cres, cstate = h.cal_finish()
            py_label = "OK" if ok else reason
            cal_rows.append({
                "scenario": name,
                "python_result": py_label,
                "cpp_result": cres,
                "cpp_state": cstate,
                "agree": bool(py_label == cres),
                "python_diagnostics": {k: round(float(v), 3) for k, v in diag.items()},
            })
            # when both accept, the derived baseline/threshold must match exactly
            if ok and cres == "OK":
                pb, pt = pp.compute_calibration(frames)
                cb, ct = h.get_cal()
                cal_rows[-1]["baseline_exact"] = bool(np.array_equal(pb, cb))
                cal_rows[-1]["threshold_exact"] = bool(np.array_equal(pt, ct))
                cal_rows[-1]["baseline_max_diff"] = int(np.abs(pb - cb).max())
                cal_rows[-1]["threshold_max_diff"] = int(np.abs(pt - ct).max())
        results["checks"].append({"name": "calibration_equivalence", "cases": cal_rows})

    # ---------------- report ----------------
    (REPORTS_DIR / "preprocess_equivalence.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    md = ["# Phase 2 - Python / C++ preprocessing equivalence\n\n"]
    md.append("Python: `ml/preprocess.py`  \nC++: `esp32/preprocess.cpp` "
              "(driven through `esp32/host_harness.exe`)\n\n")

    c0 = results["checks"][0]
    md.append("## 1. Full pipeline on real recorded frames\n\n")
    md.append("| metric | value |\n|---|---|\n")
    md.append("| frames compared | %d |\n" % c0["frames"])
    md.append("| bit-identical frames | %d / %d |\n"
              % (c0["bitwise_identical_frames"], c0["frames"]))
    md.append("| max absolute difference | %.3g |\n" % c0["max_abs_diff"])
    md.append("| max ULP difference | %d |\n" % c0["max_ulp_diff"])
    md.append("| calibration used | `%s` (recovered in Phase 1) |\n\n"
              % c0["calibration_source"])

    c1, c2 = results["checks"][1], results["checks"][2]
    md.append("## 2. Per-stage\n\n| stage | frames | exact |\n|---|---|---|\n")
    md.append("| 1-3 raw -> integer pressure | %d | %d |\n" % (c1["frames"], c1["exact_frames"]))
    md.append("| 4 pressure -> float features | %d | %d |\n\n"
              % (c2["frames"], c2["bitwise_identical_frames"]))

    md.append("## 3. Edge cases\n\n| case | bit-identical | max abs diff | non-zero features |\n")
    md.append("|---|---|---|---|\n")
    for r in results["checks"][3]["cases"]:
        md.append("| %s | %s | %.3g | %d |\n" % (r["case"], r["bitwise_identical"],
                                                 r["max_abs_diff"], r["py_nonzero"]))

    md.append("\n## 4. Calibration state machine\n\n")
    md.append("| scenario | Python | C++ | agree | baseline exact | threshold exact |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in results["checks"][4]["cases"]:
        md.append("| %s | %s | %s | %s | %s | %s |\n" % (
            r["scenario"], r["python_result"], r["cpp_result"], r["agree"],
            r.get("baseline_exact", "-"), r.get("threshold_exact", "-")))

    md.append("\n## Documented tolerance\n\n")
    if results["max_ulp_diff_overall"] == 0:
        md.append("**Exact.** Every compared frame was bit-identical in IEEE-754 "
                  "float32 (0 ULP). The integer stages run in int32 on both sides "
                  "and the only float operation is a multiply by an exact power of "
                  "two (1/1024), which is exact in binary floating point.\n")
    else:
        md.append("Max absolute difference %.3g, max %d ULP.\n"
                  % (results["max_abs_diff_overall"], results["max_ulp_diff_overall"]))

    (REPORTS_DIR / "preprocess_equivalence.md").write_text("".join(md), encoding="utf-8")

    print("".join(md))
    print("wrote reports/preprocess_equivalence.{md,json}")

    bad = [r for r in results["checks"][4]["cases"] if not r["agree"]]
    if bad:
        print("\nCALIBRATION MISMATCHES:", [b["scenario"] for b in bad])
        return 1
    if results["max_ulp_diff_overall"] != 0:
        print("\nWARNING: not bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
