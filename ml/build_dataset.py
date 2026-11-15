"""Build the model-ready arrays from the original (and augmented) CSV recordings.

Produces `models/dataset.npz` with, for every frame:
    X_pressure (N,504) int32  - archived pressure counts (stages 1-3 already applied)
    X_raw      (N,504) int32  - archived raw ADC counts
    y          (N,)    int32  - class index 0..5
    participant(N,)    str    - P1..P4 (real) / P5_SYNTHETIC.. (synthetic)
    posture    (N,)    str    - C0_Normal ..
    rep        (N,)    int32  - repetition number inside the recording
    group      (N,)    str    - "<participant>|<posture>|rep<k>" = one recording
                                 take; this is the unit that must never be split
                                 across train and test
    is_synthetic(N,)   bool
    source_participant (N,) str - provenance for synthetic rows, else same as participant

Also recovers, per real recording, the per-sensor baseline that the collection
script used, by exploiting pressure == 0 <=> |raw - baseline| <= threshold. The
recovered baselines let Phase 2 verify the raw -> features path against the
archived pressure CSVs on real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AUGMENTED_DIR, CLASS_NAMES, CLASS_TO_INDEX, DATASET_DIR,
                    MODELS_DIR, N_SENSORS, REPORTS_DIR, discover_recordings,
                    repetition_column, sensor_columns)
from preprocess import MIN_THRESHOLD, NOISE_MULTIPLIER

OUT_NPZ = MODELS_DIR / "dataset.npz"
BASELINE_JSON = MODELS_DIR / "recovered_baselines.json"


def recover_baseline(raw, pressure):
    """Recover, per sensor, the exact integer baseline the collector used.

    For every ungated cell the collector wrote pressure = |raw - baseline|, so
    the true baseline is one of {raw - pressure, raw + pressure}. Every such
    cell therefore casts two candidate votes, and the true baseline is the one
    candidate consistent with *all* ungated frames of that sensor. We score
    each candidate by how many frames it reproduces exactly and keep the best.

    Gated cells (pressure == 0) only bound the baseline to within the unknown
    threshold, so they are used solely as a fallback for sensors that are
    gated in every frame (dead sensors, light contact).
    """
    raw = np.asarray(raw, dtype=np.int64)
    pressure = np.asarray(pressure, dtype=np.int64)
    gated = pressure == 0
    n_gated = gated.sum(axis=0)
    baseline = np.zeros(N_SENSORS, dtype=np.int64)

    for i in range(N_SENSORS):
        ri, pi = raw[:, i], pressure[:, i]
        active = ~gated[:, i]
        if active.sum() == 0:
            # never ungated: the median raw over gated frames is the best we can do
            baseline[i] = int(round(float(np.median(ri))))
            continue
        cands = np.unique(np.concatenate([ri[active] - pi[active],
                                          ri[active] + pi[active]]))
        # score: number of ungated frames reproduced exactly by this candidate
        recon = np.abs(ri[active][None, :] - cands[:, None])   # (C, F_active)
        score = (recon == pi[active][None, :]).sum(axis=1)
        best = int(score.argmax())
        baseline[i] = int(cands[best])

    return baseline.astype(np.int32), n_gated.astype(np.int32)


def recover_threshold(raw, pressure, baseline):
    """Lower bound on the threshold map implied by the gated cells.

    threshold[i] >= max over gated frames of |raw - baseline|, and
    threshold[i] <  min over ungated frames of pressure. The collector used
    max(3*std, 20), so we report the implied interval and take the tightest
    consistent integer.
    """
    raw = np.asarray(raw, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    delta = np.abs(raw - baseline.reshape(1, -1))
    gated = pressure == 0
    lower = np.where(gated, delta, -np.inf).max(axis=0)
    lower = np.where(np.isfinite(lower), lower, float(MIN_THRESHOLD))
    upper = np.where(~gated, pressure, np.inf).min(axis=0)
    upper = np.where(np.isfinite(upper), upper, lower + 1.0)
    thr = np.maximum(np.rint(lower), float(MIN_THRESHOLD))
    return thr.astype(np.int32), lower, upper


def load_folder(rec):
    """Load one recording folder -> dict of arrays, or None if unusable."""
    raw_df = pd.read_csv(rec["raw_csv"])
    pr_df = pd.read_csv(rec["pressure_csv"])
    scols_r = sensor_columns(raw_df)
    scols_p = sensor_columns(pr_df)
    raw = raw_df[scols_r].to_numpy(dtype=np.int32)
    pres = pr_df[scols_p].to_numpy(dtype=np.int32)
    if raw.shape[0] != pres.shape[0]:
        n = min(raw.shape[0], pres.shape[0])
        raw, pres = raw[:n], pres[:n]
    repcol = repetition_column(raw_df)
    rep = raw_df[repcol].to_numpy(dtype=np.int32)[:raw.shape[0]]
    return {"raw": raw, "pressure": pres, "rep": rep}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    blocks = []
    baseline_info = {}
    verify_rows = []

    sources = [(DATASET_DIR, False)]
    if AUGMENTED_DIR.is_dir() and any(AUGMENTED_DIR.iterdir()):
        sources.append((AUGMENTED_DIR, True))

    for root, is_syn in sources:
        for rec in discover_recordings(root):
            d = load_folder(rec)
            n = d["raw"].shape[0]
            part = rec["participant"]
            cls = rec["posture_class"]

            # provenance: synthetic folders carry a provenance.csv written in Phase 3
            src_part = part
            prov = rec["dir"] / "provenance.csv"
            src_per_frame = None
            if is_syn and prov.is_file():
                pv = pd.read_csv(prov)
                if "source_participant" in pv.columns and len(pv) >= n:
                    src_per_frame = pv["source_participant"].to_numpy(dtype=object)[:n]

            baseline, n_gated = recover_baseline(d["raw"], d["pressure"])
            thr, lo, hi = recover_threshold(d["raw"], d["pressure"], baseline)
            key = part + "|" + cls
            baseline_info[key] = {
                "baseline": baseline.tolist(),
                "threshold_lower_bound": thr.tolist(),
                "sensors_never_gated": int((n_gated == 0).sum()),
                "frames": int(n),
            }

            # verification: does stages 1-3 on raw reproduce the archived pressure?
            delta = np.abs(d["raw"].astype(np.int64) - baseline.reshape(1, -1))
            recon = np.where(delta <= thr.reshape(1, -1), 0, delta)
            recon = np.clip(recon, 0, 4095)
            agree = (recon == d["pressure"])
            verify_rows.append({
                "participant": part, "posture_class": cls, "frames": int(n),
                "cells": int(agree.size),
                "exact_match_pct": float(agree.mean() * 100),
                "max_abs_err": int(np.abs(recon - d["pressure"]).max()),
                "mean_abs_err": float(np.abs(recon - d["pressure"]).mean()),
                "sensors_never_gated": int((n_gated == 0).sum()),
            })

            blocks.append({
                "X_pressure": d["pressure"],
                "X_raw": d["raw"],
                "y": np.full(n, CLASS_TO_INDEX[cls], dtype=np.int32),
                "participant": np.full(n, part, dtype=object),
                "posture": np.full(n, cls, dtype=object),
                "rep": d["rep"],
                "group": np.array([part + "|" + cls + "|rep" + str(r)
                                   for r in d["rep"]], dtype=object),
                "is_synthetic": np.full(n, is_syn, dtype=bool),
                "source_participant": (src_per_frame if src_per_frame is not None
                                       else np.full(n, src_part, dtype=object)),
            })
            print("  loaded %-14s %-17s n=%4d  recon=%.3f%%" % (
                part, cls, n, verify_rows[-1]["exact_match_pct"]))

    merged = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in blocks[0]}
    np.savez_compressed(
        OUT_NPZ,
        X_pressure=merged["X_pressure"].astype(np.int32),
        X_raw=merged["X_raw"].astype(np.int32),
        y=merged["y"],
        participant=merged["participant"].astype(str),
        posture=merged["posture"].astype(str),
        rep=merged["rep"],
        group=merged["group"].astype(str),
        is_synthetic=merged["is_synthetic"],
        source_participant=merged["source_participant"].astype(str),
        class_names=np.array(CLASS_NAMES),
    )
    BASELINE_JSON.write_text(json.dumps(baseline_info), encoding="utf-8")

    vdf = pd.DataFrame(verify_rows)
    vdf.to_csv(REPORTS_DIR / "dataset_report" / "pressure_reconstruction_check.csv",
               index=False)

    print("\nTotal frames: %d" % merged["y"].shape[0])
    print("Real frames : %d" % int((~merged["is_synthetic"]).sum()))
    print("Synth frames: %d" % int(merged["is_synthetic"].sum()))
    print("Groups (recording takes): %d" % len(set(merged["group"].tolist())))
    print("Wrote " + str(OUT_NPZ))
    print("\nraw->pressure reconstruction, mean exact-match: %.3f%%"
          % vdf["exact_match_pct"].mean())
    print(vdf[["participant", "posture_class", "exact_match_pct", "max_abs_err"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
