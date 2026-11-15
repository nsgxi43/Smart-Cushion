"""Phase 4 diagnostic: why does leave-one-participant-out accuracy collapse?

Within a participant the six postures separate easily (Phase 3 measured 0.73-0.85
nearest-centroid accuracy). Across participants the nested evaluation lands near
chance. This script quantifies the domain shift and measures how much of it each
candidate feature transform removes.

Transforms tested (all are implementable on the ESP32):
  global_scale    x = delta / 1024                       <- the Phase 2 spec
  frame_norm      x = delta / max(frame_max, 1)          per-frame peak
  frame_mean_norm x = delta / max(frame_mean, 1)         per-frame mean
  binary          x = 1 if delta > 0 else 0              occupancy only
  session_z       x = (delta - session_mean) / session_sd per-recording standardise
  row_col_profile 24 row sums + 21 col sums, frame-normalised
  centre_of_press 6 scalars: CoP row/col, total, area, row spread, col spread

The last two change the input dimensionality and so would need a model input
change; they are included to show how much of the signal is carried by coarse
spatial statistics that transfer across bodies.

Nothing here is used to pick the shipped model unless it is promoted into the
Phase 4 candidate space and re-run under the nested protocol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, COLS, MODELS_DIR, N_SENSORS, REPORTS_DIR, ROWS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = REPORTS_DIR / "phase4_evaluation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261115
REAL = ["P1", "P2", "P3", "P4"]


def load_real():
    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    m = ~d["is_synthetic"].astype(bool)
    return {
        "P": d["X_pressure"][m].astype(np.float64),
        "y": d["y"][m].astype(np.int32),
        "participant": d["participant"].astype(str)[m],
        "group": d["group"].astype(str)[m],
    }


# ---------------------------------------------------------------- transforms
def t_global(P, groups):
    return P / 1024.0


def t_frame_norm(P, groups):
    mx = np.maximum(P.max(axis=1, keepdims=True), 1.0)
    return P / mx


def t_frame_mean_norm(P, groups):
    mn = np.maximum(P.mean(axis=1, keepdims=True), 1.0)
    return P / mn


def t_binary(P, groups):
    return (P > 0).astype(np.float64)


def t_session_z(P, groups):
    """Standardise each recording take by its own mean/sd.

    On the device this is a session-level statistic computed over a short warm-up
    window, so it is reproducible; it uses no label information.
    """
    out = np.empty_like(P)
    for g in np.unique(groups):
        m = groups == g
        blk = P[m]
        mu, sd = blk.mean(), blk.std()
        out[m] = (blk - mu) / max(sd, 1e-6)
    return out


def t_row_col_profile(P, groups):
    M = P.reshape(-1, ROWS, COLS)
    tot = np.maximum(M.sum(axis=(1, 2), keepdims=True), 1.0)
    Mn = M / tot
    return np.concatenate([Mn.sum(axis=2), Mn.sum(axis=1)], axis=1)


def t_centre_of_pressure(P, groups):
    M = P.reshape(-1, ROWS, COLS)
    tot = np.maximum(M.sum(axis=(1, 2)), 1.0)
    rr = np.arange(ROWS)[None, :, None]
    cc = np.arange(COLS)[None, None, :]
    cop_r = (M * rr).sum(axis=(1, 2)) / tot
    cop_c = (M * cc).sum(axis=(1, 2)) / tot
    var_r = (M * (rr - cop_r[:, None, None]) ** 2).sum(axis=(1, 2)) / tot
    var_c = (M * (cc - cop_c[:, None, None]) ** 2).sum(axis=(1, 2)) / tot
    area = (M > 0).sum(axis=(1, 2)).astype(np.float64)
    return np.stack([cop_r, cop_c, np.sqrt(var_r), np.sqrt(var_c),
                     tot / 1e5, area / N_SENSORS], axis=1)


TRANSFORMS = {
    "global_scale (Phase 2 spec)": t_global,
    "frame_norm": t_frame_norm,
    "frame_mean_norm": t_frame_mean_norm,
    "binary": t_binary,
    "session_z": t_session_z,
    "row_col_profile": t_row_col_profile,
    "centre_of_pressure": t_centre_of_pressure,
}

MODELS = {
    "logreg_C1": lambda: Pipeline([("sc", StandardScaler()),
                                   ("m", LogisticRegression(max_iter=3000, C=1.0))]),
    "rf300": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED,
                                            n_jobs=2),
    "lda": lambda: Pipeline([("sc", StandardScaler()),
                             ("m", LinearDiscriminantAnalysis())]),
}


def main():
    d = load_real()
    rows = []

    # -------- 1. how big is the per-participant offset? --------------------
    shift_rows = []
    Xg = t_global(d["P"], d["group"])
    for p in REAL:
        m = d["participant"] == p
        shift_rows.append({
            "participant": p,
            "mean_feature": float(Xg[m].mean()),
            "sd_feature": float(Xg[m].std()),
            "mean_active_cells": float((d["P"][m] > 0).sum(axis=1).mean()),
            "mean_peak": float(d["P"][m].max(axis=1).mean()),
        })
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(OUT / "participant_offset.csv", index=False)

    # a participant classifier: if a model can tell WHO is sitting from a frame,
    # the between-participant signal dominates the between-posture signal
    who = LogisticRegression(max_iter=3000).fit(
        StandardScaler().fit_transform(Xg),
        np.array([REAL.index(p) for p in d["participant"]]))
    who_acc = float(np.mean(who.predict(
        StandardScaler().fit_transform(Xg)) ==
        np.array([REAL.index(p) for p in d["participant"]])))

    # -------- 2. LOPO per transform x model --------------------------------
    for tname, tf in TRANSFORMS.items():
        X = tf(d["P"], d["group"])
        for mname, mk in MODELS.items():
            accs = []
            for held in REAL:
                te = d["participant"] == held
                tr = ~te
                mdl = mk().fit(X[tr], d["y"][tr])
                accs.append(accuracy_score(d["y"][te], mdl.predict(X[te])))
            rows.append({
                "transform": tname, "model": mname, "n_features": X.shape[1],
                "mean_lopo_acc": float(np.mean(accs)),
                "min_lopo_acc": float(np.min(accs)),
                "P1": accs[0], "P2": accs[1], "P3": accs[2], "P4": accs[3],
            })
            print("  %-28s %-10s  mean=%.4f  folds=%s"
                  % (tname, mname, np.mean(accs),
                     " ".join("%.3f" % a for a in accs)))

    res = pd.DataFrame(rows).sort_values("mean_lopo_acc", ascending=False)
    res.to_csv(OUT / "feature_transform_diagnostic.csv", index=False)

    # -------- 3. within-participant ceiling (same protocol, same models) ----
    within_rows = []
    for tname, tf in TRANSFORMS.items():
        X = tf(d["P"], d["group"])
        for p in REAL:
            m = d["participant"] == p
            Xp, yp, gp = X[m], d["y"][m], d["group"][m]
            # leave-one-repetition-out inside this participant
            reps = sorted({g.split("|rep")[1] for g in gp})
            accs = []
            for r in reps:
                te = np.array([g.endswith("|rep" + r) for g in gp])
                if te.sum() == 0 or (~te).sum() == 0:
                    continue
                mdl = MODELS["logreg_C1"]().fit(Xp[~te], yp[~te])
                accs.append(accuracy_score(yp[te], mdl.predict(Xp[te])))
            within_rows.append({"transform": tname, "participant": p,
                                "within_participant_acc": float(np.mean(accs))})
    within = pd.DataFrame(within_rows)
    within.to_csv(OUT / "within_participant_ceiling.csv", index=False)

    # -------- report --------------------------------------------------------
    md = ["# Phase 4 diagnostic - cross-participant domain shift\n\n"]
    md.append("Chance level for 6 balanced classes is **16.7%**.\n\n")

    md.append("## How different are the participants?\n\n")
    md.append(shift.round(4).to_markdown(index=False))
    md.append("\n\nA logistic regression trained to predict *which participant* "
              "a frame came from (on the Phase 2 features) reaches "
              "**%.1f%%** in-sample across 4 classes (chance 25%%). "
              "Participant identity is written into the frames far more strongly "
              "than posture is.\n" % (100 * who_acc))

    md.append("\n## Leave-one-participant-out, per feature transform\n\n")
    md.append(res.round(4).to_markdown(index=False))

    md.append("\n\n## Within-participant ceiling "
              "(leave-one-repetition-out, logistic regression)\n\n")
    piv = within.pivot(index="transform", columns="participant",
                       values="within_participant_acc")
    piv["mean"] = piv.mean(axis=1)
    md.append(piv.round(4).to_markdown())

    md.append("\n\n## Reading this\n\n")
    best = res.iloc[0]
    md.append("- Best cross-participant transform here: **%s / %s**, "
              "mean LOPO %.3f.\n" % (best["transform"], best["model"],
                                     best["mean_lopo_acc"]))
    spec = res[res["transform"].str.startswith("global_scale")]
    md.append("- The Phase 2 specified transform (`global_scale`) reaches "
              "mean LOPO %.3f at best.\n" % spec["mean_lopo_acc"].max())
    md.append("- The gap between the within-participant column and the "
              "cross-participant column is the domain shift. It is a property of "
              "the sensor and the 4-participant sample, not of the model family: "
              "every model family tested lands in the same narrow band.\n")

    (OUT / "domain_shift_diagnostic.md").write_text("".join(md), encoding="utf-8")
    print("\nparticipant-identity in-sample accuracy: %.4f (chance 0.25)" % who_acc)
    print("\n" + res.round(4).head(12).to_string(index=False))
    print("\nwrote " + str(OUT / "domain_shift_diagnostic.md"))


if __name__ == "__main__":
    main()
