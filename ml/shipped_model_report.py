"""Per-class metrics and confusion matrix for the SHIPPED architecture.

Phase 4's nested table averages over whichever candidate the inner loop picked
in each fold, so its per-class numbers do not describe the model that actually
ships. This refits the chosen architecture (and only that one) in each
leave-one-real-participant-out fold and reports what it does per class.

The architecture was chosen on inner-validation score in ml/decide_model.py;
nothing here feeds back into that choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, MODELS_DIR, REPORTS_DIR
from evaluate import COL20_IDX, KerasWrapper, training_mask
from preprocess import pressure_to_features

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

OUT = REPORTS_DIR / "phase4_evaluation"
REAL = ["P1", "P2", "P3", "P4"]


def main():
    dec = json.loads((MODELS_DIR / "phase4_decision.json").read_text())
    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    data = {
        "X": pressure_to_features(d["X_pressure"]),
        "y": d["y"].astype(np.int32),
        "participant": d["participant"].astype(str),
        "is_synthetic": d["is_synthetic"].astype(bool),
        "source_participant": d["source_participant"].astype(str),
    }
    if dec["mask_col20"]:
        data["X"] = data["X"].copy()
        data["X"][:, COL20_IDX] = 0.0

    rows, preds, trues, fold_rows = [], [], [], []
    for held in REAL:
        tm = training_mask(data, held, dec["use_synthetic"])
        te = (data["participant"] == held) & ~data["is_synthetic"]
        mdl = KerasWrapper(dec["hidden"], dropout=dec["dropout"], lr=1e-3,
                           l2=dec["l2"], epochs=dec["epochs"],
                           batch=dec["batch"]).fit(data["X"][tm], data["y"][tm])
        p = mdl.predict(data["X"][te])
        t = data["y"][te]
        preds.append(p); trues.append(t)
        fold_rows.append({"held_out": held, "accuracy": accuracy_score(t, p),
                          "macro_f1": f1_score(t, p, average="macro", zero_division=0),
                          "train_frames": int(tm.sum()), "test_frames": int(te.sum())})
        pr, rc, f1, sup = precision_recall_fscore_support(
            t, p, labels=list(range(6)), zero_division=0)
        for ci, cn in enumerate(CLASS_NAMES):
            rows.append({"held_out": held, "class": cn, "precision": pr[ci],
                         "recall": rc[ci], "f1": f1[ci], "support": int(sup[ci])})
        print("  %s acc=%.4f" % (held, fold_rows[-1]["accuracy"]))

    yt, yp = np.concatenate(trues), np.concatenate(preds)
    cm = confusion_matrix(yt, yp, labels=list(range(6)))
    np.save(OUT / "cm_shipped_architecture.npy", cm)
    pd.DataFrame(rows).to_csv(OUT / "shipped_per_class_per_fold.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "shipped_per_fold.csv", index=False)

    agg = (pd.DataFrame(rows).groupby("class")
           .apply(lambda g: pd.Series({
               "precision": np.average(g["precision"], weights=g["support"]),
               "recall": np.average(g["recall"], weights=g["support"]),
               "f1": np.average(g["f1"], weights=g["support"]),
               "support": int(g["support"].sum())}), include_groups=False)
           .reindex(CLASS_NAMES))
    agg.to_csv(OUT / "shipped_per_class.csv")

    md = ["# Shipped architecture - per-class leave-one-participant-out\n\n"]
    md.append("Architecture: `504-%s-6`, column 20 masked=%s, trained on %s.\n\n"
              % ("-".join(map(str, dec["hidden"])), dec["mask_col20"],
                 dec["training_data_setting"]))
    md.append("Pooled over the 4 held-out folds. Chance = 16.7%.\n\n")
    md.append(agg.round(4).to_markdown())
    md.append("\n\n## Per fold\n\n")
    md.append(pd.DataFrame(fold_rows).round(4).to_markdown(index=False))
    md.append("\n\n## Confusion matrix (rows = true, columns = predicted)\n\n")
    md.append(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_markdown())
    md.append("\n\n**Overall accuracy %.4f, macro F1 %.4f.**\n"
              % (accuracy_score(yt, yp), f1_score(yt, yp, average="macro",
                                                  zero_division=0)))
    (OUT / "shipped_model_per_class.md").write_text("".join(md), encoding="utf-8")

    print("\n" + agg.round(3).to_string())
    print("\npooled accuracy %.4f" % accuracy_score(yt, yp))
    print("wrote " + str(OUT / "shipped_model_per_class.md"))


if __name__ == "__main__":
    main()
