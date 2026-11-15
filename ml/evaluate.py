"""Phase 4 - model comparison under strict participant-independent evaluation.

Protocol
--------
Outer loop: leave-one-REAL-participant-out (4 folds: P1..P4).
  The held-out participant contributes NOTHING to training, to hyperparameter
  or architecture selection, to the masking decision, or to the preprocessing
  constants (which are fixed a priori from the ADC contract anyway). Every
  synthetic participant DERIVED from the held-out person is also removed from
  training (ml/augment.py::SYNTHETIC_SOURCES records the derivation).

Inner loop: leave-one-participant-out over the 3 remaining real participants.
  This is the only thing allowed to choose a model. Two numbers come out of
  each outer fold:

  * NESTED  - the single candidate with the best inner score is refit on the
              whole training set and scored once on the held-out participant.
              This is the unbiased estimate and the headline number.
  * PER-CANDIDATE - every candidate is also refit and scored on the held-out
              participant. This table is diagnostic only. Choosing the best row
              of it would be selection on the test set, so it is never used to
              pick the shipped model.

Splitting is by recording take (`participant|posture|repN`): no frame from one
10-second recording can land on both sides of any split. The inner loop groups
by participant, which is strictly coarser, so the guarantee holds there too.

Everything runs twice: REAL_ONLY and REAL_PLUS_SYNTHETIC.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, COLS, MODELS_DIR, N_SENSORS, REPORTS_DIR, ROWS
from preprocess import pressure_to_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = REPORTS_DIR / "phase4_evaluation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261115
REAL_PARTICIPANTS = ["P1", "P2", "P3", "P4"]

# column 20 is an intermittent hardware fault (Phase 1): dead for P2 and P4,
# alive for P3. Masking it is a *candidate*, selected by the inner loop.
COL20_IDX = np.array([r * COLS + 20 for r in range(ROWS)])

KERAS_EPOCHS = 60
KERAS_BATCH = 64


def load():
    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    return {
        "X": pressure_to_features(d["X_pressure"]),
        "y": d["y"].astype(np.int32),
        "participant": d["participant"].astype(str),
        "group": d["group"].astype(str),
        "is_synthetic": d["is_synthetic"].astype(bool),
        "source_participant": d["source_participant"].astype(str),
    }


def mask_col20(X):
    X = X.copy()
    X[:, COL20_IDX] = 0.0
    return X


# ---------------------------------------------------------------------------
# Keras
# ---------------------------------------------------------------------------
def build_keras(hidden, dropout, lr, l2=1e-4):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)
    layers = [keras.layers.Input(shape=(N_SENSORS,))]
    for h in hidden:
        layers.append(keras.layers.Dense(h, activation="relu",
                                         kernel_regularizer=keras.regularizers.l2(l2)))
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(len(CLASS_NAMES), activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class KerasWrapper:
    def __init__(self, hidden, dropout=0.3, lr=1e-3, l2=1e-4,
                 epochs=KERAS_EPOCHS, batch=KERAS_BATCH):
        self.kw = dict(hidden=hidden, dropout=dropout, lr=lr, l2=l2)
        self.epochs, self.batch = epochs, batch
        self.model = None
        self.history = None

    def fit(self, X, y):
        self.model = build_keras(**self.kw)
        self.history = self.model.fit(X, y, epochs=self.epochs,
                                      batch_size=self.batch, verbose=0)
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0, batch_size=512)


# ---------------------------------------------------------------------------
# candidate space. One flat list; the inner loop chooses among all of it.
# ---------------------------------------------------------------------------
def candidates():
    cands = []

    def add(family, cfg, mask20, builder):
        cands.append({
            "id": "%s|%s|%s" % (family, json.dumps(cfg, sort_keys=True),
                                "mask20" if mask20 else "all504"),
            "family": family, "cfg": cfg, "mask20": mask20, "builder": builder,
        })

    for mask20 in (False, True):
        for C in (0.01, 0.1, 1.0):
            add("logreg", {"C": C}, mask20,
                lambda C=C: Pipeline([("sc", StandardScaler()),
                                      ("m", LogisticRegression(max_iter=2000, C=C))]))
        for cfg in ({"n_estimators": 200, "max_depth": 8},
                    {"n_estimators": 300, "max_depth": 14},
                    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2}):
            add("random_forest", cfg, mask20,
                lambda cfg=cfg: RandomForestClassifier(random_state=SEED, n_jobs=-1, **cfg))
        for cfg in ({"C": 1.0}, {"C": 10.0}):
            add("svm_rbf", cfg, mask20,
                lambda cfg=cfg: Pipeline([("sc", StandardScaler()),
                                          ("m", SVC(kernel="rbf", gamma="scale",
                                                    random_state=SEED, **cfg))]))
        for cfg in ({"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
                    {"hidden_layer_sizes": (32, 16), "alpha": 1e-3},
                    {"hidden_layer_sizes": (32, 16), "alpha": 1e-2}):
            add("mlp_sklearn", cfg, mask20,
                lambda cfg=cfg: Pipeline([("sc", StandardScaler()),
                                          ("m", MLPClassifier(random_state=SEED,
                                                              max_iter=600, **cfg))]))
        for name, hidden in (("504_64_32_6", [64, 32]), ("504_32_16_6", [32, 16]),
                             ("504_24_6", [24]), ("504_16_6", [16])):
            for dropout in (0.3,):
                add("keras_" + name, {"dropout": dropout, "lr": 1e-3}, mask20,
                    lambda hidden=hidden, dropout=dropout:
                        KerasWrapper(hidden, dropout=dropout, lr=1e-3))
    return cands


# ---------------------------------------------------------------------------
def training_mask(data, held_out, use_synthetic):
    m = data["participant"] != held_out
    m &= ~(data["is_synthetic"] & (data["source_participant"] == held_out))
    if not use_synthetic:
        m &= ~data["is_synthetic"]
    return m


def fit_score(cand, data, tr_mask, te_mask):
    Xtr = data["X"][tr_mask]
    Xte = data["X"][te_mask]
    if cand["mask20"]:
        Xtr, Xte = mask_col20(Xtr), mask_col20(Xte)
    mdl = cand["builder"]().fit(Xtr, data["y"][tr_mask])
    pred = mdl.predict(Xte)
    return pred, data["y"][te_mask]


def evaluate_all():
    data = load()
    cands = candidates()
    print("candidates: %d   (x %d outer folds x 2 data settings)"
          % (len(cands), len(REAL_PARTICIPANTS)))

    per_cand_rows = []      # diagnostic table
    nested_rows = []        # headline
    fold_class_rows = []
    inner_rows = []

    for use_syn in (False, True):
        data_tag = "REAL_PLUS_SYNTHETIC" if use_syn else "REAL_ONLY"
        nested_preds, nested_true = [], []

        for held in REAL_PARTICIPANTS:
            t_fold = time.time()
            tm = training_mask(data, held, use_syn)
            te = (data["participant"] == held) & ~data["is_synthetic"]
            inner_parts = sorted(set(data["participant"][tm & ~data["is_synthetic"]]))

            inner_scores = {}
            for cand in cands:
                accs = []
                for vp in inner_parts:
                    itr = tm & (data["participant"] != vp)
                    itr &= ~(data["is_synthetic"] & (data["source_participant"] == vp))
                    iva = tm & (data["participant"] == vp) & ~data["is_synthetic"]
                    if itr.sum() == 0 or iva.sum() == 0:
                        continue
                    pred, ytrue = fit_score(cand, data, itr, iva)
                    accs.append(accuracy_score(ytrue, pred))
                inner_scores[cand["id"]] = float(np.mean(accs)) if accs else 0.0
                inner_rows.append({"data": data_tag, "held_out": held,
                                   "candidate": cand["id"],
                                   "inner_mean_acc": inner_scores[cand["id"]],
                                   "inner_folds": json.dumps([round(a, 4) for a in accs])})

            # --- per-candidate outer score (diagnostic only) ---
            outer_pred_cache = {}
            for cand in cands:
                pred, ytrue = fit_score(cand, data, tm, te)
                outer_pred_cache[cand["id"]] = pred
                acc = accuracy_score(ytrue, pred)
                f1 = f1_score(ytrue, pred, average="macro", zero_division=0)
                per_cand_rows.append({
                    "data": data_tag, "held_out": held, "candidate": cand["id"],
                    "family": cand["family"], "mask20": cand["mask20"],
                    "cfg": json.dumps(cand["cfg"], sort_keys=True),
                    "outer_accuracy": acc, "outer_macro_f1": f1,
                    "inner_mean_acc": inner_scores[cand["id"]],
                    "train_frames": int(tm.sum()), "test_frames": int(te.sum()),
                })

            # --- nested selection ---
            best_id = max(inner_scores, key=inner_scores.get)
            best = next(c for c in cands if c["id"] == best_id)
            pred = outer_pred_cache[best_id]
            ytrue = data["y"][te]
            acc = accuracy_score(ytrue, pred)
            f1 = f1_score(ytrue, pred, average="macro", zero_division=0)
            nested_preds.append(pred); nested_true.append(ytrue)
            nested_rows.append({
                "data": data_tag, "held_out": held, "selected_candidate": best_id,
                "selected_family": best["family"], "selected_mask20": best["mask20"],
                "inner_mean_acc": inner_scores[best_id],
                "outer_accuracy": acc, "outer_macro_f1": f1,
                "train_frames": int(tm.sum()), "test_frames": int(te.sum()),
                "seconds": round(time.time() - t_fold, 1),
            })
            pr, rc, f1c, sup = precision_recall_fscore_support(
                ytrue, pred, labels=list(range(6)), zero_division=0)
            for ci, cname in enumerate(CLASS_NAMES):
                fold_class_rows.append({
                    "data": data_tag, "held_out": held, "class": cname,
                    "precision": pr[ci], "recall": rc[ci], "f1": f1c[ci],
                    "support": int(sup[ci]),
                })
            print("  %-20s held-out %s -> nested acc %.4f  (selected %s)  [%.0fs]"
                  % (data_tag, held, acc, best_id, time.time() - t_fold))

        yt = np.concatenate(nested_true); yp = np.concatenate(nested_preds)
        np.save(OUT / ("cm_nested_%s.npy" % data_tag),
                confusion_matrix(yt, yp, labels=list(range(6))))

    pd.DataFrame(per_cand_rows).to_csv(OUT / "per_candidate_per_fold.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "inner_selection_scores.csv", index=False)
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(OUT / "nested_results.csv", index=False)
    pd.DataFrame(fold_class_rows).to_csv(OUT / "nested_per_class.csv", index=False)
    return nested, pd.DataFrame(per_cand_rows)


if __name__ == "__main__":
    print("Phase 4 - nested leave-one-real-participant-out evaluation")
    t0 = time.time()
    nested, per_cand = evaluate_all()

    print("\n=== NESTED (headline, unbiased) ===")
    summ = nested.groupby("data")["outer_accuracy"].agg(["mean", "std", "min", "max"])
    print(summ.round(4).to_string())
    print()
    print(nested[["data", "held_out", "outer_accuracy", "outer_macro_f1",
                  "selected_candidate"]].to_string(index=False))

    print("\n=== per-candidate mean over folds (DIAGNOSTIC ONLY) ===")
    d = (per_cand.groupby(["data", "family", "mask20", "cfg"])
         .agg(outer_mean=("outer_accuracy", "mean"),
              outer_min=("outer_accuracy", "min"),
              inner_mean=("inner_mean_acc", "mean"))
         .reset_index().sort_values("outer_mean", ascending=False))
    print(d.head(20).round(4).to_string(index=False))
    print("\ntotal %.1f min   wrote %s" % ((time.time() - t0) / 60, OUT))
