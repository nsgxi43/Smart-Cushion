"""Phase 4 decision - pick what actually ships, and write down why.

Two decisions, both made from inner-validation scores only.

1. Synthetic augmentation. The rule stated in the spec is applied verbatim:
   keep synthetic data only if held-out real performance improves or stays
   equal, otherwise ship the real-only model. This one rule is deliberately
   evaluated on the held-out folds because that is what the spec asks for, and
   it is a single binary decision, not a search over a large space.

2. Architecture. The shipped model must be exportable to INT8 TFLite, so the
   choice is among the Keras candidates. It is made on the **inner** LOPO score
   averaged over the four outer folds - never on the held-out score. Among
   architectures within one standard error of the best inner score, the
   smallest (fewest parameters) wins, since size, RAM and latency all scale
   with it.

The held-out numbers for the chosen architecture are then *reported*, not used
to choose it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, MODELS_DIR, N_SENSORS, REPORTS_DIR

EVAL = REPORTS_DIR / "phase4_evaluation"
OUT_JSON = MODELS_DIR / "phase4_decision.json"

ARCH_HIDDEN = {
    "keras_504_64_32_6": [64, 32],
    "keras_504_32_16_6": [32, 16],
    "keras_504_24_6":    [24],
    "keras_504_16_6":    [16],
}


def n_params(hidden):
    dims = [N_SENSORS] + list(hidden) + [len(CLASS_NAMES)]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def main():
    nested = pd.read_csv(EVAL / "nested_results.csv")
    per_cand = pd.read_csv(EVAL / "per_candidate_per_fold.csv")

    # ---------------- decision 1: synthetic or not ----------------------
    means = nested.groupby("data")["outer_accuracy"].mean()
    ro = float(means["REAL_ONLY"])
    rs = float(means["REAL_PLUS_SYNTHETIC"])
    use_syn = rs >= ro
    print("REAL_ONLY mean held-out accuracy          : %.4f" % ro)
    print("REAL_PLUS_SYNTHETIC mean held-out accuracy: %.4f" % rs)
    print("decision rule -> %s synthetic augmentation"
          % ("KEEP" if use_syn else "DROP"))

    data_tag = "REAL_PLUS_SYNTHETIC" if use_syn else "REAL_ONLY"

    # ---------------- decision 2: architecture --------------------------
    k = per_cand[(per_cand["data"] == data_tag)
                 & (per_cand["family"].str.startswith("keras_"))].copy()
    if k.empty:
        raise SystemExit("no keras candidates in the evaluation output")

    # average the INNER score across outer folds; this never touches held-out
    g = (k.groupby(["family", "mask20"])
         .agg(inner_mean=("inner_mean_acc", "mean"),
              inner_sd=("inner_mean_acc", "std"),
              outer_mean=("outer_accuracy", "mean"),
              outer_min=("outer_accuracy", "min"),
              outer_sd=("outer_accuracy", "std"),
              folds=("outer_accuracy", "size"))
         .reset_index())
    g["params"] = g["family"].map(lambda f: n_params(ARCH_HIDDEN[f]))
    g = g.sort_values("inner_mean", ascending=False)

    best_inner = float(g["inner_mean"].iloc[0])
    se = float(g["inner_mean"].std() / max(np.sqrt(len(g)), 1)) if len(g) > 1 else 0.0
    within = g[g["inner_mean"] >= best_inner - se]
    chosen = within.sort_values(["params", "inner_mean"],
                                ascending=[True, False]).iloc[0]

    print("\narchitecture candidates (ranked by INNER validation score):")
    print(g[["family", "mask20", "inner_mean", "outer_mean", "outer_min",
             "params"]].round(4).to_string(index=False))
    print("\nbest inner %.4f, 1 SE = %.4f, %d within 1 SE -> smallest wins"
          % (best_inner, se, len(within)))
    print("CHOSEN: %s  mask_col20=%s  (%d parameters)"
          % (chosen["family"], chosen["mask20"], chosen["params"]))

    fold_rows = nested[nested["data"] == data_tag][
        ["held_out", "outer_accuracy", "outer_macro_f1", "selected_candidate"]]

    # held-out performance OF THE CHOSEN ARCHITECTURE (reported, not selected on)
    chosen_folds = k[(k["family"] == chosen["family"])
                     & (k["mask20"] == chosen["mask20"])][
        ["held_out", "outer_accuracy", "outer_macro_f1"]].sort_values("held_out")

    decision = {
        "decision_date": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "version": "1.0.0",
        "use_synthetic": bool(use_syn),
        "synthetic_decision_rule":
            "keep synthetic only if held-out real performance improves or stays equal",
        "nested_mean_real_only": ro,
        "nested_mean_real_plus_synthetic": rs,
        "training_data_setting": data_tag,
        "architecture_family": chosen["family"],
        "hidden": ARCH_HIDDEN[chosen["family"]],
        "mask_col20": bool(chosen["mask20"]),
        "dropout": 0.3,
        "l2": 1e-4,
        "epochs": 120,
        "batch": 64,
        "architecture_selection_rule":
            "best mean INNER leave-one-participant-out score across the 4 outer "
            "folds; ties within 1 SE broken by fewest parameters. The held-out "
            "participant was never used to choose the architecture.",
        "chosen_inner_mean_acc": float(chosen["inner_mean"]),
        "chosen_parameters": int(chosen["params"]),
        "chosen_heldout_per_fold": chosen_folds.to_dict("records"),
        "chosen_heldout_mean_acc": float(chosen_folds["outer_accuracy"].mean()),
        "chosen_heldout_min_acc": float(chosen_folds["outer_accuracy"].min()),
        "chosen_heldout_macro_f1": float(chosen_folds["outer_macro_f1"].mean()),
        "nested_protocol_per_fold": fold_rows.to_dict("records"),
        "nested_protocol_mean_acc": float(fold_rows["outer_accuracy"].mean()),
        "chance_level": 1.0 / len(CLASS_NAMES),
        "heldout_real_accuracy_human": "%.1f%% mean over 4 leave-one-participant-out folds (chance 16.7%%)"
                                       % (100 * float(chosen_folds["outer_accuracy"].mean())),
        "caveat": ("Four real participants only. Fold-to-fold spread is large; "
                   "treat the mean as indicative, not as a measurement. "
                   "Synthetic participants add sensor-variation robustness, not "
                   "new body types."),
    }
    OUT_JSON.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    g.round(5).to_csv(EVAL / "architecture_selection.csv", index=False)

    print("\nchosen architecture held-out per fold:")
    print(chosen_folds.round(4).to_string(index=False))
    print("mean %.4f, min %.4f"
          % (decision["chosen_heldout_mean_acc"], decision["chosen_heldout_min_acc"]))
    print("\nwrote " + str(OUT_JSON))


if __name__ == "__main__":
    main()
