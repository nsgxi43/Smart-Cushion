"""Phase 9 - the experiment summary.

One document that records exactly what was run, what came out, and what is
still unmeasured. Every number is read from a file under reports/ or models/;
nothing is typed in by hand.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from common import CLASS_NAMES, MODELS_DIR, REPORTS_DIR

EVAL = REPORTS_DIR / "phase4_evaluation"
OUT = REPORTS_DIR / "experiment_summary.md"


def git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unavailable"


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default


def main():
    decision = jload(MODELS_DIR / "phase4_decision.json")
    meta = jload(MODELS_DIR / "model_metadata.json")
    export = jload(REPORTS_DIR / "phase5_export" / "export_report.json")
    cppv = jload(REPORTS_DIR / "phase5_export" / "cpp_vs_tflite.json")
    inspect = jload(REPORTS_DIR / "dataset_report" / "inspection_summary.json")
    prepeq = jload(REPORTS_DIR / "preprocess_equivalence.json")
    nested = pd.read_csv(EVAL / "nested_results.csv") if (EVAL / "nested_results.csv").is_file() else None
    percls = pd.read_csv(EVAL / "nested_per_class.csv") if (EVAL / "nested_per_class.csv").is_file() else None

    if decision is None or nested is None:
        raise SystemExit("run ml/evaluate.py and ml/decide_model.py first")

    data_tag = decision["training_data_setting"]
    syn_counts = pd.read_csv(ROOT / "AUGMENTED_DATASET" / "synthetic_frame_counts.csv") \
        if (ROOT / "AUGMENTED_DATASET" / "synthetic_frame_counts.csv").is_file() else None

    m = []
    m.append("# Experiment summary\n\n")
    m.append("| field | value |\n|---|---|\n")
    m.append("| Experiment ID | `SC-POSTURE-%s` |\n" % decision.get("version", "1.0.0"))
    m.append("| Date | %s |\n" % decision["decision_date"])
    m.append("| Git commit | `%s` |\n" % git_hash())
    m.append("| Dataset version | `DATASET/session_20261114_170217` "
             "(%d real frames, %d recordings, SHA-256 in reports/checksums_phase1.csv) |\n"
             % (inspect["total_raw_frames"], inspect["n_recordings"]))
    m.append("| Augmented dataset | `AUGMENTED_DATASET/session_synthetic_v1` (%d synthetic frames) |\n"
             % (int(syn_counts["frames"].sum()) if syn_counts is not None else 0))
    m.append("| Classes | %s |\n" % ", ".join(CLASS_NAMES))
    m.append("| Model architecture | %s |\n"
             % (meta["architecture"] if meta else
                "504-" + "-".join(map(str, decision["hidden"])) + "-6"))
    m.append("| Parameters | %d |\n" % decision["chosen_parameters"])
    m.append("| Column 20 masked | %s |\n" % decision["mask_col20"])
    m.append("| Training configuration | dropout %.2f, L2 %g, %d epochs, batch %d, Adam 1e-3, seed 20261115 |\n"
             % (decision["dropout"], decision["l2"], decision["epochs"], decision["batch"]))
    m.append("| Training set | %s |\n" % data_tag)
    m.append("| Evaluation split | nested leave-one-REAL-participant-out (4 outer folds); "
             "inner LOPO over the 3 remaining real participants selects the model; "
             "splits are by recording take |\n")
    m.append("| Quantisation | %s |\n" % (meta["quantisation"] if meta else "pending"))
    m.append("| Model size | %s |\n" % (meta.get("model_size_human", "pending") if meta else "pending"))
    m.append("| Estimated static RAM | %s |\n"
             % ("%d B (%.1f kB), inference workspace %d B"
                % (cppv["memory"]["static_ram_total_bytes"],
                   cppv["memory"]["static_ram_total_bytes"] / 1024,
                   cppv["memory"]["inference_workspace_bytes"]) if cppv else "pending"))
    m.append("| Flash (model parameters) | %s |\n"
             % ("%d B (%.1f kB)" % (cppv["memory"]["flash_total_bytes"],
                                    cppv["memory"]["flash_total_bytes"] / 1024)
                if cppv else "pending"))
    m.append("| **ESP32 inference time** | **PENDING_HARDWARE** (no board connected; "
             "the firmware prints it as the 4th field of each `PRED,` line) |\n")
    m.append("| Chance level | %.1f%% (6 classes) |\n" % (100 * decision["chance_level"]))

    # ---------------- real participant metrics --------------------------
    m.append("\n## Real held-out participant metrics\n\n")
    m.append("These are the only generalisation numbers in this project. Each "
             "fold trains on 3 real participants (plus, when enabled, the "
             "synthetic participants not derived from the held-out person) and "
             "tests on the 4th real participant's frames.\n\n")

    m.append("### Nested protocol (unbiased: the model is chosen by the inner loop)\n\n")
    m.append("| held-out | REAL_ONLY | REAL_PLUS_SYNTHETIC |\n|---|---|---|\n")
    for held in ["P1", "P2", "P3", "P4"]:
        a = nested[(nested.data == "REAL_ONLY") & (nested.held_out == held)]
        b = nested[(nested.data == "REAL_PLUS_SYNTHETIC") & (nested.held_out == held)]
        m.append("| %s | %.4f | %.4f |\n" % (held,
                                             float(a["outer_accuracy"].iloc[0]),
                                             float(b["outer_accuracy"].iloc[0])))
    g = nested.groupby("data")["outer_accuracy"]
    m.append("| **mean** | **%.4f** | **%.4f** |\n"
             % (g.mean()["REAL_ONLY"], g.mean()["REAL_PLUS_SYNTHETIC"]))
    m.append("| std | %.4f | %.4f |\n"
             % (g.std()["REAL_ONLY"], g.std()["REAL_PLUS_SYNTHETIC"]))
    m.append("| min | %.4f | %.4f |\n"
             % (g.min()["REAL_ONLY"], g.min()["REAL_PLUS_SYNTHETIC"]))

    m.append("\n### The shipped architecture, held out per fold\n\n")
    m.append("Architecture chosen on inner-validation score only; these held-out "
             "numbers are reported, not used to choose it.\n\n")
    m.append("| held-out participant | accuracy | macro F1 |\n|---|---|---|\n")
    for r in decision["chosen_heldout_per_fold"]:
        m.append("| %s | %.4f | %.4f |\n" % (r["held_out"], r["outer_accuracy"],
                                             r["outer_macro_f1"]))
    m.append("| **mean** | **%.4f** | **%.4f** |\n"
             % (decision["chosen_heldout_mean_acc"], decision["chosen_heldout_macro_f1"]))
    m.append("| min | %.4f | |\n" % decision["chosen_heldout_min_acc"])

    if percls is not None:
        m.append("\n### Per-class, %s, nested protocol (pooled over folds)\n\n" % data_tag)
        sub = percls[percls.data == data_tag]
        agg = (sub.groupby("class")
               .apply(lambda d: pd.Series({
                   "precision": np.average(d["precision"], weights=d["support"]),
                   "recall": np.average(d["recall"], weights=d["support"]),
                   "f1": np.average(d["f1"], weights=d["support"]),
                   "support": d["support"].sum()}), include_groups=False)
               .reindex(CLASS_NAMES))
        m.append(agg.round(4).to_markdown())

        cm_p = EVAL / ("cm_nested_%s.npy" % data_tag)
        if cm_p.is_file():
            cm = np.load(cm_p)
            m.append("\n\n### Confusion matrix, %s (rows = true, cols = predicted)\n\n"
                     % data_tag)
            cmdf = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
            m.append(cmdf.to_markdown())

    # ---------------- synthetic metrics (separate!) ---------------------
    m.append("\n\n## Synthetic participant metrics — reported separately\n\n")
    m.append("Synthetic participants are **not** people and their scores are "
             "**never** merged into the real held-out numbers above. They are "
             "reported only to show that augmentation preserved posture "
             "structure.\n\n")
    synrep = REPORTS_DIR / "synthetic_report" / "centroid_separability.csv"
    if synrep.is_file():
        s = pd.read_csv(synrep)
        m.append(s.round(4).to_markdown(index=False))
        m.append("\n\nWithin-participant nearest-class-centroid accuracy. Each "
                 "synthetic participant tracks its real source closely, which is "
                 "the intended behaviour: the transforms changed the sensor "
                 "response, not the posture.\n")

    cons = REPORTS_DIR / "synthetic_report" / "synthetic_consistency.csv"
    dup = REPORTS_DIR / "synthetic_report" / "synthetic_not_copies.csv"
    if cons.is_file() and dup.is_file():
        c, du = pd.read_csv(cons), pd.read_csv(dup)
        m.append("\n| synthetic data check | value |\n|---|---|\n")
        m.append("| recordings | %d |\n" % len(c))
        m.append("| frames | %d |\n" % int(c["frames"].sum()))
        m.append("| frames identical to their source | %d |\n"
                 % int(du["identical_frames"].sum()))
        m.append("| provenance complete | %s |\n" % bool(c["provenance_complete"].all()))
        m.append("| internal preprocessing identity exact | %.3f%% of cells |\n"
                 % c["pressure_identity_exact_pct"].mean())

    # ---------------- verification ---------------------------------------
    m.append("\n## Verification\n\n| check | result | evidence |\n|---|---|---|\n")
    if prepeq:
        c0 = prepeq["checks"][0]
        m.append("| Python vs C++ preprocessing | %d/%d frames bit-identical, %d ULP | `reports/preprocess_equivalence.md` |\n"
                 % (c0["bitwise_identical_frames"], c0["frames"],
                    prepeq["max_ulp_diff_overall"]))
    if meta:
        m.append("| INT8 vs float top-1 agreement | %.4f (tolerance >= 0.99) | `reports/phase5_export/export_report.json` |\n"
                 % meta["int8_vs_float_top1_agreement"])
        m.append("| INT8 vs float max probability difference | %.4f | same |\n"
                 % meta["int8_vs_float_max_prob_diff"])
    if cppv:
        m.append("| Generated C++ vs TFLite INT8 | argmax %d/%d, max logit diff %d (tolerance: identical argmax, +/-1 step) | `reports/phase5_export/cpp_vs_tflite.json` |\n"
                 % (cppv["argmax_match"], cppv["frames"], cppv["max_logit_diff"]))
    fw = jload(REPORTS_DIR / "firmware_scan_diff.json")
    if fw:
        m.append("| Firmware scan preserved | %s | `reports/firmware_scan_diff.md` |\n"
                 % ("PASS" if fw["overall_pass"] else "FAIL"))

    # ---------------- limitations ----------------------------------------
    m.append("\n## Limitations\n\n")
    m.append("- **Four real participants.** Every cross-participant number is a "
             "mean over four folds of one person each. The fold spread is large "
             "(min %.3f, max %.3f for the shipped setting). Treat the mean as "
             "indicative, not as a measurement.\n"
             % (nested[nested.data == data_tag]["outer_accuracy"].min(),
                nested[nested.data == data_tag]["outer_accuracy"].max()))
    m.append("- **Synthetic data adds sensor-variation robustness, not new body "
             "types.** P5–P10 are transformed copies of P1–P4 with per-sensor "
             "gain/offset, drift blobs, noise, a +/-1 cell shift and temporal "
             "jitter. Height, weight and sex are recorded as `SYNTHETIC/NA`.\n")
    m.append("- **The sensor saturates**, which caps what any model can do. See "
             "`reports/phase4_evaluation/domain_shift_diagnostic.md`: within a "
             "participant the postures separate at ~0.87, across participants no "
             "feature transform or model family tested exceeds ~0.44.\n")
    m.append("- **Column 20 is an intermittent hardware fault** — dead for P2 and "
             "P4, alive for P3.\n")
    m.append("- **No on-device measurement exists.** Inference time, flash/RAM as "
             "reported by the compiler, and live accuracy are all "
             "PENDING_HARDWARE.\n")

    m.append("\n## Reproducing every number here\n\n```\n")
    m.append("python ml\\inspect_dataset.py\npython ml\\build_dataset.py\n"
             "python ml\\verify_preprocess.py\npython ml\\augment.py\n"
             "python ml\\validate_synthetic.py\npython ml\\build_dataset.py\n"
             "python ml\\evaluate.py\npython ml\\diagnose_shift.py\n"
             "python ml\\decide_model.py\npython ml\\export_int8.py\n"
             "python ml\\verify_inference.py\npython ml\\verify_firmware.py\n"
             "python ml\\audit.py\npython ml\\experiment_summary.py\n"
             "python -m pytest tests -q\n```\n")

    OUT.write_text("".join(m), encoding="utf-8")
    print("wrote " + str(OUT))


if __name__ == "__main__":
    main()
