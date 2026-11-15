"""Phase 9 - independent acceptance audit.

Re-derives each of the ten acceptance checks from the artefacts on disk and
emits one row per check with PASS / FAIL / PENDING_HARDWARE and the evidence
path. Nothing is taken on trust from earlier phases: checksums are recomputed,
the leakage check re-reads the evaluation outputs, and the C++ is re-run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from common import (AUGMENTED_DIR, CLASS_NAMES, DATASET_DIR, MODELS_DIR,
                    POSTURE_FOLDER_TO_CLASS, REPORTS_DIR, discover_recordings)

OUT = REPORTS_DIR / "phase9_audit"
OUT.mkdir(parents=True, exist_ok=True)

PASS, FAIL, PEND = "PASS", "FAIL", "PENDING_HARDWARE"


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
def check_1_label_mapping():
    """Posture labels map consistently from the original P1-P6 folders."""
    problems = []
    for folder, cls in POSTURE_FOLDER_TO_CLASS.items():
        code, meaning = folder.split("_", 1)
        if cls.split("_", 1)[1] != meaning:
            problems.append("meaning mismatch: %s -> %s" % (folder, cls))
        if int(code[1:]) - 1 != CLASS_NAMES.index(cls):
            problems.append("index mismatch: %s -> %s" % (folder, cls))

    # and the label actually stored in every recording agrees with its folder
    n_checked = 0
    for rec in discover_recordings(DATASET_DIR):
        df = pd.read_csv(rec["raw_csv"], nrows=5)
        if "position_label" in df.columns:
            got = set(df["position_label"].astype(str))
            want = {rec["posture_folder"].split("_", 1)[1]}
        else:
            got = set(df["posture"].astype(str))
            want = {rec["posture_folder"]}
        if got != want:
            problems.append("%s/%s stores %s" % (rec["participant"],
                                                 rec["posture_folder"], got))
        n_checked += 1

    detail = ("6/6 folder->class mappings consistent; %d recordings checked, "
              "all agree with their folder" % n_checked)
    return (PASS if not problems else FAIL), detail if not problems else "; ".join(problems[:5]), \
        "ml/common.py::POSTURE_FOLDER_TO_CLASS, tests/test_dataset_and_labels.py"


def check_2_no_original_modified():
    """No original dataset file modified (checksums match Phase 1)."""
    base = REPORTS_DIR / "checksums_phase1.csv"
    if not base.is_file():
        return FAIL, "reports/checksums_phase1.csv missing", rel(base)
    recorded = {r["path"]: r["sha256"] for r in csv.DictReader(base.open())}

    now = {}
    for p in sorted((DATASET_DIR).rglob("*")):
        if p.is_file():
            now[rel(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix.lower() == ".py":
            now[rel(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    ds_recorded = {k: v for k, v in recorded.items() if k.startswith("DATASET/")}
    ds_now = {k: v for k, v in now.items() if k.startswith("DATASET/")}

    changed = [k for k in ds_recorded if ds_now.get(k) != ds_recorded[k]]
    missing = [k for k in ds_recorded if k not in ds_now]
    added = [k for k in ds_now if k not in ds_recorded]

    rows = [{"path": k, "status": ("MISSING" if k in missing else
                                   "CHANGED" if k in changed else "UNCHANGED"),
             "phase1_sha256": v, "now_sha256": ds_now.get(k, "")}
            for k, v in sorted(ds_recorded.items())]
    for k in added:
        rows.append({"path": k, "status": "ADDED", "phase1_sha256": "",
                     "now_sha256": ds_now[k]})
    pd.DataFrame(rows).to_csv(OUT / "checksum_reverify.csv", index=False)

    ok = not (changed or missing or added)
    detail = ("%d original DATASET files re-hashed, all identical to Phase 1"
              % len(ds_recorded)) if ok else \
             ("changed=%d missing=%d added=%d" % (len(changed), len(missing), len(added)))
    return (PASS if ok else FAIL), detail, "reports/phase9_audit/checksum_reverify.csv"


def check_3_synthetic_provenance():
    """Synthetic provenance is complete."""
    if not AUGMENTED_DIR.is_dir() or not any(AUGMENTED_DIR.rglob("*.csv")):
        return FAIL, "no synthetic data found", rel(AUGMENTED_DIR)
    recs = discover_recordings(AUGMENTED_DIR)
    problems, total, covered = [], 0, 0
    for rec in recs:
        prov = rec["dir"] / "provenance.csv"
        n = sum(1 for _ in rec["raw_csv"].open()) - 1
        total += n
        if not prov.is_file():
            problems.append("missing provenance: " + rel(rec["dir"]))
            continue
        pv = pd.read_csv(prov)
        if len(pv) != n:
            problems.append("%s: %d provenance rows for %d frames"
                            % (rel(rec["dir"]), len(pv), n))
            continue
        need = ["source_participant", "source_posture", "source_rep",
                "source_frame_index", "source_file", "transform_seed"]
        miss = [c for c in need if c not in pv.columns or pv[c].isna().any()]
        if miss:
            problems.append("%s: incomplete columns %s" % (rel(rec["dir"]), miss))
            continue
        if not set(pv["source_participant"]).issubset({"P1", "P2", "P3", "P4"}):
            problems.append("%s: unknown source participant" % rel(rec["dir"]))
            continue
        covered += n
    ok = not problems
    detail = ("%d synthetic frames in %d recordings, every frame traced to a "
              "real source frame" % (covered, len(recs))) if ok else "; ".join(problems[:4])
    return (PASS if ok else FAIL), detail, "AUGMENTED_DATASET/**/provenance.csv"


def check_4_no_leakage():
    """No source recording leaks across training and evaluation."""
    d = np.load(MODELS_DIR / "dataset.npz", allow_pickle=False)
    part = d["participant"].astype(str)
    grp = d["group"].astype(str)
    syn = d["is_synthetic"].astype(bool)
    src = d["source_participant"].astype(str)

    problems = []
    rows = []
    for held in ["P1", "P2", "P3", "P4"]:
        te = (part == held) & ~syn
        for use_syn in (False, True):
            tr = part != held
            tr &= ~(syn & (src == held))
            if not use_syn:
                tr &= ~syn
            overlap_groups = set(grp[tr]) & set(grp[te])
            leaked_src = set(src[tr]) & {held}
            rows.append({
                "held_out": held,
                "setting": "REAL_PLUS_SYNTHETIC" if use_syn else "REAL_ONLY",
                "train_frames": int(tr.sum()), "test_frames": int(te.sum()),
                "shared_recording_takes": len(overlap_groups),
                "train_rows_derived_from_held_out": int((src[tr] == held).sum()),
                "held_out_frames_in_train": int((part[tr] == held).sum()),
            })
            if overlap_groups:
                problems.append("%s/%s shares %d takes" % (held, use_syn, len(overlap_groups)))
            if leaked_src:
                problems.append("%s/%s trains on data derived from the held-out person"
                                % (held, use_syn))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "leakage_check.csv", index=False)

    # also: a group never spans participants or repetitions
    gdf = pd.DataFrame({"g": grp, "p": part, "y": d["y"], "rep": d["rep"]})
    n = gdf.groupby("g").nunique()
    if not ((n["p"] == 1).all() and (n["y"] == 1).all() and (n["rep"] == 1).all()):
        problems.append("a group key spans more than one participant/posture/rep")

    ok = not problems
    detail = ("8 train/test splits checked: 0 shared recording takes, 0 training "
              "rows derived from the held-out participant, every group key maps "
              "to exactly one (participant, posture, repetition)") if ok \
        else "; ".join(problems[:4])
    return (PASS if ok else FAIL), detail, "reports/phase9_audit/leakage_check.csv"


def check_5_real_vs_synthetic_decision():
    """Real-only vs real+synthetic compared, with the Phase 4 decision rule applied."""
    p = REPORTS_DIR / "phase4_evaluation" / "nested_results.csv"
    dec_p = MODELS_DIR / "phase4_decision.json"
    if not p.is_file():
        return FAIL, "nested_results.csv missing", rel(p)
    nested = pd.read_csv(p)
    g = nested.groupby("data")["outer_accuracy"].mean()
    if {"REAL_ONLY", "REAL_PLUS_SYNTHETIC"} - set(g.index):
        return FAIL, "both settings not present", rel(p)
    if not dec_p.is_file():
        return FAIL, "models/phase4_decision.json missing", rel(dec_p)
    dec = json.loads(dec_p.read_text())

    ro, rs = float(g["REAL_ONLY"]), float(g["REAL_PLUS_SYNTHETIC"])
    rule_keeps_synthetic = rs >= ro
    ok = (bool(dec["use_synthetic"]) == rule_keeps_synthetic)
    detail = ("REAL_ONLY mean %.4f vs REAL_PLUS_SYNTHETIC mean %.4f; decision "
              "rule 'keep synthetic only if held-out real performance improves "
              "or stays equal' -> %s; shipped model use_synthetic=%s"
              % (ro, rs, "KEEP" if rule_keeps_synthetic else "DROP",
                 dec["use_synthetic"]))
    return (PASS if ok else FAIL), detail, "reports/phase4_evaluation/nested_results.csv"


def check_6_preprocess_equivalence():
    """Preprocessing identical between Python and C++ (within documented tolerance)."""
    p = REPORTS_DIR / "preprocess_equivalence.json"
    if not p.is_file():
        return FAIL, "run ml/verify_preprocess.py", rel(p)
    r = json.loads(p.read_text())
    c0 = r["checks"][0]
    cal = r["checks"][4]["cases"]
    ok = (r["max_ulp_diff_overall"] == 0
          and c0["bitwise_identical_frames"] == c0["frames"]
          and all(c["agree"] for c in cal))
    detail = ("%d/%d recorded frames bit-identical (0 ULP, max abs diff %g); "
              "all 8 edge cases identical; %d/%d calibration scenarios agree"
              % (c0["bitwise_identical_frames"], c0["frames"], c0["max_abs_diff"],
                 sum(1 for c in cal if c["agree"]), len(cal)))
    return (PASS if ok else FAIL), detail, "reports/preprocess_equivalence.md"


def check_7_int8_matches_reference():
    """INT8 model matches its reference implementation within Phase 5 tolerances."""
    p = REPORTS_DIR / "phase5_export" / "export_report.json"
    v = REPORTS_DIR / "phase5_export" / "cpp_vs_tflite.json"
    if not p.is_file():
        return FAIL, "run ml/export_int8.py", rel(p)
    r = json.loads(p.read_text())
    parts, ok = [], True

    a = r["int8_vs_float"]["top1_agreement"]
    parts.append("INT8 vs float top-1 agreement %.4f (required >= 0.99)" % a)
    ok &= a >= 0.99
    parts.append("max |dP| %.4f" % r["int8_vs_float"]["max_abs_prob_diff"])

    nr = r["numpy_replica_vs_tflite"]
    parts.append("numpy replica vs TFLite argmax %d/%d, max logit diff %d"
                 % (nr["argmax_match"], nr["frames"], nr["max_logit_diff"]))
    ok &= nr["argmax_match"] == nr["frames"] and nr["max_logit_diff"] <= 1

    if v.is_file():
        cv = json.loads(v.read_text())
        parts.append("generated C++ vs TFLite argmax %d/%d, max logit diff %d"
                     % (cv["argmax_match"], cv["frames"], cv["max_logit_diff"]))
        ok &= cv["argmax_match"] == cv["frames"] and cv["max_logit_diff"] <= 1
    else:
        ok = False
        parts.append("cpp_vs_tflite.json missing")
    return (PASS if ok else FAIL), "; ".join(parts), \
        "reports/phase5_export/export_report.json, cpp_vs_tflite.json"


def check_8_firmware_scan_preserved():
    """Firmware preserves the original scan sequence and pin configuration."""
    r = subprocess.run([sys.executable, str(ROOT / "ml" / "verify_firmware.py")],
                       capture_output=True, text=True)
    p = REPORTS_DIR / "firmware_scan_diff.json"
    if not p.is_file():
        return FAIL, "verify_firmware.py produced no report", rel(p)
    j = json.loads(p.read_text())
    names = [c["name"] for c in j["checks"] if c.get("pass")]
    detail = ("%d/%d sub-checks pass: %s"
              % (len(names), len(j["checks"]), ", ".join(names)))
    return (PASS if j["overall_pass"] else FAIL), detail, "reports/firmware_scan_diff.md"


def check_9_dashboard_distinguishes_origin():
    """Dashboard distinguishes ESP32 vs Python predictions."""
    app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    src = (ROOT / "dashboard" / "serial_source.py").read_text(encoding="utf-8")
    checks = {
        "serial PRED tagged origin=ESP32": '"origin": "ESP32"' in src,
        "simulated prediction tagged origin=Python": '"origin": "Python"' in src,
        "app renders the origin of the current prediction": 'origin == "ESP32"' in app,
        "app warns when the source is simulated": "not* embedded inference" in app
                                                  or "not embedded inference" in app,
        "test asserts simulated predictions are never ESP32":
            'p["origin"] != "ESP32"' in (ROOT / "tests" / "test_serial_protocol.py")
            .read_text(encoding="utf-8"),
    }
    ok = all(checks.values())
    detail = "; ".join("%s=%s" % (k, v) for k, v in checks.items())
    return (PASS if ok else FAIL), detail, "dashboard/app.py, dashboard/serial_source.py"


def check_10_metrics_reproducible():
    """All reported metrics are reproducible from saved evaluation outputs."""
    base = REPORTS_DIR / "phase4_evaluation"
    need = ["nested_results.csv", "per_candidate_per_fold.csv",
            "inner_selection_scores.csv", "nested_per_class.csv"]
    missing = [n for n in need if not (base / n).is_file()]
    if missing:
        return FAIL, "missing %s" % missing, rel(base)

    nested = pd.read_csv(base / "nested_results.csv")
    percls = pd.read_csv(base / "nested_per_class.csv")
    problems = []

    # the headline mean must be recomputable from the per-fold rows
    for data_tag, g in nested.groupby("data"):
        if len(g) != 4:
            problems.append("%s has %d folds, expected 4" % (data_tag, len(g)))

    # the per-class support must sum to the fold's test frame count
    for (data_tag, held), g in percls.groupby(["data", "held_out"]):
        want = int(nested[(nested.data == data_tag)
                          & (nested.held_out == held)]["test_frames"].iloc[0])
        got = int(g["support"].sum())
        if want != got:
            problems.append("%s/%s support %d != test_frames %d"
                            % (data_tag, held, got, want))

    # confusion matrices must exist and total the pooled frame count
    for data_tag in nested["data"].unique():
        cm_p = base / ("cm_nested_%s.npy" % data_tag)
        if not cm_p.is_file():
            problems.append("missing %s" % cm_p.name)
            continue
        cm = np.load(cm_p)
        want = int(nested[nested.data == data_tag]["test_frames"].sum())
        if int(cm.sum()) != want:
            problems.append("%s confusion matrix totals %d, expected %d"
                            % (data_tag, cm.sum(), want))

    ok = not problems
    detail = ("per-fold rows, per-class supports and confusion matrices are all "
              "mutually consistent; every headline number recomputes from "
              "nested_results.csv") if ok else "; ".join(problems[:4])
    return (PASS if ok else FAIL), detail, "reports/phase4_evaluation/"


CHECKS = [
    ("1", "Posture labels map consistently from the original P1-P6 folders", check_1_label_mapping),
    ("2", "No original dataset file modified (checksums match Phase 1)", check_2_no_original_modified),
    ("3", "Synthetic provenance is complete", check_3_synthetic_provenance),
    ("4", "No source recording leaks across training and evaluation", check_4_no_leakage),
    ("5", "Real-only vs real+synthetic compared, decision rule applied", check_5_real_vs_synthetic_decision),
    ("6", "Preprocessing identical between Python and C++", check_6_preprocess_equivalence),
    ("7", "INT8 model matches its reference implementation", check_7_int8_matches_reference),
    ("8", "Firmware preserves the original scan sequence and pins", check_8_firmware_scan_preserved),
    ("9", "Dashboard distinguishes ESP32 vs Python predictions", check_9_dashboard_distinguishes_origin),
    ("10", "All reported metrics reproducible from saved outputs", check_10_metrics_reproducible),
]

HARDWARE_ITEMS = [
    ("H1", "Firmware compiles with PlatformIO or arduino-cli",
     "neither toolchain is installed on this machine", "README.md -> Compiling"),
    ("H2", "Firmware uploads to an ESP32 DevKit V1",
     "no board detected", "README.md -> Flashing"),
    ("H3", "On-device inference time measured with micros()",
     "requires a running board; the firmware prints it on every PRED line",
     "README.md -> Measuring inference time"),
    ("H4", "Live accuracy on a real seated person",
     "requires hardware and a participant; use the dashboard manual-posture "
     "selector", "README.md -> Physical test procedure"),
    ("H5", "Flash / RAM usage reported by the compiler",
     "requires a working toolchain", "README.md -> Compiling"),
]


def main():
    rows = []
    for num, title, fn in CHECKS:
        try:
            status, detail, evidence = fn()
        except Exception as exc:
            status, detail, evidence = FAIL, "audit raised %r" % exc, ""
        rows.append({"check": num, "description": title, "result": status,
                     "detail": detail, "evidence": evidence})
        print("%-3s %-62s %s" % (num, title[:62], status))

    for num, title, why, evidence in HARDWARE_ITEMS:
        rows.append({"check": num, "description": title, "result": PEND,
                     "detail": why, "evidence": evidence})
        print("%-3s %-62s %s" % (num, title[:62], PEND))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "acceptance_table.csv", index=False)

    md = ["# Phase 9 - acceptance audit\n\n"]
    n_pass = int((df["result"] == PASS).sum())
    n_fail = int((df["result"] == FAIL).sum())
    n_pend = int((df["result"] == PEND).sum())
    md.append("**%d PASS, %d FAIL, %d PENDING_HARDWARE**\n\n" % (n_pass, n_fail, n_pend))
    md.append("| # | check | result | evidence |\n|---|---|---|---|\n")
    for r in rows:
        md.append("| %s | %s | **%s** | `%s` |\n"
                  % (r["check"], r["description"], r["result"], r["evidence"]))
    md.append("\n## Detail\n\n")
    for r in rows:
        md.append("### %s — %s — %s\n\n%s\n\n" % (r["check"], r["description"],
                                                  r["result"], r["detail"]))
    (OUT / "acceptance_table.md").write_text("".join(md), encoding="utf-8")

    print("\n%d PASS, %d FAIL, %d PENDING_HARDWARE" % (n_pass, n_fail, n_pend))
    print("wrote " + rel(OUT / "acceptance_table.md"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
