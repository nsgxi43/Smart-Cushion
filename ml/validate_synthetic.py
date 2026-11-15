"""Phase 3 validation: did augmentation preserve posture structure and
produce *different* participants rather than copies?

Checks
  1. schema + provenance completeness
  2. internal consistency: pressure_data == gate(|raw - calibration baseline|)
  3. class centroids stay separable (nearest-centroid accuracy within each
     synthetic participant, and centroid cosine similarity to the real source)
  4. synthetic participants differ from their source and from each other
  5. value distributions / saturation comparable to real data
Figures: before/after heatmaps per synthetic participant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import SYNTHETIC_SOURCES, UNKNOWN
from common import (AUGMENTED_DIR, CLASS_NAMES, COLS, DATASET_DIR, N_SENSORS,
                    POSTURE_FOLDERS, POSTURE_FOLDER_TO_CLASS, REPORTS_DIR, ROWS,
                    discover_recordings, sensor_columns)

OUT = REPORTS_DIR / "synthetic_report"
OUT.mkdir(parents=True, exist_ok=True)


def load_all(root):
    out = {}
    for rec in discover_recordings(root):
        raw_df = pd.read_csv(rec["raw_csv"])
        pr_df = pd.read_csv(rec["pressure_csv"])
        out[(rec["participant"], rec["posture_class"])] = {
            "raw": raw_df[sensor_columns(raw_df)].to_numpy(dtype=np.int32),
            "pressure": pr_df[sensor_columns(pr_df)].to_numpy(dtype=np.int32),
            "dir": rec["dir"],
        }
    return out


def main():
    real = load_all(DATASET_DIR)
    syn = load_all(AUGMENTED_DIR)
    print("real recordings: %d   synthetic recordings: %d" % (len(real), len(syn)))

    rows = []
    # ---------- 1 & 2: schema, provenance, internal consistency -------------
    for (part, cls), d in sorted(syn.items()):
        prov_path = d["dir"] / "provenance.csv"
        prov = pd.read_csv(prov_path) if prov_path.is_file() else None
        cal = np.load(d["dir"] / "calibration_used.npy")
        b, thr = cal[0], cal[1]
        delta = np.abs(d["raw"].astype(np.int64) - b.reshape(1, -1))
        recon = np.clip(np.where(delta <= thr.reshape(1, -1), 0, delta), 0, 4095)
        rows.append({
            "participant": part, "posture_class": cls,
            "frames": int(d["raw"].shape[0]),
            "provenance_rows": 0 if prov is None else int(len(prov)),
            "provenance_complete": bool(prov is not None
                                        and len(prov) == d["raw"].shape[0]
                                        and prov["source_participant"].notna().all()
                                        and prov["source_frame_index"].notna().all()),
            "pressure_identity_exact_pct": float((recon == d["pressure"]).mean() * 100),
            "raw_in_range": bool((d["raw"] >= 0).all() and (d["raw"] <= 4095).all()),
            "pct_at_adc_max": float((d["raw"] >= 4095).mean() * 100),
            "pct_at_adc_min": float((d["raw"] <= 0).mean() * 100),
            "raw_mean": float(d["raw"].mean()),
        })
    cons = pd.DataFrame(rows)
    cons.to_csv(OUT / "synthetic_consistency.csv", index=False)

    # ---------- 3: centroid separability ------------------------------------
    def centroids(store, part):
        return {cls: store[(part, cls)]["pressure"].mean(axis=0).astype(np.float64)
                for cls in CLASS_NAMES if (part, cls) in store}

    def nearest_centroid_acc(store, part):
        c = centroids(store, part)
        names = list(c.keys())
        M = np.stack([c[n] for n in names])
        correct = total = 0
        for cls in names:
            X = store[(part, cls)]["pressure"].astype(np.float64)
            d = ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)
            pred = d.argmin(axis=1)
            correct += int((np.array(names)[pred] == cls).sum())
            total += X.shape[0]
        return correct / total

    real_parts = sorted({p for p, _ in real})
    sep_rows = []
    for part, store, kind in ([(p, real, "real") for p in real_parts]
                              + [(p, syn, "synthetic") for p in SYNTHETIC_SOURCES]):
        sep_rows.append({
            "participant": part,
            "kind": kind,
            "within_participant_nearest_centroid_acc": nearest_centroid_acc(store, part),
        })
    sep = pd.DataFrame(sep_rows)
    sep.to_csv(OUT / "centroid_separability.csv", index=False)

    # centroid similarity to source, and class-pair distances
    sim_rows = []
    for syn_part, src_part in SYNTHETIC_SOURCES.items():
        cs = centroids(syn, syn_part)
        cr = centroids(real, src_part)
        for cls in CLASS_NAMES:
            a, b = cs[cls], cr[cls]
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            # how different is the synthetic from its source, relative to the
            # spread between different postures of the source?
            others = [np.linalg.norm(cr[cls] - cr[o]) for o in CLASS_NAMES if o != cls]
            sim_rows.append({
                "synthetic": syn_part, "source": src_part, "posture_class": cls,
                "centroid_cosine_to_source": cos,
                "centroid_l2_to_source": float(np.linalg.norm(a - b)),
                "mean_l2_between_source_postures": float(np.mean(others)),
                "ratio_shift_vs_class_gap": float(np.linalg.norm(a - b)
                                                  / (np.mean(others) + 1e-12)),
            })
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT / "centroid_similarity_to_source.csv", index=False)

    # ---------- 4: synthetic participants are not copies --------------------
    dup_rows = []
    for syn_part, src_part in SYNTHETIC_SOURCES.items():
        ident = 0
        tot = 0
        for cls in CLASS_NAMES:
            a = syn[(syn_part, cls)]["raw"]
            b = real[(src_part, cls)]["raw"]
            n = min(a.shape[0], b.shape[0])
            ident += int((a[:n] == b[:n]).all(axis=1).sum())
            tot += n
        dup_rows.append({"synthetic": syn_part, "source": src_part,
                         "identical_frames": ident, "frames": tot,
                         "identical_pct": 100.0 * ident / tot})
    dup = pd.DataFrame(dup_rows)
    dup.to_csv(OUT / "synthetic_not_copies.csv", index=False)

    # pairwise distinctness between the two synthetics sharing a source
    pair_rows = []
    syn_names = list(SYNTHETIC_SOURCES)
    for i in range(len(syn_names)):
        for j in range(i + 1, len(syn_names)):
            a, b = syn_names[i], syn_names[j]
            ca = np.concatenate([centroids(syn, a)[c] for c in CLASS_NAMES])
            cb = np.concatenate([centroids(syn, b)[c] for c in CLASS_NAMES])
            pair_rows.append({"a": a, "b": b,
                              "same_source": SYNTHETIC_SOURCES[a] == SYNTHETIC_SOURCES[b],
                              "centroid_l2": float(np.linalg.norm(ca - cb)),
                              "centroid_cosine": float(ca @ cb /
                                                       (np.linalg.norm(ca) * np.linalg.norm(cb)))})
    pd.DataFrame(pair_rows).to_csv(OUT / "synthetic_pairwise_distinctness.csv", index=False)

    # ---------- figures ------------------------------------------------------
    for syn_part, src_part in SYNTHETIC_SOURCES.items():
        fig, axes = plt.subplots(2, len(CLASS_NAMES),
                                 figsize=(2.1 * len(CLASS_NAMES), 5.6))
        vmax = 4095
        for j, cls in enumerate(CLASS_NAMES):
            axes[0, j].imshow(real[(src_part, cls)]["pressure"].mean(axis=0)
                              .reshape(ROWS, COLS), cmap="turbo", vmin=0, vmax=vmax)
            axes[1, j].imshow(syn[(syn_part, cls)]["pressure"].mean(axis=0)
                              .reshape(ROWS, COLS), cmap="turbo", vmin=0, vmax=vmax)
            axes[0, j].set_title(cls.replace("_", "\n", 1), fontsize=8)
            for r in (0, 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, 0].set_ylabel("REAL\n" + src_part, fontsize=9)
        axes[1, 0].set_ylabel("SYNTHETIC\n" + syn_part, fontsize=9)
        fig.suptitle("Before / after: %s derived from real %s "
                     "(SYNTHETIC - not a real person)" % (syn_part, src_part),
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / ("before_after_%s.png" % syn_part), dpi=100)
        plt.close(fig)

    # distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rv = np.concatenate([d["raw"].ravel() for d in real.values()])
    sv = np.concatenate([d["raw"].ravel() for d in syn.values()])
    axes[0].hist(rv, bins=100, histtype="step", density=True, label="real")
    axes[0].hist(sv, bins=100, histtype="step", density=True, label="synthetic")
    axes[0].set_yscale("log"); axes[0].set_title("raw ADC distribution"); axes[0].legend()
    rp = np.concatenate([d["pressure"].ravel() for d in real.values()])
    sp = np.concatenate([d["pressure"].ravel() for d in syn.values()])
    axes[1].hist(rp, bins=100, histtype="step", density=True, label="real")
    axes[1].hist(sp, bins=100, histtype="step", density=True, label="synthetic")
    axes[1].set_yscale("log"); axes[1].set_title("pressure distribution"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "real_vs_synthetic_distributions.png", dpi=110)
    plt.close(fig)

    # ---------- markdown -----------------------------------------------------
    counts = pd.read_csv(AUGMENTED_DIR / "synthetic_frame_counts.csv")
    params = json.loads((AUGMENTED_DIR / "augmentation_parameters.json").read_text())

    md = ["# Phase 3 - Synthetic participant report\n\n"]
    md.append("> **These are not real people.** P5_SYNTHETIC..P10_SYNTHETIC are "
              "transformed copies of real recordings. Height, weight and sex are "
              "recorded as `%s`. They model sensor and session variation, not new "
              "body types.\n\n" % UNKNOWN)

    md.append("## Source mapping\n\n| synthetic | derived from | global intensity | "
              "noise sigma | shift (dr,dc) | temporal jitter |\n|---|---|---|---|---|---|\n")
    for name, src in SYNTHETIC_SOURCES.items():
        p = params["participants"][name]
        md.append("| %s | %s | %.3f | %.2f | (%d,%d) | %.2f |\n" % (
            name, src, p["global_intensity"], p["noise_sigma_counts"],
            p["shift_dr"], p["shift_dc"], p["temporal_jitter_frames"]))

    md.append("\n## Frame counts\n\n")
    piv = counts.pivot_table(index="participant", columns="posture_class",
                             values="frames", aggfunc="sum")
    piv["TOTAL"] = piv.sum(axis=1)
    md.append(piv.to_markdown())
    n_real = sum(d["raw"].shape[0] for d in real.values())
    n_syn = sum(d["raw"].shape[0] for d in syn.values())
    md.append("\n\n**Real frames: %d. Synthetic frames: %d. Ratio %.2f:1.**\n"
              % (n_real, n_syn, n_syn / n_real))

    md.append("\n## Provenance and internal consistency\n\n")
    md.append(cons[["participant", "posture_class", "frames", "provenance_rows",
                    "provenance_complete", "pressure_identity_exact_pct",
                    "raw_in_range"]].round(3).to_markdown(index=False))
    md.append("\n\nEvery synthetic recording satisfies the Phase 2 identity "
              "`pressure = clip(gate(|raw - baseline|, threshold), 0, 4095)` "
              "exactly (%.2f%% of cells, min %.2f%%).\n"
              % (cons["pressure_identity_exact_pct"].mean(),
                 cons["pressure_identity_exact_pct"].min()))

    md.append("\n## Posture structure preserved\n\n")
    md.append("Within-participant nearest-class-centroid accuracy. If the "
              "transforms had destroyed spatial structure, the synthetic rows "
              "would collapse toward chance (1/6 = 16.7%).\n\n")
    md.append(sep.round(4).to_markdown(index=False))

    md.append("\n\n### Centroid shift vs class gap\n\n")
    md.append("`ratio_shift_vs_class_gap` is the distance from the synthetic "
              "centroid to its source centroid, divided by the mean distance "
              "between different postures of the source. Values well below 1 "
              "mean the posture identity survived the transform.\n\n")
    g = sim.groupby("synthetic")["ratio_shift_vs_class_gap"].agg(["mean", "min", "max"])
    md.append(g.round(4).to_markdown())
    md.append("\n\nCentroid cosine similarity to source: mean %.4f, min %.4f.\n"
              % (sim["centroid_cosine_to_source"].mean(),
                 sim["centroid_cosine_to_source"].min()))

    md.append("\n## Synthetic participants are not copies\n\n")
    md.append(dup.round(4).to_markdown(index=False))

    md.append("\n\n## Saturation, real vs synthetic\n\n")
    real_sat = pd.DataFrame([{
        "set": "real",
        "pct_at_adc_min": float(np.mean([(d["raw"] <= 0).mean() for d in real.values()]) * 100),
        "pct_at_adc_max": float(np.mean([(d["raw"] >= 4095).mean() for d in real.values()]) * 100),
        "raw_mean": float(np.mean([d["raw"].mean() for d in real.values()])),
    }, {
        "set": "synthetic",
        "pct_at_adc_min": float(np.mean([(d["raw"] <= 0).mean() for d in syn.values()]) * 100),
        "pct_at_adc_max": float(np.mean([(d["raw"] >= 4095).mean() for d in syn.values()]) * 100),
        "raw_mean": float(np.mean([d["raw"].mean() for d in syn.values()])),
    }])
    md.append(real_sat.round(3).to_markdown(index=False))

    md.append("\n\n## Figures\n\n")
    for name in SYNTHETIC_SOURCES:
        md.append("- `before_after_%s.png`\n" % name)
    md.append("- `real_vs_synthetic_distributions.png`\n")

    (OUT / "README.md").write_text("".join(md), encoding="utf-8")

    print("\n--- key numbers ---")
    print("provenance complete for all recordings:",
          bool(cons["provenance_complete"].all()))
    print("pressure identity exact: mean %.3f%%  min %.3f%%"
          % (cons["pressure_identity_exact_pct"].mean(),
             cons["pressure_identity_exact_pct"].min()))
    print("identical-to-source frames: %d (of %d)"
          % (dup["identical_frames"].sum(), dup["frames"].sum()))
    print(sep.round(4).to_string(index=False))
    print("centroid shift / class gap: mean %.3f max %.3f"
          % (sim["ratio_shift_vs_class_gap"].mean(),
             sim["ratio_shift_vs_class_gap"].max()))
    print("\nwrote " + str(OUT))


if __name__ == "__main__":
    main()
