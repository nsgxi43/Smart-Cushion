"""Phase 1 - inspect the original dataset. Read-only: never writes into DATASET/."""
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
from common import (ADC_MAX, ADC_MIN, CLASS_NAMES, COLS, DATASET_DIR, N_SENSORS,
                    POSTURE_FOLDER_TO_CLASS, REPORTS_DIR, ROWS, detect_schema,
                    discover_recordings, frame_index_column, load_recording,
                    repetition_column, sensor_columns)

OUT = REPORTS_DIR / "dataset_report"
OUT.mkdir(parents=True, exist_ok=True)


def inspect_one(rec):
    row = {
        "session": rec["session"],
        "participant": rec["participant"],
        "posture_folder": rec["posture_folder"],
        "posture_class": rec["posture_class"],
        "has_metadata_csv": rec["has_metadata"],
        "has_npy": rec["has_npy"],
    }
    out_extras = {}

    for kind in ("raw", "pressure"):
        path = rec[kind + "_csv"]
        if not path.is_file():
            row[kind + "_present"] = False
            continue
        row[kind + "_present"] = True

        df = pd.read_csv(path)
        scols = sensor_columns(df)
        schema = detect_schema(df)
        frames = df[scols].to_numpy(dtype=np.float64)

        repcol = repetition_column(df)
        fidxcol = frame_index_column(df)

        row[kind + "_schema"] = schema
        row[kind + "_n_cols"] = df.shape[1]
        row[kind + "_n_meta_cols"] = df.shape[1] - len(scols)
        row[kind + "_sensor_col_first"] = scols[0]
        row[kind + "_sensor_col_last"] = scols[-1]
        row[kind + "_frames"] = int(frames.shape[0])
        row[kind + "_nan_cells"] = int(np.isnan(frames).sum())
        row[kind + "_nan_meta"] = int(df.drop(columns=scols).isna().sum().sum())
        # corrupted = any value outside the 12-bit ADC range
        row[kind + "_out_of_range_cells"] = int(((frames < ADC_MIN) | (frames > ADC_MAX)).sum())
        row[kind + "_frames_with_bad_cells"] = int(
            (((frames < ADC_MIN) | (frames > ADC_MAX)) | np.isnan(frames)).any(axis=1).sum())
        row[kind + "_non_integer_cells"] = int((frames != np.round(frames)).sum())
        row[kind + "_min"] = float(np.nanmin(frames))
        row[kind + "_max"] = float(np.nanmax(frames))
        row[kind + "_mean"] = float(np.nanmean(frames))
        row[kind + "_std"] = float(np.nanstd(frames))
        row[kind + "_pct_at_adc_min"] = float((frames <= ADC_MIN).mean() * 100)
        row[kind + "_pct_at_adc_max"] = float((frames >= ADC_MAX).mean() * 100)
        row[kind + "_pct_zero"] = float((frames == 0).mean() * 100)

        # repetition structure
        reps = df[repcol].to_numpy()
        row[kind + "_n_reps"] = int(pd.unique(reps).size)
        row[kind + "_rep_values"] = ",".join(str(v) for v in sorted(pd.unique(reps)))
        counts = df.groupby(repcol).size()
        row[kind + "_frames_per_rep_min"] = int(counts.min())
        row[kind + "_frames_per_rep_max"] = int(counts.max())
        row[kind + "_frames_per_rep_mean"] = float(counts.mean())

        # frame index monotonic within each repetition
        mono = all(df.loc[df[repcol] == r, fidxcol].is_monotonic_increasing
                   for r in pd.unique(reps))
        row[kind + "_frameidx_monotonic_within_rep"] = bool(mono)

        # elapsed seconds sanity
        el = df["elapsed_seconds"].to_numpy(dtype=np.float64)
        row[kind + "_elapsed_min"] = float(el.min())
        row[kind + "_elapsed_max"] = float(el.max())
        durs, rates = [], []
        for r in pd.unique(reps):
            sub = df.loc[df[repcol] == r, "elapsed_seconds"].to_numpy(dtype=np.float64)
            if sub.size > 1:
                durs.append(sub.max() - sub.min())
                rates.append((sub.size - 1) / max(sub.max() - sub.min(), 1e-9))
        row[kind + "_rep_duration_mean_s"] = float(np.mean(durs)) if durs else float("nan")
        row[kind + "_fps_mean"] = float(np.mean(rates)) if rates else float("nan")

        # timestamp monotonicity
        tcol = "frame_timestamp" if "frame_timestamp" in df.columns else "timestamp"
        ts = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
        row[kind + "_timestamp_parse_failures"] = int(ts.isna().sum())
        row[kind + "_timestamps_monotonic"] = bool(ts.is_monotonic_increasing)
        row[kind + "_duplicate_timestamps"] = int(df[tcol].duplicated().sum())

        # label consistency against the folder name
        if schema == "P1_LEGACY":
            codes = set(df["position_code"].astype(str).unique())
            labels = set(df["position_label"].astype(str).unique())
            expect_code, expect_label = rec["posture_folder"].split("_", 1)
            row[kind + "_label_consistent"] = bool(codes == {expect_code}
                                                   and labels == {expect_label})
            row[kind + "_label_values"] = str(sorted(codes)) + "|" + str(sorted(labels))
            row[kind + "_participant_in_file"] = ""
        else:
            postures = set(df["posture"].astype(str).unique())
            pids = set(df["participant_id"].astype(str).unique())
            row[kind + "_label_consistent"] = bool(postures == {rec["posture_folder"]}
                                                   and pids == {rec["participant"]})
            row[kind + "_label_values"] = str(sorted(postures))
            row[kind + "_participant_in_file"] = ",".join(sorted(pids))

        # duplicate frames (identical 504-vectors) - a sign of a stalled serial read
        row[kind + "_duplicate_frames"] = int(pd.DataFrame(frames).duplicated().sum())

        out_extras[kind + "_frames"] = frames

    return row, out_extras


def main():
    recs = discover_recordings()
    print("Discovered " + str(len(recs)) + " recordings")

    rows = []
    per_sensor_raw = {}
    mean_pressure_maps = {}
    mean_raw_maps = {}
    raw_by_participant = {}

    for rec in recs:
        row, extras = inspect_one(rec)
        rows.append(row)
        key = (rec["participant"], rec["posture_class"])
        if "raw_frames" in extras:
            rf = extras["raw_frames"]
            per_sensor_raw[key] = (rf.mean(axis=0), rf.std(axis=0))
            mean_raw_maps[key] = rf.mean(axis=0).reshape(ROWS, COLS)
            raw_by_participant.setdefault(rec["participant"], []).append(rf)
        if "pressure_frames" in extras:
            mean_pressure_maps[key] = extras["pressure_frames"].mean(axis=0).reshape(ROWS, COLS)
        print("  %3s %-17s frames=%4s schema=%s" % (
            rec["participant"], rec["posture_folder"],
            row.get("raw_frames", "?"), row.get("raw_schema", "?")))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "per_recording_summary.csv", index=False)

    ps_rows = []
    for (part, posture), (m, s) in sorted(per_sensor_raw.items()):
        for i in range(N_SENSORS):
            ps_rows.append({"participant": part, "posture_class": posture,
                            "sensor_index": i, "row": i // COLS, "col": i % COLS,
                            "raw_mean": m[i], "raw_std": s[i]})
    pd.DataFrame(ps_rows).to_csv(OUT / "per_sensor_stats.csv", index=False)

    dead_rows = []
    dead_sets = {}
    for part, lst in sorted(raw_by_participant.items()):
        allf = np.concatenate(lst, axis=0)
        std = allf.std(axis=0)
        mean = allf.mean(axis=0)
        dead = np.where(std == 0)[0]
        near_dead = np.where(std < 1.0)[0]
        stuck_low = np.where(mean <= 1.0)[0]
        stuck_high = np.where(mean >= ADC_MAX - 1)[0]
        dead_sets[part] = set(dead.tolist())
        dead_rows.append({
            "participant": part,
            "frames_total": int(allf.shape[0]),
            "dead_sensors_std0": len(dead),
            "dead_sensor_indices": ",".join(map(str, dead.tolist())),
            "near_dead_std_lt_1": len(near_dead),
            "stuck_low_mean_le_1": len(stuck_low),
            "stuck_high_mean_ge_4094": len(stuck_high),
            "raw_global_min": float(allf.min()),
            "raw_global_max": float(allf.max()),
            "raw_global_mean": float(allf.mean()),
            "pct_at_adc_max": float((allf >= ADC_MAX).mean() * 100),
            "pct_at_adc_min": float((allf <= ADC_MIN).mean() * 100),
        })
    dead_df = pd.DataFrame(dead_rows)
    dead_df.to_csv(OUT / "dead_sensors_by_participant.csv", index=False)

    parts = sorted(raw_by_participant.keys())
    corr_rows = []
    for a in parts:
        for b in parts:
            ka, kb = (a, "C0_Normal"), (b, "C0_Normal")
            if ka in mean_raw_maps and kb in mean_raw_maps:
                va, vb = mean_raw_maps[ka].ravel(), mean_raw_maps[kb].ravel()
                corr_rows.append({"a": a, "b": b,
                                  "pearson_mean_raw_C0": float(np.corrcoef(va, vb)[0, 1])})
    corr = pd.DataFrame(corr_rows).pivot(index="a", columns="b",
                                         values="pearson_mean_raw_C0")
    corr.to_csv(OUT / "sensor_indexing_consistency_C0.csv")

    for which, maps, fname in (("pressure", mean_pressure_maps, "heatmaps_mean_pressure.png"),
                               ("raw", mean_raw_maps, "heatmaps_mean_raw.png")):
        fig, axes = plt.subplots(len(parts), len(CLASS_NAMES),
                                 figsize=(2.1 * len(CLASS_NAMES), 2.6 * len(parts)))
        axes = np.atleast_2d(axes)
        vmax = max((m.max() for m in maps.values()), default=1.0)
        for i, part in enumerate(parts):
            for j, cls in enumerate(CLASS_NAMES):
                ax = axes[i, j]
                m = maps.get((part, cls))
                if m is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                ax.imshow(m, cmap="turbo", vmin=0, vmax=vmax,
                          interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(cls.replace("_", "\n", 1), fontsize=8)
                if j == 0:
                    ax.set_ylabel(part, fontsize=10)
        fig.suptitle("Mean " + which + " map per participant x posture "
                     "(shared scale 0.." + ("%.0f" % vmax) + ")", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(parts), figsize=(3.0 * len(parts), 3.6))
    axes = np.atleast_1d(axes)
    for i, part in enumerate(parts):
        allf = np.concatenate(raw_by_participant[part], axis=0)
        im = axes[i].imshow(allf.std(axis=0).reshape(ROWS, COLS), cmap="magma",
                            interpolation="nearest")
        axes[i].set_title(part + ": raw std", fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "per_sensor_raw_std.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for part in parts:
        allf = np.concatenate(raw_by_participant[part], axis=0).ravel()
        axes[0].hist(allf, bins=120, histtype="step", label=part, density=True)
    axes[0].set_title("raw ADC value distribution"); axes[0].set_yscale("log")
    axes[0].legend(fontsize=8); axes[0].set_xlabel("ADC counts")
    allp = np.concatenate([m.ravel() for m in mean_pressure_maps.values()])
    axes[1].hist(allp, bins=120, color="tab:red")
    axes[1].set_title("mean-pressure cell values (all recordings)")
    axes[1].set_yscale("log"); axes[1].set_xlabel("pressure counts")
    fig.tight_layout()
    fig.savefig(OUT / "value_distributions.png", dpi=110)
    plt.close(fig)

    md = []
    md.append("# Phase 1 - Dataset Inspection Report\n\n")
    md.append("Dataset root: `" + str(DATASET_DIR) + "`\n\n")
    md.append("Recordings discovered: **" + str(len(recs)) + "**\n\n")
    md.append("Participants: **" + ", ".join(parts) + "**\n\n")
    md.append("Postures per participant: **" + str(len(CLASS_NAMES)) + "**\n")

    md.append("\n## Posture folder -> class mapping\n\n")
    md.append(pd.DataFrame([{"original_folder": k, "canonical_class": v,
                             "label_index": CLASS_NAMES.index(v)}
                            for k, v in POSTURE_FOLDER_TO_CLASS.items()]
                           ).to_markdown(index=False))

    md.append("\n\n## Frame counts\n\n")
    piv = summary.pivot_table(index="participant", columns="posture_class",
                              values="raw_frames", aggfunc="sum")
    piv["TOTAL"] = piv.sum(axis=1)
    md.append(piv.to_markdown())
    md.append("\n\nTotal raw frames in dataset: **"
              + str(int(summary["raw_frames"].sum())) + "**\n")

    md.append("\n## Schema differences\n\n")
    md.append(summary.groupby(["participant", "raw_schema"]).size()
              .reset_index(name="recordings").to_markdown(index=False))
    md.append("\n\n")
    md.append(summary[["participant", "posture_folder", "raw_n_meta_cols",
                       "raw_sensor_col_first", "raw_sensor_col_last",
                       "has_metadata_csv", "has_npy"]].to_markdown(index=False))

    md.append("\n\n## Data integrity\n\n")
    md.append(summary[["participant", "posture_folder", "raw_frames",
                       "raw_nan_cells", "raw_nan_meta", "raw_out_of_range_cells",
                       "raw_non_integer_cells", "raw_duplicate_frames",
                       "raw_timestamp_parse_failures", "raw_timestamps_monotonic",
                       "raw_frameidx_monotonic_within_rep", "raw_label_consistent"]
                      ].to_markdown(index=False))

    md.append("\n\n## Saturation (percentage of all cells)\n\n")
    md.append(summary[["participant", "posture_folder",
                       "raw_pct_at_adc_min", "raw_pct_at_adc_max",
                       "pressure_pct_zero", "raw_min", "raw_max", "raw_mean"]
                      ].round(3).to_markdown(index=False))

    md.append("\n\n## Repetition structure\n\n")
    md.append(summary[["participant", "posture_folder", "raw_n_reps", "raw_rep_values",
                       "raw_frames_per_rep_min", "raw_frames_per_rep_max",
                       "raw_rep_duration_mean_s", "raw_fps_mean"]
                      ].round(2).to_markdown(index=False))

    md.append("\n\n## Dead / stuck sensors per participant\n\n")
    md.append(dead_df.round(3).to_markdown(index=False))

    md.append("\n\n## Sensor indexing consistency\n\n")
    md.append("Pearson correlation of the mean *raw* 504-vector for posture "
              "C0_Normal between participants. High values indicate the same "
              "physical sensor order.\n\n")
    md.append(corr.round(4).to_markdown())

    md.append("\n\n## Figures\n\n")
    for f in ("heatmaps_mean_pressure.png", "heatmaps_mean_raw.png",
              "per_sensor_raw_std.png", "value_distributions.png"):
        md.append("- `" + f + "`\n")

    (OUT / "README.md").write_text("".join(md), encoding="utf-8")

    (OUT / "inspection_summary.json").write_text(json.dumps({
        "n_recordings": len(recs),
        "participants": parts,
        "classes": CLASS_NAMES,
        "posture_folder_to_class": POSTURE_FOLDER_TO_CLASS,
        "total_raw_frames": int(summary["raw_frames"].sum()),
        "frames_per_participant": {p: int(summary.loc[summary.participant == p,
                                                      "raw_frames"].sum()) for p in parts},
        "dead_sensors": {p: sorted(dead_sets[p]) for p in parts},
        "schemas": {p: sorted(summary.loc[summary.participant == p, "raw_schema"].unique())
                    for p in parts},
    }, indent=2), encoding="utf-8")

    print("\nWrote report to " + str(OUT))


if __name__ == "__main__":
    main()
