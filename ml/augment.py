"""Phase 3 - generate synthetic participants P5_SYNTHETIC .. P10_SYNTHETIC.

What this does and does NOT do
------------------------------
Every synthetic frame is a transformed copy of one specific real frame, and the
provenance of that frame (source participant, posture, repetition, frame index)
is written to `provenance.csv` next to it. Nothing is sampled from scratch, so
the spatial structure of a real seated posture is preserved.

The transforms model *sensor and session variation*, not new bodies:

  per-sensor gain        smooth multiplicative field, mean 1
  per-sensor offset      smooth additive field on the baseline
  localized drift        2-3 smooth Gaussian blobs of altered sensitivity
  global intensity       one scalar per synthetic participant
  sensor noise           per-frame additive Gaussian, per-participant sigma
  +/-1 cell shift        one constant (dr,dc) per participant - a slightly
                         different seating position, never more than one cell
  temporal jitter        sub-frame resampling within a repetition

This adds robustness to sensor variation. It does **not** add new body types,
and the reports must never present these as real people.

Generation order (keeps every synthetic recording internally consistent with
the Phase 2 preprocessing contract):
  1. recover the source recording's integer baseline b_src (Phase 1 method)
  2. signed response  d = raw_src - b_src
  3. transform d, build a synthetic baseline b_syn
  4. raw_syn = clip(round(b_syn + d_syn), 0, 4095)
  5. pressure_syn = clip(gate(|raw_syn - b_syn|, thr_syn), 0, 4095)
so `pressure_data.csv` of a synthetic recording satisfies exactly the same
identity as a real one.

Each synthetic participant derives from exactly ONE real participant, so a
leave-one-real-participant-out fold can drop every frame derived from the
held-out person.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AUGMENTED_DIR, CLASS_NAMES, COLS, DATASET_DIR, MODELS_DIR,
                    N_SENSORS, POSTURE_FOLDERS, POSTURE_FOLDER_TO_CLASS,
                    REPORTS_DIR, ROWS, discover_recordings, repetition_column,
                    sensor_columns)
from build_dataset import recover_baseline
from preprocess import MIN_THRESHOLD, NOISE_MULTIPLIER

SESSION_NAME = "session_synthetic_v1"
OUT_SESSION = AUGMENTED_DIR / SESSION_NAME
PARAMS_JSON = AUGMENTED_DIR / "augmentation_parameters.json"
MASTER_SEED = 20261115

# synthetic participant -> the single real participant it is derived from
SYNTHETIC_SOURCES = {
    "P5_SYNTHETIC":  "P1",
    "P6_SYNTHETIC":  "P2",
    "P7_SYNTHETIC":  "P3",
    "P8_SYNTHETIC":  "P4",
    "P9_SYNTHETIC":  "P1",
    "P10_SYNTHETIC": "P2",
}

META_COLS = ["timestamp", "participant_id", "height_cm", "weight_kg", "sex",
             "posture", "rep", "frame_index", "elapsed_seconds"]
SENSOR_COLS = ["sensor_%03d" % i for i in range(N_SENSORS)]
UNKNOWN = "SYNTHETIC/NA"


# ---------------------------------------------------------------------------
# smooth random fields - physically plausible, unlike white noise per sensor
# ---------------------------------------------------------------------------
def _smooth2d(field, passes=2):
    """Cheap separable box blur on a (24,21) field; keeps edges sane."""
    f = field.astype(np.float64).copy()
    for _ in range(passes):
        p = np.pad(f, 1, mode="edge")
        f = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4 * f) / 8.0
    return f


def smooth_gain_field(rng, sigma=0.06, lo=0.82, hi=1.18):
    g = 1.0 + rng.normal(0.0, sigma, (ROWS, COLS))
    g = _smooth2d(g, passes=2)
    g = 1.0 + (g - g.mean())          # re-centre on 1 after blurring
    return np.clip(g, lo, hi)


def smooth_offset_field(rng, sigma=10.0):
    o = rng.normal(0.0, sigma, (ROWS, COLS))
    o = _smooth2d(o, passes=3)
    return o - o.mean()


def drift_field(rng, n_blobs=None, amp=0.15):
    """A few smooth blobs of raised / lowered sensitivity."""
    if n_blobs is None:
        n_blobs = int(rng.integers(2, 4))
    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float64)
    f = np.ones((ROWS, COLS))
    blobs = []
    for _ in range(n_blobs):
        cy = rng.uniform(0, ROWS - 1)
        cx = rng.uniform(0, COLS - 1)
        sy = rng.uniform(3.0, 8.0)
        sx = rng.uniform(3.0, 7.0)
        a = rng.uniform(-amp, amp)
        f += a * np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2)
                          + ((xx - cx) ** 2) / (2 * sx ** 2)))
        blobs.append({"cy": round(cy, 3), "cx": round(cx, 3),
                      "sy": round(sy, 3), "sx": round(sx, 3),
                      "amplitude": round(a, 4)})
    return f, blobs


def shift_map(mat, dr, dc):
    """Shift a (24,21) map by at most one cell, replicating the edge."""
    out = np.roll(mat, dr, axis=0)
    if dr > 0:
        out[0, :] = mat[0, :]
    elif dr < 0:
        out[-1, :] = mat[-1, :]
    out2 = np.roll(out, dc, axis=1)
    if dc > 0:
        out2[:, 0] = out[:, 0]
    elif dc < 0:
        out2[:, -1] = out[:, -1]
    return out2


def make_profile(rng, name, source):
    gain = smooth_gain_field(rng)
    offset = smooth_offset_field(rng)
    drift, blobs = drift_field(rng)
    prof = {
        "synthetic_participant": name,
        "source_participant": source,
        "global_intensity": float(rng.uniform(0.85, 1.20)),
        "noise_sigma_counts": float(rng.uniform(3.0, 9.0)),
        "shift_dr": int(rng.integers(-1, 2)),
        "shift_dc": int(rng.integers(-1, 2)),
        "temporal_jitter_frames": float(rng.uniform(0.15, 0.45)),
        "gain_field_summary": {"min": float(gain.min()), "max": float(gain.max()),
                               "mean": float(gain.mean()), "std": float(gain.std())},
        "offset_field_summary": {"min": float(offset.min()), "max": float(offset.max()),
                                 "std": float(offset.std())},
        "drift_blobs": blobs,
        "height_cm": UNKNOWN, "weight_kg": UNKNOWN, "sex": UNKNOWN,
        "is_synthetic": True,
    }
    fields = {"gain": gain, "offset": offset, "drift": drift}
    return prof, fields


# ---------------------------------------------------------------------------
def generate_recording(rng, fields, prof, src_raw, src_pressure, src_rep, src_frame_idx):
    """Transform one source recording into one synthetic recording."""
    b_src, _ = recover_baseline(src_raw, src_pressure)
    d_src = (src_raw.astype(np.float64) - b_src.reshape(1, -1))

    gain, offset, drift = fields["gain"], fields["offset"], fields["drift"]
    combined = (gain * drift).reshape(1, -1)
    g = prof["global_intensity"]
    dr, dc = prof["shift_dr"], prof["shift_dc"]
    jitter = prof["temporal_jitter_frames"]
    sigma = prof["noise_sigma_counts"]

    # --- temporal jitter: resample inside each repetition only ---------------
    out_rows = []
    for rep in np.unique(src_rep):
        sel = np.where(src_rep == rep)[0]
        m = sel.size
        pos = np.arange(m, dtype=np.float64) + rng.normal(0.0, jitter, m)
        pos = np.clip(pos, 0, m - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, m - 1)
        w = (pos - lo).reshape(-1, 1)
        d_rep = d_src[sel]
        blended = d_rep[lo] * (1.0 - w) + d_rep[hi] * w
        for k in range(m):
            out_rows.append((rep, k, sel[lo[k]], blended[k]))

    d_syn = np.stack([r[3] for r in out_rows], axis=0)

    # --- spatial: +/-1 cell shift, then gain/drift/global intensity ----------
    if dr or dc:
        d_syn = np.stack([shift_map(f.reshape(ROWS, COLS), dr, dc).ravel()
                          for f in d_syn], axis=0)
    d_syn = d_syn * combined * g

    # --- sensor noise --------------------------------------------------------
    d_syn = d_syn + rng.normal(0.0, sigma, d_syn.shape)

    # --- synthetic baseline + reconstructed raw ------------------------------
    b_syn = np.clip(np.rint(b_src + offset.ravel()), 0, 4095).astype(np.int32)
    raw_syn = np.clip(np.rint(b_syn.reshape(1, -1) + d_syn), 0, 4095).astype(np.int32)

    # --- threshold from the synthetic noise level, same formula as the rig ---
    thr_syn = np.full(N_SENSORS,
                      max(int(round(NOISE_MULTIPLIER * sigma)), MIN_THRESHOLD),
                      dtype=np.int32)

    delta = np.abs(raw_syn.astype(np.int64) - b_syn.reshape(1, -1).astype(np.int64))
    pressure_syn = np.where(delta <= thr_syn.reshape(1, -1), 0, delta)
    pressure_syn = np.clip(pressure_syn, 0, 4095).astype(np.int32)

    meta = {
        "rep": np.array([r[0] for r in out_rows], dtype=np.int32),
        "frame_index": np.array([r[1] for r in out_rows], dtype=np.int32),
        "source_frame_row": np.array([r[2] for r in out_rows], dtype=np.int32),
        "source_rep": np.array([src_rep[r[2]] for r in out_rows], dtype=np.int32),
        "source_frame_index": np.array([src_frame_idx[r[2]] for r in out_rows],
                                       dtype=np.int32),
    }
    return raw_syn, pressure_syn, b_syn, thr_syn, meta


def main():
    if OUT_SESSION.exists() and any(OUT_SESSION.rglob("*.csv")):
        raise SystemExit("AUGMENTED_DATASET already populated; delete %s to regenerate"
                         % OUT_SESSION)

    real = {}
    for rec in discover_recordings(DATASET_DIR):
        raw_df = pd.read_csv(rec["raw_csv"])
        pr_df = pd.read_csv(rec["pressure_csv"])
        raw = raw_df[sensor_columns(raw_df)].to_numpy(dtype=np.int32)
        pres = pr_df[sensor_columns(pr_df)].to_numpy(dtype=np.int32)
        n = min(raw.shape[0], pres.shape[0])
        raw, pres = raw[:n], pres[:n]
        repcol = repetition_column(raw_df)
        fcol = "frame_in_repetition" if "frame_in_repetition" in raw_df.columns else "frame_index"
        real[(rec["participant"], rec["posture_folder"])] = {
            "raw": raw, "pressure": pres,
            "rep": raw_df[repcol].to_numpy(dtype=np.int32)[:n],
            "frame_index": raw_df[fcol].to_numpy(dtype=np.int32)[:n],
        }
    print("loaded %d real recordings" % len(real))

    all_params = {"master_seed": MASTER_SEED, "generated": datetime.now().isoformat(),
                  "session": SESSION_NAME, "participants": {}}
    counts = []

    for s_idx, (syn_name, src_part) in enumerate(SYNTHETIC_SOURCES.items()):
        rng = np.random.default_rng(MASTER_SEED + 1000 * (s_idx + 1))
        prof, fields = make_profile(rng, syn_name, src_part)
        part_dir = OUT_SESSION / syn_name
        part_dir.mkdir(parents=True, exist_ok=True)

        with open(part_dir / "participant_profile.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["participant_id", "height_cm", "weight_kg", "sex",
                        "is_synthetic", "derived_from_participant"])
            w.writerow([syn_name, UNKNOWN, UNKNOWN, UNKNOWN, "TRUE", src_part])

        for posture in POSTURE_FOLDERS:
            src = real[(src_part, posture)]
            raw_syn, press_syn, b_syn, thr_syn, meta = generate_recording(
                rng, fields, prof, src["raw"], src["pressure"],
                src["rep"], src["frame_index"])

            pdir = part_dir / posture
            pdir.mkdir(parents=True, exist_ok=True)
            n = raw_syn.shape[0]
            ts = datetime.now().isoformat(timespec="milliseconds")

            for fname, arr in (("raw_data.csv", raw_syn),
                               ("pressure_data.csv", press_syn)):
                with open(pdir / fname, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(META_COLS + SENSOR_COLS)
                    for k in range(n):
                        w.writerow([ts, syn_name, UNKNOWN, UNKNOWN, UNKNOWN,
                                    posture, int(meta["rep"][k]),
                                    int(meta["frame_index"][k]),
                                    round(float(meta["frame_index"][k]) * 0.70, 4)]
                                   + arr[k].tolist())

            with open(pdir / "provenance.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "synthetic_participant", "posture_folder",
                            "posture_class", "source_participant", "source_posture",
                            "source_rep", "source_frame_index", "source_row_in_csv",
                            "source_file", "transform_seed"])
                for k in range(n):
                    w.writerow([
                        int(meta["frame_index"][k]), syn_name, posture,
                        POSTURE_FOLDER_TO_CLASS[posture], src_part, posture,
                        int(meta["source_rep"][k]), int(meta["source_frame_index"][k]),
                        int(meta["source_frame_row"][k]),
                        str(Path("DATASET") / "session_20261114_170217" / src_part
                            / posture / "raw_data.csv").replace("\\", "/"),
                        MASTER_SEED + 1000 * (s_idx + 1),
                    ])

            np.save(pdir / "calibration_used.npy",
                    np.stack([b_syn, thr_syn]).astype(np.int32))
            counts.append({"participant": syn_name, "posture_folder": posture,
                           "posture_class": POSTURE_FOLDER_TO_CLASS[posture],
                           "frames": n, "source_participant": src_part})
            print("  %-14s %-17s n=%4d  (from %s)" % (syn_name, posture, n, src_part))

        all_params["participants"][syn_name] = prof

    PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(all_params, indent=2), encoding="utf-8")
    pd.DataFrame(counts).to_csv(AUGMENTED_DIR / "synthetic_frame_counts.csv", index=False)

    print("\nwrote %s" % OUT_SESSION)
    print("parameters: %s" % PARAMS_JSON)
    print("total synthetic frames: %d" % sum(c["frames"] for c in counts))


if __name__ == "__main__":
    main()
