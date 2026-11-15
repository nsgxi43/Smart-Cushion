#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 - preprocessing and session-aware dataset splitting.

Preprocessing
-------------
Each screenshot is decoded from the renderer's discrete palette back to a
single-channel *band index* map (see common.decode_band_image):

    RGB screenshot -> nearest palette entry -> band index 0..7 -> [0, 1]

This is the faithful reduction for these images. The renderer uses eight
evenly spaced samples of matplotlib's "turbo" colormap, and 99.96 % of all
pixels land exactly on that palette, so the decode is lossless with respect to
what the render actually encodes. Plain RGB or luminance would be worse:
turbo is not monotonic in brightness, so a grayscale conversion scrambles the
pressure ordering, while raw RGB forces the network to relearn the colormap
from 280 images.

Splitting
---------
Recording sessions are recovered from the screenshot timestamps (a gap longer
than 20 s starts a new session). Frames inside one session are near-identical,
so sessions are kept whole and never span two splits.

Exactly 70/15/15 is unreachable at session granularity because each class has
only 3-5 sessions; leakage prevention is prioritised over hitting the ratio,
and the achieved ratio is reported. Classes with a single session fall back to
a contiguous temporal split with a guard band, which is documented as a
limitation rather than hidden.

Usage:
    python training/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from common import (CLASS_NAMES, DEFAULT_DATASET, INPUT_H, INPUT_W, NUM_CLASSES,
                    ROOT, decode_band_image, list_dataset, set_seeds)

PROCESSED = os.path.join(ROOT, "processed_dataset")
EVAL = os.path.join(ROOT, "evaluation")

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
GUARD = 2  # frames dropped either side of a within-session split boundary


def split_sessions(sess_sizes: dict[str, int]) -> dict[str, str]:
    """Assign whole sessions to train/val/test, deterministically.

    Picks the session whose size is closest to the 15 % target for test, then
    the closest of the remainder for val; everything else trains. Requires at
    least three sessions.
    """
    total = sum(sess_sizes.values())
    target = VAL_FRAC * total
    remaining = dict(sess_sizes)
    assign = {}
    for split in ("test", "val"):
        pick = min(sorted(remaining), key=lambda s: (abs(remaining[s] - target), s))
        assign[pick] = split
        del remaining[pick]
    for s in remaining:
        assign[s] = "train"
    return assign


def split_single_session(records: list[dict]) -> dict[str, str]:
    """Contiguous temporal split for a class that has only one session.

    Guard bands are dropped at both boundaries so that the frames immediately
    adjacent across a split are not near-duplicates of each other.
    """
    rs = sorted(records, key=lambda r: r["order"])
    n = len(rs)
    usable = n - 2 * GUARD
    n_tr = int(round(usable * TRAIN_FRAC))
    n_va = int(round(usable * VAL_FRAC))
    n_te = usable - n_tr - n_va
    assign = {}
    i = 0
    for name, k in (("train", n_tr), (None, GUARD), ("val", n_va), (None, GUARD), ("test", n_te)):
        for r in rs[i:i + k]:
            assign[r["path"]] = name if name else "drop"
        i += k
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    set_seeds(args.seed)
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(os.path.join(EVAL, "figures"), exist_ok=True)

    recs = list_dataset(args.dataset)
    print(f"decoding {len(recs)} screenshots to {INPUT_H}x{INPUT_W} band-index maps ...")

    X = np.zeros((len(recs), INPUT_H, INPUT_W, 1), dtype=np.float32)
    off_palette = []
    for i, r in enumerate(recs):
        arr, st = decode_band_image(r["path"])
        X[i, :, :, 0] = arr
        off_palette.append(st["off_palette_frac"])
        r["mean_band"] = st["mean_band"]
    y = np.array([r["label"] for r in recs], dtype=np.int32)
    print(f"  mean off-palette pixel fraction: {100*np.mean(off_palette):.4f} %")
    print(f"  value range after normalisation: [{X.min():.3f}, {X.max():.3f}]  mean {X.mean():.3f}")

    # ---- assign splits ----------------------------------------------------
    print("\n== SPLIT ASSIGNMENT (session-aware) ==")
    split_of: dict[str, str] = {}
    notes = {}
    for cls in CLASS_NAMES:
        rs = [r for r in recs if r["cls"] == cls]
        sizes = Counter(r["session"] for r in rs)
        if len(sizes) >= 3:
            assign = split_sessions(dict(sizes))
            for r in rs:
                split_of[r["path"]] = assign[r["session"]]
            notes[cls] = "session-level split (sessions kept whole)"
            detail = "  ".join(f"{s.split('_')[1]}->{assign[s]}({sizes[s]})" for s in sorted(sizes))
        else:
            a = split_single_session(rs)
            for r in rs:
                split_of[r["path"]] = a[r["path"]]
            notes[cls] = (f"SINGLE SESSION - contiguous temporal split with a {GUARD}-frame "
                          f"guard band; train/val/test are not independent recordings")
            detail = f"1 session of {len(rs)} frames -> contiguous temporal split"
        n = Counter(split_of[r["path"]] for r in rs)
        print(f"  {cls:<10} {detail}")
        print(f"             train={n['train']:3d} val={n['val']:3d} test={n['test']:3d} "
              f"dropped={n['drop']:2d}   [{notes[cls]}]")

    keep = np.array([split_of[r["path"]] != "drop" for r in recs])
    splits = np.array([split_of[r["path"]] for r in recs])

    print("\n== SPLIT TOTALS ==")
    tot = keep.sum()
    for s in ("train", "val", "test"):
        m = splits == s
        print(f"  {s:<6} {m.sum():3d} images ({100*m.sum()/tot:4.1f} %)   "
              f"per class: {[int(((splits==s)&(y==k)).sum()) for k in range(NUM_CLASSES)]}")
    print(f"  dropped (guard bands): {(splits=='drop').sum()}")
    print("  target was 70/15/15; session integrity takes precedence - see README")

    # ---- leakage check -----------------------------------------------------
    print("\n== CROSS-SPLIT NEAR-DUPLICATE CHECK ==")
    flat = X.reshape(len(X), -1)
    worst = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia = np.where(splits == a)[0]
        ib = np.where(splits == b)[0]
        # similarity = 1 - normalised mean absolute difference
        sim = 1.0 - np.abs(flat[ia][:, None, :] - flat[ib][None, :, :]).mean(-1)
        mx = float(sim.max())
        n99 = int((sim > 0.99).sum())
        worst.append((a, b, mx, n99))
        print(f"  {a:<6} vs {b:<5} max similarity {mx:.4f}   pairs >0.99: {n99}")
    if max(w[2] for w in worst) > 0.995:
        print("  WARNING: very similar frames appear in two different splits")
    else:
        print("  no cross-split pair exceeds 0.995 similarity")

    # ---- save ---------------------------------------------------------------
    np.save(os.path.join(PROCESSED, "X.npy"), X)
    np.save(os.path.join(PROCESSED, "y.npy"), y)
    np.save(os.path.join(PROCESSED, "split.npy"), splits)

    manifest = dict(
        input_shape=[INPUT_H, INPUT_W, 1],
        normalisation="band index 0..7 decoded from the 8-level turbo palette, divided by 7",
        class_names=CLASS_NAMES,
        train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, guard_frames=GUARD,
        achieved={s: int((splits == s).sum()) for s in ("train", "val", "test", "drop")},
        per_class_notes=notes,
        cross_split_max_similarity={f"{a}_vs_{b}": m for a, b, m, _ in worst},
        items=[
            dict(path=os.path.relpath(r["path"], ROOT).replace("\\", "/"),
                 name=r["name"], cls=r["cls"], label=r["label"],
                 session=r["session"], split=split_of[r["path"]])
            for r in recs
        ],
    )
    mpath = os.path.join(PROCESSED, "splits.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nsaved X.npy {X.shape}, y.npy, split.npy and splits.json -> {PROCESSED}")

    # ---- preview figure -------------------------------------------------------
    pad = 4
    tiles = []
    for k, cls in enumerate(CLASS_NAMES):
        i = int(np.where(y == k)[0][0])
        g = (X[i, :, :, 0] * 255).astype(np.uint8)
        tiles.append(np.repeat(g[:, :, None], 3, axis=2))
    canvas = Image.new("RGB", (len(tiles) * (INPUT_W + pad) + pad, INPUT_H + 2 * pad), (24, 24, 28))
    for i, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t), (pad + i * (INPUT_W + pad), pad))
    canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)
    out = os.path.join(EVAL, "figures", "preprocessed_samples.png")
    canvas.save(out)
    print(f"preview ({', '.join(CLASS_NAMES)}) -> {out}")


if __name__ == "__main__":
    main()
