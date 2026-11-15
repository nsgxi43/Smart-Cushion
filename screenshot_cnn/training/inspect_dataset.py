#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 - dataset inspection.

Reports everything required by the project brief section 2:
  * images per class, dimensions, formats, corrupt files
  * exact duplicates (MD5) and near-duplicates
  * class imbalance and effective (non-redundant) sample counts
  * heat-map layout and colour-scale consistency
  * recording sessions derived from the screenshot timestamps

Writes evaluation/dataset_summary.json, evaluation/figures/*.png and prints a
full analysis. The narrative report lives in evaluation/dataset_report.md.

Usage:
    python training/inspect_dataset.py

Opens the dataset read-only; nothing in dataset/ is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from common import (CLASS_NAMES, NATIVE_GRID_H, NATIVE_GRID_W, PALETTE, ROOT,
                    DEFAULT_DATASET, list_dataset, quantize_to_ids)

TILE_W, TILE_H, PAD = 180, 206, 6


def dominant_period(score: np.ndarray, lo: float = 8.0, hi: float = 40.0,
                    step: float = 0.01, win: float = 0.8) -> tuple[float, float]:
    """Period at which edge energy is most concentrated (the render-cell size)."""
    best_val, best_p = 0.0, float("nan")
    idx = np.arange(len(score))
    mean = score.mean() + 1e-9
    for p in np.arange(lo, hi, step):
        ph = idx % p
        m = (ph < win) | (ph > p - win)
        v = score[m].mean() / mean
        if v > best_val:
            best_val, best_p = v, p
    return best_p, best_val


def montage(tiles: list[np.ndarray]) -> Image.Image:
    cols = len(tiles)
    canvas = Image.new("RGB",
                       (cols * TILE_W + (cols + 1) * PAD, TILE_H + 2 * PAD),
                       (24, 24, 28))
    for i, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t), (PAD + i * (TILE_W + PAD), PAD))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation"))
    args = ap.parse_args()
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    recs = list_dataset(args.dataset)
    for r in recs:
        with open(r["path"], "rb") as fh:
            r["md5"] = hashlib.md5(fh.read()).hexdigest()
    print(f"TOTAL FILES: {len(recs)}\n")

    # ---- integrity, dimensions, formats ----------------------------------
    sizes, modes, fmts, corrupt = Counter(), Counter(), Counter(), []
    for r in recs:
        try:
            im = Image.open(r["path"])
            im.load()
            sizes[im.size] += 1
            modes[im.mode] += 1
            fmts[im.format] += 1
        except Exception as exc:
            corrupt.append((r["path"], str(exc)))
    print("== INTEGRITY ==")
    print(f"  dimensions: {dict(sizes)}")
    print(f"  formats:    {dict(fmts)}")
    print(f"  modes:      {dict(modes)}")
    print(f"  corrupt / unreadable: {len(corrupt)}")
    print(f"  files without a parsable timestamp: {sum(1 for r in recs if r['ts'] is None)}")

    # ---- duplicates -------------------------------------------------------
    md5c = Counter(r["md5"] for r in recs)
    dup_hashes = {h for h, c in md5c.items() if c > 1}
    print("\n== EXACT DUPLICATES (MD5) ==")
    print(f"  groups: {len(dup_hashes)}   files involved: {sum(md5c[h] for h in dup_hashes)}")

    # ---- band-index maps at the native grid, for near-duplicate analysis --
    M = []
    for r in recs:
        a = np.asarray(Image.open(r["path"]).convert("RGB")
                       .resize((NATIVE_GRID_W, NATIVE_GRID_H), Image.NEAREST)).astype(np.float32)
        M.append(quantize_to_ids(a)[0].ravel())
    M = np.array(M)
    n = len(M)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i] = (M == M[i]).mean(1)

    print(f"\n== NEAR-DUPLICATES ({NATIVE_GRID_W}x{NATIVE_GRID_H} band-index maps) ==")
    for thr in (0.99, 0.95, 0.90):
        pairs = np.argwhere(np.triu(sim, 1) >= thr)
        ss = sum(1 for i, j in pairs if recs[i]["session"] == recs[j]["session"])
        sc = sum(1 for i, j in pairs if recs[i]["cls"] == recs[j]["cls"])
        print(f"  >= {thr*100:.0f}% identical cells: {len(pairs):4d} pairs "
              f"({ss} same-session, {sc} same-class, {len(pairs)-sc} cross-class)")

    effective = {}
    for cls in CLASS_NAMES:
        ii = [k for k, r in enumerate(recs) if r["cls"] == cls]
        used, clusters = set(), 0
        for i in ii:
            if i in used:
                continue
            clusters += 1
            used.update(j for j in ii if sim[i, j] >= 0.95)
        effective[cls] = clusters

    # ---- per-class summary -------------------------------------------------
    print("\n== PER-CLASS SUMMARY ==")
    print("  id class      files unique_md5 dup_files sessions effective session_sizes")
    per_class = {}
    for label, cls in enumerate(CLASS_NAMES):
        rs = [r for r in recs if r["cls"] == cls]
        mc = Counter(r["md5"] for r in rs)
        ss = Counter(r["session"] for r in rs)
        per_class[cls] = dict(
            label=label, files=len(rs), unique=len(mc),
            dup_files=int(sum(c for c in mc.values() if c > 1)),
            sessions=len(ss), session_sizes=sorted(ss.values(), reverse=True),
            effective=effective[cls])
        p = per_class[cls]
        print(f"  {label}  {cls:<10}{p['files']:5d} {p['unique']:9d} {p['dup_files']:9d} "
              f"{p['sessions']:8d} {p['effective']:9d} {p['session_sizes']}")
    counts = [per_class[c]["files"] for c in CLASS_NAMES]
    print(f"\n  class imbalance: min={min(counts)} max={max(counts)} "
          f"ratio={max(counts)/min(counts):.2f} (nominally balanced)")
    eff = [effective[c] for c in CLASS_NAMES]
    print(f"  effective imbalance: min={min(eff)} max={max(eff)} ratio={max(eff)/min(eff):.2f}")

    # ---- sessions ----------------------------------------------------------
    print("\n== RECORDING SESSIONS (new session when gap > 20 s) ==")
    for cls in CLASS_NAMES:
        by = defaultdict(list)
        for r in sorted([x for x in recs if x["cls"] == cls], key=lambda r: r["ts"]):
            by[r["session"]].append(r["ts"])
        parts = [f"{s.split('_')[1]}:{t[0]:%H:%M:%S}-{t[-1]:%H:%M:%S}({len(t)})"
                 for s, t in sorted(by.items())]
        print(f"  {cls:<10}{'  '.join(parts)}")

    # ---- palette / colour-scale consistency --------------------------------
    print("\n== COLOUR SCALE AND LAYOUT CONSISTENCY ==")
    hist, off = [], []
    for r in recs:
        a = np.asarray(Image.open(r["path"]).convert("RGB")
                       .resize((426, 487), Image.NEAREST)).astype(np.float32)
        ids, dist = quantize_to_ids(a)
        off.append(float((dist > 1200).mean()))
        hist.append(np.bincount(ids.ravel(), minlength=10) / ids.size)
    hist = np.array(hist)
    print(f"  pixels NOT on the 10-entry palette: {100*float(np.mean(off)):.4f} %")
    print("  -> every frame uses the SAME discrete palette: 8 turbo bands + black + white")
    print("\n  class      " + " ".join(f"  L{i} " for i in range(8)) + " BLACK  WHITE")
    for cls in CLASS_NAMES:
        ii = [i for i, r in enumerate(recs) if r["cls"] == cls]
        print(f"  {cls:<10}" + " ".join(f"{v*100:5.1f}" for v in hist[ii].mean(0)))

    print("\n  dynamic range per frame (auto-scaling would leave every frame spanning L0..L7)")
    present = hist[:, :8] > 0.0005
    print("  class      has_L0 has_L7 has_both")
    for cls in CLASS_NAMES:
        ii = [i for i, r in enumerate(recs) if r["cls"] == cls]
        m = len(ii)
        print(f"  {cls:<10} {present[ii,0].sum():2d}/{m:<3d} {present[ii,7].sum():2d}/{m:<3d} "
              f"{(present[ii,0]&present[ii,7]).sum():2d}/{m}")

    print("\n  mean band index per class (0 = lowest band, 7 = highest)")
    w = np.arange(8)
    band_stats = {}
    for cls in CLASS_NAMES:
        ii = [i for i, r in enumerate(recs) if r["cls"] == cls]
        lv = hist[ii][:, :8]
        lv = lv / lv.sum(1, keepdims=True)
        mi = (lv * w).sum(1)
        band_stats[cls] = dict(mean=float(mi.mean()), min=float(mi.min()), max=float(mi.max()),
                               black_pct=float(hist[ii][:, 8].mean() * 100))
        print(f"  {cls:<10} mean={mi.mean():.2f} min={mi.min():.2f} max={mi.max():.2f} "
              f"black={hist[ii][:,8].mean()*100:.1f}%")

    # ---- native render grid -------------------------------------------------
    print("\n== NATIVE RENDER GRID ==")
    periods = []
    for r in recs[::20]:
        a = np.asarray(Image.open(r["path"]).convert("RGB")).astype(np.int32)
        H, W, _ = a.shape
        dx = (np.abs(np.diff(a, axis=1)).sum(2) > 12).sum(0).astype(np.float64)
        dy = (np.abs(np.diff(a, axis=0)).sum(2) > 12).sum(1).astype(np.float64)
        periods.append((dominant_period(dx)[0], dominant_period(dy)[0]))
    pc = np.array(periods)
    cw, ch = float(np.median(pc[:, 0])), float(np.median(pc[:, 1]))
    W, H = next(iter(sizes))
    print(f"  median column period {cw:.2f} px -> {W/cw:.1f} cells across")
    print(f"  median row    period {ch:.2f} px -> {H/ch:.1f} cells down")
    print(f"  => native render grid is approximately {NATIVE_GRID_W} x {NATIVE_GRID_H}")

    # ---- figures ------------------------------------------------------------
    rgb_tiles, idx_tiles = [], []
    for cls in CLASS_NAMES:
        path = [r["path"] for r in recs if r["cls"] == cls][0]
        im = Image.open(path).convert("RGB").resize((TILE_W, TILE_H), Image.NEAREST)
        a = np.asarray(im)
        rgb_tiles.append(a)
        ids = quantize_to_ids(a.astype(np.float32))[0]
        g = np.where(ids < 8, ids * (255 // 7), np.where(ids == 8, 0, 255)).astype(np.uint8)
        idx_tiles.append(np.repeat(g[:, :, None], 3, axis=2))
    montage(rgb_tiles).save(os.path.join(figdir, "class_montage.png"))
    montage(idx_tiles).save(os.path.join(figdir, "band_index_montage.png"))
    Image.open(recs[-1]["path"]).convert("RGB").crop((400, 500, 1100, 1100)).save(
        os.path.join(figdir, "zoom_detail.png"))
    print(f"\nfigures -> {figdir}")

    # ---- machine-readable summary --------------------------------------------
    summary = dict(
        total_files=len(recs),
        dimensions={f"{k[0]}x{k[1]}": v for k, v in sizes.items()},
        formats=dict(fmts), modes=dict(modes),
        corrupt_files=corrupt,
        exact_duplicate_groups=len(dup_hashes),
        exact_duplicate_files=int(sum(md5c[h] for h in dup_hashes)),
        per_class=per_class,
        band_stats=band_stats,
        median_col_period_px=cw, median_row_period_px=ch,
        native_render_grid=[NATIVE_GRID_W, NATIVE_GRID_H],
        palette="8-level matplotlib turbo + black (below lowest band) + white (contour strokes)",
    )
    out = os.path.join(args.out, "dataset_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
