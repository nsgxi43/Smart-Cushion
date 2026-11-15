#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the Smart Cushion posture pipeline.

Everything that more than one stage needs lives here so that inspection,
preparation, training and evaluation can never drift apart:

  * the seven-class mapping, loaded from models/classes.json
  * the renderer's discrete colour palette and the band-index decoding
  * recording-session identification from the screenshot timestamps
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CLASSES_JSON = os.path.join(ROOT, "models", "classes.json")
DEFAULT_DATASET = os.path.join(ROOT, "dataset")

# --------------------------------------------------------------------------
# Class mapping - single source of truth
# --------------------------------------------------------------------------


def load_classes(path: str = CLASSES_JSON) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    entries = sorted(spec["classes"], key=lambda c: c["id"])
    assert [c["id"] for c in entries] == list(range(len(entries))), "class ids must be 0..N-1"
    return [c["name"] for c in entries]


CLASS_NAMES = load_classes()
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------
# Renderer palette
#
# The screenshots are a filled-contour render using eight evenly spaced
# samples of matplotlib's "turbo" colormap, plus pure black for area below the
# lowest filled band and pure white for the contour strokes themselves.
# 99.96 % of all pixels in the dataset fall exactly on these ten colours, so
# decoding a pixel back to its band index is exact rather than approximate.
#
# Band index is monotonic in pressure; raw RGB is not (turbo is not monotonic
# in luminance). Decoding to the band index is therefore the faithful way to
# reduce these screenshots to a single channel.
# --------------------------------------------------------------------------
TURBO_LEVELS = np.array(
    [
        [44, 20, 60],     # L0 lowest filled band
        [84, 116, 228],   # L1
        [100, 204, 212],  # L2
        [140, 252, 124],  # L3
        [212, 232, 92],   # L4
        [244, 160, 72],   # L5
        [200, 68, 36],    # L6
        [112, 20, 12],    # L7 highest filled band
    ],
    dtype=np.float32,
)
BLACK_ID, WHITE_ID = 8, 9
PALETTE = np.vstack([TURBO_LEVELS, [[0, 0, 0], [255, 255, 255]]]).astype(np.float32)
MAX_BAND = 7.0

# Native render grid measured from the screenshots (27.41 px / 27.44 px period).
NATIVE_GRID_W, NATIVE_GRID_H = 62, 71
# Working resolution: 4x the native grid, so NEAREST downscaling cannot blur
# two palette colours into an unmatched intermediate.
WORK_W, WORK_H = NATIVE_GRID_W * 4, NATIVE_GRID_H * 4
# CNN input: close to the native grid, divisible by 4 for two pooling stages.
INPUT_W, INPUT_H = 64, 72
INPUT_SHAPE = (INPUT_H, INPUT_W, 1)


def quantize_to_ids(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map every pixel to the nearest palette entry. Returns (ids, sq_distance)."""
    d = ((rgb[:, :, None, :] - PALETTE[None, None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8), d.min(-1)


def _fill_contour_strokes(value: np.ndarray, mask: np.ndarray, iters: int = 12) -> np.ndarray:
    """Replace white contour-stroke pixels with the mean of their known neighbours.

    A contour stroke sits exactly on the boundary between two bands, so the
    local mean is the natural estimate for what it covers.
    """
    v = value.copy()
    known = (~mask).astype(np.float32)
    v[mask] = 0.0
    for _ in range(iters):
        if not mask.any():
            break
        num = np.zeros_like(v)
        den = np.zeros_like(v)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            num += np.roll(np.roll(v * known, dy, 0), dx, 1)
            den += np.roll(np.roll(known, dy, 0), dx, 1)
        fillable = mask & (den > 0)
        if not fillable.any():
            break
        v[fillable] = num[fillable] / den[fillable]
        known[fillable] = 1.0
        mask = mask & ~fillable
    return v


def decode_band_image(path: str,
                      out_w: int = INPUT_W,
                      out_h: int = INPUT_H) -> tuple[np.ndarray, dict]:
    """Decode one heat-map screenshot into a normalized single-channel array.

    Steps:
      1. NEAREST downscale to 4x the native render grid (palette colours survive exactly).
      2. Decode each pixel to its band index 0..7, or black / white.
      3. Black (below the lowest filled band) -> band 0, the lowest pressure.
      4. White (contour stroke) -> mean of known neighbours.
      5. Area-average down to the CNN input size.
      6. Scale to [0, 1] by dividing by 7.

    Returns (float32 array of shape (out_h, out_w), per-image statistics).
    """
    im = Image.open(path).convert("RGB")
    orig_size = im.size
    small = np.asarray(im.resize((WORK_W, WORK_H), Image.NEAREST)).astype(np.float32)

    ids, dist = quantize_to_ids(small)
    off_palette = float((dist > 1200).mean())

    value = ids.astype(np.float32)
    value[ids == BLACK_ID] = 0.0
    value = _fill_contour_strokes(value, ids == WHITE_ID)
    np.clip(value, 0.0, MAX_BAND, out=value)

    # Area-average to the input grid (BOX == exact area resampling).
    arr = np.array(
        Image.fromarray(value, mode="F").resize((out_w, out_h), Image.BOX),
        dtype=np.float32, copy=True,
    )
    arr /= MAX_BAND

    stats = dict(
        orig_w=orig_size[0],
        orig_h=orig_size[1],
        off_palette_frac=off_palette,
        black_frac=float((ids == BLACK_ID).mean()),
        white_frac=float((ids == WHITE_ID).mean()),
        mean_band=float(value.mean()),
    )
    return arr, stats


# --------------------------------------------------------------------------
# Recording sessions
# --------------------------------------------------------------------------
# macOS screenshot filenames, e.g. "Screenshot 2026-11-15 at 3.08.55 PM.png".
# Non-ASCII bytes in the names are mojibake introduced by the Windows copy, so
# the filename is reduced to ASCII before matching.
TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*(AM|PM)", re.I)
SESSION_GAP_S = 20.0


def ascii_name(name: str) -> str:
    return "".join(ch if ord(ch) < 127 else " " for ch in name)


def parse_timestamp(name: str):
    m = TS_RE.search(ascii_name(name))
    if not m:
        return None
    y, mo, d, hh, mm, ss, ap = m.groups()
    hh = int(hh) % 12
    if ap.upper() == "PM":
        hh += 12
    return dt.datetime(int(y), int(mo), int(d), hh, int(mm), int(ss))


def list_dataset(dataset_dir: str = DEFAULT_DATASET) -> list[dict]:
    """Enumerate every image, with class, label id, timestamp and session id."""
    recs = []
    for label, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(dataset_dir, cls)
        if not os.path.isdir(folder):
            raise SystemExit(f"missing class folder: {folder}")
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not paths:
            raise SystemExit(f"no PNG files in {folder}")
        for p in paths:
            recs.append(
                dict(
                    path=p,
                    name=ascii_name(os.path.basename(p)),
                    cls=cls,
                    label=label,
                    ts=parse_timestamp(os.path.basename(p)),
                )
            )
    assign_sessions(recs)
    return recs


def assign_sessions(recs: list[dict]) -> None:
    """Group each class's frames into recording sessions using the timestamps.

    A gap longer than SESSION_GAP_S starts a new session. Frames without a
    usable timestamp fall back to one shared pseudo-session per class, which
    the split logic then treats as un-splittable.
    """
    for cls in CLASS_NAMES:
        rs = [r for r in recs if r["cls"] == cls]
        dated = sorted([r for r in rs if r["ts"] is not None], key=lambda r: r["ts"])
        sid = 0
        for i, r in enumerate(dated):
            if i > 0 and (r["ts"] - dated[i - 1]["ts"]).total_seconds() > SESSION_GAP_S:
                sid += 1
            r["session"] = f"{cls}_s{sid}"
            r["order"] = i
        for i, r in enumerate(r for r in rs if r["ts"] is None):
            r["session"] = f"{cls}_sNA"
            r["order"] = i


def set_seeds(seed: int = 1337) -> None:
    """Seed Python, NumPy and TensorFlow for reproducibility."""
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
