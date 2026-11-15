"""Shared constants and dataset loading helpers for the Smart Cushion project.

Sensor geometry and the ADC contract are fixed by the ESP32 firmware and must
not be changed here: ROWS=24, COLS=21, row-major flattening (index = r*21 + c),
12-bit ADC (0..4095).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 24
COLS = 21
N_SENSORS = ROWS * COLS          # 504
ADC_MIN = 0
ADC_MAX = 4095

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "DATASET"
AUGMENTED_DIR = PROJECT_ROOT / "AUGMENTED_DATASET"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Posture label mapping.
#
# The original dataset stores postures in folders named with a *position code*
# P1..P6 that collides visually with participant ids P1..P4. The canonical
# class names required by the project are C0..C5. This mapping is the single
# source of truth and is asserted in tests/test_label_mapping.py.
# ---------------------------------------------------------------------------
POSTURE_FOLDER_TO_CLASS = {
    "P1_Normal":        "C0_Normal",
    "P2_Forward_Lean":  "C1_Forward_Lean",
    "P3_Slouching":     "C2_Slouching",
    "P4_Left_Lean":     "C3_Left_Lean",
    "P5_Right_Lean":    "C4_Right_Lean",
    "P6_Backward_Lean": "C5_Backward_Lean",
}

CLASS_NAMES = [
    "C0_Normal",
    "C1_Forward_Lean",
    "C2_Slouching",
    "C3_Left_Lean",
    "C4_Right_Lean",
    "C5_Backward_Lean",
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

POSTURE_FOLDERS = list(POSTURE_FOLDER_TO_CLASS.keys())

# Metadata column sets seen in the two collection script generations.
SCHEMA_P1_META = [
    "session_timestamp", "frame_timestamp", "position_code", "position_label",
    "repetition", "frame_in_repetition", "elapsed_seconds",
]
SCHEMA_P2PLUS_META = [
    "timestamp", "participant_id", "height_cm", "weight_kg", "sex",
    "posture", "rep", "frame_index", "elapsed_seconds",
]


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return the sensor columns of a recording in firmware (row-major) order.

    Both schemas name columns sensor_NNN but P1 is 1-indexed (001..504) and
    P2+ is 0-indexed (000..503). Sorting by the numeric suffix recovers the
    firmware transmission order in both cases.
    """
    cols = [c for c in df.columns if re.fullmatch(r"sensor_\d+", c)]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    if len(cols) != N_SENSORS:
        raise ValueError(f"expected {N_SENSORS} sensor columns, found {len(cols)}")
    return cols


def detect_schema(df: pd.DataFrame) -> str:
    if "position_code" in df.columns:
        return "P1_LEGACY"
    if "participant_id" in df.columns:
        return "P2PLUS"
    raise ValueError(f"unrecognised schema, columns start: {list(df.columns)[:6]}")


def repetition_column(df: pd.DataFrame) -> str:
    return "repetition" if "repetition" in df.columns else "rep"


def frame_index_column(df: pd.DataFrame) -> str:
    return "frame_in_repetition" if "frame_in_repetition" in df.columns else "frame_index"


def discover_recordings(dataset_dir: Path = DATASET_DIR) -> list[dict]:
    """Find every (session, participant, posture) recording folder."""
    found = []
    for session_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for part_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            for posture in POSTURE_FOLDERS:
                pdir = part_dir / posture
                if not pdir.is_dir():
                    continue
                found.append({
                    "session": session_dir.name,
                    "participant": part_dir.name,
                    "posture_folder": posture,
                    "posture_class": POSTURE_FOLDER_TO_CLASS[posture],
                    "label": CLASS_TO_INDEX[POSTURE_FOLDER_TO_CLASS[posture]],
                    "dir": pdir,
                    "raw_csv": pdir / "raw_data.csv",
                    "pressure_csv": pdir / "pressure_data.csv",
                    "has_metadata": (pdir / "metadata.csv").is_file(),
                    "has_npy": (pdir / "raw_data.npy").is_file(),
                })
    return found


def load_recording(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load one recording CSV. Returns (metadata df, frames (N,504) float32, sensor col names)."""
    df = pd.read_csv(csv_path)
    scols = sensor_columns(df)
    frames = df[scols].to_numpy(dtype=np.float32)
    meta = df.drop(columns=scols)
    return meta, frames, scols


def frames_to_matrix(frames: np.ndarray) -> np.ndarray:
    """(N,504) -> (N,24,21) using the firmware's row-major ordering."""
    return frames.reshape(-1, ROWS, COLS)
