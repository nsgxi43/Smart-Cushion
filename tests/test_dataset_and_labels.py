"""Feature shape, sensor ordering, and label mapping."""
import re

import numpy as np
import pandas as pd
import pytest

import common
from common import (CLASS_NAMES, CLASS_TO_INDEX, COLS, N_SENSORS,
                    POSTURE_FOLDER_TO_CLASS, ROWS, sensor_columns)


def test_geometry_constants():
    assert ROWS == 24 and COLS == 21
    assert N_SENSORS == 504 == ROWS * COLS


def test_label_mapping_is_complete_and_injective():
    assert len(POSTURE_FOLDER_TO_CLASS) == 6
    assert sorted(POSTURE_FOLDER_TO_CLASS.values()) == sorted(CLASS_NAMES)
    assert len(set(POSTURE_FOLDER_TO_CLASS.values())) == 6
    assert [CLASS_TO_INDEX[c] for c in CLASS_NAMES] == [0, 1, 2, 3, 4, 5]


def test_label_mapping_matches_folder_semantics():
    """The original position code must map to the class of the same meaning."""
    for folder, cls in POSTURE_FOLDER_TO_CLASS.items():
        code, meaning = folder.split("_", 1)
        assert re.fullmatch(r"P[1-6]", code)
        assert cls.split("_", 1)[1] == meaning, (folder, cls)
        # P1..P6 maps to C0..C5 in order
        assert int(code[1:]) - 1 == CLASS_TO_INDEX[cls]


def test_recordings_discovered():
    recs = common.discover_recordings()
    assert len(recs) == 24, "expected 4 real participants x 6 postures"
    parts = sorted({r["participant"] for r in recs})
    assert parts == ["P1", "P2", "P3", "P4"]
    for r in recs:
        assert r["posture_class"] in CLASS_NAMES


def test_sensor_columns_order_both_schemas():
    """P1 is 1-indexed, P2+ is 0-indexed; both must sort to firmware order."""
    p1 = pd.DataFrame(columns=["session_timestamp"]
                      + ["sensor_%03d" % i for i in range(1, N_SENSORS + 1)])
    p2 = pd.DataFrame(columns=["timestamp"]
                      + ["sensor_%03d" % i for i in range(N_SENSORS)])
    c1, c2 = sensor_columns(p1), sensor_columns(p2)
    assert len(c1) == len(c2) == N_SENSORS
    assert c1[0] == "sensor_001" and c1[-1] == "sensor_504"
    assert c2[0] == "sensor_000" and c2[-1] == "sensor_503"
    # strictly increasing numeric suffix in both cases
    for cols in (c1, c2):
        nums = [int(c.split("_")[1]) for c in cols]
        assert nums == sorted(nums)


def test_row_major_feature_index():
    """index = row * 21 + col, matching the firmware print order."""
    flat = np.arange(N_SENSORS).reshape(ROWS, COLS)
    for r in range(ROWS):
        for c in range(COLS):
            assert flat[r, c] == r * COLS + c
    assert common.frames_to_matrix(np.arange(N_SENSORS)[None, :]).shape == (1, ROWS, COLS)


def test_dataset_shapes(dataset):
    n = dataset["y"].shape[0]
    assert dataset["X_pressure"].shape == (n, N_SENSORS)
    assert dataset["X_raw"].shape == (n, N_SENSORS)
    for k in ("participant", "posture", "group", "rep", "is_synthetic",
              "source_participant"):
        assert dataset[k].shape[0] == n
    assert set(np.unique(dataset["y"]).tolist()) == set(range(6))
    assert list(dataset["class_names"]) == CLASS_NAMES


def test_dataset_values_in_adc_range(dataset):
    for k in ("X_pressure", "X_raw"):
        v = dataset[k]
        assert v.min() >= 0 and v.max() <= 4095, k


def test_group_key_identifies_one_recording_take(dataset):
    """A group must never span two participants, postures or repetitions."""
    df = pd.DataFrame({
        "group": dataset["group"], "participant": dataset["participant"],
        "posture": dataset["posture"], "rep": dataset["rep"],
    })
    g = df.groupby("group").nunique()
    assert (g["participant"] == 1).all()
    assert (g["posture"] == 1).all()
    assert (g["rep"] == 1).all()


def test_posture_label_matches_group_key(dataset):
    posture_from_group = np.array([g.split("|")[1] for g in dataset["group"]])
    assert (posture_from_group == dataset["posture"]).all()
    idx = np.array([CLASS_TO_INDEX[p] for p in dataset["posture"]])
    assert (idx == dataset["y"]).all()


def test_synthetic_rows_declare_a_real_source(dataset):
    syn = dataset["is_synthetic"]
    if not syn.any():
        pytest.skip("no synthetic data built")
    srcs = set(np.unique(dataset["source_participant"][syn]).tolist())
    assert srcs <= {"P1", "P2", "P3", "P4"}, srcs
    # real rows are their own source
    real = ~syn
    assert (dataset["source_participant"][real] == dataset["participant"][real]).all()
