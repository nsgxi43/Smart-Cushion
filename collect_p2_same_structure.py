#!/usr/bin/env python3
"""
StrideSense participant collector — participant-first folder layout.

Target structure:
DATASET/
└── session_20261114_170217/
    ├── P1/
    │   ├── P1_Normal/
    │   ├── P2_Forward_Lean/
    │   ├── P3_Slouching/
    │   ├── P4_Left_Lean/
    │   ├── P5_Right_Lean/
    │   └── P6_Backward_Lean/
    └── P2/
        ├── P1_Normal/
        ├── P2_Forward_Lean/
        ├── P3_Slouching/
        ├── P4_Left_Lean/
        ├── P5_Right_Lean/
        └── P6_Backward_Lean/

This script:
1. Moves the six existing posture folders into P1/ if they are currently
   directly inside the session folder.
2. Creates the matching six folders inside P2/.
3. Collects P2's actual measurements and records.
4. Saves separate raw_data.csv and pressure_data.csv inside each P2 posture.
5. Does not modify ESP32 firmware or PLOT.PY.
6. Refuses to overwrite existing P2 data.

Run from the FY folder:
    pip install pyserial numpy
    python collect_p2_same_structure.py

Check PORT and SESSION_DIR below before running.
"""

from pathlib import Path
from datetime import datetime
import csv
import re
import shutil
import sys
import time

import numpy as np
import serial

# ---------------- SETTINGS TO CHECK ----------------

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

ROWS = 24
COLS = 21
TOTAL_SENSORS = ROWS * COLS

# Use the exact session folder shown in your VS Code screenshot.
SESSION_DIR = Path(__file__).resolve().parent / "DATASET" / "session_20261114_170217"

P1_ID = "P1"
P2_ID = "P2"

REPETITIONS = 5
SECONDS_PER_REP = 10
CALIBRATION_FRAMES = 20
COUNTDOWN_SECONDS = 5

POSTURES = [
    "P1_Normal",
    "P2_Forward_Lean",
    "P3_Slouching",
    "P4_Left_Lean",
    "P5_Right_Lean",
    "P6_Backward_Lean",
]

# ---------------------------------------------------


def prepare_folders():
    """Move current root posture folders into P1, then create P2 folders."""
    if not SESSION_DIR.exists():
        raise FileNotFoundError(
            f"Session folder not found:\n{SESSION_DIR}\n"
            "Check SESSION_DIR at the top of this script."
        )

    p1_dir = SESSION_DIR / P1_ID
    p2_dir = SESSION_DIR / P2_ID
    p1_dir.mkdir(exist_ok=True)

    # If the existing layout is session/P1_Normal ... move those into session/P1/.
    for posture in POSTURES:
        old_location = SESSION_DIR / posture
        new_location = p1_dir / posture

        if old_location.exists():
            if new_location.exists():
                raise FileExistsError(
                    f"Both locations exist:\n{old_location}\n{new_location}\n"
                    "Resolve this manually; refusing to merge or overwrite."
                )
            shutil.move(str(old_location), str(new_location))

    # Confirm all P1 posture folders are present before collecting P2.
    missing_p1 = [p for p in POSTURES if not (p1_dir / p).is_dir()]
    if missing_p1:
        raise FileNotFoundError(
            "Could not find these existing P1 posture folders:\n"
            + "\n".join(missing_p1)
            + "\nNo P2 data has been collected."
        )

    # Refuse to mix with a previous/partial P2 collection.
    if p2_dir.exists() and any(p2_dir.iterdir()):
        raise FileExistsError(
            f"{p2_dir} already contains files. "
            "Choose a new participant ID or move the old P2 folder first."
        )

    p2_dir.mkdir(exist_ok=True)
    for posture in POSTURES:
        (p2_dir / posture).mkdir(exist_ok=True)

    print("\nFolder structure ready:")
    print(f"  Existing participant data: {p1_dir}")
    print(f"  New participant data:      {p2_dir}")


def read_frame(ser):
    """Read exactly 504 integer ADC values; skip boot/debug/non-frame lines."""
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != TOTAL_SENSORS:
            continue
        try:
            values = np.asarray([int(v.strip()) for v in parts], dtype=np.float32)
        except ValueError:
            continue
        if np.any(values < 0) or np.any(values > 4095):
            continue
        return values


def collect_calibration(ser):
    print(f"\nKeep the sensor mat/insole UNLOADED.")
    input(f"Press ENTER to collect {CALIBRATION_FRAMES} baseline frames...")
    frames = []
    while len(frames) < CALIBRATION_FRAMES:
        frames.append(read_frame(ser))
        print(f"\rCalibration: {len(frames)}/{CALIBRATION_FRAMES}", end="", flush=True)
    print("\nCalibration complete.")
    stack = np.stack(frames)
    baseline = stack.mean(axis=0)
    threshold = np.maximum(3.0 * stack.std(axis=0), 20.0)
    return baseline, threshold


def make_pressure(raw, baseline, threshold):
    pressure = np.abs(raw - baseline)
    pressure[pressure < threshold] = 0
    return np.clip(pressure, 0, 4095)


def countdown():
    for n in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Starting in {n}...", end="\r", flush=True)
        time.sleep(1)
    print("RECORDING NOW!       ")


def collect_posture(ser, participant, profile, posture, posture_dir,
                    baseline, threshold):
    raw_path = posture_dir / "raw_data.csv"
    pressure_path = posture_dir / "pressure_data.csv"

    # Never overwrite a previous capture.
    if raw_path.exists() or pressure_path.exists():
        raise FileExistsError(
            f"Data already exists in {posture_dir}. "
            "Refusing to overwrite."
        )

    metadata = [
        "timestamp", "participant_id", "height_cm", "weight_kg", "sex",
        "posture", "rep", "frame_index", "elapsed_seconds"
    ]
    sensor_headers = [f"sensor_{i:03d}" for i in range(TOTAL_SENSORS)]

    posture_frames = 0
    with open(raw_path, "w", newline="") as raw_file, \
         open(pressure_path, "w", newline="") as pressure_file:
        raw_writer = csv.writer(raw_file)
        pressure_writer = csv.writer(pressure_file)
        raw_writer.writerow(metadata + sensor_headers)
        pressure_writer.writerow(metadata + sensor_headers)

        for rep in range(1, REPETITIONS + 1):
            print(f"\n{participant} | {posture} | Rep {rep}/{REPETITIONS}")
            input("Position participant in this posture; press ENTER when ready...")
            countdown()

            start = time.monotonic()
            frame_index = 0

            while time.monotonic() - start < SECONDS_PER_REP:
                raw = read_frame(ser)
                elapsed = time.monotonic() - start
                pressure = make_pressure(raw.copy(), baseline, threshold)
                timestamp = datetime.now().isoformat(timespec="milliseconds")

                common = [
                    timestamp, participant, profile["height_cm"],
                    profile["weight_kg"], profile["sex"], posture,
                    rep, frame_index, round(elapsed, 4)
                ]
                raw_writer.writerow(common + raw.astype(int).tolist())
                pressure_writer.writerow(common + pressure.astype(int).tolist())

                frame_index += 1
                posture_frames += 1
                print(
                    f"\r{posture} | Rep {rep}/{REPETITIONS} | "
                    f"{frame_index} frames | {elapsed:4.1f}/{SECONDS_PER_REP}s",
                    end="", flush=True
                )

            print(f"\nSaved {frame_index} frames for Rep {rep}.")
            raw_file.flush()
            pressure_file.flush()

    return posture_frames


def main():
    prepare_folders()

    print("\n========== P2 PARTICIPANT DETAILS ==========")
    print("Enter the NEW participant's actual self-reported measurements.")
    height = float(input("Height (cm): ").strip())
    weight = float(input("Weight (kg): ").strip())
    sex = input("Sex (Female/Male/Other/Prefer_not_to_say): ").strip()

    if not 100 <= height <= 230:
        raise ValueError("Height should be in cm (expected 100–230).")
    if not 25 <= weight <= 250:
        raise ValueError("Weight should be in kg (expected 25–250).")
    if not sex:
        raise ValueError("Enter a value, or Prefer_not_to_say.")

    profile = {"height_cm": height, "weight_kg": weight, "sex": sex}
    p2_dir = SESSION_DIR / P2_ID
    profile_path = p2_dir / "participant_profile.csv"
    if profile_path.exists():
        raise FileExistsError(f"{profile_path} already exists; refusing to overwrite.")
    with open(profile_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "height_cm", "weight_kg", "sex"])
        writer.writerow([P2_ID, height, weight, sex])

    print(f"\nConnecting to {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        print("Check PORT and close PLOT.PY/Serial Monitor before retrying.")
        sys.exit(1)

    total_frames = 0
    try:
        time.sleep(2)  # ESP32 may reset on serial connection.
        ser.reset_input_buffer()
        baseline, threshold = collect_calibration(ser)

        for i, posture in enumerate(POSTURES, start=1):
            print(f"\n\n========== P2 POSTURE {i}/6: {posture} ==========")
            print("Keep the sensor arrangement unchanged from P1.")
            frames = collect_posture(
                ser, P2_ID, profile, posture, p2_dir / posture,
                baseline, threshold
            )
            total_frames += frames
            print(f"Completed {posture}: {frames} frames.")

    finally:
        ser.close()

    print("\n========== P2 COLLECTION COMPLETE ==========")
    print(f"Participant folder: {p2_dir}")
    print(f"Total captured frames: {total_frames}")
    print("Each posture folder contains raw_data.csv and pressure_data.csv.")
    print("P1's original CSV files were moved, not edited.")


if __name__ == "__main__":
    main()
