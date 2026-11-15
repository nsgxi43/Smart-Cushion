#!/usr/bin/env python3
"""
StrideSense — P3 collection WITH LIVE HEATMAP.

Participant-first layout:
DATASET/session_20261114_170217/
    P1/<six posture folders>/
    P2/<six posture folders>/
    P3/<six posture folders>/

For every P3 posture folder this saves:
    raw_data.csv
    pressure_data.csv

It also displays a live pressure heatmap during every repetition, using the
same key visualization settings from the original PLOT.PY workflow:
- 24 x 21 sensor matrix
- 20-frame unloaded baseline
- threshold_map = max(3 * noise_std, 20)
- zoom factor 6, interpolation order 1
- Gaussian smoothing sigma 0.6
- values below 20 set to zero
- live peak decay factor 0.95, minimum scale 100
- turbo colour map, 8 colour levels

Does NOT modify ESP32 firmware or PLOT.PY.
Does NOT move, edit, or overwrite P1/P2 data.

Install:
    pip install pyserial numpy matplotlib scipy

Run:
    python collect_p3_with_heatmap.py

Check PORT and SESSION_DIR below before running.
Press Q in the heatmap window to close the visualizer; collection continues.
"""

from pathlib import Path
from datetime import datetime
import csv
import sys
import time

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import zoom, gaussian_filter, binary_erosion

# ---------------- SETTINGS ----------------

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

ROWS = 24
COLS = 21
TOTAL_SENSORS = ROWS * COLS

SESSION_DIR = (
    Path(__file__).resolve().parent
    / "DATASET"
    / "session_20261114_170217"
)

PARTICIPANT_ID = "P3"

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

# Original PLOT.PY visualization settings
NOISE_MULTIPLIER = 3
MIN_PRESSURE = 20
AUTO_SCALE = True
FIXED_MAX = 1000
MIN_SCALE = 100
NUM_COLOUR_LEVELS = 8
COLOURMAP = "turbo"
UPSCALE = 6
SMOOTHING = 0.6

# ------------------------------------------


def check_existing_participants():
    if not SESSION_DIR.is_dir():
        raise FileNotFoundError(
            f"Session folder not found:\n{SESSION_DIR}\n"
            "Correct SESSION_DIR near the top of this script."
        )

    for pid in ("P1", "P2"):
        folder = SESSION_DIR / pid
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected participant folder missing: {folder}")
        missing = [name for name in POSTURES if not (folder / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{pid} is missing posture folders: {', '.join(missing)}"
            )

    p3_dir = SESSION_DIR / PARTICIPANT_ID
    if p3_dir.exists() and any(p3_dir.iterdir()):
        raise FileExistsError(
            f"{p3_dir} already contains data. Refusing to overwrite it. "
            "Back it up or choose a different participant ID."
        )

    p3_dir.mkdir(exist_ok=True)
    for posture in POSTURES:
        (p3_dir / posture).mkdir(exist_ok=True)

    print(f"P1 preserved: {SESSION_DIR / 'P1'}")
    print(f"P2 preserved: {SESSION_DIR / 'P2'}")
    print(f"P3 folders ready: {p3_dir}")


def read_frame(ser):
    """Read one valid 504-integer CSV frame; ignore boot/debug lines."""
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != TOTAL_SENSORS:
            continue
        try:
            values = np.asarray([int(x.strip()) for x in parts], dtype=np.float32)
        except ValueError:
            continue
        if np.any(values < 0) or np.any(values > 4095):
            continue
        return values


def collect_calibration(ser):
    print("\nKeep the sensor mat/insole UNLOADED.")
    input(f"Press ENTER to collect {CALIBRATION_FRAMES} baseline frames...")
    frames = []
    while len(frames) < CALIBRATION_FRAMES:
        frames.append(read_frame(ser))
        print(f"\rCalibration: {len(frames)}/{CALIBRATION_FRAMES}", end="", flush=True)
    print("\nCalibration complete.")

    stack = np.stack(frames)
    baseline = stack.mean(axis=0)
    noise_std = stack.std(axis=0)
    threshold_map = np.maximum(NOISE_MULTIPLIER * noise_std, MIN_PRESSURE)
    return baseline, threshold_map


def make_pressure(raw, baseline, threshold_map):
    pressure = np.abs(raw - baseline)
    pressure[pressure < threshold_map] = 0
    return np.clip(pressure, 0, 4095)


class LiveHeatmap:
    """Live view that follows the original PLOT.PY processing sequence."""

    def __init__(self, posture):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 8))
        self.fig.canvas.manager.set_window_title("StrideSense — P3 Live Heatmap")
        self.ax.set_title(f"P3 | {posture} | Live pressure")
        self.ax.set_xlabel("Sensor column")
        self.ax.set_ylabel("Sensor row")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.peak = float(MIN_SCALE)
        self.closed = False

        initial = np.zeros((ROWS * UPSCALE, COLS * UPSCALE), dtype=float)
        self.image = self.ax.imshow(
            initial,
            cmap=COLOURMAP,
            vmin=0,
            vmax=100,
            interpolation="nearest",
            origin="upper",
            aspect="auto",
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax, fraction=0.046, pad=0.04)
        self.colorbar.set_label("Relative pressure (%)")
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, pressure_flat, elapsed, rep):
        if self.closed:
            return

        matrix = pressure_flat.reshape(ROWS, COLS)
        enlarged = zoom(matrix, UPSCALE, order=1)
        smooth = gaussian_filter(enlarged, sigma=SMOOTHING)
        smooth[smooth < MIN_PRESSURE] = 0

        current_max = float(np.max(matrix)) if matrix.size else 0.0
        if AUTO_SCALE:
            self.peak = max(self.peak * 0.95, current_max, MIN_SCALE)
        else:
            self.peak = max(FIXED_MAX, MIN_SCALE)

        percent = np.clip((smooth / self.peak) * 100.0, 0, 100)

        # Match the original heatmap's active-region edge calculation.
        inside = smooth > 0
        edge = inside & ~binary_erosion(inside)
        self.image.set_data(percent)
        self.image.set_clim(0, 100)

        # Replace prior edge overlay cleanly.
        for artist in list(self.ax.lines):
            artist.remove()
        if np.any(edge):
            self.ax.contour(
                edge.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.45,
                alpha=0.8,
            )

        self.ax.set_title(
            f"P3 | Rep {rep}/{REPETITIONS} | {elapsed:4.1f}/{SECONDS_PER_REP}s"
            f" | Peak scale: {self.peak:.0f}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if not self.closed:
            plt.close(self.fig)
            self.closed = True


def countdown():
    for n in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Starting in {n}...", end="\r", flush=True)
        time.sleep(1)
    print("RECORDING NOW!       ")


def collect_one_posture(ser, posture, posture_dir, profile, baseline, threshold):
    raw_path = posture_dir / "raw_data.csv"
    pressure_path = posture_dir / "pressure_data.csv"
    if raw_path.exists() or pressure_path.exists():
        raise FileExistsError(f"CSV already exists in {posture_dir}; refusing to overwrite.")

    metadata = [
        "timestamp", "participant_id", "height_cm", "weight_kg", "sex",
        "posture", "rep", "frame_index", "elapsed_seconds",
    ]
    sensor_headers = [f"sensor_{i:03d}" for i in range(TOTAL_SENSORS)]
    saved_frames = 0

    with open(raw_path, "w", newline="") as raw_file, \
         open(pressure_path, "w", newline="") as pressure_file:

        raw_writer = csv.writer(raw_file)
        pressure_writer = csv.writer(pressure_file)
        raw_writer.writerow(metadata + sensor_headers)
        pressure_writer.writerow(metadata + sensor_headers)

        for rep in range(1, REPETITIONS + 1):
            print(f"\nP3 | {posture} | Rep {rep}/{REPETITIONS}")
            input("Position Person 3 in this posture, then press ENTER...")
            countdown()

            heatmap = LiveHeatmap(posture)
            start = time.monotonic()
            frame_index = 0

            try:
                while time.monotonic() - start < SECONDS_PER_REP:
                    raw = read_frame(ser)
                    elapsed = time.monotonic() - start
                    pressure = make_pressure(raw.copy(), baseline, threshold)

                    heatmap.update(pressure, elapsed, rep)

                    common = [
                        datetime.now().isoformat(timespec="milliseconds"),
                        PARTICIPANT_ID,
                        profile["height_cm"],
                        profile["weight_kg"],
                        profile["sex"],
                        posture,
                        rep,
                        frame_index,
                        round(elapsed, 4),
                    ]
                    raw_writer.writerow(common + raw.astype(int).tolist())
                    pressure_writer.writerow(common + pressure.astype(int).tolist())

                    frame_index += 1
                    saved_frames += 1
                    print(
                        f"\r{posture} | Rep {rep}/{REPETITIONS} | "
                        f"{frame_index} frames | {elapsed:4.1f}/{SECONDS_PER_REP}s",
                        end="",
                        flush=True,
                    )
                    raw_file.flush()
                    pressure_file.flush()

            finally:
                heatmap.close()

            print(f"\nSaved {frame_index} frames for Rep {rep}.")

    return saved_frames


def main():
    check_existing_participants()

    print("\n========== PERSON 3 DETAILS ==========")
    height = float(input("Actual height (cm): ").strip())
    weight = float(input("Actual weight (kg): ").strip())
    sex = input("Sex (Female/Male/Other/Prefer_not_to_say): ").strip()

    if not 100 <= height <= 230:
        raise ValueError("Height should be in cm (expected 100–230).")
    if not 25 <= weight <= 250:
        raise ValueError("Weight should be in kg (expected 25–250).")
    if not sex:
        raise ValueError("Enter a value, or Prefer_not_to_say.")

    profile = {"height_cm": height, "weight_kg": weight, "sex": sex}
    p3_dir = SESSION_DIR / PARTICIPANT_ID
    profile_path = p3_dir / "participant_profile.csv"
    with open(profile_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "height_cm", "weight_kg", "sex"])
        writer.writerow([PARTICIPANT_ID, height, weight, sex])

    print(f"\nConnecting to {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        print("Check PORT and close PLOT.PY/Serial Monitor before retrying.")
        sys.exit(1)

    total_frames = 0
    try:
        time.sleep(2)
        ser.reset_input_buffer()
        baseline, threshold = collect_calibration(ser)

        for index, posture in enumerate(POSTURES, start=1):
            print(f"\n\n========== P3 POSTURE {index}/6: {posture} ==========")
            print("Keep the physical sensor arrangement unchanged from P1 and P2.")
            total_frames += collect_one_posture(
                ser,
                posture,
                p3_dir / posture,
                profile,
                baseline,
                threshold,
            )
            print(f"Completed {posture}.")

    finally:
        ser.close()

    print("\n========== P3 COLLECTION COMPLETE ==========")
    print(f"Participant folder: {p3_dir}")
    print(f"Total frames saved: {total_frames}")
    print("Each posture folder contains raw_data.csv and pressure_data.csv.")
    print("P1 and P2 were not modified.")


if __name__ == "__main__":
    main()
