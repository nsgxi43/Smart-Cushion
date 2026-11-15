
import csv
import shutil
import time
import serial
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import zoom, gaussian_filter, binary_erosion
from datetime import datetime
from pathlib import Path


# ============================================================
# P5 RIGHT LEAN ONLY — ORIGINAL PLOT.PY HEATMAP
# ============================================================

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

ROWS = 24
COLS = 21
TOTAL_SENSORS = ROWS * COLS

CALIBRATION_FRAMES = 20

NOISE_MULTIPLIER = 3
MIN_PRESSURE = 20

AUTO_SCALE = True
FIXED_MAX = 1000
MIN_SCALE = 100

NUM_COLOUR_LEVELS = 8
COLOURMAP = "turbo"
UPSCALE = 6
SMOOTHING = 0.6

CAPTURE_DURATION_SECONDS = 10
REPETITIONS = 5
COUNTDOWN_SECONDS = 3

SESSION_TIMESTAMP = "20261114_170217"

DATASET_ROOT = Path(__file__).resolve().parent / "DATASET"

POSITION_CODE = "P5"
POSITION_NAME = "Right_Lean"


# ============================================================
# FRAME READING
# ============================================================

def read_frame(ser):
    start = time.time()

    while time.time() - start < 10:
        line = ser.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if not line:
            continue

        try:
            values = [int(x) for x in line.split(",")]

            if len(values) == TOTAL_SENSORS:
                return np.array(values).reshape(ROWS, COLS)

        except ValueError:
            continue

    raise TimeoutError(
        f"No valid {TOTAL_SENSORS}-value frame received."
    )


def calculate_pressure(raw, baseline):
    return np.abs(raw - baseline)


def apply_threshold(pressure, threshold_map):
    return np.where(
        pressure > threshold_map,
        pressure,
        0
    )


# ============================================================
# ORIGINAL PLOT.PY HEATMAP PROCESSING
# ============================================================

def create_heatmap(data):
    enlarged = zoom(data, UPSCALE, order=1)
    enlarged = gaussian_filter(
        enlarged,
        sigma=SMOOTHING
    )
    enlarged[enlarged < MIN_PRESSURE] = 0

    return enlarged


def find_outline(smooth):
    inside = smooth > 0
    edge = inside & ~binary_erosion(inside)

    return np.ma.masked_where(
        ~edge,
        edge.astype(float)
    )


def setup_original_heatmap():
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 9))

    height = ROWS * UPSCALE
    width = COLS * UPSCALE

    levels = np.linspace(
        0, 100, NUM_COLOUR_LEVELS + 1
    )
    levels[0] = 0.001

    cmap = plt.get_cmap(
        COLOURMAP,
        NUM_COLOUR_LEVELS
    ).copy()

    cmap.set_under("black")
    cmap.set_over("white")

    norm = BoundaryNorm(
        levels,
        NUM_COLOUR_LEVELS
    )

    heat_img = ax.imshow(
        np.zeros((height, width)),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )

    outline_img = ax.imshow(
        np.ma.masked_all((height, width)),
        cmap=ListedColormap(["white"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )

    colorbar = plt.colorbar(heat_img, ax=ax)
    colorbar.set_label(
        "Pressure (% of current peak)"
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    fig.tight_layout()

    return fig, ax, heat_img, outline_img


# ============================================================
# SAVE P5 DATA
# ============================================================

def save_position(session_dir, records):

    final_dir = session_dir / (
        f"{POSITION_CODE}_{POSITION_NAME}"
    )

    temp_dir = session_dir / (
        f".{POSITION_CODE}_{POSITION_NAME}_recapture_tmp"
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    raw_array = np.stack([
        r["raw"] for r in records
    ]).astype(np.int32)

    pressure_array = np.stack([
        r["pressure"] for r in records
    ]).astype(np.float32)

    np.save(temp_dir / "raw_data.npy", raw_array)
    np.save(
        temp_dir / "pressure_data.npy",
        pressure_array
    )

    sensor_columns = [
        f"sensor_{i:03d}"
        for i in range(1, TOTAL_SENSORS + 1)
    ]

    metadata_columns = [
        "session_timestamp",
        "frame_timestamp",
        "position_code",
        "position_label",
        "repetition",
        "frame_in_repetition",
        "elapsed_seconds",
    ]

    for filename, key in [
        ("raw_data.csv", "raw"),
        ("pressure_data.csv", "pressure"),
    ]:

        with open(
            temp_dir / filename,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                metadata_columns + sensor_columns
            )

            for r in records:
                writer.writerow([
                    SESSION_TIMESTAMP,
                    r["frame_timestamp"],
                    POSITION_CODE,
                    POSITION_NAME,
                    r["repetition"],
                    r["frame_in_repetition"],
                    f'{r["elapsed_seconds"]:.4f}',
                    *np.asarray(r[key]).reshape(-1).tolist(),
                ])

    with open(
        temp_dir / "metadata.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "session_timestamp",
            "position_code",
            "position_label",
            "repetitions_planned",
            "capture_duration_seconds",
            "captured_frames",
            "matrix_rows",
            "matrix_columns",
        ])

        writer.writerow([
            SESSION_TIMESTAMP,
            POSITION_CODE,
            POSITION_NAME,
            REPETITIONS,
            CAPTURE_DURATION_SECONDS,
            len(records),
            ROWS,
            COLS,
        ])

    # Replace P5 only after successful completion.
    if final_dir.exists():
        shutil.rmtree(final_dir)

    temp_dir.rename(final_dir)

    print("\nP5 RIGHT LEAN SAVED SUCCESSFULLY!")
    print("Frames:", len(records))
    print("Location:", final_dir)


# ============================================================
# MAIN
# ============================================================

def main():

    session_dir = (
        DATASET_ROOT /
        f"session_{SESSION_TIMESTAMP}"
    )

    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"Session folder not found:\n{session_dir}\n"
            "Check SESSION_TIMESTAMP. No data was changed."
        )

    print("\n======================================")
    print(" P5 — RIGHT LEAN RECAPTURE")
    print("======================================")

    print("Session:", session_dir)
    print("Position: RIGHT LEAN")
    print("Repetitions:", REPETITIONS)
    print("Duration:", CAPTURE_DURATION_SECONDS, "seconds")
    print("\nP1-P4 and P6 will NOT be modified.")
    print("\nKeep the cushion EMPTY for calibration.\n")

    ser = None
    fig = None
    records = []

    try:

        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=3
        )

        time.sleep(2)
        ser.reset_input_buffer()

        print("Connected to ESP32.")

        # ---------------- CALIBRATION ----------------

        calibration = []

        for i in range(CALIBRATION_FRAMES):

            calibration.append(read_frame(ser))

            print(
                f"Calibration: "
                f"{i + 1}/{CALIBRATION_FRAMES}"
            )

        calibration = np.array(calibration)

        baseline = np.mean(calibration, axis=0)
        noise = np.std(calibration, axis=0)

        threshold_map = np.maximum(
            NOISE_MULTIPLIER * noise,
            MIN_PRESSURE
        )

        print("\nCalibration complete!")

        print(
            "Baseline min / mean / max:",
            round(float(np.min(baseline))),
            round(float(np.mean(baseline))),
            round(float(np.max(baseline)))
        )

        print(
            "Average noise:",
            round(float(np.mean(noise)), 1)
        )

        print("\nYou can sit down now.\n")

        # ---------------- HEATMAP ----------------

        fig, ax, heat_img, outline_img = (
            setup_original_heatmap()
        )

        peak = MIN_SCALE

        # ---------------- RIGHT LEAN CAPTURE ----------------

        for repetition in range(1, REPETITIONS + 1):

            input(
                f"\nAssume RIGHT LEAN posture. "
                f"Press ENTER for repetition "
                f"{repetition}/{REPETITIONS}..."
            )

            for count in range(
                COUNTDOWN_SECONDS,
                0,
                -1
            ):
                print(f"Starting in {count}...")
                plt.pause(0.01)

            print("CAPTURING RIGHT LEAN...")

            capture_start = time.perf_counter()
            frame_number = 0

            while (
                time.perf_counter() - capture_start
                < CAPTURE_DURATION_SECONDS
            ):

                raw = read_frame(ser)

                raw_change = calculate_pressure(
                    raw,
                    baseline
                )

                pressure = apply_threshold(
                    raw_change,
                    threshold_map
                )

                if AUTO_SCALE:
                    peak = max(
                        peak * 0.95,
                        float(np.max(pressure)),
                        MIN_SCALE
                    )
                else:
                    peak = FIXED_MAX

                smooth = create_heatmap(pressure)
                percent = smooth / peak * 100

                heat_img.set_data(percent)
                outline_img.set_data(
                    find_outline(smooth)
                )

                elapsed = (
                    time.perf_counter() - capture_start
                )

                frame_number += 1

                ax.set_title(
                    f"RIGHT LEAN | Rep "
                    f"{repetition}/{REPETITIONS} | "
                    f"{elapsed:.1f}/"
                    f"{CAPTURE_DURATION_SECONDS}s | "
                    f"Peak: {np.max(pressure):.0f} | "
                    f"Active points: "
                    f"{np.count_nonzero(pressure)}"
                )

                fig.canvas.draw_idle()
                plt.pause(0.01)

                records.append({
                    "raw": raw.copy(),
                    "pressure": pressure.copy(),
                    "frame_timestamp": datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    "repetition": repetition,
                    "frame_in_repetition": frame_number,
                    "elapsed_seconds": elapsed,
                })

            print(
                f"Repetition {repetition} complete: "
                f"{frame_number} frames."
            )

        # Save only after all 5 repetitions complete.
        save_position(session_dir, records)

        print("\n======================================")
        print("RIGHT LEAN RECAPTURE COMPLETE!")
        print("P1-P4 and P6 remain untouched.")
        print("======================================")

    except KeyboardInterrupt:

        print("\nCapture interrupted.")
        print("Incomplete P5 data was NOT saved.")
        print("The previous P5 folder remains intact.")

    finally:

        if ser is not None and ser.is_open:
            ser.close()

        if fig is not None:
            plt.close(fig)

        print("Serial connection closed.")


if __name__ == "__main__":
    main()
