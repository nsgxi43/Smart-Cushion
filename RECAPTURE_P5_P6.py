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
# P5 + P6 RECAPTURE — ORIGINAL PLOT.PY HEATMAP LOGIC PRESERVED
# ============================================================

# ---------------- ORIGINAL CONFIGURATION ----------------
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

# ---------------- RECAPTURE CONFIGURATION ----------------
CAPTURE_DURATION_SECONDS = 10
REPETITIONS_PER_POSITION = 5
COUNTDOWN_SECONDS = 3

# Exact existing session shown in your VS Code Explorer.
SESSION_TIMESTAMP = "20261114_170217"
DATASET_ROOT = Path(__file__).resolve().parent / "DATASET"
POSITIONS_TO_RECAPTURE = [
    ("P5", "Right_Lean"),
    ("P6", "Backward_Lean"),
]

# --------------------------------------------------------


# These four functions intentionally match the original PLOT.PY.
def read_frame(ser):
    start = time.time()

    while time.time() - start < 10:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        try:
            values = [int(x) for x in line.split(",")]
            if len(values) == TOTAL_SENSORS:
                return np.array(values).reshape(ROWS, COLS)
        except ValueError:
            continue

    raise TimeoutError(f"No valid {TOTAL_SENSORS}-value frame received.")


def calculate_pressure(raw, baseline):
    return np.abs(raw - baseline)


def apply_threshold(pressure, threshold_map):
    return np.where(pressure > threshold_map, pressure, 0)


def create_heatmap(data):
    enlarged = zoom(data, UPSCALE, order=1)
    enlarged = gaussian_filter(enlarged, sigma=SMOOTHING)
    enlarged[enlarged < MIN_PRESSURE] = 0
    return enlarged


def find_outline(smooth):
    inside = smooth > 0
    edge = inside & ~binary_erosion(inside)
    return np.ma.masked_where(~edge, edge.astype(float))


def setup_original_heatmap():
    # Exact plotting setup from original PLOT.PY.
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 9))

    height = ROWS * UPSCALE
    width = COLS * UPSCALE

    levels = np.linspace(0, 100, NUM_COLOUR_LEVELS + 1)
    levels[0] = 0.001

    cmap = plt.get_cmap(COLOURMAP, NUM_COLOUR_LEVELS).copy()
    cmap.set_under("black")
    cmap.set_over("white")
    norm = BoundaryNorm(levels, NUM_COLOUR_LEVELS)

    heat_img = ax.imshow(
        np.zeros((height, width)),
        cmap=cmap, norm=norm,
        interpolation="nearest", origin="upper", aspect="equal",
    )

    outline_img = ax.imshow(
        np.ma.masked_all((height, width)),
        cmap=ListedColormap(["white"]), vmin=0, vmax=1,
        interpolation="nearest", origin="upper", aspect="equal",
    )

    colorbar = plt.colorbar(heat_img, ax=ax)
    colorbar.set_label("Pressure (% of current peak)")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.tight_layout()

    return fig, ax, heat_img, outline_img


def save_position(session_dir, code, name, records, session_timestamp):
    """Write to a temporary folder, then replace this position only."""
    final_dir = session_dir / f"{code}_{name}"
    temp_dir = session_dir / f".{code}_{name}_recapture_tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    raw_array = np.stack([r["raw"] for r in records]).astype(np.int32)
    pressure_array = np.stack([r["pressure"] for r in records]).astype(np.float32)

    np.save(temp_dir / "raw_data.npy", raw_array)
    np.save(temp_dir / "pressure_data.npy", pressure_array)

    sensor_columns = [f"sensor_{i:03d}" for i in range(1, TOTAL_SENSORS + 1)]
    metadata_columns = [
        "session_timestamp", "frame_timestamp", "position_code",
        "position_label", "repetition", "frame_in_repetition",
        "elapsed_seconds",
    ]

    for filename, key in [
        ("raw_data.csv", "raw"),
        ("pressure_data.csv", "pressure"),
    ]:
        with open(temp_dir / filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metadata_columns + sensor_columns)
            for r in records:
                writer.writerow([
                    session_timestamp,
                    r["frame_timestamp"],
                    code,
                    name,
                    r["repetition"],
                    r["frame_in_repetition"],
                    f'{r["elapsed_seconds"]:.4f}',
                    *np.asarray(r[key]).reshape(-1).tolist(),
                ])

    with open(temp_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_timestamp", "position_code", "position_label",
            "repetitions_planned", "capture_duration_seconds",
            "captured_frames", "matrix_rows", "matrix_columns",
        ])
        writer.writerow([
            session_timestamp, code, name, REPETITIONS_PER_POSITION,
            CAPTURE_DURATION_SECONDS, len(records), ROWS, COLS,
        ])

    # Replace only the position that has just been fully recaptured.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temp_dir.rename(final_dir)

    print(f"\nSaved {code}_{name}: {len(records)} frames")
    print(f"Replaced folder: {final_dir}")


def main():
    session_dir = DATASET_ROOT / f"session_{SESSION_TIMESTAMP}"
    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"Expected existing session folder was not found:\n{session_dir}\n"
            "Check SESSION_TIMESTAMP. No dataset folders were changed."
        )

    print("\n==============================================")
    print(" P5 + P6 RECAPTURE — ORIGINAL HEATMAP")
    print("==============================================")
    print(f"Session: {session_dir}")
    print("Will recapture ONLY:")
    print("  P5_Right_Lean")
    print("  P6_Backward_Lean")
    print("P1-P4 will not be touched.")
    print("\nKeep the cushion EMPTY during the 20-frame calibration.\n")

    ser = None
    fig = None
    current_records = None
    current_code = None
    current_name = None

    try:
        ser = serial.Serial(PORT, BAUD, timeout=3)
        time.sleep(2)
        ser.reset_input_buffer()
        print("Connected to ESP32.")

        calibration = []
        for i in range(CALIBRATION_FRAMES):
            calibration.append(read_frame(ser))
            print(f"Calibration: {i + 1}/{CALIBRATION_FRAMES}")

        calibration = np.array(calibration)
        baseline = np.mean(calibration, axis=0)
        noise = np.std(calibration, axis=0)
        threshold_map = np.maximum(
            NOISE_MULTIPLIER * noise, MIN_PRESSURE
        )

        print("\nCalibration complete!")
        print("Baseline min / mean / max:",
              round(float(np.min(baseline))),
              round(float(np.mean(baseline))),
              round(float(np.max(baseline))))
        print("Average noise:", round(float(np.mean(noise)), 1))
        print("Average threshold:", round(float(np.mean(threshold_map)), 1))
        print("\nYou can sit down now.\n")

        fig, ax, heat_img, outline_img = setup_original_heatmap()

        # Same persistent peak behavior as the original PLOT.PY live loop.
        peak = MIN_SCALE

        for code, name in POSITIONS_TO_RECAPTURE:
            current_code, current_name = code, name
            current_records = []

            print("\n----------------------------------------------")
            print(f"{code}: {name.replace('_', ' ')}")
            print("----------------------------------------------")

            for repetition in range(1, REPETITIONS_PER_POSITION + 1):
                input(
                    f"\nAssume {name.replace('_', ' ')} posture. "
                    f"Press ENTER for repetition {repetition}/"
                    f"{REPETITIONS_PER_POSITION}..."
                )

                for count in range(COUNTDOWN_SECONDS, 0, -1):
                    print(f"Starting in {count}...")
                    plt.pause(0.01)

                print("CAPTURING — hold the posture steady.")

                capture_start = time.perf_counter()
                frame_number = 0

                while time.perf_counter() - capture_start < CAPTURE_DURATION_SECONDS:
                    raw = read_frame(ser)
                    raw_change = calculate_pressure(raw, baseline)
                    pressure = apply_threshold(raw_change, threshold_map)

                    # Exact peak update and heatmap processing from PLOT.PY.
                    if AUTO_SCALE:
                        peak = max(
                            peak * 0.95,
                            float(np.max(pressure)),
                            MIN_SCALE,
                        )
                    else:
                        peak = FIXED_MAX

                    smooth = create_heatmap(pressure)
                    percent = smooth / peak * 100

                    heat_img.set_data(percent)
                    outline_img.set_data(find_outline(smooth))

                    elapsed = time.perf_counter() - capture_start
                    frame_number += 1

                    ax.set_title(
                        f"{name.replace('_', ' ')} | Rep "
                        f"{repetition}/{REPETITIONS_PER_POSITION} | "
                        f"{elapsed:.1f}/{CAPTURE_DURATION_SECONDS}s | "
                        f"Peak: {np.max(pressure):.0f} | "
                        f"Active points: {np.count_nonzero(pressure)}"
                    )

                    fig.canvas.draw_idle()
                    plt.pause(0.01)

                    current_records.append({
                        "raw": raw.copy(),
                        "pressure": pressure.copy(),
                        "frame_timestamp": datetime.now().isoformat(
                            timespec="milliseconds"
                        ),
                        "repetition": repetition,
                        "frame_in_repetition": frame_number,
                        "elapsed_seconds": elapsed,
                    })

                print(f"Rep {repetition} complete: {frame_number} frames.")

            # Save/replace only after all 5 repetitions of this position finish.
            save_position(
                session_dir, code, name, current_records, SESSION_TIMESTAMP
            )
            current_records = None

        print("\n==============================================")
        print("P5 RIGHT LEAN AND P6 BACKWARD LEAN COMPLETE.")
        print("P1-P4 were left untouched.")
        print(f"Dataset: {session_dir}")
        print("==============================================")

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print("P1-P4 were not modified.")
        print("An old P5/P6 folder is retained unless its full recapture completed.")
        print("The currently incomplete position was not saved.")

    finally:
        if ser is not None and ser.is_open:
            ser.close()
        if fig is not None:
            plt.close(fig)
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
