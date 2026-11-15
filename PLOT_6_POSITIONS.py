import csv
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import zoom, gaussian_filter, binary_erosion
from datetime import datetime
from pathlib import Path

# ============================================================
# ESP32 24x21 PRESSURE MATRIX: 6-POSITION DATA ACQUISITION
# ============================================================

# ---------------- CONFIGURATION ----------------

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

ROWS = 24
COLS = 21
TOTAL_SENSORS = ROWS * COLS

CALIBRATION_FRAMES = 20

# Noise filtering
NOISE_MULTIPLIER = 3
MIN_PRESSURE = 20

# Heatmap scaling
AUTO_SCALE = True
FIXED_MAX = 1000
MIN_SCALE = 100

NUM_COLOUR_LEVELS = 8
COLOURMAP = "turbo"
UPSCALE = 6
SMOOTHING = 0.6

# Acquisition settings
CAPTURE_DURATION_SECONDS = 10
REPETITIONS_PER_POSITION = 5
POSITIONS = [
    ("P1", "Normal"),
    ("P2", "Forward_Lean"),
    ("P3", "Slouching"),
    ("P4", "Left_Lean"),
    ("P5", "Right_Lean"),
    ("P6", "Backward_Lean"),
]

COUNTDOWN_SECONDS = 3
DATASET_ROOT = Path(__file__).resolve().parent / "DATASET"

# ------------------------------------------------------------


def read_frame(ser):
    """Read one complete comma-separated 504-value sensor frame."""
    start = time.time()

    while time.time() - start < 10:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        try:
            values = [int(x.strip()) for x in line.split(",")]
            if len(values) == TOTAL_SENSORS:
                return np.asarray(values, dtype=np.int32).reshape(ROWS, COLS)
        except ValueError:
            # Ignore status messages or incomplete/non-numeric lines.
            continue

    raise TimeoutError(
        f"No valid {TOTAL_SENSORS}-value frame received within 10 seconds. "
        "Check ESP32 output and serial connection."
    )


def calculate_pressure(raw, baseline):
    return np.abs(raw.astype(np.float32) - baseline)


def apply_threshold(pressure, threshold_map):
    return np.where(pressure > threshold_map, pressure, 0).astype(np.float32)


def create_heatmap(data):
    enlarged = zoom(data, UPSCALE, order=1)
    enlarged = gaussian_filter(enlarged, sigma=SMOOTHING)
    enlarged[enlarged < MIN_PRESSURE] = 0
    return enlarged


def find_outline(smooth):
    inside = smooth > 0
    edge = inside & ~binary_erosion(inside)
    return np.ma.masked_where(~edge, edge.astype(float))


def make_heatmap_window():
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
    colorbar.set_label("Pressure (% of current peak)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_title("Live Pressure Map")
    fig.tight_layout()

    return fig, ax, heat_img, outline_img


def update_heatmap(fig, ax, heat_img, outline_img, pressure, peak,
                   position_name, repetition, elapsed):
    smooth = create_heatmap(pressure)
    percent = smooth / max(peak, 1) * 100

    heat_img.set_data(percent)
    outline_img.set_data(find_outline(smooth))

    active = int(np.count_nonzero(pressure))
    ax.set_title(
        f"{position_name} | Rep {repetition}/{REPETITIONS_PER_POSITION} | "
        f"{elapsed:.1f}/{CAPTURE_DURATION_SECONDS}s | "
        f"Peak: {np.max(pressure):.0f} | Active: {active}"
    )

    fig.canvas.draw_idle()
    plt.pause(0.001)


def save_position_data(session_dir, position_code, position_name,
                       records, session_timestamp):
    """
    Save one position's records as:
      raw_data.csv / raw_data.npy
      pressure_data.csv / pressure_data.npy
      metadata.csv

    Each NPY array has shape: (captured_frames, 24, 21).
    """
    position_dir = session_dir / f"{position_code}_{position_name}"
    position_dir.mkdir(parents=True, exist_ok=True)

    raw_array = np.stack([r["raw"] for r in records]).astype(np.int32)
    pressure_array = np.stack([r["pressure"] for r in records]).astype(np.float32)

    np.save(position_dir / "raw_data.npy", raw_array)
    np.save(position_dir / "pressure_data.npy", pressure_array)

    sensor_columns = [f"sensor_{i:03d}" for i in range(1, TOTAL_SENSORS + 1)]
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
        with open(position_dir / filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metadata_columns + sensor_columns)

            for record in records:
                metadata = [
                    session_timestamp,
                    record["frame_timestamp"],
                    position_code,
                    position_name,
                    record["repetition"],
                    record["frame_in_repetition"],
                    f'{record["elapsed_seconds"]:.4f}',
                ]
                flattened = np.asarray(record[key]).reshape(-1).tolist()
                writer.writerow(metadata + flattened)

    with open(position_dir / "metadata.csv", "w", newline="") as f:
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
            session_timestamp,
            position_code,
            position_name,
            REPETITIONS_PER_POSITION,
            CAPTURE_DURATION_SECONDS,
            len(records),
            ROWS,
            COLS,
        ])

    print(f"Saved {position_code}_{position_name}: {len(records)} frames")
    print(f"  {position_dir}")


def main():
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = DATASET_ROOT / f"session_{session_timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print("\n================================================")
    print("  ESP32 SIX-POSITION PRESSURE DATA ACQUISITION")
    print("================================================")
    print(f"Serial port : {PORT}")
    print(f"Matrix      : {ROWS} x {COLS} ({TOTAL_SENSORS} sensors)")
    print(f"Positions   : {len(POSITIONS)}")
    print(f"Repetitions : {REPETITIONS_PER_POSITION} per position")
    print(f"Duration    : {CAPTURE_DURATION_SECONDS} seconds per repetition")
    print(f"Dataset     : {session_dir}")
    print("================================================\n")

    ser = None
    fig = None

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("Connected to ESP32.")
        print("\nKeep the cushion EMPTY and untouched for calibration.")

        calibration = []
        for i in range(CALIBRATION_FRAMES):
            calibration.append(read_frame(ser))
            print(f"Calibration: {i + 1}/{CALIBRATION_FRAMES}")

        calibration = np.asarray(calibration, dtype=np.float32)
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

        fig, ax, heat_img, outline_img = make_heatmap_window()
        peak = MIN_SCALE

        print("\nIMPORTANT:")
        print("- Assume the requested posture before each repetition.")
        print("- Press ENTER in this terminal when ready.")
        print("- Remain as still as possible during the 10-second capture.")
        print("- The live heatmap updates during each capture.")
        print("- Press Ctrl+C to stop early.\n")

        for position_index, (position_code, position_name) in enumerate(POSITIONS, 1):
            position_records = []

            print("\n------------------------------------------------")
            print(f"{position_index}/{len(POSITIONS)}: {position_name.replace('_', ' ')}")
            print("------------------------------------------------")

            for repetition in range(1, REPETITIONS_PER_POSITION + 1):
                input(
                    f"\nAssume {position_name.replace('_', ' ')} posture. "
                    f"Press ENTER for repetition {repetition}/"
                    f"{REPETITIONS_PER_POSITION}..."
                )

                for count in range(COUNTDOWN_SECONDS, 0, -1):
                    print(f"Starting in {count}...")
                    plt.pause(0.01)

                print("CAPTURING — hold your position.")

                capture_start = time.perf_counter()
                frame_in_repetition = 0

                while (time.perf_counter() - capture_start) < CAPTURE_DURATION_SECONDS:
                    raw = read_frame(ser)
                    frame_time = datetime.now().isoformat(timespec="milliseconds")
                    elapsed = time.perf_counter() - capture_start

                    raw_change = calculate_pressure(raw, baseline)
                    pressure = apply_threshold(raw_change, threshold_map)

                    if AUTO_SCALE:
                        peak = max(
                            peak * 0.95,
                            float(np.max(pressure)),
                            MIN_SCALE,
                        )
                    else:
                        peak = FIXED_MAX

                    frame_in_repetition += 1
                    position_records.append({
                        "raw": raw.copy(),
                        "pressure": pressure.copy(),
                        "frame_timestamp": frame_time,
                        "repetition": repetition,
                        "frame_in_repetition": frame_in_repetition,
                        "elapsed_seconds": elapsed,
                    })

                    update_heatmap(
                        fig, ax, heat_img, outline_img, pressure, peak,
                        position_name.replace("_", " "),
                        repetition, elapsed
                    )

                print(
                    f"Repetition {repetition} complete: "
                    f"{frame_in_repetition} frames captured."
                )

            save_position_data(
                session_dir,
                position_code,
                position_name,
                position_records,
                session_timestamp,
            )

        print("\n================================================")
        print("ALL SIX POSITIONS CAPTURED SUCCESSFULLY!")
        print(f"Dataset saved at:\n{session_dir}")
        print("================================================")

    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
        print("Completed positions/repetitions already saved remain on disk.")
        print("The currently running repetition may not have been saved.")

    except Exception as error:
        print(f"\nERROR: {error}")
        print("Check the ESP32 connection, serial port and frame format.")

    finally:
        if ser is not None and ser.is_open:
            ser.close()
        if fig is not None:
            plt.close(fig)
        print("\nSerial connection closed.")


if __name__ == "__main__":
    main()
