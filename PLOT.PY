import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import zoom, gaussian_filter, binary_erosion

# ---------------- CONFIGURATION ----------------

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200            # must match Serial.begin() in the ESP32 sketch

ROWS = 24
COLS = 21
TOTAL_SENSORS = ROWS * COLS

CALIBRATION_FRAMES = 20

NOISE_MULTIPLIER = 3     # ignore changes smaller than 3x the noise
MIN_PRESSURE = 20        # and smaller than this many ADC counts

AUTO_SCALE = True        # colours adjust to the current pressure
FIXED_MAX = 1000         # used only if AUTO_SCALE = False
MIN_SCALE = 100          # stops noise filling the whole colour range

NUM_COLOUR_LEVELS = 8
COLOURMAP = "turbo"
UPSCALE = 6
SMOOTHING = 0.6

# ------------------------------------------------


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


def main():
    ser = serial.Serial(PORT, BAUD, timeout=3)
    time.sleep(2)
    ser.reset_input_buffer()

    print("\nConnected to ESP32.")
    print("Nobody touch the cushion until calibration is complete.\n")

    calibration = []
    for i in range(CALIBRATION_FRAMES):
        calibration.append(read_frame(ser))
        print(f"Calibration: {i + 1}/{CALIBRATION_FRAMES}")

    calibration = np.array(calibration)
    baseline = np.mean(calibration, axis=0)
    noise = np.std(calibration, axis=0)
    threshold_map = np.maximum(NOISE_MULTIPLIER * noise, MIN_PRESSURE)

    print("\nCalibration complete!")
    print("Baseline min / mean / max:",
          round(float(np.min(baseline))),
          round(float(np.mean(baseline))),
          round(float(np.max(baseline))))
    print("Average noise:", round(float(np.mean(noise)), 1))
    print("Average threshold:", round(float(np.mean(threshold_map)), 1))
    print("\nYou can sit down now.\n")

    # ---------------- HEATMAP SETUP ----------------

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

    peak = MIN_SCALE

    # ---------------- LIVE LOOP ----------------

    try:
        while plt.fignum_exists(fig.number):

            raw = read_frame(ser)
            raw_change = calculate_pressure(raw, baseline)
            pressure = apply_threshold(raw_change, threshold_map)

            if AUTO_SCALE:
                peak = max(peak * 0.95, float(np.max(pressure)), MIN_SCALE)
            else:
                peak = FIXED_MAX

            smooth = create_heatmap(pressure)
            percent = smooth / peak * 100

            heat_img.set_data(percent)
            outline_img.set_data(find_outline(smooth))

            active = int(np.count_nonzero(pressure))
            ax.set_title(
                f"Live Pressure Map | Peak: {np.max(pressure):.0f} | "
                f"Active points: {active}"
            )

            print(
                f"Raw min: {np.min(raw):4d} | Raw max: {np.max(raw):4d} | "
                f"Change before filter: {np.max(raw_change):5.0f} | "
                f"After filter: {np.max(pressure):5.0f} | "
                f"Active: {active}"
            )

            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        ser.close()
        plt.close(fig)
        print("Serial connection closed.")


if __name__ == "__main__":
    main()