"""CLI tester for the Smart Cushion firmware.

    python dashboard/serial_tester.py --port COM5
    python dashboard/serial_tester.py --simulate --participant P3
    python dashboard/serial_tester.py --list-ports

Prints one line per frame: a compact ASCII pressure map, the predicted label,
confidence, and where the prediction came from. Predictions from the board are
tagged [ESP32]; predictions computed locally are tagged [Python] and are never
presented as embedded inference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from serial_source import (SimulatedSource, SerialSource, list_ports,
                           load_replay, ROWS, COLS)

RAMP = " .:-=+*#%@"


def ascii_map(mat, width=COLS):
    m = np.asarray(mat, dtype=np.float64).reshape(ROWS, COLS)
    hi = max(m.max(), 1.0)
    rows = []
    for r in range(0, ROWS, 2):          # halve vertically so it fits a terminal
        chunk = m[r:r + 2].mean(axis=0)
        rows.append("".join(RAMP[min(int(v / hi * (len(RAMP) - 1)), len(RAMP) - 1)]
                            for v in chunk))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true",
                    help="replay recorded CSV frames instead of using hardware")
    ap.add_argument("--participant", default=None)
    ap.add_argument("--posture", default=None)
    ap.add_argument("--fps", type=float, default=1.4)
    ap.add_argument("--python-model", action="store_true",
                    help="also run the exported model locally (tagged [Python])")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds")
    ap.add_argument("--map", action="store_true", help="draw an ASCII pressure map")
    ap.add_argument("--list-ports", action="store_true")
    args = ap.parse_args()

    if args.list_ports:
        ports = list_ports()
        if not ports:
            print("no serial ports found")
        for dev, desc in ports:
            print("%-12s %s" % (dev, desc))
        return 0

    predictor = None
    if args.python_model:
        try:
            from python_reference import PythonReferenceModel
            ref = PythonReferenceModel()
            predictor = ref.predict_from_raw
            print("local model loaded: %s" % ref.describe())
        except Exception as exc:
            print("could not load local model (%s); continuing without it" % exc)

    if args.simulate:
        frames, labels = load_replay(args.participant, args.posture)
        print("SIMULATED source: %d recorded frames%s%s"
              % (len(frames),
                 " participant=" + args.participant if args.participant else "",
                 " posture=" + args.posture if args.posture else ""))
        src = SimulatedSource(frames, labels=labels, fps=args.fps,
                              predictor=predictor)
    else:
        if not args.port:
            print("--port is required (or use --simulate / --list-ports)")
            return 2
        print("opening %s at %d baud ..." % (args.port, args.baud))
        src = SerialSource(args.port, args.baud)

    src.start()
    t0 = time.monotonic()
    n_frames = 0
    last_pred = None
    gt = None

    try:
        while True:
            if args.seconds and (time.monotonic() - t0) > args.seconds:
                break
            evs = src.drain()
            if not evs:
                time.sleep(0.05)
                continue
            for ev in evs:
                k = ev["kind"]
                if k == "frame":
                    n_frames += 1
                    raw = ev["raw"]
                    line = ("frame %4d  min=%4d max=%4d mean=%6.1f"
                            % (n_frames, raw.min(), raw.max(), raw.mean()))
                    if last_pred:
                        line += ("   %-18s conf=%.3f  %6d us  [%s]"
                                 % (last_pred["label"], last_pred["confidence"],
                                    last_pred["inference_us"], last_pred["origin"]))
                    if gt:
                        line += "   truth=%s" % gt
                    print(line)
                    if args.map:
                        for row in ascii_map(raw):
                            print("    |" + row + "|")
                elif k == "pred":
                    last_pred = ev
                elif k == "info" and ev["text"].startswith("ground_truth="):
                    gt = ev["text"].split("=", 1)[1]
                else:
                    print("[%s] %s" % (k.upper(), ev.get("text", "")))
            if src.error:
                print("source error: %s" % src.error)
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        src.stop()

    print("\n%d frames in %.1f s (%.2f fps)"
          % (n_frames, time.monotonic() - t0,
             n_frames / max(time.monotonic() - t0, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
