"""Frame sources for the tester and the dashboard.

Two interchangeable sources, both producing the same event dicts:

  {"kind": "frame", "raw": (504,) int32, "t": monotonic}
  {"kind": "pred",  "label": str, "confidence": float, "inference_us": int,
   "origin": "ESP32"}
  {"kind": "state"|"cal"|"info", "text": str}

`SerialSource`    - a real ESP32 on a COM port, speaking the Phase 6 protocol.
`SimulatedSource` - replays recorded CSV frames at the real capture rate, so
                    every part of the UI can be exercised with no hardware.

Both run their read loop on a background thread and hand events to a bounded
queue; the UI thread never blocks on I/O.

A prediction is labelled `origin="ESP32"` only when it arrived as a `PRED,`
line from the board. Predictions computed here in Python are labelled
`origin="Python"` by the caller and must never be presented as embedded
inference.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

N_SENSORS = 504
ROWS, COLS = 24, 21

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_line(line):
    """One protocol line -> event dict, or None if it is not recognised."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("FRAME,"):
        parts = line[6:].split(",")
        if len(parts) != N_SENSORS:
            return {"kind": "info", "text": "bad FRAME length %d" % len(parts)}
        try:
            vals = np.array([int(p) for p in parts], dtype=np.int32)
        except ValueError:
            return {"kind": "info", "text": "non-integer FRAME"}
        if (vals < 0).any() or (vals > 4095).any():
            return {"kind": "info", "text": "FRAME out of ADC range"}
        return {"kind": "frame", "raw": vals, "t": time.monotonic()}
    if line.startswith("PRED,"):
        parts = line[5:].split(",")
        if len(parts) != 3:
            return {"kind": "info", "text": "bad PRED line"}
        try:
            return {"kind": "pred", "label": parts[0],
                    "confidence": float(parts[1]),
                    "inference_us": int(parts[2]),
                    "origin": "ESP32", "t": time.monotonic()}
        except ValueError:
            return {"kind": "info", "text": "bad PRED fields"}
    for prefix, kind in (("STATE,", "state"), ("CAL,", "cal"), ("INFO,", "info")):
        if line.startswith(prefix):
            return {"kind": kind, "text": line[len(prefix):], "t": time.monotonic()}
    # the original sketch's banner and anything else
    return {"kind": "info", "text": line, "t": time.monotonic()}


class _BaseSource:
    def __init__(self, maxsize=400):
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def _put(self, ev):
        if ev is None:
            return
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            # drop the oldest so a slow UI never stalls acquisition
            try:
                self.q.get_nowait()
                self.q.put_nowait(ev)
            except queue.Empty:
                pass

    def drain(self, limit=200):
        out = []
        for _ in range(limit):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def send(self, command):
        raise NotImplementedError

    def _run(self):
        raise NotImplementedError

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()


class SerialSource(_BaseSource):
    """A real board. Requires pyserial and a connected ESP32."""

    def __init__(self, port, baud=115200, timeout=2.0, **kw):
        super().__init__(**kw)
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = None
        self._tx = queue.Queue()

    def send(self, command):
        self._tx.put(command)

    def _run(self):
        import serial
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except Exception as exc:
            self.error = str(exc)
            self._put({"kind": "info", "text": "serial open failed: %s" % exc})
            return
        time.sleep(2.0)                 # the ESP32 resets when the port opens
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        try:
            while not self._stop.is_set():
                while not self._tx.empty():
                    cmd = self._tx.get_nowait()
                    self.ser.write((cmd + "\n").encode("ascii", errors="ignore"))
                    self.ser.flush()
                line = self.ser.readline().decode("utf-8", errors="ignore")
                if line:
                    self._put(parse_line(line))
        except Exception as exc:
            self.error = str(exc)
            self._put({"kind": "info", "text": "serial read failed: %s" % exc})
        finally:
            try:
                self.ser.close()
            except Exception:
                pass


class SimulatedSource(_BaseSource):
    """Replays recorded raw frames, optionally with a local model's predictions.

    This is a *simulator*. Any prediction it emits is tagged origin="Python"
    (or "Simulated") and the UI must display it as such.
    """

    def __init__(self, frames, labels=None, fps=1.4, loop=True,
                 predictor=None, **kw):
        super().__init__(**kw)
        self.frames = np.asarray(frames, dtype=np.int32)
        self.labels = labels
        self.fps = float(fps)
        self.loop = loop
        self.predictor = predictor
        self._cmds = []

    def send(self, command):
        self._cmds.append(command)
        self._put({"kind": "info", "text": "simulated source ignored command: %s"
                                           % command})

    def _run(self):
        self._put({"kind": "info", "text": "SIMULATED SOURCE - replaying recorded CSV frames"})
        self._put({"kind": "state", "text": "READY (simulated)"})
        period = 1.0 / max(self.fps, 0.01)
        i = 0
        n = self.frames.shape[0]
        while not self._stop.is_set() and n > 0:
            raw = self.frames[i % n]
            self._put({"kind": "frame", "raw": raw.copy(), "t": time.monotonic()})
            if self.predictor is not None:
                t0 = time.perf_counter()
                label, conf = self.predictor(raw)
                dt = int((time.perf_counter() - t0) * 1e6)
                self._put({"kind": "pred", "label": label, "confidence": float(conf),
                           "inference_us": dt, "origin": "Python",
                           "t": time.monotonic()})
            if self.labels is not None:
                self._put({"kind": "info",
                           "text": "ground_truth=%s" % self.labels[i % n]})
            i += 1
            if not self.loop and i >= n:
                break
            self._stop.wait(period)


def load_replay(participant=None, posture=None, limit=None):
    """Load recorded raw frames for the simulator, from models/dataset.npz."""
    p = PROJECT_ROOT / "models" / "dataset.npz"
    if not p.is_file():
        raise FileNotFoundError("models/dataset.npz not built; run ml/build_dataset.py")
    d = np.load(p, allow_pickle=False)
    m = np.ones(d["y"].shape[0], dtype=bool)
    if participant:
        m &= (d["participant"].astype(str) == participant)
    if posture:
        m &= (d["posture"].astype(str) == posture)
    raw = d["X_raw"][m]
    lab = d["posture"].astype(str)[m]
    if limit:
        raw, lab = raw[:limit], lab[:limit]
    return raw, lab


def list_ports():
    try:
        from serial.tools import list_ports as lp
        return [(p.device, p.description) for p in lp.comports()]
    except Exception:
        return []
