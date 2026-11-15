"""Compile and drive esp32/host_harness.exe so Python can compare against the
exact C++ the firmware runs. Used by Phase 2/5 verification and by pytest.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ESP32_DIR = Path(__file__).resolve().parent.parent / "esp32"
HARNESS_EXE = ESP32_DIR / "host_harness.exe"

# g++ 14.2 shipped by WinLibs mis-resolves its own include path when the
# toolchain lives under a directory containing spaces; clang++ from the same
# package works. Try clang++ first, then g++, then cc.
COMPILERS = ["clang++", "g++", "c++"]


def _sources(with_inference: bool):
    srcs = [ESP32_DIR / "host_harness.cpp", ESP32_DIR / "preprocess.cpp"]
    if with_inference:
        srcs.append(ESP32_DIR / "inference.cpp")
    return srcs


def build(with_inference=None, force=False):
    """Compile the harness. with_inference=None -> auto-detect inference.cpp."""
    if with_inference is None:
        with_inference = (ESP32_DIR / "inference.cpp").is_file()

    srcs = _sources(with_inference)
    missing = [s for s in srcs if not s.is_file()]
    if missing:
        raise FileNotFoundError("missing C++ sources: %s" % missing)

    if HARNESS_EXE.is_file() and not force:
        newest = max(s.stat().st_mtime for s in srcs + [ESP32_DIR / "preprocess.h"])
        if HARNESS_EXE.stat().st_mtime >= newest:
            return HARNESS_EXE

    last_err = None
    for cc in COMPILERS:
        if shutil.which(cc) is None:
            continue
        cmd = [cc, "-O2", "-std=c++17", "-o", str(HARNESS_EXE)]
        if with_inference:
            cmd.append("-DSC_WITH_INFERENCE")
        cmd += [str(s) for s in srcs]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ESP32_DIR))
        if proc.returncode == 0:
            return HARNESS_EXE
        last_err = "%s failed:\n%s" % (cc, proc.stderr[-4000:])
    raise RuntimeError("no working C++ compiler. " + (last_err or "none found"))


class Harness:
    """One long-lived harness process; cheaper than re-spawning per frame."""

    def __init__(self, with_inference=None):
        exe = build(with_inference=with_inference)
        self.p = subprocess.Popen([str(exe)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1,
                                  cwd=str(ESP32_DIR))

    def _cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline()
        if not out:
            raise RuntimeError("harness died")
        return out.strip()

    @staticmethod
    def _ints(arr):
        return " ".join(str(int(v)) for v in np.asarray(arr).ravel())

    @staticmethod
    def _floats(arr):
        return " ".join(repr(float(v)) for v in np.asarray(arr).ravel())

    # ---- calibration ----
    def cal_begin(self):
        return self._cmd("CAL_BEGIN")

    def cal_add(self, frame):
        return self._cmd("CAL_FRAME " + self._ints(frame))

    def cal_finish(self):
        out = self._cmd("CAL_FINISH").split()
        return out[1], out[3]          # (result, state)

    def set_cal(self, baseline, threshold):
        return self._cmd("SET_CAL " + self._ints(baseline) + " " + self._ints(threshold))

    def get_cal(self):
        parts = self._cmd("GET_CAL").split()[1:]
        v = np.array(parts, dtype=np.int64)
        return v[:504].astype(np.int32), v[504:].astype(np.int32)

    # ---- transforms ----
    def preprocess(self, raw):
        out = self._cmd("PRE " + self._ints(raw)).split()
        assert out[0] == "FEAT", out[:3]
        return np.array(out[1:], dtype=np.float32)

    def raw_to_pressure(self, raw):
        out = self._cmd("PRESSURE " + self._ints(raw)).split()
        assert out[0] == "PRESS", out[:3]
        return np.array(out[1:], dtype=np.int32)

    def pressure_to_features(self, pressure):
        out = self._cmd("FEAT_FROM_PRESSURE " + self._ints(pressure)).split()
        assert out[0] == "FEAT", out[:3]
        return np.array(out[1:], dtype=np.float32)

    # ---- inference (only when inference.cpp was compiled in) ----
    def infer(self, raw):
        return self._parse_infer(self._cmd("INFER " + self._ints(raw)))

    def infer_features(self, feat):
        return self._parse_infer(self._cmd("INFER_FEAT " + self._floats(feat)))

    @staticmethod
    def _parse_infer(line):
        t = line.split()
        if t[0] != "LOGITS":
            raise RuntimeError("unexpected inference reply: " + line[:120])
        i_arg = t.index("ARGMAX")
        i_prob = t.index("PROB")
        return {
            "logits": np.array(t[1:i_arg], dtype=np.int32),
            "argmax": int(t[i_arg + 1]),
            "prob": np.array(t[i_prob + 1:], dtype=np.float32),
        }

    def close(self):
        try:
            self._cmd("QUIT")
        except Exception:
            pass
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    print("built:", build(force=True))
    with Harness() as h:
        rng = np.random.default_rng(0)
        base = rng.integers(1500, 3000, 504)
        thr = np.full(504, 20)
        h.set_cal(base, thr)
        b2, t2 = h.get_cal()
        assert np.array_equal(b2, base) and np.array_equal(t2, thr)
        print("calibration round-trip OK")
        raw = rng.integers(0, 4096, 504)
        print("sample features:", h.preprocess(raw)[:6])
