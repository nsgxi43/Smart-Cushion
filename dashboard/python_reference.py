"""Python-side inference, used only by the simulator and for comparison.

Anything this produces is a **Python** prediction. It is not embedded
inference and must always be labelled as such in the UI and in reports.

Loads, in order of preference:
  1. models/model_int8.tflite   - the exact artefact the firmware was built from
  2. models/model_float.keras   - the float model it was quantised from
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

MODELS = PROJECT_ROOT / "models"


class PythonReferenceModel:
    def __init__(self, prefer="int8"):
        from preprocess import pressure_to_features  # noqa: F401
        self.meta = {}
        meta_path = MODELS / "model_metadata.json"
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta.get("class_names") or [
            "C0_Normal", "C1_Forward_Lean", "C2_Slouching",
            "C3_Left_Lean", "C4_Right_Lean", "C5_Backward_Lean"]

        self.kind = None
        self.interp = None
        self.keras = None

        tfl = MODELS / "model_int8.tflite"
        kf = MODELS / "model_float.keras"
        if prefer == "int8" and tfl.is_file():
            self._load_tflite(tfl)
        elif kf.is_file():
            self._load_keras(kf)
        elif tfl.is_file():
            self._load_tflite(tfl)
        else:
            raise FileNotFoundError(
                "no exported model in models/ (run ml/export_int8.py)")

        # calibration used to turn raw frames into features
        self.baseline = None
        self.threshold = None
        cal = MODELS / "dashboard_calibration.npy"
        if cal.is_file():
            arr = np.load(cal)
            self.baseline, self.threshold = arr[0], arr[1]

    def _load_tflite(self, path):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.kind = "INT8 TFLite"

    def _load_keras(self, path):
        from tensorflow import keras
        self.keras = keras.models.load_model(path)
        self.kind = "float Keras"

    def describe(self):
        return "%s | %s | %s" % (
            self.kind,
            self.meta.get("architecture", "unknown architecture"),
            self.meta.get("training_set", "unknown training set"))

    # ---------------------------------------------------------------
    def set_calibration(self, baseline, threshold):
        self.baseline = np.asarray(baseline, dtype=np.int32)
        self.threshold = np.asarray(threshold, dtype=np.int32)

    def features_from_raw(self, raw):
        from preprocess import preprocess_frame, pressure_to_features
        if self.baseline is None:
            # no calibration captured yet: treat the frame as already
            # baseline-subtracted, which is what the archived CSVs hold
            return pressure_to_features(np.asarray(raw, dtype=np.int32))
        return preprocess_frame(raw, self.baseline, self.threshold)

    def predict_proba_features(self, feats):
        feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        if self.interp is not None:
            s_in, z_in = self.inp["quantization"]
            s_out, z_out = self.out["quantization"]
            q = np.clip(np.round(feats / s_in) + z_in, -128, 127).astype(np.int8)
            self.interp.set_tensor(self.inp["index"], q)
            self.interp.invoke()
            raw_out = self.interp.get_tensor(self.out["index"])[0].astype(np.float64)
            z = (raw_out - z_out) * s_out
            if self.meta.get("output_is_logits", True):
                z = z - z.max()
                e = np.exp(z)
                return e / e.sum()
            return raw_out * 0 + z / max(z.sum(), 1e-9)
        p = self.keras.predict(feats, verbose=0)[0]
        if self.meta.get("output_is_logits", False):
            p = p - p.max()
            e = np.exp(p)
            return e / e.sum()
        return p

    def predict_from_raw(self, raw):
        probs = self.predict_proba_features(self.features_from_raw(raw))
        i = int(np.argmax(probs))
        return self.class_names[i], float(probs[i])

    def predict_proba_from_raw(self, raw):
        return self.predict_proba_features(self.features_from_raw(raw))
