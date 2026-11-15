"""Smart Cushion live dashboard.

    streamlit run dashboard/app.py

Works with a connected ESP32 or with the simulated source (recorded CSV
frames), so the whole UI can be exercised without hardware.

Prediction provenance is shown explicitly everywhere:
  ESP32   - arrived as a PRED, line from the board (embedded inference)
  Python  - computed here from the exported model (NOT embedded inference)
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ml"))

from serial_source import (COLS, ROWS, SerialSource, SimulatedSource,
                           list_ports, load_replay)

CLASS_NAMES = ["C0_Normal", "C1_Forward_Lean", "C2_Slouching",
               "C3_Left_Lean", "C4_Right_Lean", "C5_Backward_Lean"]
HISTORY_SECONDS = 60

st.set_page_config(page_title="Smart Cushion", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------- state
def S():
    s = st.session_state
    s.setdefault("source", None)
    s.setdefault("source_kind", None)
    s.setdefault("running", False)
    s.setdefault("recording", False)
    s.setdefault("frames", deque(maxlen=4000))      # (t, raw) for recording
    s.setdefault("history", deque(maxlen=4000))     # (t, label, conf, origin)
    s.setdefault("last_frame", None)
    s.setdefault("last_pred", None)
    s.setdefault("log", deque(maxlen=200))
    s.setdefault("manual_label", "(not set)")
    s.setdefault("manual_log", [])
    s.setdefault("frame_count", 0)
    s.setdefault("py_model", None)
    s.setdefault("py_model_error", None)
    s.setdefault("state_text", "unknown")
    return s


def reset_all():
    s = S()
    s.frames.clear(); s.history.clear(); s.log.clear()
    s.last_frame = None; s.last_pred = None
    s.frame_count = 0; s.manual_log = []


def load_py_model():
    s = S()
    if s.py_model is not None or s.py_model_error is not None:
        return s.py_model
    try:
        from python_reference import PythonReferenceModel
        s.py_model = PythonReferenceModel()
    except Exception as exc:
        s.py_model_error = str(exc)
    return s.py_model


def model_info():
    p = ROOT / "models" / "model_metadata.json"
    if p.is_file():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------- sidebar
s = S()
st.sidebar.title("Smart Cushion")

mode = st.sidebar.radio("Frame source", ["Simulated (recorded CSV)", "ESP32 serial"],
                        index=0, disabled=s.running)

if mode == "ESP32 serial":
    ports = list_ports()
    opts = [p[0] for p in ports] or ["(no ports found)"]
    port = st.sidebar.selectbox("Port", opts, disabled=s.running)
    baud = st.sidebar.number_input("Baud", value=115200, step=1, disabled=s.running)
else:
    try:
        d = np.load(ROOT / "models" / "dataset.npz", allow_pickle=False)
        parts = ["(all)"] + sorted(set(d["participant"].astype(str).tolist()))
        posts = ["(all)"] + CLASS_NAMES
    except Exception:
        parts, posts = ["(all)"], ["(all)"] + CLASS_NAMES
    sim_part = st.sidebar.selectbox("Replay participant", parts, disabled=s.running)
    sim_post = st.sidebar.selectbox("Replay posture", posts, disabled=s.running)
    sim_fps = st.sidebar.slider("Replay rate (fps)", 0.2, 10.0, 1.4, 0.2,
                                disabled=s.running)
    sim_predict = st.sidebar.checkbox(
        "Run the exported model here (labelled Python)", value=True,
        disabled=s.running)

c1, c2 = st.sidebar.columns(2)
start = c1.button("Start test", disabled=s.running, use_container_width=True)
stop = c2.button("Stop test", disabled=not s.running, use_container_width=True)

c3, c4 = st.sidebar.columns(2)
rec_start = c3.button("Start recording", disabled=s.recording, use_container_width=True)
rec_stop = c4.button("Stop recording", disabled=not s.recording, use_container_width=True)

if st.sidebar.button("Reset", use_container_width=True):
    reset_all()
    st.sidebar.success("cleared")

st.sidebar.divider()
st.sidebar.subheader("Device commands")
cc1, cc2 = st.sidebar.columns(2)
if cc1.button("Recalibrate (CAL)", use_container_width=True, disabled=not s.running):
    s.source.send("CAL")
if cc2.button("STATUS", use_container_width=True, disabled=not s.running):
    s.source.send("STATUS")
cm1, cm2 = st.sidebar.columns(2)
if cm1.button("MODE RAW", use_container_width=True, disabled=not s.running):
    s.source.send("MODE RAW")
if cm2.button("MODE PREDICT", use_container_width=True, disabled=not s.running):
    s.source.send("MODE PREDICT")

st.sidebar.divider()
st.sidebar.subheader("Manual posture (for comparison)")
s.manual_label = st.sidebar.selectbox("What posture is actually being held?",
                                      ["(not set)"] + CLASS_NAMES,
                                      index=(["(not set)"] + CLASS_NAMES)
                                      .index(s.manual_label))
if st.sidebar.button("Log this comparison", use_container_width=True):
    if s.last_pred and s.manual_label != "(not set)":
        s.manual_log.append({
            "time": time.strftime("%H:%M:%S"),
            "manual_posture": s.manual_label,
            "predicted": s.last_pred["label"],
            "confidence": s.last_pred["confidence"],
            "origin": s.last_pred["origin"],
            "correct": s.manual_label == s.last_pred["label"],
        })
        st.sidebar.success("logged")
    else:
        st.sidebar.warning("need a prediction and a manual posture")


# ---------------------------------------------------------------- control
if start:
    reset_all()
    if mode == "ESP32 serial":
        if port.startswith("("):
            st.sidebar.error("no serial port available")
        else:
            s.source = SerialSource(port, int(baud)).start()
            s.source_kind = "ESP32"
            s.running = True
    else:
        try:
            frames, labels = load_replay(
                None if sim_part == "(all)" else sim_part,
                None if sim_post == "(all)" else sim_post)
            predictor = None
            if sim_predict:
                m = load_py_model()
                if m is not None:
                    predictor = m.predict_from_raw
            s.source = SimulatedSource(frames, labels=labels, fps=sim_fps,
                                       predictor=predictor).start()
            s.source_kind = "Simulated"
            s.running = True
        except Exception as exc:
            st.sidebar.error(str(exc))

if stop and s.source is not None:
    s.source.stop()
    s.running = False

if rec_start:
    s.recording = True
    s.frames.clear()
if rec_stop:
    s.recording = False


# ---------------------------------------------------------------- pump
if s.running and s.source is not None:
    for ev in s.source.drain():
        k = ev["kind"]
        if k == "frame":
            s.last_frame = ev["raw"]
            s.frame_count += 1
            if s.recording:
                s.frames.append((time.time(), ev["raw"]))
        elif k == "pred":
            s.last_pred = ev
            s.history.append((ev.get("t", time.monotonic()), ev["label"],
                              ev["confidence"], ev["origin"]))
        elif k == "state":
            s.state_text = ev["text"]
            s.log.append("[STATE] " + ev["text"])
        else:
            s.log.append("[%s] %s" % (k.upper(), ev.get("text", "")))
    if s.source.error:
        st.error("source error: %s" % s.source.error)


# ---------------------------------------------------------------- layout
st.title("Smart Cushion — live posture")

banner = {"ESP32": ("Connected to ESP32 — predictions below are **embedded "
                    "inference running on the board**.", "success"),
          "Simulated": ("**SIMULATED SOURCE.** Frames are replayed from recorded "
                        "CSV files and any prediction shown is computed in "
                        "Python on this PC — this is *not* embedded inference.",
                        "warning")}.get(s.source_kind)
if banner:
    (st.success if banner[1] == "success" else st.warning)(banner[0])
else:
    st.info("Pick a frame source in the sidebar and press **Start test**.")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Pressure map (24 × 21)")
    if s.last_frame is not None:
        mat = np.asarray(s.last_frame, dtype=float).reshape(ROWS, COLS)
        fig_df = pd.DataFrame(mat)
        st.dataframe(
            fig_df.style.background_gradient(cmap="turbo", vmin=0, vmax=4095)
            .format("{:.0f}"),
            use_container_width=True, height=560)
        st.caption("frame %d | min %d | max %d | mean %.1f"
                   % (s.frame_count, mat.min(), mat.max(), mat.mean()))
    else:
        st.info("no frames yet")

with right:
    st.subheader("Current posture")
    if s.last_pred:
        origin = s.last_pred["origin"]
        tag = "🟢 ESP32 (on-device)" if origin == "ESP32" else "🟡 Python (PC)"
        st.markdown("<div style='font-size:2.6rem;font-weight:700;line-height:1.1'>"
                    + s.last_pred["label"].replace("_", " ") + "</div>",
                    unsafe_allow_html=True)
        st.progress(min(max(s.last_pred["confidence"], 0.0), 1.0),
                    text="confidence %.1f%%" % (100 * s.last_pred["confidence"]))
        m1, m2 = st.columns(2)
        m1.metric("Inference", "%d µs" % s.last_pred["inference_us"])
        m2.metric("Source of prediction", tag)
        if origin != "ESP32":
            st.caption("This label was produced on this PC, not on the ESP32.")
    else:
        st.info("no prediction yet")

    st.subheader("Last %d s" % HISTORY_SECONDS)
    if s.history:
        now = time.monotonic()
        rows = [{"t": now - t, "posture": lab, "confidence": conf, "origin": org}
                for t, lab, conf, org in s.history if now - t <= HISTORY_SECONDS]
        if rows:
            hdf = pd.DataFrame(rows).sort_values("t")
            st.line_chart(hdf.set_index("t")[["confidence"]], height=160)
            tip = (hdf.groupby("posture").size() /
                   max(len(hdf), 1) * 100).rename("% of window")
            st.bar_chart(tip, height=200)
        else:
            st.caption("no predictions inside the window yet")
    else:
        st.caption("no history yet")

st.divider()
t1, t2, t3 = st.columns(3)

with t1:
    st.subheader("Model")
    meta = model_info()
    if meta:
        st.markdown(
            "\n".join([
                "- **Architecture:** %s" % meta.get("architecture", "?"),
                "- **Version:** %s" % meta.get("version", "?"),
                "- **Quantisation:** %s" % meta.get("quantisation", "?"),
                "- **Model size:** %s" % meta.get("model_size_human", "?"),
                "- **Training set:** %s" % meta.get("training_set", "?"),
                "- **Held-out real accuracy:** %s"
                % meta.get("heldout_real_accuracy_human", "?"),
                "- **ESP32 inference time:** %s"
                % meta.get("esp32_inference_us", "PENDING_HARDWARE"),
            ]))
    else:
        st.caption("models/model_metadata.json not found (run ml/export_int8.py)")
    if s.py_model_error:
        st.caption("Python model unavailable: %s" % s.py_model_error)

with t2:
    st.subheader("Device log")
    st.code("\n".join(list(s.log)[-18:]) or "(empty)", language=None)
    st.caption("calibration state: **%s**" % s.state_text)

with t3:
    st.subheader("Manual comparison")
    if s.manual_log:
        mdf = pd.DataFrame(s.manual_log)
        st.dataframe(mdf, use_container_width=True, height=220)
        acc = 100.0 * mdf["correct"].mean()
        st.metric("Agreement with manual labels", "%.1f%% (n=%d)" % (acc, len(mdf)))
        st.download_button("Download comparison CSV",
                           mdf.to_csv(index=False).encode(),
                           file_name="manual_comparison.csv", mime="text/csv")
    else:
        st.caption("pick the real posture in the sidebar and press "
                   "**Log this comparison**")

st.divider()
st.subheader("Recording")
rc1, rc2 = st.columns([1, 2])
rc1.metric("Recorded frames", len(s.frames))
rc1.caption("recording: **%s**" % ("ON" if s.recording else "off"))
if s.frames:
    buf = io.StringIO()
    cols = ["timestamp"] + ["sensor_%03d" % i for i in range(ROWS * COLS)]
    rows = [[t] + list(map(int, raw)) for t, raw in s.frames]
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False)
    rc2.download_button("Export frames to CSV", buf.getvalue().encode(),
                        file_name="smart_cushion_capture.csv", mime="text/csv")

if s.running:
    time.sleep(0.4)
    st.rerun()
