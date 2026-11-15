// Host-side harness: runs the *same* preprocess.cpp / inference.cpp the ESP32
// runs, driven from stdin, so the Python tests can compare numerically without
// a board attached. Not compiled into the firmware.
//
// Build:  g++ -O2 -std=c++17 -o host_harness host_harness.cpp preprocess.cpp \
//             [inference.cpp]
//
// Protocol (line based, stdin -> stdout):
//   CAL_BEGIN
//   CAL_FRAME <504 ints>          repeat >= 20 times
//   CAL_FINISH                    -> "CAL_RESULT <name>"
//   SET_CAL <504 baseline ints> <504 threshold ints>   (bypass, for tests)
//   GET_CAL                       -> "CAL <504 baseline> <504 threshold>"
//   PRE <504 raw ints>            -> "FEAT <504 floats, %.9g>"
//   PRESSURE <504 raw ints>       -> "PRESS <504 ints>"
//   FEAT_FROM_PRESSURE <504 ints> -> "FEAT <504 floats>"
//   INFER <504 raw ints>          -> "LOGITS <6 ints> ARGMAX <i> PROB <6 floats>"
//   INFER_FEAT <504 floats>       -> same, skipping preprocessing
//   QUIT

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "preprocess.h"

#ifdef SC_WITH_INFERENCE
#include "inference.h"
#endif

static bool readInts(std::istringstream &is, int32_t *dst, int n) {
  for (int i = 0; i < n; i++) {
    long v;
    if (!(is >> v)) return false;
    dst[i] = (int32_t)v;
  }
  return true;
}

int main() {
  static int32_t raw[SC_N_FEATURES];
  static int32_t tmp[SC_N_FEATURES];
  static float   feat[SC_N_FEATURES];

  std::string line;

  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string cmd;
    if (!(is >> cmd)) continue;

    if (cmd == "QUIT") break;

    if (cmd == "CAL_BEGIN") {
      scCalibrationBegin();
      std::cout << "OK\n";
    } else if (cmd == "CAL_FRAME") {
      if (!readInts(is, raw, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
      bool enough = scCalibrationAddFrame(raw);
      std::cout << "OK " << scCalFrameCount() << " " << (enough ? 1 : 0) << "\n";
    } else if (cmd == "CAL_FINISH") {
      ScCalResult r = scCalibrationFinish();
      std::cout << "CAL_RESULT " << scCalResultName(r)
                << " STATE " << scCalStateName(scCalState()) << "\n";
    } else if (cmd == "SET_CAL") {
      if (!readInts(is, scBaseline, SC_N_FEATURES) ||
          !readInts(is, scThreshold, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
      std::cout << "OK\n";
    } else if (cmd == "GET_CAL") {
      std::cout << "CAL";
      for (int i = 0; i < SC_N_FEATURES; i++) std::cout << " " << scBaseline[i];
      for (int i = 0; i < SC_N_FEATURES; i++) std::cout << " " << scThreshold[i];
      std::cout << "\n";
    } else if (cmd == "PRE") {
      if (!readInts(is, raw, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
      scPreprocessFrame(raw, feat);
      std::cout << "FEAT";
      for (int i = 0; i < SC_N_FEATURES; i++) printf(" %.9g", (double)feat[i]);
      std::cout << "\n";
    } else if (cmd == "PRESSURE") {
      if (!readInts(is, raw, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
      scRawToPressure(raw, tmp);
      std::cout << "PRESS";
      for (int i = 0; i < SC_N_FEATURES; i++) std::cout << " " << tmp[i];
      std::cout << "\n";
    } else if (cmd == "FEAT_FROM_PRESSURE") {
      if (!readInts(is, tmp, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
      scPressureToFeatures(tmp, feat);
      std::cout << "FEAT";
      for (int i = 0; i < SC_N_FEATURES; i++) printf(" %.9g", (double)feat[i]);
      std::cout << "\n";
    }
#ifdef SC_WITH_INFERENCE
    else if (cmd == "INFER" || cmd == "INFER_FEAT") {
      if (cmd == "INFER") {
        if (!readInts(is, raw, SC_N_FEATURES)) { std::cout << "ERR_PARSE\n"; continue; }
        scPreprocessFrame(raw, feat);
      } else {
        for (int i = 0; i < SC_N_FEATURES; i++) {
          double v;
          if (!(is >> v)) { std::cout << "ERR_PARSE\n"; goto next; }
          feat[i] = (float)v;
        }
      }
      {
        ScInferResult r;
        scInfer(feat, &r);
        std::cout << "LOGITS";
        for (int i = 0; i < SC_N_CLASSES; i++) std::cout << " " << r.logits[i];
        std::cout << " ARGMAX " << (int)r.argmax << " PROB";  // int8_t streams as a char
        for (int i = 0; i < SC_N_CLASSES; i++) printf(" %.9g", (double)r.prob[i]);
        std::cout << "\n";
      }
      next:;
    }
#endif
    else {
      std::cout << "ERR_CMD\n";
    }
    std::cout.flush();
  }
  return 0;
}
