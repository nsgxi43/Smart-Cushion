// Smart Cushion - preprocessing contract, C++ side.
//
// Literal translation of ml/preprocess.py. Any change here must be mirrored
// there and re-verified by tests/test_preprocess_equivalence.py.
//
// No dynamic allocation. All buffers are static and sized at compile time.

#ifndef SMART_CUSHION_PREPROCESS_H
#define SMART_CUSHION_PREPROCESS_H

#include <stdint.h>

// ---- geometry (fixed by the firmware scan loop) ----------------------------
#define SC_ROWS        24
#define SC_COLS        21
#define SC_N_FEATURES  (SC_ROWS * SC_COLS)   // 504
#define SC_ADC_MIN     0
#define SC_ADC_MAX     4095

// feature_index = row * SC_COLS + col   (row-major, same order the FRAME line
// is printed in and the same order the sensor_* CSV columns are in)

// ---- fixed pipeline constants (mirror ml/preprocess.py) --------------------
#define SC_CLIP_MAX    4095
#define SC_SCALE       1024.0f
#define SC_INV_SCALE   (1.0f / SC_SCALE)     // exactly 0.0009765625f

// ---- calibration constants ------------------------------------------------
#define SC_CAL_FRAMES        20
#define SC_NOISE_MULTIPLIER  3
#define SC_MIN_THRESHOLD     20

// ---- calibration acceptance rules -----------------------------------------
#define SC_CAL_MAX_MEAN_ABS_DEV   25.0f
#define SC_CAL_MAX_LOADED_CELLS   10
#define SC_CAL_LOAD_DELTA         150.0f
#define SC_CAL_MAX_NOISE_STD      120.0f
#define SC_CAL_MIN_BASELINE       1.0f
#define SC_CAL_MAX_BASELINE       4045.0f    // SC_ADC_MAX - 50

// ---- calibration state machine --------------------------------------------
typedef enum {
  SC_CAL_IDLE = 0,
  SC_CAL_COLLECTING,
  SC_CAL_READY,
  SC_CAL_FAILED
} ScCalState;

typedef enum {
  SC_CAL_OK = 0,
  SC_CAL_ERR_TOO_FEW_FRAMES,
  SC_CAL_ERR_PRESSURE_DETECTED,
  SC_CAL_ERR_UNSTABLE_BASELINE,
  SC_CAL_ERR_EXCESSIVE_NOISE,
  SC_CAL_ERR_NO_SIGNAL,
  SC_CAL_ERR_RAIL_SATURATED
} ScCalResult;

// Accepted calibration. Valid only while scCalState() == SC_CAL_READY.
extern int32_t scBaseline[SC_N_FEATURES];
extern int32_t scThreshold[SC_N_FEATURES];

const char *scCalResultName(ScCalResult r);
const char *scCalStateName(ScCalState s);

// --- calibration API --------------------------------------------------------
void        scCalibrationBegin(void);                 // clear and enter COLLECTING
bool        scCalibrationAddFrame(const int32_t *raw);// true once enough frames
ScCalResult scCalibrationFinish(void);                // validate + accept/reject
ScCalState  scCalState(void);
uint16_t    scCalFrameCount(void);

// --- transform --------------------------------------------------------------
// raw[504] -> out[504] float features. Requires SC_CAL_READY.
void scPreprocessFrame(const int32_t *raw, float *out);

// Stage 1-3 only: raw -> integer pressure counts (what pressure_data.csv holds).
void scRawToPressure(const int32_t *raw, int32_t *pressure);

// Stage 4 only: integer pressure counts -> float features.
void scPressureToFeatures(const int32_t *pressure, float *out);

#endif  // SMART_CUSHION_PREPROCESS_H
