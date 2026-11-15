// Smart Cushion - preprocessing contract, C++ side.
// Literal translation of ml/preprocess.py. See preprocess.h.

#include "preprocess.h"

#include <math.h>
#include <string.h>

int32_t scBaseline[SC_N_FEATURES];
int32_t scThreshold[SC_N_FEATURES];

// Calibration accumulators. Static, no heap. Two passes are avoided by
// accumulating sum and sum-of-squares plus a per-frame deviation statistic.
static ScCalState  gState = SC_CAL_IDLE;
static uint16_t    gFrames = 0;
static double      gSum[SC_N_FEATURES];
static double      gSumSq[SC_N_FEATURES];
// running mean-absolute-deviation of each frame from the mean of the frames
// seen so far; used to detect a drifting or disturbed calibration window.
static double      gDevAccum = 0.0;
static uint32_t    gDevCount = 0;

const char *scCalResultName(ScCalResult r) {
  switch (r) {
    case SC_CAL_OK:                    return "OK";
    case SC_CAL_ERR_TOO_FEW_FRAMES:    return "TOO_FEW_FRAMES";
    case SC_CAL_ERR_PRESSURE_DETECTED: return "PRESSURE_DETECTED";
    case SC_CAL_ERR_UNSTABLE_BASELINE: return "UNSTABLE_BASELINE";
    case SC_CAL_ERR_EXCESSIVE_NOISE:   return "EXCESSIVE_NOISE";
    case SC_CAL_ERR_NO_SIGNAL:         return "NO_SIGNAL";
    case SC_CAL_ERR_RAIL_SATURATED:    return "RAIL_SATURATED";
  }
  return "UNKNOWN";
}

const char *scCalStateName(ScCalState s) {
  switch (s) {
    case SC_CAL_IDLE:       return "IDLE";
    case SC_CAL_COLLECTING: return "COLLECTING";
    case SC_CAL_READY:      return "READY";
    case SC_CAL_FAILED:     return "CALIBRATION_FAILED";
  }
  return "UNKNOWN";
}

ScCalState scCalState(void)    { return gState; }
uint16_t   scCalFrameCount(void) { return gFrames; }

void scCalibrationBegin(void) {
  memset(gSum,   0, sizeof(gSum));
  memset(gSumSq, 0, sizeof(gSumSq));
  gFrames   = 0;
  gDevAccum = 0.0;
  gDevCount = 0;
  gState    = SC_CAL_COLLECTING;
}

bool scCalibrationAddFrame(const int32_t *raw) {
  if (gState != SC_CAL_COLLECTING) return false;

  // deviation of this frame from the mean of the frames already seen
  if (gFrames > 0) {
    double dev = 0.0;
    for (int i = 0; i < SC_N_FEATURES; i++) {
      double mean_so_far = gSum[i] / (double)gFrames;
      dev += fabs((double)raw[i] - mean_so_far);
    }
    gDevAccum += dev / (double)SC_N_FEATURES;
    gDevCount += 1;
  }

  for (int i = 0; i < SC_N_FEATURES; i++) {
    double v = (double)raw[i];
    gSum[i]   += v;
    gSumSq[i] += v * v;
  }
  gFrames++;
  return gFrames >= SC_CAL_FRAMES;
}

// Insertion-free median of the per-sensor means, via a copy + partial sort.
static float medianOfMeans(const double *mean) {
  static float tmp[SC_N_FEATURES];
  for (int i = 0; i < SC_N_FEATURES; i++) tmp[i] = (float)mean[i];
  // selection up to the midpoint is enough and avoids qsort's stack use
  const int mid = SC_N_FEATURES / 2;
  for (int i = 0; i <= mid; i++) {
    int m = i;
    for (int j = i + 1; j < SC_N_FEATURES; j++) {
      if (tmp[j] < tmp[m]) m = j;
    }
    float t = tmp[i]; tmp[i] = tmp[m]; tmp[m] = t;
  }
  return tmp[mid];
}

ScCalResult scCalibrationFinish(void) {
  if (gFrames < SC_CAL_FRAMES) {
    gState = SC_CAL_FAILED;
    return SC_CAL_ERR_TOO_FEW_FRAMES;
  }

  static double mean[SC_N_FEATURES];
  static double sd[SC_N_FEATURES];

  const double n = (double)gFrames;
  float minMean = 1e9f, maxMean = -1e9f;
  double meanNoise = 0.0;

  for (int i = 0; i < SC_N_FEATURES; i++) {
    mean[i] = gSum[i] / n;
    double var = gSumSq[i] / n - mean[i] * mean[i];
    if (var < 0.0) var = 0.0;              // guard float cancellation
    sd[i] = sqrt(var);
    meanNoise += sd[i];
    if (mean[i] < minMean) minMean = (float)mean[i];
    if (mean[i] > maxMean) maxMean = (float)mean[i];
  }
  meanNoise /= (double)SC_N_FEATURES;

  const float median = medianOfMeans(mean);
  int loaded = 0;
  for (int i = 0; i < SC_N_FEATURES; i++) {
    if (fabs(mean[i] - (double)median) > (double)SC_CAL_LOAD_DELTA) loaded++;
  }

  const double meanAbsDev = (gDevCount > 0) ? (gDevAccum / (double)gDevCount) : 0.0;

  // --- acceptance rules, same order as validate_calibration() in Python ---
  ScCalResult res = SC_CAL_OK;
  if (loaded > SC_CAL_MAX_LOADED_CELLS)              res = SC_CAL_ERR_PRESSURE_DETECTED;
  else if (meanAbsDev > (double)SC_CAL_MAX_MEAN_ABS_DEV) res = SC_CAL_ERR_UNSTABLE_BASELINE;
  else if (meanNoise > (double)SC_CAL_MAX_NOISE_STD) res = SC_CAL_ERR_EXCESSIVE_NOISE;
  else if (maxMean < SC_CAL_MIN_BASELINE)            res = SC_CAL_ERR_NO_SIGNAL;
  else if (minMean > SC_CAL_MAX_BASELINE)            res = SC_CAL_ERR_RAIL_SATURATED;

  if (res != SC_CAL_OK) {
    gState = SC_CAL_FAILED;
    return res;
  }

  for (int i = 0; i < SC_N_FEATURES; i++) {
    // nearbyint with the default rounding mode == numpy rint (banker's rounding)
    scBaseline[i] = (int32_t)nearbyint(mean[i]);
    double t = (double)SC_NOISE_MULTIPLIER * sd[i];
    if (t < (double)SC_MIN_THRESHOLD) t = (double)SC_MIN_THRESHOLD;
    scThreshold[i] = (int32_t)nearbyint(t);
  }

  gState = SC_CAL_READY;
  return SC_CAL_OK;
}

void scRawToPressure(const int32_t *raw, int32_t *pressure) {
  for (int i = 0; i < SC_N_FEATURES; i++) {
    int32_t d = raw[i] - scBaseline[i];
    if (d < 0) d = -d;                        // stage 1: abs
    if (d <= scThreshold[i]) d = 0;           // stage 2: noise gate
    if (d > SC_CLIP_MAX) d = SC_CLIP_MAX;     // stage 3: clamp
    pressure[i] = d;
  }
}

void scPressureToFeatures(const int32_t *pressure, float *out) {
  for (int i = 0; i < SC_N_FEATURES; i++) {
    int32_t p = pressure[i];
    if (p < 0) p = 0;
    if (p > SC_CLIP_MAX) p = SC_CLIP_MAX;
    out[i] = (float)p * SC_INV_SCALE;         // stage 4: fixed global scale
  }
}

void scPreprocessFrame(const int32_t *raw, float *out) {
  for (int i = 0; i < SC_N_FEATURES; i++) {
    int32_t d = raw[i] - scBaseline[i];
    if (d < 0) d = -d;
    if (d <= scThreshold[i]) d = 0;
    if (d > SC_CLIP_MAX) d = SC_CLIP_MAX;
    out[i] = (float)d * SC_INV_SCALE;
  }
}
