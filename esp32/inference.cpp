// Smart Cushion - dependency-free INT8 inference. See inference.h.
//
// The requantisation below reproduces TFLite's single-rounding integer
// FULLY_CONNECTED kernel exactly, which is what lets the generated C++ match
// the TFLite INT8 interpreter bit-for-bit rather than approximately.

#include "inference.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// Two ping-pong activation buffers, sized for the widest layer. Static, so the
// inference loop performs no allocation at all.
static int32_t gBufA[SC_MAX_UNITS];
static int32_t gBufB[SC_MAX_UNITS];

// ---------------------------------------------------------------------------
// TFLite requantisation, single-rounding variant (TFLITE_SINGLE_ROUNDING).
//
// Rounds ONCE on a 64-bit intermediate. The older gemmlowp path
// (SaturatingRoundingDoublingHighMul + RoundingDivideByPOT) rounds twice and
// pushes values just below .5 up to the next integer, which flips occasional
// near-tie argmaxes. ml/export_int8.py implements exactly this function.
// ---------------------------------------------------------------------------
static inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t qm,
                                                    int32_t shift) {
  const int32_t total_shift = 31 - shift;              // shift is in [-31, 30]
  const int64_t rounding = (int64_t)1 << (total_shift - 1);
  int64_t result = (int64_t)x * (int64_t)qm + rounding;
  result >>= total_shift;
  if (result < INT32_MIN) return INT32_MIN;
  if (result > INT32_MAX) return INT32_MAX;
  return (int32_t)result;
}

// ---------------------------------------------------------------------------
void scQuantiseInput(const float *features, int8_t *q_out) {
  for (int i = 0; i < SC_MODEL_INPUT_DIM; i++) {
    // lrintf uses the default rounding mode (round-half-to-even), which is
    // what numpy.round does, and the division is float32 on both sides.
    float v = features[i] / SC_INPUT_SCALE;
    int32_t q = (int32_t)lrintf(v) + SC_INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    q_out[i] = (int8_t)q;
  }
}

static void runLayer(const ScLayer *L, const int32_t *in, int32_t *out) {
  const int8_t *w = L->w;
  for (int u = 0; u < L->units; u++) {
    int32_t acc = L->b[u];
    const int8_t *wrow = w + (int32_t)u * L->in_dim;
    for (int j = 0; j < L->in_dim; j++) {
      acc += (int32_t)wrow[j] * (in[j] - L->input_zero_point);
    }
    int32_t v = MultiplyByQuantizedMultiplier(acc, L->mult[u], L->shift[u]);
    v += L->output_zero_point;
    const int32_t lo = L->relu ? L->output_zero_point : -128;
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    out[u] = v;
  }
}

void scInferQuantised(const int8_t *q_in, ScInferResult *out) {
  int32_t *cur = gBufA;
  int32_t *nxt = gBufB;

  for (int i = 0; i < SC_MODEL_INPUT_DIM; i++) cur[i] = (int32_t)q_in[i];

  for (int li = 0; li < SC_N_LAYERS; li++) {
    runLayer(&SC_LAYERS[li], cur, nxt);
    int32_t *t = cur; cur = nxt; nxt = t;
  }

  // cur now holds SC_N_CLASSES quantised logits
  for (int c = 0; c < SC_N_CLASSES; c++) out->logits[c] = cur[c];

  // dequantise, then softmax in float
  float z[SC_N_CLASSES];
  float zmax = -1e30f;
  for (int c = 0; c < SC_N_CLASSES; c++) {
    z[c] = (float)(cur[c] - SC_OUTPUT_ZERO_POINT) * SC_OUTPUT_SCALE;
    if (z[c] > zmax) zmax = z[c];
  }
  float sum = 0.0f;
  for (int c = 0; c < SC_N_CLASSES; c++) {
    z[c] = expf(z[c] - zmax);
    sum += z[c];
  }
  int8_t best = 0;
  for (int c = 0; c < SC_N_CLASSES; c++) {
    out->prob[c] = z[c] / sum;
    if (out->prob[c] > out->prob[best]) best = (int8_t)c;
  }
  out->argmax = best;
  out->confidence = out->prob[best];
}

void scInfer(const float *features, ScInferResult *out) {
  static int8_t q[SC_MODEL_INPUT_DIM];
  scQuantiseInput(features, q);
  scInferQuantised(q, out);
}

uint32_t scModelFlashBytes(void) {
  uint32_t total = 0;
  for (int li = 0; li < SC_N_LAYERS; li++) {
    const ScLayer *L = &SC_LAYERS[li];
    total += (uint32_t)L->in_dim * L->units;   // int8 weights
    total += (uint32_t)L->units * 4;           // int32 bias
    total += (uint32_t)L->units * 4;           // int32 multiplier
    total += (uint32_t)L->units * 4;           // int32 shift
  }
  total += sizeof(SC_LAYERS);
  return total;
}

uint32_t scInferenceRamBytes(void) {
  return (uint32_t)(sizeof(gBufA) + sizeof(gBufB) + SC_MODEL_INPUT_DIM);
}
