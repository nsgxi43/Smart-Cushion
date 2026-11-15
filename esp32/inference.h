// Smart Cushion - dependency-free INT8 inference.
//
// No TensorFlow, no dynamic allocation, no floating point in the matmul.
// Arithmetic matches TFLite's reference FULLY_CONNECTED kernel exactly:
// int32 accumulators and gemmlowp requantisation.

#ifndef SMART_CUSHION_INFERENCE_H
#define SMART_CUSHION_INFERENCE_H

#include <stdint.h>

#include "model_data.h"
#include "preprocess.h"

typedef struct {
  int32_t logits[SC_N_CLASSES];   // raw int8 output values, widened
  float   prob[SC_N_CLASSES];     // softmax over the dequantised logits
  int8_t  argmax;
  float   confidence;             // prob[argmax]
} ScInferResult;

// features[504] (output of scPreprocessFrame) -> result.
void scInfer(const float *features, ScInferResult *out);

// Same, starting from an already-quantised int8 input vector. Used by tests.
void scInferQuantised(const int8_t *q_in, ScInferResult *out);

// Quantise 504 float features to int8 using the model's input scale/zero point.
void scQuantiseInput(const float *features, int8_t *q_out);

// Total static bytes the model and workspace occupy, for the memory report.
uint32_t scModelFlashBytes(void);
uint32_t scInferenceRamBytes(void);

#endif  // SMART_CUSHION_INFERENCE_H
