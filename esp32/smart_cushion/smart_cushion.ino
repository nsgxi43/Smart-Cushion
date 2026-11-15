// Smart Cushion - 24x21 Velostat matrix scanner + on-device posture inference.
//
// The scan loop below is byte-for-byte the original
// esp32/backup_20261115/matrix_scanner_original.ino: same pins, same order,
// same SAMPLES/SETTLE_US, same readAverage(). Inference runs only AFTER the
// full 504-value frame has been acquired. See esp32/FIRMWARE_CHANGES.md for
// the diff and the reason for every added line.
//
// Serial protocol, 115200 baud
//   FRAME,<504 raw comma-separated values>
//   PRED,<label>,<confidence 0-1>,<inference_us>
//   STATE,<calibration state>
//   CAL,<progress or result>
//   INFO,<free-form>
// Commands (one per line, case-insensitive)
//   MODE RAW      - stream FRAME only
//   MODE PREDICT  - stream FRAME + PRED (default)
//   MODE BOTH     - same as PREDICT
//   CAL           - restart baseline calibration
//   STATUS        - print state, mode, model info
//   HELP

#include "preprocess.h"
#include "inference.h"

// ======================= ORIGINAL SCAN CONFIGURATION =======================
// (unchanged from matrix_scanner_original.ino)
const int ROWS = 24;
const int COLS = 21;

const int ROW_S0 = 25;
const int ROW_S1 = 26;
const int ROW_S2 = 27;
const int ROW_S3 = 14;

const int COL_S0 = 16;
const int COL_S1 = 17;
const int COL_S2 = 18;
const int COL_S3 = 19;

const int ROW_EN_1 = 21;
const int ROW_EN_2 = 22;

const int ROW_SIG = 13;

const int COL_ADC_1 = 34;
const int COL_ADC_2 = 35;

const int SAMPLES = 16;
const int SETTLE_US = 300;

int readings[ROWS][COLS];
// ===========================================================================

// ------------------------- added: runtime state ----------------------------
enum Mode { MODE_RAW, MODE_PREDICT };
static Mode gMode = MODE_PREDICT;

static int32_t gFrame[SC_N_FEATURES];      // the frame flattened row-major
static float   gFeatures[SC_N_FEATURES];

// probability smoothing: moving average over the last N frames.
// The scan runs at roughly 1.4 Hz, so keep the window short or the label
// lags the sitter by several seconds.
#define SC_SMOOTH_WINDOW 3
#define SC_CONFIDENCE_THRESHOLD 0.55f
static float gProbHistory[SC_SMOOTH_WINDOW][SC_N_CLASSES];
static uint8_t gHistCount = 0;
static uint8_t gHistHead = 0;

static const char *const STATE_UNKNOWN = "UNKNOWN";

// ---------------------- ORIGINAL SCAN HELPERS (unchanged) ------------------
void setAddress(int channel, int s0, int s1, int s2, int s3) {
  digitalWrite(s0, channel & 1);
  digitalWrite(s1, (channel >> 1) & 1);
  digitalWrite(s2, (channel >> 2) & 1);
  digitalWrite(s3, (channel >> 3) & 1);
}

int readAverage(int pin) {
  // Discard first reading after switching channels
  analogRead(pin);
  delayMicroseconds(200);

  long sum = 0;

  for (int i = 0; i < SAMPLES; i++) {
    sum += analogRead(pin);
    delayMicroseconds(50);
  }

  return sum / SAMPLES;
}

// Exactly the original loop() scan body, extracted into a function so that
// inference can run after it without touching the scan itself.
void scanMatrix() {
  for (int r = 0; r < ROWS; r++) {

    // Disable both row MUXes before changing row
    digitalWrite(ROW_SIG, LOW);
    digitalWrite(ROW_EN_1, HIGH);
    digitalWrite(ROW_EN_2, HIGH);

    int rowChannel = r % 16;

    setAddress(
      rowChannel,
      ROW_S0, ROW_S1, ROW_S2, ROW_S3
    );

    // Enable the appropriate row MUX
    if (r < 16) {
      digitalWrite(ROW_EN_1, LOW);
    } else {
      digitalWrite(ROW_EN_2, LOW);
    }

    delayMicroseconds(SETTLE_US);

    // Drive selected row HIGH
    digitalWrite(ROW_SIG, HIGH);

    delayMicroseconds(SETTLE_US);

    // Scan columns
    for (int c = 0; c < 16; c++) {

      setAddress(
        c,
        COL_S0, COL_S1, COL_S2, COL_S3
      );

      delayMicroseconds(SETTLE_US);

      readings[r][c] = readAverage(COL_ADC_1);

      // Second column MUX handles columns 16-20
      if (c < 5) {
        readings[r][c + 16] = readAverage(COL_ADC_2);
      }
    }

    // Disable selected row
    digitalWrite(ROW_SIG, LOW);
    digitalWrite(ROW_EN_1, HIGH);
    digitalWrite(ROW_EN_2, HIGH);
  }
}
// ---------------------------------------------------------------------------

static void flattenFrame() {
  // row-major, index = r * COLS + c - the ordering the model was trained on
  int i = 0;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      gFrame[i++] = (int32_t)readings[r][c];
    }
  }
}

static void printFrame() {
  Serial.print("FRAME,");
  for (int i = 0; i < SC_N_FEATURES; i++) {
    Serial.print(gFrame[i]);
    if (i != SC_N_FEATURES - 1) Serial.print(",");
  }
  Serial.println();
}

static void resetSmoothing() {
  gHistCount = 0;
  gHistHead = 0;
}

// Returns the smoothed argmax and writes its confidence.
static int smoothedPrediction(const float *prob, float *confOut) {
  for (int c = 0; c < SC_N_CLASSES; c++) gProbHistory[gHistHead][c] = prob[c];
  gHistHead = (uint8_t)((gHistHead + 1) % SC_SMOOTH_WINDOW);
  if (gHistCount < SC_SMOOTH_WINDOW) gHistCount++;

  float avg[SC_N_CLASSES];
  for (int c = 0; c < SC_N_CLASSES; c++) {
    float s = 0.0f;
    for (int k = 0; k < gHistCount; k++) s += gProbHistory[k][c];
    avg[c] = s / (float)gHistCount;
  }
  int best = 0;
  for (int c = 1; c < SC_N_CLASSES; c++) if (avg[c] > avg[best]) best = c;
  *confOut = avg[best];
  return best;
}

// --------------------------- calibration ------------------------------------
static void runCalibration() {
  Serial.println("STATE,CALIBRATING");
  Serial.println("INFO,Keep the cushion COMPLETELY UNLOADED until calibration finishes.");
  scCalibrationBegin();
  resetSmoothing();

  while (true) {
    scanMatrix();
    flattenFrame();
    bool enough = scCalibrationAddFrame(gFrame);
    Serial.print("CAL,");
    Serial.print(scCalFrameCount());
    Serial.print("/");
    Serial.println(SC_CAL_FRAMES);
    if (enough) break;
  }

  ScCalResult r = scCalibrationFinish();
  Serial.print("CAL,");
  Serial.println(scCalResultName(r));
  Serial.print("STATE,");
  Serial.println(scCalStateName(scCalState()));

  if (r != SC_CAL_OK) {
    Serial.println("INFO,Calibration refused. Clear the cushion and send CAL to retry.");
  } else {
    Serial.println("INFO,Calibration accepted. You can sit down now.");
  }
}

// --------------------------- serial commands --------------------------------
static void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd == "CAL") {
    runCalibration();
  } else if (cmd == "MODE RAW") {
    gMode = MODE_RAW;
    Serial.println("INFO,mode=RAW");
  } else if (cmd == "MODE PREDICT" || cmd == "MODE BOTH") {
    gMode = MODE_PREDICT;
    resetSmoothing();
    Serial.println("INFO,mode=PREDICT");
  } else if (cmd == "STATUS") {
    Serial.print("STATE,");
    Serial.println(scCalStateName(scCalState()));
    Serial.print("INFO,mode=");
    Serial.println(gMode == MODE_RAW ? "RAW" : "PREDICT");
    Serial.print("INFO,model_flash_bytes=");
    Serial.println(scModelFlashBytes());
    Serial.print("INFO,inference_ram_bytes=");
    Serial.println(scInferenceRamBytes());
    Serial.print("INFO,classes=");
    for (int c = 0; c < SC_N_CLASSES; c++) {
      Serial.print(SC_CLASS_NAMES[c]);
      if (c != SC_N_CLASSES - 1) Serial.print("|");
    }
    Serial.println();
    Serial.print("INFO,smoothing_window=");
    Serial.println(SC_SMOOTH_WINDOW);
    Serial.print("INFO,confidence_threshold=");
    Serial.println(SC_CONFIDENCE_THRESHOLD, 3);
  } else if (cmd == "HELP") {
    Serial.println("INFO,commands: MODE RAW | MODE PREDICT | CAL | STATUS | HELP");
  } else {
    Serial.print("INFO,unknown command: ");
    Serial.println(cmd);
  }
}

static void pollSerial() {
  static String buf;
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (buf.length() > 0) {
        String c = buf;
        buf = "";
        handleCommand(c);
      }
    } else if (buf.length() < 64) {
      buf += ch;
    }
  }
}

// ------------------------------- setup --------------------------------------
void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  analogSetPinAttenuation(COL_ADC_1, ADC_11db);
  analogSetPinAttenuation(COL_ADC_2, ADC_11db);

  // Address pins
  pinMode(ROW_S0, OUTPUT);
  pinMode(ROW_S1, OUTPUT);
  pinMode(ROW_S2, OUTPUT);
  pinMode(ROW_S3, OUTPUT);

  pinMode(COL_S0, OUTPUT);
  pinMode(COL_S1, OUTPUT);
  pinMode(COL_S2, OUTPUT);
  pinMode(COL_S3, OUTPUT);

  // Row MUX enable pins
  pinMode(ROW_EN_1, OUTPUT);
  pinMode(ROW_EN_2, OUTPUT);

  // Row drive
  pinMode(ROW_SIG, OUTPUT);

  // ADC inputs
  pinMode(COL_ADC_1, INPUT);
  pinMode(COL_ADC_2, INPUT);

  // Disable both row MUXes initially
  digitalWrite(ROW_EN_1, HIGH);
  digitalWrite(ROW_EN_2, HIGH);

  digitalWrite(ROW_SIG, LOW);

  Serial.println("ESP32 matrix scanner ready");
  Serial.println("INFO,smart-cushion firmware with on-device INT8 inference");
  Serial.println("INFO,send HELP for commands");

  runCalibration();
}

// ------------------------------- loop ---------------------------------------
void loop() {
  pollSerial();

  scanMatrix();          // unchanged original scan
  flattenFrame();

  printFrame();

  if (gMode == MODE_PREDICT) {
    if (scCalState() != SC_CAL_READY) {
      Serial.print("PRED,");
      Serial.print(STATE_UNKNOWN);
      Serial.print(",0.000,0");
      Serial.println();
    } else {
      uint32_t t0 = micros();
      scPreprocessFrame(gFrame, gFeatures);
      ScInferResult res;
      scInfer(gFeatures, &res);
      uint32_t dt = micros() - t0;

      float conf = 0.0f;
      int cls = smoothedPrediction(res.prob, &conf);

      Serial.print("PRED,");
      if (conf < SC_CONFIDENCE_THRESHOLD) {
        Serial.print(STATE_UNKNOWN);
      } else {
        Serial.print(SC_CLASS_NAMES[cls]);
      }
      Serial.print(",");
      Serial.print(conf, 4);
      Serial.print(",");
      Serial.println(dt);
    }
  }
}
