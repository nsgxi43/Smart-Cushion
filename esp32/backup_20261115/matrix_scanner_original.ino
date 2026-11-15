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
}

void loop() {

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

  // Send exactly 504 comma-separated readings
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {

      Serial.print(readings[r][c]);

      if (!(r == ROWS - 1 && c == COLS - 1)) {
        Serial.print(",");
      }
    }
  }

  Serial.println();
}
