# Firmware scan-preservation check

Original: `esp32\backup_20261115\matrix_scanner_original.ino`  
New: `esp32\smart_cushion\smart_cushion.ino`

**Overall: PASS**

## Pin and timing constants

| constant | original | new | required | match |
|---|---|---|---|---|
| ROWS | 24 | 24 | 24 | True |
| COLS | 21 | 21 | 21 | True |
| ROW_S0 | 25 | 25 | 25 | True |
| ROW_S1 | 26 | 26 | 26 | True |
| ROW_S2 | 27 | 27 | 27 | True |
| ROW_S3 | 14 | 14 | 14 | True |
| COL_S0 | 16 | 16 | 16 | True |
| COL_S1 | 17 | 17 | 17 | True |
| COL_S2 | 18 | 18 | 18 | True |
| COL_S3 | 19 | 19 | 19 | True |
| ROW_EN_1 | 21 | 21 | 21 | True |
| ROW_EN_2 | 22 | 22 | 22 | True |
| ROW_SIG | 13 | 13 | 13 | True |
| COL_ADC_1 | 34 | 34 | 34 | True |
| COL_ADC_2 | 35 | 35 | 35 | True |
| SAMPLES | 16 | 16 | 16 | True |
| SETTLE_US | 300 | 300 | 300 | True |

## ADC / serial configuration

| call | original | new | match |
|---|---|---|---|
| `analogReadResolution(12)` | True | True | True |
| `analogSetPinAttenuation(COL_ADC_1,ADC_11db)` | True | True | True |
| `analogSetPinAttenuation(COL_ADC_2,ADC_11db)` | True | True | True |
| `Serial.begin(115200)` | True | True | True |

## Scan helper functions

| function | byte-identical after comment/whitespace normalisation |
|---|---|
| `setAddress()` | True |
| `readAverage()` | True |

## Scan operation sequence

Ordered list of `digitalWrite` / `setAddress` / `delayMicroseconds` / `readAverage` operations inside the scan.

| | function | operations |
|---|---|---|
| original | `loop()` | 18 |
| new | `scanMatrix()` | 18 |

Identical: **True**

## pinMode calls

Identical set and order: **True**

```
COL_ADC_1,INPUT
COL_ADC_2,INPUT
COL_S0,OUTPUT
COL_S1,OUTPUT
COL_S2,OUTPUT
COL_S3,OUTPUT
ROW_EN_1,OUTPUT
ROW_EN_2,OUTPUT
ROW_S0,OUTPUT
ROW_S1,OUTPUT
ROW_S2,OUTPUT
ROW_S3,OUTPUT
ROW_SIG,OUTPUT
```

## What changed, and why

See `esp32/FIRMWARE_CHANGES.md` for the full annotated diff. In summary, the original `loop()` body was moved verbatim into `scanMatrix()` so that inference can run *after* a complete 504-value frame has been acquired, without a single statement of the scan being reordered, retimed or removed.
