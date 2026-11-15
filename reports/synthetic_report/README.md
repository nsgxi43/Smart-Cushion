# Phase 3 - Synthetic participant report

> **These are not real people.** P5_SYNTHETIC..P10_SYNTHETIC are transformed copies of real recordings. Height, weight and sex are recorded as `SYNTHETIC/NA`. They model sensor and session variation, not new body types.

## Source mapping

| synthetic | derived from | global intensity | noise sigma | shift (dr,dc) | temporal jitter |
|---|---|---|---|---|---|
| P5_SYNTHETIC | P1 | 1.026 | 6.16 | (1,1) | 0.22 |
| P6_SYNTHETIC | P2 | 0.854 | 4.33 | (-1,0) | 0.24 |
| P7_SYNTHETIC | P3 | 1.055 | 7.03 | (-1,1) | 0.36 |
| P8_SYNTHETIC | P4 | 1.022 | 8.12 | (0,-1) | 0.26 |
| P9_SYNTHETIC | P1 | 1.012 | 3.58 | (1,1) | 0.42 |
| P10_SYNTHETIC | P2 | 1.089 | 8.46 | (0,1) | 0.33 |

## Frame counts

| participant   |   C0_Normal |   C1_Forward_Lean |   C2_Slouching |   C3_Left_Lean |   C4_Right_Lean |   C5_Backward_Lean |   TOTAL |
|:--------------|------------:|------------------:|---------------:|---------------:|----------------:|-------------------:|--------:|
| P10_SYNTHETIC |          81 |                80 |             75 |             78 |              82 |                 75 |     471 |
| P5_SYNTHETIC  |          78 |                72 |             72 |             70 |              73 |                 75 |     440 |
| P6_SYNTHETIC  |          81 |                80 |             75 |             78 |              82 |                 75 |     471 |
| P7_SYNTHETIC  |          83 |                77 |             74 |             75 |              76 |                 78 |     463 |
| P8_SYNTHETIC  |          83 |                75 |             76 |             77 |              80 |                 83 |     474 |
| P9_SYNTHETIC  |          78 |                72 |             72 |             70 |              73 |                 75 |     440 |

**Real frames: 1848. Synthetic frames: 2759. Ratio 1.49:1.**

## Provenance and internal consistency

| participant   | posture_class    |   frames |   provenance_rows | provenance_complete   |   pressure_identity_exact_pct | raw_in_range   |
|:--------------|:-----------------|---------:|------------------:|:----------------------|------------------------------:|:---------------|
| P10_SYNTHETIC | C0_Normal        |       81 |                81 | True                  |                           100 | True           |
| P10_SYNTHETIC | C1_Forward_Lean  |       80 |                80 | True                  |                           100 | True           |
| P10_SYNTHETIC | C2_Slouching     |       75 |                75 | True                  |                           100 | True           |
| P10_SYNTHETIC | C3_Left_Lean     |       78 |                78 | True                  |                           100 | True           |
| P10_SYNTHETIC | C4_Right_Lean    |       82 |                82 | True                  |                           100 | True           |
| P10_SYNTHETIC | C5_Backward_Lean |       75 |                75 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C0_Normal        |       78 |                78 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C1_Forward_Lean  |       72 |                72 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C2_Slouching     |       72 |                72 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C3_Left_Lean     |       70 |                70 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C4_Right_Lean    |       73 |                73 | True                  |                           100 | True           |
| P5_SYNTHETIC  | C5_Backward_Lean |       75 |                75 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C0_Normal        |       81 |                81 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C1_Forward_Lean  |       80 |                80 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C2_Slouching     |       75 |                75 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C3_Left_Lean     |       78 |                78 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C4_Right_Lean    |       82 |                82 | True                  |                           100 | True           |
| P6_SYNTHETIC  | C5_Backward_Lean |       75 |                75 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C0_Normal        |       83 |                83 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C1_Forward_Lean  |       77 |                77 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C2_Slouching     |       74 |                74 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C3_Left_Lean     |       75 |                75 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C4_Right_Lean    |       76 |                76 | True                  |                           100 | True           |
| P7_SYNTHETIC  | C5_Backward_Lean |       78 |                78 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C0_Normal        |       83 |                83 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C1_Forward_Lean  |       75 |                75 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C2_Slouching     |       76 |                76 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C3_Left_Lean     |       77 |                77 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C4_Right_Lean    |       80 |                80 | True                  |                           100 | True           |
| P8_SYNTHETIC  | C5_Backward_Lean |       83 |                83 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C0_Normal        |       78 |                78 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C1_Forward_Lean  |       72 |                72 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C2_Slouching     |       72 |                72 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C3_Left_Lean     |       70 |                70 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C4_Right_Lean    |       73 |                73 | True                  |                           100 | True           |
| P9_SYNTHETIC  | C5_Backward_Lean |       75 |                75 | True                  |                           100 | True           |

Every synthetic recording satisfies the Phase 2 identity `pressure = clip(gate(|raw - baseline|, threshold), 0, 4095)` exactly (100.00% of cells, min 100.00%).

## Posture structure preserved

Within-participant nearest-class-centroid accuracy. If the transforms had destroyed spatial structure, the synthetic rows would collapse toward chance (1/6 = 16.7%).

| participant   | kind      |   within_participant_nearest_centroid_acc |
|:--------------|:----------|------------------------------------------:|
| P1            | real      |                                    0.7273 |
| P2            | real      |                                    0.7686 |
| P3            | real      |                                    0.851  |
| P4            | real      |                                    0.7595 |
| P5_SYNTHETIC  | synthetic |                                    0.7273 |
| P6_SYNTHETIC  | synthetic |                                    0.7665 |
| P7_SYNTHETIC  | synthetic |                                    0.8423 |
| P8_SYNTHETIC  | synthetic |                                    0.7637 |
| P9_SYNTHETIC  | synthetic |                                    0.7227 |
| P10_SYNTHETIC | synthetic |                                    0.7665 |

### Centroid shift vs class gap

`ratio_shift_vs_class_gap` is the distance from the synthetic centroid to its source centroid, divided by the mean distance between different postures of the source. Values well below 1 mean the posture identity survived the transform.

| synthetic     |   mean |    min |    max |
|:--------------|-------:|-------:|-------:|
| P10_SYNTHETIC | 0.6449 | 0.515  | 0.7621 |
| P5_SYNTHETIC  | 0.6871 | 0.4224 | 0.8058 |
| P6_SYNTHETIC  | 0.5517 | 0.4752 | 0.6022 |
| P7_SYNTHETIC  | 0.5815 | 0.3475 | 0.7459 |
| P8_SYNTHETIC  | 0.4961 | 0.4672 | 0.5193 |
| P9_SYNTHETIC  | 0.6871 | 0.431  | 0.7938 |

Centroid cosine similarity to source: mean 0.9765, min 0.9373.

## Synthetic participants are not copies

| synthetic     | source   |   identical_frames |   frames |   identical_pct |
|:--------------|:---------|-------------------:|---------:|----------------:|
| P5_SYNTHETIC  | P1       |                  0 |      440 |               0 |
| P6_SYNTHETIC  | P2       |                  0 |      471 |               0 |
| P7_SYNTHETIC  | P3       |                  0 |      463 |               0 |
| P8_SYNTHETIC  | P4       |                  0 |      474 |               0 |
| P9_SYNTHETIC  | P1       |                  0 |      440 |               0 |
| P10_SYNTHETIC | P2       |                  0 |      471 |               0 |

## Saturation, real vs synthetic

| set       |   pct_at_adc_min |   pct_at_adc_max |   raw_mean |
|:----------|-----------------:|-----------------:|-----------:|
| real      |           10.076 |           12.379 |    2856.86 |
| synthetic |            4.213 |           28.254 |    2880.12 |

## Figures

- `before_after_P5_SYNTHETIC.png`
- `before_after_P6_SYNTHETIC.png`
- `before_after_P7_SYNTHETIC.png`
- `before_after_P8_SYNTHETIC.png`
- `before_after_P9_SYNTHETIC.png`
- `before_after_P10_SYNTHETIC.png`
- `real_vs_synthetic_distributions.png`
