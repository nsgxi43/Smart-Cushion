# Phase 1 - Dataset Inspection Report

Dataset root: `C:\Users\Niranjana Gopinath\Projects\Smart-Cushion\DATASET`

Recordings discovered: **24**

Participants: **P1, P2, P3, P4**

Postures per participant: **6**

## Posture folder -> class mapping

| original_folder   | canonical_class   |   label_index |
|:------------------|:------------------|--------------:|
| P1_Normal         | C0_Normal         |             0 |
| P2_Forward_Lean   | C1_Forward_Lean   |             1 |
| P3_Slouching      | C2_Slouching      |             2 |
| P4_Left_Lean      | C3_Left_Lean      |             3 |
| P5_Right_Lean     | C4_Right_Lean     |             4 |
| P6_Backward_Lean  | C5_Backward_Lean  |             5 |

## Frame counts

| participant   |   C0_Normal |   C1_Forward_Lean |   C2_Slouching |   C3_Left_Lean |   C4_Right_Lean |   C5_Backward_Lean |   TOTAL |
|:--------------|------------:|------------------:|---------------:|---------------:|----------------:|-------------------:|--------:|
| P1            |          78 |                72 |             72 |             70 |              73 |                 75 |     440 |
| P2            |          81 |                80 |             75 |             78 |              82 |                 75 |     471 |
| P3            |          83 |                77 |             74 |             75 |              76 |                 78 |     463 |
| P4            |          83 |                75 |             76 |             77 |              80 |                 83 |     474 |

Total raw frames in dataset: **1848**

## Schema differences

| participant   | raw_schema   |   recordings |
|:--------------|:-------------|-------------:|
| P1            | P1_LEGACY    |            6 |
| P2            | P2PLUS       |            6 |
| P3            | P2PLUS       |            6 |
| P4            | P2PLUS       |            6 |

| participant   | posture_folder   |   raw_n_meta_cols | raw_sensor_col_first   | raw_sensor_col_last   | has_metadata_csv   | has_npy   |
|:--------------|:-----------------|------------------:|:-----------------------|:----------------------|:-------------------|:----------|
| P1            | P1_Normal        |                 7 | sensor_001             | sensor_504            | True               | True      |
| P1            | P2_Forward_Lean  |                 7 | sensor_001             | sensor_504            | True               | True      |
| P1            | P3_Slouching     |                 7 | sensor_001             | sensor_504            | True               | True      |
| P1            | P4_Left_Lean     |                 7 | sensor_001             | sensor_504            | True               | True      |
| P1            | P5_Right_Lean    |                 7 | sensor_001             | sensor_504            | True               | True      |
| P1            | P6_Backward_Lean |                 7 | sensor_001             | sensor_504            | True               | True      |
| P2            | P1_Normal        |                 9 | sensor_000             | sensor_503            | False              | False     |
| P2            | P2_Forward_Lean  |                 9 | sensor_000             | sensor_503            | False              | False     |
| P2            | P3_Slouching     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P2            | P4_Left_Lean     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P2            | P5_Right_Lean    |                 9 | sensor_000             | sensor_503            | False              | False     |
| P2            | P6_Backward_Lean |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P1_Normal        |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P2_Forward_Lean  |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P3_Slouching     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P4_Left_Lean     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P5_Right_Lean    |                 9 | sensor_000             | sensor_503            | False              | False     |
| P3            | P6_Backward_Lean |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P1_Normal        |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P2_Forward_Lean  |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P3_Slouching     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P4_Left_Lean     |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P5_Right_Lean    |                 9 | sensor_000             | sensor_503            | False              | False     |
| P4            | P6_Backward_Lean |                 9 | sensor_000             | sensor_503            | False              | False     |

## Data integrity

| participant   | posture_folder   |   raw_frames |   raw_nan_cells |   raw_nan_meta |   raw_out_of_range_cells |   raw_non_integer_cells |   raw_duplicate_frames |   raw_timestamp_parse_failures | raw_timestamps_monotonic   | raw_frameidx_monotonic_within_rep   | raw_label_consistent   |
|:--------------|:-----------------|-------------:|----------------:|---------------:|-------------------------:|------------------------:|-----------------------:|-------------------------------:|:---------------------------|:------------------------------------|:-----------------------|
| P1            | P1_Normal        |           78 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P1            | P2_Forward_Lean  |           72 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P1            | P3_Slouching     |           72 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P1            | P4_Left_Lean     |           70 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P1            | P5_Right_Lean    |           73 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P1            | P6_Backward_Lean |           75 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P1_Normal        |           81 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P2_Forward_Lean  |           80 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P3_Slouching     |           75 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P4_Left_Lean     |           78 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P5_Right_Lean    |           82 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P2            | P6_Backward_Lean |           75 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P1_Normal        |           83 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P2_Forward_Lean  |           77 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P3_Slouching     |           74 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P4_Left_Lean     |           75 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P5_Right_Lean    |           76 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P3            | P6_Backward_Lean |           78 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P1_Normal        |           83 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P2_Forward_Lean  |           75 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P3_Slouching     |           76 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P4_Left_Lean     |           77 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P5_Right_Lean    |           80 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |
| P4            | P6_Backward_Lean |           83 |               0 |              0 |                        0 |                       0 |                      0 |                              0 | True                       | True                                | True                   |

## Saturation (percentage of all cells)

| participant   | posture_folder   |   raw_pct_at_adc_min |   raw_pct_at_adc_max |   pressure_pct_zero |   raw_min |   raw_max |   raw_mean |
|:--------------|:-----------------|---------------------:|---------------------:|--------------------:|----------:|----------:|-----------:|
| P1            | P1_Normal        |               18.284 |                8.796 |              22.713 |         0 |      4095 |    2576.36 |
| P1            | P2_Forward_Lean  |               13.156 |                0.992 |              17.59  |         0 |      4095 |    2712.51 |
| P1            | P3_Slouching     |                4.847 |                4.947 |               6.322 |         0 |      4095 |    3070.96 |
| P1            | P4_Left_Lean     |                6.431 |                0.87  |               7.985 |         0 |      4095 |    2908.5  |
| P1            | P5_Right_Lean    |               24.301 |               18.17  |              29.017 |         0 |      4095 |    2371.95 |
| P1            | P6_Backward_Lean |               15.034 |                6.14  |              18.653 |         0 |      4095 |    2060.05 |
| P2            | P1_Normal        |               14.354 |               29.365 |              22.19  |         0 |      4095 |    2879.84 |
| P2            | P2_Forward_Lean  |               12.326 |               18.867 |              10.699 |         0 |      4095 |    3118.21 |
| P2            | P3_Slouching     |                5.259 |               34.386 |               5.175 |         0 |      4095 |    3746.24 |
| P2            | P4_Left_Lean     |                4.79  |               14.591 |               4.866 |         0 |      4095 |    3264.95 |
| P2            | P5_Right_Lean    |                4.856 |               41.601 |               4.905 |         0 |      4095 |    3602.05 |
| P2            | P6_Backward_Lean |                7.667 |               25.265 |               7.833 |         0 |      4095 |    2991.66 |
| P3            | P1_Normal        |                6.87  |               12.165 |              15.569 |         0 |      4095 |    3057.81 |
| P3            | P2_Forward_Lean  |                4.056 |                2.791 |              15.78  |         0 |      4095 |    2786.9  |
| P3            | P3_Slouching     |                2.574 |               11.905 |               8.878 |         0 |      4095 |    3244.49 |
| P3            | P4_Left_Lean     |                2.204 |                8.804 |               3.563 |         0 |      4095 |    3290.59 |
| P3            | P5_Right_Lean    |                5.42  |               24.178 |               7.696 |         0 |      4095 |    3111.96 |
| P3            | P6_Backward_Lean |                2.511 |                4.271 |               6.654 |         0 |      4095 |    3083.15 |
| P4            | P1_Normal        |               14.24  |                5.943 |              20.678 |         0 |      4095 |    2725.69 |
| P4            | P2_Forward_Lean  |               12.667 |                3.693 |              16.526 |         0 |      4095 |    2417.73 |
| P4            | P3_Slouching     |                5.446 |                9.234 |               6.837 |         0 |      4095 |    3186.41 |
| P4            | P4_Left_Lean     |               13.348 |                1.041 |              15.783 |         0 |      4095 |    2026.83 |
| P4            | P5_Right_Lean    |               21.89  |                2.949 |              25.246 |         0 |      4095 |    1982.82 |
| P4            | P6_Backward_Lean |               19.282 |                6.127 |              20.491 |         0 |      4095 |    2346.85 |

## Repetition structure

| participant   | posture_folder   |   raw_n_reps | raw_rep_values   |   raw_frames_per_rep_min |   raw_frames_per_rep_max |   raw_rep_duration_mean_s |   raw_fps_mean |
|:--------------|:-----------------|-------------:|:-----------------|-------------------------:|-------------------------:|--------------------------:|---------------:|
| P1            | P1_Normal        |            5 | 1,2,3,4,5        |                       12 |                       21 |                     10.24 |           1.43 |
| P1            | P2_Forward_Lean  |            5 | 1,2,3,4,5        |                       13 |                       16 |                     10.71 |           1.25 |
| P1            | P3_Slouching     |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.17 |           1.32 |
| P1            | P4_Left_Lean     |            5 | 1,2,3,4,5        |                       10 |                       15 |                     10.43 |           1.25 |
| P1            | P5_Right_Lean    |            5 | 1,2,3,4,5        |                       11 |                       21 |                     10.5  |           1.29 |
| P1            | P6_Backward_Lean |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.04 |           1.4  |
| P2            | P1_Normal        |            5 | 1,2,3,4,5        |                       15 |                       19 |                     10.35 |           1.47 |
| P2            | P2_Forward_Lean  |            5 | 1,2,3,4,5        |                       15 |                       17 |                     10.48 |           1.43 |
| P2            | P3_Slouching     |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.42 |           1.34 |
| P2            | P4_Left_Lean     |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.4  |           1.4  |
| P2            | P5_Right_Lean    |            5 | 1,2,3,4,5        |                       16 |                       17 |                     10.73 |           1.43 |
| P2            | P6_Backward_Lean |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.44 |           1.34 |
| P3            | P1_Normal        |            5 | 1,2,3,4,5        |                       16 |                       19 |                     10.74 |           1.45 |
| P3            | P2_Forward_Lean  |            5 | 1,2,3,4,5        |                       15 |                       16 |                     10.58 |           1.36 |
| P3            | P3_Slouching     |            5 | 1,2,3,4,5        |                       14 |                       15 |                     10.57 |           1.31 |
| P3            | P4_Left_Lean     |            5 | 1,2,3,4,5        |                       14 |                       16 |                     10.55 |           1.33 |
| P3            | P5_Right_Lean    |            5 | 1,2,3,4,5        |                       13 |                       16 |                     10.21 |           1.39 |
| P3            | P6_Backward_Lean |            5 | 1,2,3,4,5        |                       14 |                       17 |                     10.55 |           1.38 |
| P4            | P1_Normal        |            5 | 1,2,3,4,5        |                       15 |                       20 |                     10.37 |           1.51 |
| P4            | P2_Forward_Lean  |            5 | 1,2,3,4,5        |                       14 |                       17 |                     10.55 |           1.33 |
| P4            | P3_Slouching     |            5 | 1,2,3,4,5        |                       15 |                       16 |                     10.24 |           1.39 |
| P4            | P4_Left_Lean     |            5 | 1,2,3,4,5        |                       15 |                       16 |                     10.36 |           1.39 |
| P4            | P5_Right_Lean    |            5 | 1,2,3,4,5        |                       16 |                       16 |                     10.72 |           1.4  |
| P4            | P6_Backward_Lean |            5 | 1,2,3,4,5        |                       15 |                       17 |                     10.34 |           1.51 |

## Dead / stuck sensors per participant

| participant   |   frames_total |   dead_sensors_std0 | dead_sensor_indices                                                                         |   near_dead_std_lt_1 |   stuck_low_mean_le_1 |   stuck_high_mean_ge_4094 |   raw_global_min |   raw_global_max |   raw_global_mean |   pct_at_adc_max |   pct_at_adc_min |
|:--------------|---------------:|--------------------:|:--------------------------------------------------------------------------------------------|---------------------:|----------------------:|--------------------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|
| P1            |            440 |                   2 | 20,83                                                                                       |                    2 |                     4 |                         0 |                0 |             4095 |           2610.49 |            6.731 |           13.805 |
| P2            |            471 |                  23 | 20,41,62,83,104,125,146,167,188,209,230,251,272,293,314,335,356,377,398,419,440,461,503     |                   23 |                    24 |                         0 |                0 |             4095 |           3265.61 |           27.412 |            8.259 |
| P3            |            463 |                   0 |                                                                                             |                    0 |                     0 |                         0 |                0 |             4095 |           3093.46 |           10.662 |            3.987 |
| P4            |            474 |                  24 | 20,41,62,83,104,125,146,167,188,209,230,251,272,293,314,335,356,377,398,419,440,461,482,503 |                   24 |                    24 |                         0 |                0 |             4095 |           2445.59 |            4.845 |           14.61  |

## Sensor indexing consistency

Pearson correlation of the mean *raw* 504-vector for posture C0_Normal between participants. High values indicate the same physical sensor order.

| a   |     P1 |     P2 |     P3 |     P4 |
|:----|-------:|-------:|-------:|-------:|
| P1  | 1      | 0.9624 | 0.9194 | 0.9862 |
| P2  | 0.9624 | 1      | 0.9786 | 0.9851 |
| P3  | 0.9194 | 0.9786 | 1      | 0.9437 |
| P4  | 0.9862 | 0.9851 | 0.9437 | 1      |

## Figures

- `heatmaps_mean_pressure.png`
- `heatmaps_mean_raw.png`
- `per_sensor_raw_std.png`
- `value_distributions.png`
