# Phase 4 diagnostic - cross-participant domain shift

Chance level for 6 balanced classes is **16.7%**.

## How different are the participants?

| participant   |   mean_feature |   sd_feature |   mean_active_cells |   mean_peak |
|:--------------|---------------:|-------------:|--------------------:|------------:|
| P1            |         2.4403 |       1.5747 |             417.296 |     3897.96 |
| P2            |         2.8587 |       1.265  |             456.803 |     4003.69 |
| P3            |         2.8097 |       1.3422 |             454.628 |     3980.61 |
| P4            |         2.2288 |       1.5949 |             414.565 |     3989.43 |

A logistic regression trained to predict *which participant* a frame came from (on the Phase 2 features) reaches **99.5%** in-sample across 4 classes (chance 25%). Participant identity is written into the frames far more strongly than posture is.

## Leave-one-participant-out, per feature transform

| transform                   | model     |   n_features |   mean_lopo_acc |   min_lopo_acc |     P1 |     P2 |     P3 |     P4 |
|:----------------------------|:----------|-------------:|----------------:|---------------:|-------:|-------:|-------:|-------:|
| centre_of_pressure          | lda       |            6 |          0.4398 |         0.3737 | 0.4591 | 0.3737 | 0.4622 | 0.4641 |
| frame_mean_norm             | rf300     |          504 |          0.4325 |         0.3386 | 0.3386 | 0.3715 | 0.3974 | 0.6224 |
| centre_of_pressure          | logreg_C1 |            6 |          0.4266 |         0.3776 | 0.4205 | 0.431  | 0.4773 | 0.3776 |
| session_z                   | rf300     |          504 |          0.4131 |         0.0998 | 0.4636 | 0.0998 | 0.4773 | 0.6118 |
| frame_norm                  | logreg_C1 |          504 |          0.4113 |         0.2866 | 0.4955 | 0.2866 | 0.3801 | 0.4831 |
| global_scale (Phase 2 spec) | rf300     |          504 |          0.4048 |         0.2463 | 0.3205 | 0.2463 | 0.5292 | 0.5232 |
| global_scale (Phase 2 spec) | logreg_C1 |          504 |          0.3949 |         0.2829 | 0.4909 | 0.3227 | 0.2829 | 0.4831 |
| session_z                   | logreg_C1 |          504 |          0.391  |         0.2611 | 0.4886 | 0.2611 | 0.4449 | 0.3692 |
| row_col_profile             | logreg_C1 |           45 |          0.3786 |         0.3273 | 0.3273 | 0.3482 | 0.419  | 0.4198 |
| centre_of_pressure          | rf300     |            6 |          0.3735 |         0.2611 | 0.5318 | 0.2611 | 0.3866 | 0.3143 |
| row_col_profile             | rf300     |           45 |          0.3718 |         0.1805 | 0.3409 | 0.1805 | 0.4363 | 0.5295 |
| frame_mean_norm             | logreg_C1 |          504 |          0.3673 |         0.3165 | 0.4614 | 0.3631 | 0.3283 | 0.3165 |
| frame_norm                  | rf300     |          504 |          0.3416 |         0.1783 | 0.3068 | 0.1783 | 0.4363 | 0.4451 |
| global_scale (Phase 2 spec) | lda       |          504 |          0.2625 |         0.1899 | 0.3727 | 0.293  | 0.1944 | 0.1899 |
| row_col_profile             | lda       |           45 |          0.2612 |         0.2081 | 0.2932 | 0.2081 | 0.3326 | 0.211  |
| binary                      | rf300     |          504 |          0.2435 |         0.1603 | 0.3136 | 0.3057 | 0.1944 | 0.1603 |
| session_z                   | lda       |          504 |          0.2297 |         0.1941 | 0.2136 | 0.3015 | 0.2095 | 0.1941 |
| frame_mean_norm             | lda       |          504 |          0.2268 |         0.1771 | 0.2091 | 0.3248 | 0.1771 | 0.1962 |
| frame_norm                  | lda       |          504 |          0.2216 |         0.1751 | 0.2091 | 0.3206 | 0.1814 | 0.1751 |
| binary                      | lda       |          504 |          0.2148 |         0.1045 | 0.1045 | 0.2123 | 0.2829 | 0.2595 |
| binary                      | logreg_C1 |          504 |          0.2054 |         0.1045 | 0.1045 | 0.2654 | 0.2873 | 0.1646 |

## Within-participant ceiling (leave-one-repetition-out, logistic regression)

| transform                   |     P1 |     P2 |     P3 |     P4 |   mean |
|:----------------------------|-------:|-------:|-------:|-------:|-------:|
| binary                      | 0.8545 | 0.5462 | 0.6969 | 0.8682 | 0.7414 |
| centre_of_pressure          | 0.8781 | 0.8794 | 0.8584 | 0.8061 | 0.8555 |
| frame_mean_norm             | 0.8922 | 0.9117 | 0.8474 | 0.8515 | 0.8757 |
| frame_norm                  | 0.8862 | 0.9118 | 0.8411 | 0.8599 | 0.8748 |
| global_scale (Phase 2 spec) | 0.8839 | 0.9118 | 0.8433 | 0.8599 | 0.8747 |
| row_col_profile             | 0.8853 | 0.9118 | 0.871  | 0.9122 | 0.8951 |
| session_z                   | 0.8837 | 0.8545 | 0.8691 | 0.908  | 0.8788 |

## Reading this

- Best cross-participant transform here: **centre_of_pressure / lda**, mean LOPO 0.440.
- The Phase 2 specified transform (`global_scale`) reaches mean LOPO 0.405 at best.
- The gap between the within-participant column and the cross-participant column is the domain shift. It is a property of the sensor and the 4-participant sample, not of the model family: every model family tested lands in the same narrow band.
