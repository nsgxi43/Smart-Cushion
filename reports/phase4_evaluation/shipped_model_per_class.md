# Shipped architecture - per-class leave-one-participant-out

Architecture: `504-32-16-6`, column 20 masked=True, trained on REAL_ONLY.

Pooled over the 4 held-out folds. Chance = 16.7%.

| class            |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| C0_Normal        |      0.4165 |   0.44   | 0.3608 |       325 |
| C1_Forward_Lean  |      0.1173 |   0.1316 | 0.1188 |       304 |
| C2_Slouching     |      0.2469 |   0.3939 | 0.3    |       297 |
| C3_Left_Lean     |      0.5676 |   0.7    | 0.6162 |       300 |
| C4_Right_Lean    |      0.6771 |   0.6431 | 0.6588 |       311 |
| C5_Backward_Lean |      0.1475 |   0.2058 | 0.1711 |       311 |

## Per fold

| held_out   |   accuracy |   macro_f1 |   train_frames |   test_frames |
|:-----------|-----------:|-----------:|---------------:|--------------:|
| P1         |     0.4136 |     0.3832 |           1408 |           440 |
| P2         |     0.3291 |     0.1957 |           1377 |           471 |
| P3         |     0.3952 |     0.3785 |           1385 |           463 |
| P4         |     0.5359 |     0.53   |           1374 |           474 |

## Confusion matrix (rows = true, columns = predicted)

|                  |   C0_Normal |   C1_Forward_Lean |   C2_Slouching |   C3_Left_Lean |   C4_Right_Lean |   C5_Backward_Lean |
|:-----------------|------------:|------------------:|---------------:|---------------:|----------------:|-------------------:|
| C0_Normal        |         143 |                 1 |            181 |              0 |               0 |                  0 |
| C1_Forward_Lean  |          60 |                40 |             42 |              3 |               0 |                159 |
| C2_Slouching     |         129 |                13 |            117 |              2 |               0 |                 36 |
| C3_Left_Lean     |          16 |                33 |             11 |            210 |               1 |                 29 |
| C4_Right_Lean    |          19 |                 0 |             16 |             75 |             200 |                  1 |
| C5_Backward_Lean |          67 |               141 |              6 |             16 |              17 |                 64 |

**Overall accuracy 0.4188, macro F1 0.4232.**
