# Does the motion wiring survive distribution shift?

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), train clean full-span, test shifted.

| arm | clean | contrast_low | contrast_high | slow | fast | noise | occlusion | partial |
|---|---|---|---|---|---|---|---|---|
| `real` | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.899 ± 0.225 | 0.499 ± 0.027 | 0.610 ± 0.220 | 0.527 ± 0.032 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `degree_preserving` | 0.504 ± 0.031 | 0.498 ± 0.014 | 0.515 ± 0.032 | 0.518 ± 0.018 | 0.506 ± 0.355 | 0.501 ± 0.021 | 0.512 ± 0.032 | 0.511 ± 0.030 |
| `random` | 0.510 ± 0.030 | 0.501 ± 0.014 | 0.521 ± 0.027 | 0.505 ± 0.027 | 0.604 ± 0.224 | 0.510 ± 0.018 | 0.502 ± 0.035 | 0.512 ± 0.029 |
| `shuffled_weights` | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.899 ± 0.225 | 0.798 ± 0.278 | 0.601 ± 0.226 | 0.588 ± 0.212 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Accuracy drops vs clean (mean over seeds):

| arm | contrast_low | contrast_high | slow | fast | noise | occlusion | partial |
|---|---|---|---|---|---|---|---|
| `real` | +0.000 | +0.101 | +0.501 | +0.390 | +0.473 | +0.000 | +0.000 |
| `degree_preserving` | +0.006 | -0.011 | -0.014 | -0.002 | +0.003 | -0.009 | -0.007 |
| `random` | +0.009 | -0.011 | +0.005 | -0.094 | -0.000 | +0.009 | -0.002 |
| `shuffled_weights` | +0.000 | +0.101 | +0.202 | +0.399 | +0.412 | +0.000 | +0.000 |
