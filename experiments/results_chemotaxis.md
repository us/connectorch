# Does the worm circuit compute chemotaxis?

411 nodes, 3534 connections. 5 seeds, 5 epoch(s), start-matched ramp direction.

| arm | test accuracy | test loss | silenced acc | trainable params |
|---|---|---|---|---|
| `real` | 0.9059 ± 0.0106 | 0.2290 | 0.5090 | 80 |
| `degree_preserving` | 0.7445 ± 0.0277 | 0.4775 | 0.5090 | 80 |
| `random` | 0.7582 ± 0.0200 | 0.4584 | 0.5090 | 80 |
| `shuffled_weights` | 0.8945 ± 0.0162 | 0.2565 | 0.5090 | 80 |
