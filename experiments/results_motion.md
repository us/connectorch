# Does the motion circuit help with motion?

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), moving-bar direction.

| arm | test accuracy | test loss | trainable params | |w| / synapse-count prior |
|---|---|---|---|---|
| `real` | 0.7422 ± 0.0173 | 0.6512 | 262,595 | 0.08 – 2.05 |
| `biological` | 0.6879 ± 0.0968 | 0.6723 | 262,595 | 0.84 – 1.20 |
| `biological_shared` | 0.5570 ± 0.0845 | 0.6898 | 362 | 0.86 – 1.17 |
| `degree_preserving` | 0.7383 ± 0.0070 | 0.4721 | 262,595 | 0.00 – 2.02 |
| `random` | 0.7555 ± 0.0137 | 0.4664 | 262,595 | 0.02 – 1.92 |
| `shuffled_weights` | 0.7328 ± 0.0186 | 0.6500 | 262,595 | 0.01 – 2.14 |
| `dense_rnn` | 0.7625 ± 0.0864 | 0.4869 | 262,154 | n/a |
