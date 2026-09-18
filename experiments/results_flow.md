# Does the motion circuit help with 2-D flow?

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), four-direction flow.

| arm | test accuracy | test loss | trainable params | |w| / synapse-count prior |
|---|---|---|---|---|
| `real` | 0.2680 ± 0.0152 | 1.3829 | 262,621 | 0.52 – 1.49 |
| `biological` | 0.2602 ± 0.0165 | 1.3846 | 262,621 | 0.92 – 1.08 |
| `biological_shared` | 0.2605 ± 0.0199 | 1.3851 | 388 | 0.91 – 1.06 |
| `degree_preserving` | 0.3012 ± 0.0116 | 1.3479 | 262,621 | 0.29 – 1.69 |
| `random` | 0.3090 ± 0.0129 | 1.3448 | 262,621 | 0.41 – 1.83 |
| `shuffled_weights` | 0.2770 ± 0.0099 | 1.3821 | 262,621 | 0.50 – 1.52 |
| `dense_rnn` | 0.3746 ± 0.0279 | 1.3009 | 262,161 | n/a |
