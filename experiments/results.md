# Does biological wiring help?

100 nodes, 3,630 connections, fingerprint `c5a303ffe2b8ff4b`. 5 seeds, 3 epoch(s), MNIST.

| arm | test accuracy | test loss | trainable params |
|---|---|---|---|
| `real` | 0.9468 ± 0.0017 | 0.1909 | 54,520 |
| `degree_preserving` | 0.9443 ± 0.0036 | 0.2038 | 54,520 |
| `random` | 0.9392 ± 0.0020 | 0.2235 | 54,520 |
| `shuffled_weights` | 0.9460 ± 0.0045 | 0.1975 | 54,520 |
| `dense_rnn` | 0.9643 ± 0.0014 | 0.1204 | 54,986 |
