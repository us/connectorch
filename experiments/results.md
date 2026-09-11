# Does biological wiring help?

100 nodes, 3,630 connections, fingerprint `711a197c4e2e41b8`. 5 seeds, 3 epoch(s), MNIST.

| arm | test accuracy | test loss | trainable params | |w| / synapse-count prior |
|---|---|---|---|---|
| `real` | 0.9611 ± 0.0024 | 0.1320 | 54,520 | 0.00 – 2.92 |
| `biological` | 0.9576 ± 0.0018 | 0.1531 | 54,520 | 0.76 – 1.35 |
| `biological_shared` | 0.9554 ± 0.0032 | 0.1580 | 54,146 | 0.77 – 1.33 |
| `degree_preserving` | 0.9616 ± 0.0035 | 0.1298 | 54,520 | 0.02 – 2.20 |
| `random` | 0.9641 ± 0.0025 | 0.1225 | 54,520 | 0.00 – 2.38 |
| `shuffled_weights` | 0.9640 ± 0.0012 | 0.1255 | 54,520 | 0.00 – 2.47 |
| `dense_rnn` | 0.9643 ± 0.0014 | 0.1204 | 54,986 | n/a |
