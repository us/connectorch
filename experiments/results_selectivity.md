# Does direction selectivity emerge from the motion wiring?

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), full-span coherent sweep.

| arm | test accuracy | mean T4 DSI | selective fraction | silenced acc | trainable params |
|---|---|---|---|---|---|
| `real` | 1.0000 ± 0.0000 | 0.6878 ± 0.1830 | 1.00 | 0.4875 | 42,188 |
| `degree_preserving` | 0.5195 ± 0.0181 | 0.5542 ± 0.1821 | 1.00 | 0.5039 | 42,412 |
| `random` | 0.6156 ± 0.2157 | 0.5028 ± 0.1798 | 1.00 | 0.5039 | 42,412 |
| `shuffled_weights` | 1.0000 ± 0.0000 | 0.9459 ± 0.0829 | 1.00 | 0.4844 | 42,188 |
