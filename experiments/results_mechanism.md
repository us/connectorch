# What in the wiring computes direction? (endpoint-free)

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), partial-span cartridge sweeps.

| arm | test accuracy | mean T4 DSI | selective fraction | silenced acc | trainable params |
|---|---|---|---|---|---|
| `real` | 0.9535 ± 0.0382 | 0.2811 ± 0.0643 | 0.75 | 0.4895 | 42,188 |
| `no_delay` | 0.9832 ± 0.0157 | 0.2513 ± 0.0729 | 0.55 | 0.4895 | 42,188 |
| `sign_shuffled` | 0.6941 ± 0.0726 | 0.0960 ± 0.1062 | 0.15 | 0.4980 | 42,188 |
| `ei_collapsed` | 0.5801 ± 0.1427 | 0.1084 ± 0.0625 | 0.15 | 0.4980 | 42,188 |
| `random` | 0.8352 ± 0.1592 | 0.1293 ± 0.1160 | 0.25 | 0.4895 | 42,412 |
