# 2-D endpoint-free flow with the correct drive (T4+T5)

41,826 nodes, 262,585 connections, fingerprint `bf9321920b61c5a0`. 5 seeds, 3 epoch(s), four-direction cartridge flow.

| arm | test accuracy | test loss | silenced acc | trainable params |
|---|---|---|---|---|
| `real` | 0.2918 ± 0.0254 | 1.3728 | 0.2520 | 42,214 |
| `no_delay` | 0.2934 ± 0.0217 | 1.3722 | 0.2520 | 42,214 |
| `sign_shuffled` | 0.2703 ± 0.0218 | 3.4652 | 0.2520 | 42,214 |
| `ei_collapsed` | 0.2590 ± 0.0232 | 199.4677 | 0.2520 | 42,214 |
| `random` | 0.2895 ± 0.0203 | 1.3813 | 0.2520 | 42,438 |
