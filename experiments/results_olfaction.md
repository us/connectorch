# Does the mushroom-body head help classification?

MNIST, 5 seeds, few-shot (256 samples, 20 epochs) and full (60,000 samples, 3 epochs). Sparse expansion is frozen; only the readout trains.

| arm | regime | test accuracy | trainable params |
|---|---|---|---|
| `fly` | few-shot (256) | 0.7830 ± 0.0165 | 20,010 |
| `dense_expansion` | few-shot (256) | 0.7381 ± 0.0145 | 20,010 |
| `mlp` | few-shot (256) | 0.6625 ± 0.0204 | 19,885 |
| `fly` | full | 0.9482 ± 0.0008 | 20,010 |
| `dense_expansion` | full | 0.9426 ± 0.0018 | 20,010 |
| `mlp` | full | 0.9283 ± 0.0013 | 19,885 |
