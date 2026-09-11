# Performance

Every number on this page came out of `benchmarks/`. Nothing here is an estimate.

## The whole fly, on one GPU

`examples/03_malecns_load.py --full --cuda` on an NVIDIA GB10 (torch 2.14.0+cu130):

| | |
|---|---|
| neurons | 164,587 |
| connections | 25,563,197 |
| synapses | 124,025,046 |
| load and compile from the published feather file | 16.3 s, 3.0 GB RAM |
| compile to GPU | 1.1 s |
| trainable parameters | 25,563,197 |
| training step, batch 4, 8 recurrent steps | 995 ms |
| peak VRAM | 8.02 GiB |
| a dense float32 adjacency of the same graph | 100.9 GiB |

## Backends

N=50,000, 500k edges, batch 32, same machine. Dense float32 `[N, N]` is 9.3 GiB:

| backend | forward | forward+backward | peak (f+b) |
|---|---|---|---|
| `sparse_mm` (CSR) | **0.16 ms** | 140.4 ms | 11.74 GiB |
| COO | 0.49 ms | 74.6 ms | 9.40 GiB |
| `scatter` | 1.76 ms | **3.3 ms** | **0.24 GiB** |

Across sizes, CUDA, forward+backward:

| N | edges | `sparse_mm` | `scatter` |
|---|---|---|---|
| 1,000 | 10k | 1.07 ms / 0.068 GiB | 0.84 ms / 0.063 GiB |
| 10,000 | 100k | 6.28 ms / 0.539 GiB | 0.46 ms / 0.066 GiB |
| 100,000 | 1M | refused (37.3 GiB) | 0.68 ms / 0.097 GiB |

CPU tells the same story: at N=10,000 forward+backward is 52.8 ms through
`sparse_mm` and 1.7 ms through `scatter`.

## Reproducing

```bash
python benchmarks/sparse_backends.py --out results.jsonl
```

Writes one JSON line per configuration, including the torch version, platform and
device, so a number can always be traced to the machine that produced it.
`benchmarks/gb10-results.jsonl` is the run quoted here.

## What has not been optimised

No custom CUDA or Triton kernels, by design. The first job was correct graph
semantics, correct gradients, a stable API and sparse memory behaviour. The
gather-scatter step is memory-bandwidth bound and there is clearly headroom in it;
profiling comes before writing a kernel, not after.

`torch.compile` has not been validated and is not claimed to work.
