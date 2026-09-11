# Experiments

Code in `src/` is the library. Code here asks questions with it, and the answers
are written to files rather than to a README, so anyone can check them.

## 01 — Does biological wiring help, or is it just sparsity?

```bash
python experiments/01_inductive_bias.py --seeds 5 --epochs 3
python experiments/01_inductive_bias.py --seeds 5 --epochs 3 --download --nodes 2000 --cuda
```

The claim "a connectome is a useful inductive bias" is only testable against
wiring that matches it in every respect except being biological. This runs five
arms on MNIST at a matched parameter budget and matched seeds:

| arm | what differs from the real connectome |
|---|---|
| `real` | nothing; the reconstruction as published |
| `degree_preserving` | wiring shuffled, every neuron's in- and out-degree intact |
| `random` | same node and edge count, degrees not preserved |
| `shuffled_weights` | wiring intact, synapse counts permuted across connections |
| `dense_rnn` | not a connectome at all; an ordinary RNN at the same budget |

Each arm gets the same encoder (`784 -> io_nodes`), the same decoder
(`io_nodes -> 10`), the same optimiser and the same number of recurrent steps.
The parameter counts are printed per run so the match can be checked rather than
assumed; `dense_rnn` is allowed to be slightly *larger*, which biases against the
connectome rather than for it.

### How to read the output

`results.jsonl` has one line per run with the full config, the environment, the
connectome's fingerprint and the training curve. `results.md` has the summary
table with a standard deviation across seeds.

A difference smaller than the seed-to-seed spread is not a difference. A single
seed is not a result. If you quote a number from here, quote the spread with it.

### What this does not show

MNIST is a static image task given to a recurrent network, which is not what
either a fly or an RNN is for. A connectome arm losing to a dense RNN here says
something about this task and this budget, not about brains. The value of the
comparison is *between* the connectome arms, where only the wiring changes.
