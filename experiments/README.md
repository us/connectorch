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

### What it found, on MNIST with the bundled 100-neuron sample

5 seeds, 3 epochs, matched parameter budget:

| arm | test accuracy | `\|w\|` / synapse-count prior |
|---|---|---|
| `dense_rnn` | 0.9643 ± 0.0014 | n/a |
| `random` | 0.9641 ± 0.0025 | 0.00 – 2.38 |
| `shuffled_weights` | 0.9640 ± 0.0012 | 0.00 – 2.47 |
| `degree_preserving` | 0.9616 ± 0.0035 | 0.02 – 2.20 |
| `real` | 0.9611 ± 0.0024 | 0.00 – 2.92 |
| `biological` | 0.9576 ± 0.0018 | 0.76 – 1.35 |
| `biological_shared` | 0.9554 ± 0.0032 | 0.77 – 1.33 |

**The real connectome is not better than a random graph of the same size here.**
It is 0.003 behind, against a seed-to-seed spread of 0.002 to 0.0035, which is to
say: indistinguishable. Degree-preserving rewiring and weight shuffling are
likewise indistinguishable from it. A dense RNN at the same budget is at the top,
by a margin that is itself small.

The biologically constrained arms are about 0.006 behind the free ones. That is
the price of the constraint, and it is the expected direction: fewer degrees of
freedom, slightly worse fit. Their gain never approached its 0.5–2.0 bound
(0.76–1.35 observed), so the bound was not what cost them the accuracy; holding
polarity fixed and sharing gains across cell types was.

Read the right-hand column alongside the accuracies. Every unconstrained arm ends
with some connections driven to zero and others near tripled, including
connections the dataset measured at hundreds of synapses. Those arms score well
*while no longer being the connectome that was loaded*.

**We are not reporting this as evidence for connectome-constrained networks. It
is evidence against, on this task.**

### What this does not show

MNIST is a static image task given to a recurrent network, which is not what
either a fly or an RNN is for. A hundred neurons drawn from across the CNS are
not a circuit that does anything in particular, the inputs are wired to whichever
neurons came first in body-id order rather than to a sensory map, and a "step"
corresponds to no amount of time. Under those conditions the honest expectation
*is* that topology barely matters, and that is what came out.

So the result rules something in as much as it rules something out: whatever
makes a connectome worth using, this setup does not expose it. The next
experiment should change the task before it changes anything else — a sequence
problem, on a circuit selected because it does something known, with inputs
mapped to the neurons that actually carry them. FlyVis got its result on motion
estimation in the visual system, not on digits.
