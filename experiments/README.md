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

## 02 — Motion direction on the real early-visual circuit

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/02_motion.py --seeds 5 --epochs 3
```

Exp 01's negative needed a follow-up on the connectome's home turf: the
early motion pathway (L1 → Mi1/Mi4/Mi9 → T4, 55k edges into T4 in
traced-only, min_synapses=5), 24 columnar types, 41,826 nodes / 262,585
edges. Input is a Gaussian bump sweeping across the 36 published optic-lobe
hex columns, injected into one L1 cell per column; each sample starts at a
random position and covers only part of the range, so direction has to come
from local motion, not endpoints. Readout is the T4a-d mean pool (the four
biological direction channels) plus a 4→2 linear layer. Same 7 arms,
5 seeds, matched budgets (262,595 vs 262,154 params).

### What it found

| arm | test accuracy | `\|w\|` / synapse-count prior |
|---|---|---|
| `dense_rnn` | 0.7625 ± 0.0864 | n/a |
| `random` | 0.7555 ± 0.0137 | 0.02 – 1.92 |
| `real` | 0.7422 ± 0.0173 | 0.08 – 2.05 |
| `degree_preserving` | 0.7383 ± 0.0070 | 0.00 – 2.02 |
| `shuffled_weights` | 0.7328 ± 0.0186 | 0.01 – 2.14 |
| `biological` | 0.6879 ± 0.0968 | 0.84 – 1.20 |
| `biological_shared` | 0.5570 ± 0.0845 | 0.86 – 1.17 |

The second honest negative: even on its home circuit and a sequential,
sensory-mapped task, the real wiring is indistinguishable from a random
graph of the same size (0.7422 vs 0.7555, spread 0.014–0.019). The dense
RNN is on top again, and the biologically constrained arms pay the
constraint price again. Unconstrained arms end with connections driven to
zero or doubled, i.e. they score while no longer being the connectome.

**We are not reporting this as evidence for connectome-constrained
networks either. Two tasks, two negatives.**

Requires downloading annotations (14 MB) + traced-only edges (508 MB);
note the sandbox-proxy 403 on GCS, use a direct connection
(`NO_PROXY=$NO_PROXY,storage.googleapis.com`).

### What this does not show

Twelve steps of a 1-D sweep is still a toy stimulus, the L1-per-column
sampling throws away most of the retina, and T5/lobula outputs are not in
the readout. The next step that could still matter: full 2-D hex layout,
naturalistic optic flow, and T4+T5 readout.

### What this does not show (exp 01)

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
