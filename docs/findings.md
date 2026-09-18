# When does a real connectome help? A motion-circuit case study with honest negatives

Status: **draft, 2026-09-18.** This is the writeup of the experiment program in
`experiments/` (experiments 01 through 07). Every number below is copied from
`experiments/README.md`, which is generated from the per-run JSONL files. If a
number here disagrees with the JSONL, the JSONL wins.

## TL;DR

On mismatched tasks (MNIST, arbitrary mappings, free per-edge weights), the real
fly wiring is indistinguishable from random graphs of the same size. On its home
circuit with the stimulus entering through the channels that actually drive it,
the same wiring solves coherent motion perfectly (1.00 vs ~0.52 rewired) with
~42k cell-type-shared parameters. E/I identity is causal, delays are not (in this
task), topology matters beyond signs, and 2-D flow at matched budget is a wall.
Separately, the mushroom-body motif wins twice: 10x retrieval at matched
compute (exp 08) and few-shot classification 0.78 vs 0.66 MLP (exp 09).
The pattern across all runs: **correct channel in, frozen wiring, biology does
the work; anything else, randomness does as well.**

## 1. Question

Is a measured biological wiring diagram a useful inductive bias for a trainable
network, and if so, under which conditions? The claim is only testable against
controls matched in every respect except being biological (same nodes, same edge
count, same parameter budget, same seeds), which is the discipline every
experiment below follows. See [Scientific caveats](caveats.md).

## 2. Setup (shared by the motion experiments)

* **Circuit:** MaleCNS v1.0 traced-only, `min_synapses=5`, 24 columnar
  early-visual types (lamina, medulla, transmedullary, T4/T5):
  **41,826 nodes / 262,585 edges**, fingerprint `bf9321920b61c5a0`.
  Requires downloading annotations (14 MB) + edges (508 MB).
* **Signs:** `infer_signs` with `DROSOPHILA_POLARITY` (glutamate inhibitory in
  the fly, acetylcholine excitatory; modulators unknown). Recorded in provenance.
* **FlyVis recipe (exps 04-07):** fixed wiring, count-proportional prior, Dale
  signs frozen, per-type-pair gains only (`BiologicalWeights`,
  `share_by="cell_type"`, ~42k params), per-type leak, heterogeneous synaptic
  delays, graded `threshold_linear` output with learnable per-neuron thresholds.
* **Controls:** degree-preserving rewiring, random topology at matched size,
  synapse-count shuffle, sign shuffle, E/I collapse (all +1), delay ablation,
  T4/T4+T5 silencing at readout. Every control records that it is not
  biological data.
* **Discipline:** 5 seeds, matched optimiser/lr/budget, fingerprint + environment
  logged per run, JSONL + summary table per experiment.

## 3. The key biological fact this program turned on

L1 is glutamatergic and inhibitory in *Drosophila*: all 1,818 L1→Mi1 edges are
sign −1. Driving L1 positive can only silence the ON pathway. Verified: with an
L1 drive the network is exactly silent past the lamina
(`frac_nonzero 0.0000`). Exps 02 and 03 drove L1. The excitatory lamina outputs
are L5/L3→Mi1 (ON) and L2/L4→Tm1/Tm2/Tm4, L3→Tm9 (OFF), all cholinergic. Exps
04-07 drive the cartridge (L2/L3/L5 per hex column) instead. Mean T4 DSI on the
same circuit went 0.00 → 0.80 on that change alone.

## 4. Results

### 4.1 Mismatched tasks: the wiring does not matter (exps 01, 02)

| exp | task | `real` | `random` | verdict |
|---|---|---|---|---|
| 01 | MNIST, 100 random neurons, free weights | 0.9611 ± 0.0024 | 0.9641 ± 0.0025 | negative |
| 02 | 1-D sweep, L1 drive (silent past lamina) | 0.7422 ± 0.0173 | 0.7555 ± 0.0137 | negative |

Differences are inside seed spread. Biologically constrained arms pay a small
constraint price (01: ~0.006; 02: `biological` 0.688, `biological_shared`
0.557). Unconstrained arms end with weights driven to zero or doubled: they
score while no longer being the connectome that was loaded.

### 4.2 First positive: selectivity emerges (exp 04)

Full-span coherent sweep through the cartridge drive, T4 readout:

| arm | accuracy | mean T4 DSI | silenced |
|---|---|---|---|
| `real` | 1.0000 ± 0.0000 | 0.69 ± 0.18 | 0.49 |
| `shuffled_weights` | 1.0000 ± 0.0000 | 0.95 ± 0.08 | 0.48 |
| `random` | 0.6156 ± 0.2157 | 0.50 ± 0.18 | 0.50 |
| `degree_preserving` | 0.5195 ± 0.0181 | 0.55 ± 0.18 | 0.50 |

One random seed hit 1.00, reported as is. Count magnitudes do not matter
(shuffled ≥ real): the computation is in which cell type connects to which,
with what sign. Silencing T4 collapses all arms to chance. Caveat: full-span
endpoints predict direction, shared by all arms, so the ranking stands but the
confound is real. Exp 06 removes it.

### 4.3 Mechanism, endpoint-free (exp 06)

Partial-span cartridge sweeps (random start, path strictly inside the range):

| arm | accuracy | mean T4 DSI | silenced |
|---|---|---|---|
| `real` | 0.9535 ± 0.0382 | 0.28 ± 0.06 | 0.49 |
| `no_delay` | 0.9832 ± 0.0157 | 0.25 ± 0.07 | 0.49 |
| `random` | 0.8352 ± 0.1592 | 0.13 ± 0.12 | 0.49 |
| `sign_shuffled` | 0.6941 ± 0.0726 | 0.10 ± 0.11 | 0.50 |
| `ei_collapsed` | 0.5801 ± 0.1427 | 0.11 ± 0.06 | 0.50 |

Three causal claims: (1) E/I identity is causal (shuffle 0.95→0.69, collapse
to 0.58). (2) Our delay priors are NOT causal (removal changes nothing, 0.98
vs 0.95). The coincidence hypothesis for those priors is dead. (3) Topology
matters beyond signs (random reaches 0.84 but stays below intact wiring on
every seed).

### 4.4 Distribution shift: no robustness win (exp 05)

Trained clean (exp 04 recipe), tested under seven shifts. Intact wiring holds
contrast, occlusion and partial sweeps (1.00) but collapses on slow (0.50),
fast (0.61) and noise (0.53) shifts. Rewired arms show ~zero drop only because they never left
the floor (~0.51), a floor artifact, not robustness. Honest comparison is real
vs shuffled (both solve clean): shuffled drops less on slow sweeps (0.20 vs
0.50). No NCP-style robustness win.

### 4.5 Bound: 2-D flow at matched budget is a wall (exps 03, 07)

| exp | drive | `real` (chance) |
|---|---|---|
| 03 | L1, 4-direction 2-D flow, T4+T5 | 0.268 (0.25), floor |
| 07 | cartridge, endpoint-free 4-direction flow, T4+T5 | 0.292 (0.25), floor, all arms |

Not dead plumbing: a 4x-budget 2-way probe on `real` reaches 0.68 (chance
0.50). 2-D position invariance (same direction at any of 892 positions) needs
far more samples than 1-D at matched protocol. Input energy ruled out as the
cause: a 2-way 2-D probe at input gain x1 vs x8 scores 0.49/0.46 (both
chance), so the floor is not whisper-quiet drive. Secondary signal: removing
inhibition destabilizes dynamics (`ei_collapsed` loss 199 vs 1.37), so E/I
balance stabilizes this recurrent network even while it learns nothing here.

### 4.6 Second positive: sparse expansion wins at matched compute (exp 08)

| arm | mean top-10 overlap | ops/query |
|---|---|---|
| `fly` (`SparseExpander`) | 4.10 ± 0.11 | 12,000 |
| `lsh_matched` (dense, same compute) | 0.39 ± 0.08 | 11,760 |
| `lsh_big` (dense, 130x compute) | 7.65 ± 0.07 | 1,568,000 |

MNIST, 200 queries vs 2000 db, cosine ground truth, 5 seeds. At matched
compute the mushroom-body motif wins 10x; dense codes only win with two
orders of magnitude more compute. The Dasgupta et al. 2017 efficiency
pattern, reproduced with the library's own module.

### 4.7 Third positive: the mushroom-body head learns few-shot (exp 09)

| arm | few-shot (256) | full MNIST |
|---|---|---|
| `fly` (sparse expansion + readout) | 0.7830 ± 0.0165 | 0.9482 ± 0.0008 |
| `dense_expansion` (same trainables, 130x compute) | 0.7381 ± 0.0145 | 0.9426 ± 0.0018 |
| `mlp` (matched trainables, ~20k) | 0.6625 ± 0.0204 | 0.9283 ± 0.0013 |

5 seeds, frozen expansion, only the readout trains. Clean ladder, gaps
larger than spreads: expansion beats the MLP (+0.12 few-shot), and
sparsity beats dense expansion (+0.045 few-shot). The motif is a
few-shot learner first, accuracy winner second.

### 4.8 Killed probes (throwaway scripts, recorded so nobody reruns them)

* Full-field translating texture, 2-way 2-D, 6 epochs, 1 seed: blob
  0.64 vs texture 0.59 (chance 0.50). Drive density is not the 2-D wall;
  no full experiment written.
* Elegans reservoir on Swimmer-v5, decoder-only random search (60 iters,
  1 seed): real 28.2 vs random 26.1 (random-action baseline ~3.5). Both
  swim, neither wins on topology. Full-ES landscape is cliffy (returns
  oscillate 24 to -15 across generations as swim direction flips), so no
  full run: the compute buys no ranking.
* Class-incremental MNIST (0-4 then 5-9, 5 epochs each, 3 seeds):
  retained accuracy on 0-4 collapses to ~0.00 in both arms (fly 0.001,
  mlp 0.000). Frozen sparse expansion does not save a shared readout
  from being overwritten; no full experiment written.
* Retrieval under query noise (200 queries, 3 seeds): fly 4.1 to 3.1 to
  1.7 at noise 0.0/0.3/0.6, matched dense 0.44 to 0.21 to 0.11. The 10x
  efficiency gap persists at every noise level, but both degrade
  proportionally: no differential robustness win, same win as exp 08.
* Speed-augmented training (mixed 0.5x/1x/2x, 2 epochs, 1 seed):
  fast-shift rescued (0.61 to 1.00) but clean drops (1.00 to 0.86) and
  slow barely moves (0.50 to 0.61). Random wiring rescued identically
  (slow 0.63, clean 0.80, fast 1.00): augmentation is a recipe fix, not
  a wiring win; no full experiment written.

## 5. Interpretation

1. **Channel in matters most.** The single biggest effect in the program is
   stimulus entry (L1 silence vs cartridge DSI 0.80), not any architectural
   choice.
2. **Frozen wiring + E/I computes; free weights memorize.** The positive
   appears only when the model starts as the measured animal (42k shared
   gains) and never when 262k free weights can rewrite it.
3. **What is causal:** E/I identity and cell-type topology. **What is not:**
   synapse-count magnitudes, our delay priors. Delays may still matter with
   learned taus or naturalistic flow; with fixed priors they do nothing here.
4. **Bounds are results.** 1-D works, 2-D does not, at matched budget. Speed
   and noise shifts break the mechanism. These are quantitative, reproducible
   walls, not failures to report.

## 6. Limits (do not cite beyond these)

* Rate neurons, no spikes, no gap junctions, no neuromodulation; a step is no
  number of milliseconds.
* (hex1, hex2) treated as Cartesian; the real lattice is hexagonal, so
  directions are stimulus labels, not visual-angle claims.
* Full-span exps carry an endpoint confound (rankings stand, shared by arms).
* Post-hoc DSI floor is high (~0.5); accuracy is the clean separator.
* Transmitter labels are predictions (Eckstein et al. 2024); signs derived
  from them are predictions on predictions. Receptor identity decides
  polarity, not the transmitter.
* One circuit, one dataset, one lab's protocol. Replication needs the
  fingerprint (`bf9321920b61c5a0`) and the commands below.

## 7. Related work

FlyVis (Lappalainen et al., Nature 2024) is the closest proof the idea works
and the source of this program's recipe; our contribution is the mechanism
dissection (sign/delay/topology ablations) and the honest negatives around
it. NCP/CfC work motivates the OOD axis we tested and did not win. See
[References](references.md).

## 8. Reproduction

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/04_selectivity.py --seeds 5 --epochs 3
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/06_mechanism.py --seeds 5 --epochs 3
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/05_ood.py --seeds 5 --epochs 3
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/07_flow_v2.py --seeds 5 --epochs 3
```

Full per-run tables: `experiments/README.md`. Raw rows: `results_*.jsonl`.
Circuit: `ct.datasets.malecns(variant="traced-only", min_synapses=5, ...)`,
41,826 nodes / 262,585 edges.
