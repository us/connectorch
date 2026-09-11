"""Load the real MaleCNS connectome and compile it, without ever densifying it.

The bundled 100-neuron sample runs offline. Pass --full to fetch the published
half-gigabyte connectivity table and compile all ~164,000 neurons; a dense
adjacency for that graph would be about 101 GiB, and this never allocates one.

    python examples/03_malecns_load.py
    python examples/03_malecns_load.py --full          # downloads ~520 MB
    python examples/03_malecns_load.py --full --cuda
"""

import argparse
import resource
import sys
import time

import torch

import connectorch as ct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true", help="download and load the whole dataset")
parser.add_argument("--variant", default="traced-only")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

start = time.perf_counter()
if args.full:
    brain = ct.datasets.malecns(variant=args.variant, download=True)
else:
    brain = ct.datasets.malecns_sample()
elapsed = time.perf_counter() - start

print(brain)
print(f"\nloaded and compiled in {elapsed:.1f}s")
# ru_maxrss is bytes on macOS and kilobytes on Linux.
scale = 1e9 if sys.platform == "darwin" else 1e6
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale:.1f} GB")
print(f"license: {brain.provenance['license']}")
print(f"a dense float32 adjacency would need {brain.num_nodes**2 * 4 / 2**30:,.1f} GiB")

model = ct.nn.ConnectomeRNN(brain, weights="trainable", output_nodes=brain.node_ids[:64])
if args.cuda:
    model = model.cuda()
print(f"\nbackend: {model.backend}, trainable synapses: {model.edge_weight.numel():,}")

x = torch.zeros(2, brain.num_nodes, device=model.edge_weight.device)
start = time.perf_counter()
y = model(x, steps=4)
y.square().mean().backward()
if args.cuda:
    torch.cuda.synchronize()
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"forward+backward over {model.num_edges:,} synapses: {time.perf_counter() - start:.2f}s")
print(f"gradients finite: {bool(torch.isfinite(model.edge_weight.grad).all())}")

for name, value in model.diagnostics().items():
    print(f"  {name}: {value}")
