"""Load the real MaleCNS connectome and compile it, without ever densifying it.

The bundled 100-neuron sample runs offline. Pass --download to fetch the published
connectivity table and compile all ~164,000 neurons of the traced-only graph; a
dense adjacency for that graph would be about 101 GiB, and this never allocates
one.

    python examples/03_malecns_load.py
    python examples/03_malecns_load.py --download                     # ~520 MB
    python examples/03_malecns_load.py --download --cuda --batch-size 4 --steps 8

The last line is the exact command behind the numbers quoted in the README.
"""

import argparse
import resource
import sys
import time

import torch

import connectorch as ct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--download",
    action="store_true",
    help="fetch and load the real dataset instead of the bundled sample",
)
parser.add_argument(
    "--variant",
    default="traced-only",
    choices=["traced-only", "significant-only", "full"],
    help="which published connectivity table to load (only with --download)",
)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output-nodes", type=int, default=64, help="how many neurons to read out")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

start = time.perf_counter()
if args.download:
    brain = ct.datasets.malecns(variant=args.variant, download=True)
else:
    brain = ct.datasets.malecns_sample()
elapsed = time.perf_counter() - start

# ru_maxrss is bytes on macOS and kilobytes on Linux.
scale = 1e9 if sys.platform == "darwin" else 1e6
print(brain)
print(f"\nloaded and compiled in {elapsed:.1f}s")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale:.1f} GB")
print(f"license: {brain.provenance['license']}")
print(f"a dense float32 adjacency would need {brain.num_nodes**2 * 4 / 2**30:,.1f} GiB")

model = ct.nn.ConnectomeRNN(
    brain, weights="trainable", output_nodes=brain.node_ids[: args.output_nodes]
)
if args.cuda:
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
print(f"\nbackend: {model.backend}, trainable connections: {model.edge_weight.numel():,}")
print(
    f"activations this call will store: "
    f"{model.activation_bytes(args.batch_size, args.steps) / 2**30:.2f} GiB"
)

x = torch.zeros(args.batch_size, brain.num_nodes, device=model.edge_weight.device)
start = time.perf_counter()
y = model(x, steps=args.steps)
y.square().mean().backward()
if args.cuda:
    torch.cuda.synchronize()
step = time.perf_counter() - start

print(
    f"forward+backward over {model.num_edges:,} connections "
    f"(batch {args.batch_size}, {args.steps} steps): {step * 1000:,.0f} ms"
)
if args.cuda:
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"gradients finite: {bool(torch.isfinite(model.edge_weight.grad).all())}")

for name, value in model.diagnostics().items():
    print(f"  {name}: {value}")
