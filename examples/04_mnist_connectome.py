"""Classify MNIST digits through a real fly connectome.

    784 pixels -> encoder -> 100 real neurons, wired as the fly is -> decoder -> 10 classes

The point is not the accuracy. The point is that a wiring diagram traced out of an
actual animal's nervous system sits in the middle of an ordinary PyTorch training
loop, receives gradients, and learns.

Needs torchvision:  pip install torchvision

    python examples/04_mnist_connectome.py
    python examples/04_mnist_connectome.py --control degree_preserving
"""

import argparse
import time

import torch
from torch import nn

import connectorch as ct
from connectorch.transforms.rewire import degree_preserving_rewire, random_topology

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--control",
    choices=["none", "degree_preserving", "random"],
    default="none",
    help="replace the real wiring with a topology-matched control, for comparison",
)
args = parser.parse_args()

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover - a user-facing message, not library code
    raise SystemExit("this example needs torchvision: pip install torchvision") from None

torch.manual_seed(args.seed)

brain = ct.datasets.malecns_sample()
if args.control == "degree_preserving":
    brain = degree_preserving_rewire(brain, seed=args.seed)
elif args.control == "random":
    brain = random_topology(brain, seed=args.seed)
print(f"{brain}\ncontrol: {args.control}\n")

core = ct.nn.ConnectomeRNN(
    brain,
    input_nodes=brain.node_ids[:64],
    output_nodes=brain.node_ids[-64:],
    weights="trainable",
    initializer="normalized_synapse_count",
    leak=0.4,
)
model = ct.nn.ConnectomeModel(
    core,
    encoder=nn.Sequential(nn.Flatten(), nn.Linear(784, core.num_input_nodes), nn.Tanh()),
    decoder=nn.Linear(core.num_output_nodes, 10),
)
print(f"parameters: {ct.nn.count_parameters(model)}")

transform = transforms.ToTensor()
train_set = datasets.MNIST(
    "~/.cache/connectorch/mnist", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    "~/.cache/connectorch/mnist", train=False, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=512)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
wiring_before = core.edge_index.clone()

for epoch in range(args.epochs):
    model.train()
    start = time.perf_counter()
    for batch, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = criterion(model(images, steps=args.steps), labels)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"epoch {epoch}  batch {batch:>4}  loss {loss.item():.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += int((model(images, steps=args.steps).argmax(1) == labels).sum())
            total += labels.numel()
    print(
        f"epoch {epoch} done in {time.perf_counter() - start:.0f}s  "
        f"test accuracy {correct / total:.4f}\n"
    )

print(f"topology unchanged after training: {torch.equal(wiring_before, core.edge_index)}")
print("A biological wiring diagram just did gradient descent.")
