"""One forward/backward pass through the explicit Apple Metal CSR backend.

    PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/apple_metal.py
    python examples/apple_metal.py --device cpu --dtype float64

The CPU option uses the reference implementation. No datasets or weights are
downloaded, and this example performs no optimizer updates.
"""

import argparse
import os

import torch

import connectorch as ct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    if args.device == "mps":
        if args.dtype != "float32":
            parser.error("metal_csr on MPS requires float32; float64 is CPU-only")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            parser.error("start Python with PYTORCH_ENABLE_MPS_FALLBACK=0")
        if not torch.backends.mps.is_available():
            parser.error("this Python environment has no available Apple MPS device")
        if not callable(getattr(torch.mps, "compile_shader", None)):
            parser.error("this PyTorch build lacks torch.mps.compile_shader")

    dtype = getattr(torch, args.dtype)
    brain = ct.Connectome.from_edges(
        source=["A", "B", "C", "C"],
        target=["B", "C", "A", "B"],
        weight=[1.0, 0.5, 0.2, 0.8],
    )
    model = ct.nn.ConnectomeRNN(
        brain,
        weights="trainable",
        initializer="weight",
        backend="metal_csr",
        leak=0.5,
        dtype=dtype,
    ).to(args.device)
    x = torch.linspace(-0.5, 0.5, 24, dtype=dtype).reshape(2, 4, brain.num_nodes)
    x = x.to(args.device).requires_grad_()
    initial_state = torch.zeros(
        2, brain.num_nodes, device=args.device, dtype=dtype, requires_grad=True
    )
    y = model(x, state=initial_state)
    y.square().mean().backward()

    for name, gradient in (
        ("input", x.grad),
        ("initial state", initial_state.grad),
        ("edge values", model.edge_weight.grad),
    ):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"missing or nonfinite {name} gradient")
        print(f"{name} gradient norm: {gradient.norm().item():.8f}")
    print(f"backend={model.backend}, device={y.device}, dtype={y.dtype}, output={tuple(y.shape)}")


if __name__ == "__main__":
    main()
