"""Wiring a connectome core into an ordinary machine learning model.

A connectome has no notion of pixels, tokens or class labels. These wrappers do
the boring translation between arbitrary feature vectors and the neurons that
receive or report them, and deliberately stay outside the core so that
:class:`~connectorch.nn.ConnectomeRNN` never learns about your task.
"""

from __future__ import annotations

from torch import Tensor, nn

from .recurrent import ConnectomeRNN

__all__ = ["InputProjection", "ConnectomeModel", "count_parameters"]


class InputProjection(nn.Module):
    """A linear map from arbitrary features onto a connectome's input neurons.

    Parameters
    ----------
    in_features:
        Width of the incoming feature vector, e.g. 784 for a flattened MNIST digit.
    core:
        The connectome runtime whose ``num_input_nodes`` sets the output width.
    bias:
        Whether the projection has a bias.

    Examples
    --------
    >>> encoder = InputProjection(784, core)      # doctest: +SKIP
    """

    def __init__(self, in_features: int, core: ConnectomeRNN, *, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, core.num_input_nodes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Project ``[..., in_features]`` to ``[..., num_input_nodes]``."""
        return self.linear(x)


class ConnectomeModel(nn.Module):
    """Encoder, connectome core, decoder: the usual shape of a task model.

    Parameters
    ----------
    core:
        The connectome runtime.
    encoder:
        Maps task features to the core's input nodes. ``None`` feeds ``x``
        straight in, which requires it to already be ``num_input_nodes`` wide.
    decoder:
        Maps the core's readout to task outputs. ``None`` returns the readout.
    readout:
        ``"last"`` decodes the final step, ``"mean"`` the average over steps, and
        ``"all"`` decodes every step and returns ``[batch, steps, out]``.

    Examples
    --------
    >>> model = ConnectomeModel(                       # doctest: +SKIP
    ...     core=ct.nn.ConnectomeRNN(brain, weights="trainable"),
    ...     encoder=nn.Linear(784, 64),
    ...     decoder=nn.Linear(64, 10),
    ... )
    >>> logits = model(images, steps=8)                # doctest: +SKIP
    """

    def __init__(
        self,
        core: ConnectomeRNN,
        *,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        readout: str = "last",
    ) -> None:
        super().__init__()
        if readout not in ("last", "mean", "all"):
            raise ValueError(f"readout must be 'last', 'mean' or 'all', got {readout!r}.")
        self.core = core
        self.encoder = encoder
        self.decoder = decoder
        self.readout = readout

    def forward(self, x: Tensor, steps: int | None = None, **kwargs: object) -> Tensor:
        """Encode, run the connectome, read out and decode."""
        if self.encoder is not None:
            x = self.encoder(x)
        trajectory = self.core(x, steps, **kwargs)  # type: ignore[arg-type]

        if self.readout == "last":
            features = trajectory[:, -1, :]
        elif self.readout == "mean":
            features = trajectory.mean(dim=1)
        else:
            features = trajectory

        return self.decoder(features) if self.decoder is not None else features

    def extra_repr(self) -> str:
        return f"readout={self.readout!r}"


def count_parameters(module: nn.Module) -> dict[str, int]:
    """Trainable parameter counts, split into the connectome and everything else.

    Comparing a connectome model against a vanilla RNN is only meaningful at a
    matched parameter budget, and it is easy to forget that the encoder and
    decoder are part of that budget.
    """
    # Count by identity: two cores can share one weight tensor, and counting it
    # twice makes "other" go negative.
    connectome_tensors = {
        id(p): p
        for m in module.modules()
        if isinstance(m, ConnectomeRNN)
        for p in m.parameters()
        if p.requires_grad
    }
    all_tensors = {id(p): p for p in module.parameters() if p.requires_grad}
    connectome = sum(p.numel() for p in connectome_tensors.values())
    total = sum(p.numel() for p in all_tensors.values())
    return {"total": total, "connectome": connectome, "other": total - connectome}
