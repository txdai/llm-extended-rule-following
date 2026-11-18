"""Causal interventions on the residual stream.

The four profiles from the paper, as edit functions you hand to
`ActivationEditor.edit`. Each takes a tensor of shape (batch, positions, hidden)
and returns an edited copy:

    steer_final_token   add alpha * direction at the last prompt token
    patch_positions     replace a span of positions with donor states
    translate_positions move a span along a direction by a fixed amount
    clamp_projection    force the projection onto a direction to a target value

`clamp_projection` is the one that falsified the 1-D causal hypothesis: it moves
the state purely along the fitted direction, leaving every orthogonal component
untouched, and it never rescued a failed sequence. That "purely along" is the
whole point of the test -- see `clamp_projection` for why the implementation must
not normalise the direction twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

Edit = Callable[["torch.Tensor"], "torch.Tensor"]


def _as_tensor(vector: np.ndarray, like: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(
        device=like.device, dtype=like.dtype
    )


def steer_final_token(direction: np.ndarray, alpha: float) -> Edit:
    """Add `alpha * direction` to the last position only.

    The last prompt position is the assistant-prefix token, i.e. the state the
    decoder actually reads. Steering here is the cheapest test of whether a
    direction is behaviourally live.
    """

    def edit(activations: "torch.Tensor") -> "torch.Tensor":
        vector = _as_tensor(direction, activations) * float(alpha)
        out = activations.clone()
        out[:, -1, :] = out[:, -1, :] + vector
        return out

    return edit


def patch_positions(donor: np.ndarray, positions: Sequence[int] | slice | None = None) -> Edit:
    """Replace states at `positions` with `donor`.

    donor is (len(positions), hidden), or (hidden,) to write one position.
    `positions=None` writes the final position, matching final-token donor
    patching; pass a slice for sequence-token patching.

    Donor and target spans routinely differ in length -- a count-14 donor has
    more sequence tokens than a count-10 base -- so resample the donor with
    `resample_positions` before calling this.
    """

    def edit(activations: "torch.Tensor") -> "torch.Tensor":
        out = activations.clone()
        block = _as_tensor(donor, activations)
        if positions is None:
            out[:, -1, :] = block.reshape(-1)
        elif isinstance(positions, slice):
            out[:, positions, :] = block.reshape(1, -1, out.shape[-1])
        else:
            index = list(positions)
            out[:, index, :] = block.reshape(1, len(index), out.shape[-1])
        return out

    return edit


def translate_positions(
    direction: np.ndarray, amount: float, positions: slice | None = None
) -> Edit:
    """Shift a span of positions by `amount * direction`."""

    def edit(activations: "torch.Tensor") -> "torch.Tensor":
        vector = _as_tensor(direction, activations) * float(amount)
        out = activations.clone()
        span = positions if positions is not None else slice(None)
        out[:, span, :] = out[:, span, :] + vector
        return out

    return edit


def clamp_projection(
    direction: np.ndarray,
    center: np.ndarray,
    target_projection: float,
    position: int = -1,
) -> Edit:
    """Force the projection onto `direction` to equal `target_projection`.

    Moves the state purely along the direction and leaves every orthogonal
    component alone, which is what makes this a clean test of the 1-D hypothesis:
    if the coordinate were sufficient, correcting it would restore the output.

    The shift is `(target - current) / (d.d) * d`. Dividing by the squared norm
    is what makes it exact for an unnormalised `direction` -- and the fitted
    direction from `fit_counter_direction` is unnormalised, so this division must
    not be dropped.
    """

    def edit(activations: "torch.Tensor") -> "torch.Tensor":
        vector = _as_tensor(direction, activations)
        origin = _as_tensor(center, activations)
        out = activations.clone()
        state = out[:, position, :]
        denominator = (vector * vector).sum()
        if float(denominator) <= 1e-12:
            return out
        current = ((state - origin) * vector).sum(dim=-1, keepdim=True)
        shift = (float(target_projection) - current) / denominator
        out[:, position, :] = state + shift * vector
        return out

    return edit


def resample_positions(donor: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a (positions, hidden) donor span to `length` positions.

    Donor prompts almost never have the same span length as the base prompt, and
    truncating or padding would compare different parts of the trajectory.
    """
    donor = np.asarray(donor, dtype=np.float32)
    if donor.ndim != 2:
        raise ValueError(f"donor must be (positions, hidden); got shape {donor.shape}")
    if donor.shape[0] == length:
        return donor
    if length <= 0:
        raise ValueError("length must be positive")
    if donor.shape[0] == 1:
        return np.repeat(donor, length, axis=0)
    source = np.linspace(0.0, 1.0, donor.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack(
        [np.interp(target, source, donor[:, dim]) for dim in range(donor.shape[1])], axis=1
    ).astype(np.float32)
