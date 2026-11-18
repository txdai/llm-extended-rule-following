"""Sparse autoencoder dictionaries in the Gemma Scope layout.

A dictionary directory holds `config.json` plus `params.safetensors` with
`w_enc`, `b_enc`, `w_dec`, `b_dec`, and `threshold`. The encoder is JumpReLU:

    pre = x @ w_enc + b_enc
    f   = pre where pre > threshold, else 0

Gemma 3 has Gemma Scope 2 dictionaries; Qwen does not have a public one, which is
why the Qwen probes stop at residual-stream projections. A spec with no `sae` on
any hook simply skips this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch


class SparseAutoencoder:
    """A JumpReLU SAE loaded from a Gemma Scope-style directory."""

    def __init__(self, root: Path, device: str) -> None:
        import json

        from safetensors import safe_open

        import torch  # noqa: F401 - device strings are validated by torch

        self.root = Path(root)
        self.device = device
        config_path = self.root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"no config.json under {self.root}; expected a Gemma Scope-style SAE directory"
            )
        config = json.loads(config_path.read_text())
        self.width = int(config["width"])

        with safe_open(self.root / "params.safetensors", framework="pt", device="cpu") as handle:
            self.b_dec = handle.get_tensor("b_dec").to(device=device)
            self.b_enc = handle.get_tensor("b_enc").to(device=device)
            self.threshold = handle.get_tensor("threshold").to(device=device)
            self.w_dec = handle.get_tensor("w_dec").to(device=device)
            self.w_enc = handle.get_tensor("w_enc").to(device=device)

        self.hidden_size = int(self.w_enc.shape[0])

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """Dense activations -> sparse feature activations."""
        import torch

        pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def decode(self, features: "torch.Tensor") -> "torch.Tensor":
        return features @ self.w_dec + self.b_dec

    def __repr__(self) -> str:
        return f"SparseAutoencoder(width={self.width}, hidden_size={self.hidden_size})"


def load_saes(spec: Any, device: str) -> dict[str, SparseAutoencoder]:
    """Load every SAE a ProbeSpec attaches to a hook, keyed by hook label."""
    if spec.sae_root is None:
        return {}
    root = Path(spec.sae_root).expanduser()
    return {
        hook.label: SparseAutoencoder(root / hook.sae, device) for hook in spec.sae_hooks
    }
