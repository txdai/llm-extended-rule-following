"""Decode-point editing shared by the subspace interventions.

One primitive: run a greedy generation for a prompt while a function
`fn(h) -> h'` (float32 numpy, shape (hidden,)) rewrites the residual state at
the decode-point token (the last prompt token) at one hooked site. Both
backends are covered: PyTorch forward hooks, and the MLX layer taps, where
`generate_step` prefills all-but-last tokens and then runs a one-token step
for the last prompt token, so the edit fires on that first one-token forward.

`intervene.py` carries its own copy of this logic; it is left untouched so the
rows it produced for the paper stay reproducible byte for byte.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..common.spec import HookSpec
from .util import decode_tokens

EditFn = Callable[[np.ndarray], np.ndarray]


class DecodeEditor:
    def __init__(self, model: Any, tokenizer: Any, device: str, max_new_tokens: int = 16) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, ids: list[int], hook: HookSpec | None, fn: EditFn | None,
                 positions: list[int] | None = None) -> str:
        """Greedy continuation of `ids` with `fn` applied at the last prompt token.

        With `positions`, `fn` receives the (n_positions, hidden) block of
        states at those prompt positions (all strictly before the decode
        point) and returns the edited block; the edit fires once, during the
        prefill.
        """
        q_idx = len(ids) - 1
        if positions is not None and any(p >= q_idx for p in positions):
            raise ValueError("multi-position edits must lie before the decode point")
        if self.device == "mlx":
            return self._generate_mlx(ids, q_idx, hook, fn, positions)
        return self._generate_torch(ids, q_idx, hook, fn, positions)

    def logits(self, ids: list[int], hook: HookSpec | None, fn: EditFn | None) -> np.ndarray:
        """Next-token logits at the decode point (no generation) under the edit."""
        q_idx = len(ids) - 1
        if self.device == "mlx":
            from ..common.mlx_backend import MlxActivationEditor, forward_logits

            with MlxActivationEditor(self.model) as editor:
                if hook is not None and fn is not None:
                    def edit(arr):
                        out = np.array(arr, copy=True)
                        if out.shape[1] > q_idx:
                            out[0, q_idx] = fn(out[0, q_idx].astype(np.float32)).astype(out.dtype)
                        return out

                    editor.edit(hook, edit)
                return forward_logits(self.model, ids)[q_idx].astype(np.float32)
        import torch

        from ..common.model import ActivationEditor

        with ActivationEditor(self.model) as editor, torch.inference_mode():
            if hook is not None and fn is not None:
                def edit(tensor):
                    if tensor.shape[1] > q_idx:
                        h = tensor[0, q_idx].float().cpu().numpy()
                        tensor[0, q_idx] = torch.as_tensor(fn(h), device=tensor.device, dtype=tensor.dtype)
                    return tensor

                editor.edit(hook, edit)
            out = self.model(torch.tensor([ids], device=self.device))
            return out.logits[0, q_idx].float().cpu().numpy()

    # -- backends -------------------------------------------------------------

    def _generate_mlx(self, ids, q_idx, hook, fn, positions=None) -> str:
        from ..common.mlx_backend import MlxActivationEditor, generate_tokens

        with MlxActivationEditor(self.model) as editor:
            if hook is not None and fn is not None:
                fired = {"once": False}
                pos = np.asarray(positions, dtype=np.int64) if positions is not None else None

                def edit(arr):
                    out = np.array(arr, copy=True)
                    t = out.shape[1]
                    if pos is not None:
                        # mlx prefill chunks are all-but-last tokens; every
                        # edited position is inside the first chunk.
                        if t > int(pos.max()) and not fired["once"]:
                            fired["once"] = True
                            out[0, pos] = fn(out[0, pos].astype(np.float32)).astype(out.dtype)
                    elif t > q_idx:
                        out[0, q_idx] = fn(out[0, q_idx].astype(np.float32)).astype(out.dtype)
                    elif t == 1 and not fired["once"]:
                        fired["once"] = True
                        out[0, 0] = fn(out[0, 0].astype(np.float32)).astype(out.dtype)
                    return out

                editor.edit(hook, edit)
            new_ids = generate_tokens(self.model, self.tokenizer, ids, self.max_new_tokens)
        return decode_tokens(self.tokenizer, new_ids).strip()

    def _generate_torch(self, ids, q_idx, hook, fn, positions=None) -> str:
        import torch

        from ..common.model import ActivationEditor

        input_ids = torch.tensor([ids], device=self.device)
        attention = torch.ones_like(input_ids)
        with ActivationEditor(self.model) as editor, torch.inference_mode():
            if hook is not None and fn is not None:
                pos = torch.tensor(positions, device=self.device) if positions is not None else None

                def edit(tensor):
                    if pos is not None:
                        if tensor.shape[1] > int(pos.max()):
                            block = tensor[0, pos].float().cpu().numpy()
                            tensor[0, pos] = torch.as_tensor(
                                fn(block), device=tensor.device, dtype=tensor.dtype
                            )
                    elif tensor.shape[1] > q_idx:
                        h = tensor[0, q_idx].float().cpu().numpy()
                        tensor[0, q_idx] = torch.as_tensor(
                            fn(h), device=tensor.device, dtype=tensor.dtype
                        )
                    return tensor

                editor.edit(hook, edit)
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return decode_tokens(self.tokenizer, sequences[0, len(ids):]).strip()


def set_state(new_state: np.ndarray) -> EditFn:
    new_state = np.asarray(new_state, dtype=np.float32)
    return lambda _h: new_state


def add_vector(vec: np.ndarray) -> EditFn:
    vec = np.asarray(vec, dtype=np.float32)
    return lambda h: h + vec


def project_delta(base: np.ndarray, delta: np.ndarray, basis: np.ndarray, complement: bool = False) -> EditFn:
    """h_base + P delta with P the projector onto the rows of `basis` (or I - P)."""
    basis = np.asarray(basis, dtype=np.float32)
    proj = basis.T @ (basis @ np.asarray(delta, dtype=np.float32))
    shift = (np.asarray(delta, dtype=np.float32) - proj) if complement else proj
    target = np.asarray(base, dtype=np.float32) + shift
    return lambda _h: target


def ablate_to_mean(basis: np.ndarray, mean: np.ndarray) -> EditFn:
    """h - P h + P mean: replace the component in the subspace with the mean's."""
    basis = np.asarray(basis, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    mean_component = basis.T @ (basis @ mean)

    def fn(h: np.ndarray) -> np.ndarray:
        return h - basis.T @ (basis @ h) + mean_component

    return fn


def rotate_pair(w: np.ndarray, x_center: np.ndarray, y_center: np.ndarray, phase: float) -> EditFn:
    """Rotate the readout of a (cos, sin) ridge pair by `phase`, minimum-norm edit.

    `w` is (2, hidden): the two ridge directions; readouts are
    y = (h - x_center) @ w.T + y_center. The edit adds the minimum-norm vector
    in span(w) that moves the readout pair from (c, s) to R(phase) (c, s).
    """
    w = np.asarray(w, dtype=np.float32)
    x_center = np.asarray(x_center, dtype=np.float32)
    y_center = np.asarray(y_center, dtype=np.float32)
    gram_inv = np.linalg.inv((w @ w.T).astype(np.float64)).astype(np.float32)
    if w.shape[0] == 1:
        # Period-2 pair has a single (parity) phase: rotation by pi negates it.
        rot = np.array([[np.cos(phase)]], dtype=np.float32)
    else:
        rot = np.array(
            [[np.cos(phase), -np.sin(phase)], [np.sin(phase), np.cos(phase)]], dtype=np.float32
        )

    def fn(h: np.ndarray) -> np.ndarray:
        y = (h - x_center) @ w.T + y_center
        target = rot @ y
        return h + w.T @ (gram_inv @ (target - y))

    return fn
