"""Probe-layer checks that need no checkpoint download.

The numeric core (direction fitting, interventions, resampling) is tested against
synthetic data, and the hook plumbing is tested against a tiny hand-built torch
module. Together that covers everything except the model weights themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counting.probes.common import (
    ProbeSpec,
    clamp_projection,
    fit_counter_direction,
    fit_principal_components,
    monotonicity_score,
    patch_positions,
    pearson,
    pearson_by_feature,
    project_position_window,
    resample_positions,
    steer_final_token,
)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise AssertionError(label)


def synthetic_states(n: int = 200, hidden: int = 24, seed: int = 0):
    """States whose first axis encodes the count, plus noise on the rest.

    Past `capacity` the count signal is replaced by a fixed "collapsed" state,
    mimicking what the residual stream does past the boundary.
    """
    rng = np.random.default_rng(seed)
    counts = np.arange(1, n + 1, dtype=np.float32)
    axis = rng.normal(size=hidden).astype(np.float32)
    axis /= np.linalg.norm(axis)
    noise = rng.normal(scale=0.05, size=(n, hidden)).astype(np.float32)
    states = counts[:, None] * axis[None, :] + noise
    return states, counts, axis


def test_direction_fit() -> None:
    print("counter direction")
    # n >> hidden so the least-squares fit is well conditioned; with fewer
    # samples than dimensions the recovered weights fit noise and need not point
    # along the planted axis, even though the projection still tracks the count.
    states, counts, axis = synthetic_states(n=200, hidden=24)
    mask = counts <= 150

    fit = fit_counter_direction(states, counts, mask)
    # The fit promises a *projection* that tracks the count, not a weight vector
    # parallel to the generating axis: least squares divides the signal direction
    # by a large eigenvalue and the noise directions by small ones, so the weights
    # pick up noise components even when the projection is near-perfect. Assert
    # what the method actually claims.
    check("direction is non-degenerate", float(np.linalg.norm(fit.direction)) > 1e-6)
    check("direction has a real component along the planted axis",
          abs(pearson(fit.direction, axis)) > 0.5,
          f"r={pearson(fit.direction, axis):.3f}")
    check("projection tracks count",
          pearson(fit.projected, counts) > 0.99,
          f"r={pearson(fit.projected, counts):.4f}")
    check("rho near 1 (constant step per item)", 0.8 < fit.rho < 1.2, f"rho={fit.rho:.3f}")

    recovered = fit.count_from_projection(fit.projected)
    check("count recoverable from projection",
          float(np.abs(recovered - counts).max()) < 1.0,
          f"max err={float(np.abs(recovered - counts).max()):.3f}")
    # target_projection maps count -> projection; that orientation is what the
    # clamping experiment extrapolates, so assert it in that direction.
    check("target_projection maps count -> projection",
          abs(float(fit.target_projection(100)) - float(fit.projected[99])) < 1.0,
          f"{float(fit.target_projection(100)):.2f} vs {float(fit.projected[99]):.2f}")
    check("count_from_projection is its inverse",
          abs(float(fit.count_from_projection(fit.target_projection(77))) - 77.0) < 1e-2)
    check("project() matches the stored projection",
          np.allclose(fit.project(states), fit.projected, atol=1e-4))

    try:
        fit_counter_direction(states, counts, counts < 0)
        check("refuses to fit on <2 points", False)
    except ValueError:
        check("refuses to fit on <2 points", True)


def test_fit_mask_matters() -> None:
    """Fitting on successes only must separate the collapsed regime; fitting on
    everything must not. This is the property the paper's claim depends on."""
    print("fit mask")
    states, counts, _ = synthetic_states(n=40)
    capacity = 25
    collapsed = states[capacity - 1].copy()
    states[capacity:] = collapsed + np.random.default_rng(1).normal(scale=0.05, size=states[capacity:].shape)

    mask = counts <= capacity
    fit = fit_counter_direction(states, counts, mask)
    stable_err = np.abs(fit.projected[mask] - fit.target_projection(counts[mask])).mean()
    failed_err = np.abs(fit.projected[~mask] - fit.target_projection(counts[~mask])).mean()
    check("projection diverges past the boundary", failed_err > 5 * stable_err,
          f"stable={stable_err:.3f} failed={failed_err:.3f}")


def test_correlations() -> None:
    print("correlations")
    rng = np.random.default_rng(3)
    target = np.arange(20, dtype=np.float32)
    features = rng.normal(size=(20, 6)).astype(np.float32)
    features[:, 0] = target                 # perfectly correlated
    features[:, 1] = -target                # perfectly anticorrelated
    features[:, 2] = 1.0                    # constant

    r = pearson_by_feature(features, target)
    check("perfect correlation -> 1", abs(r[0] - 1.0) < 1e-4, f"{r[0]:.4f}")
    check("perfect anticorrelation -> -1", abs(r[1] + 1.0) < 1e-4, f"{r[1]:.4f}")
    check("constant feature -> 0, not NaN", r[2] == 0.0 and np.isfinite(r).all())

    check("monotone series scores 1", monotonicity_score(np.arange(10)) == 1.0)
    check("descending series scores 1", monotonicity_score(-np.arange(10)) == 1.0)
    check("alternating series scores ~0.5",
          abs(monotonicity_score(np.array([0, 1, 0, 1, 0, 1.0])) - 0.5) < 0.15)

    coords, explained = fit_principal_components(np.outer(np.arange(20), np.ones(5)) + rng.normal(scale=0.01, size=(20, 5)))
    check("PCA: one dominant component", explained[0] > 0.99, f"{explained[0]:.4f}")
    check("PCA coords shape", coords.shape == (20, 2), str(coords.shape))


def test_position_window() -> None:
    print("position window")
    states, counts, _ = synthetic_states(n=12, hidden=8)
    fit = fit_counter_direction(states, counts)
    window = np.stack([states, states], axis=1)          # (12, 2, 8)
    projected = project_position_window(window, fit.center, fit.direction)
    check("window projection shape", projected.shape == (12, 2), str(projected.shape))
    check("window projection matches 1-D projection",
          np.allclose(projected[:, 0], fit.projected, atol=1e-3))


def test_resampling() -> None:
    print("donor resampling")
    donor = np.arange(20, dtype=np.float32).reshape(10, 2)
    check("identity when lengths match", np.allclose(resample_positions(donor, 10), donor))
    up = resample_positions(donor, 25)
    check("upsamples to requested length", up.shape == (25, 2), str(up.shape))
    check("endpoints preserved", np.allclose(up[0], donor[0]) and np.allclose(up[-1], donor[-1]))
    down = resample_positions(donor, 4)
    check("downsamples to requested length", down.shape == (4, 2), str(down.shape))
    check("single-position donor broadcasts",
          resample_positions(donor[:1], 5).shape == (5, 2))


def test_interventions() -> None:
    print("interventions")
    import torch

    hidden = 12
    rng = np.random.default_rng(5)
    direction = rng.normal(size=hidden).astype(np.float32) * 3.0   # deliberately unnormalised
    center = rng.normal(size=hidden).astype(np.float32)
    activations = torch.from_numpy(rng.normal(size=(2, 6, hidden)).astype(np.float32))

    steered = steer_final_token(direction, 2.0)(activations)
    delta = (steered - activations).numpy()
    check("steering touches only the last position",
          np.allclose(delta[:, :-1, :], 0.0))
    check("steering adds alpha*direction",
          np.allclose(delta[:, -1, :], 2.0 * direction, atol=1e-4))

    donor = rng.normal(size=hidden).astype(np.float32)
    patched = patch_positions(donor)(activations)
    check("final-token patch replaces the last position",
          np.allclose(patched[:, -1, :].numpy(), donor, atol=1e-5))
    check("final-token patch leaves earlier positions alone",
          np.allclose(patched[:, :-1, :].numpy(), activations[:, :-1, :].numpy()))

    span_donor = rng.normal(size=(3, hidden)).astype(np.float32)
    patched = patch_positions(span_donor, slice(1, 4))(activations)
    check("span patch writes the span",
          np.allclose(patched[0, 1:4, :].numpy(), span_donor, atol=1e-5))
    check("span patch leaves the rest alone",
          np.allclose(patched[0, 4:, :].numpy(), activations[0, 4:, :].numpy()))

    target = 7.5
    clamped = clamp_projection(direction, center, target)(activations)
    achieved = ((clamped[:, -1, :].numpy() - center) * direction).sum(axis=-1)
    check("clamp hits the target projection exactly",
          np.allclose(achieved, target, atol=1e-3), f"{achieved}")

    # The defining property: the move is purely along `direction`.
    shift = (clamped - activations).numpy()[:, -1, :]
    unit = direction / np.linalg.norm(direction)
    orthogonal = shift - np.outer(shift @ unit, unit)
    check("clamp moves purely along the direction",
          float(np.abs(orthogonal).max()) < 1e-4, f"max orth={float(np.abs(orthogonal).max()):.2e}")
    check("clamp leaves earlier positions alone",
          np.allclose(shift.shape, (2, hidden)) and
          np.allclose((clamped - activations).numpy()[:, :-1, :], 0.0))


def test_hooks_on_a_real_module() -> None:
    print("hook plumbing")
    import torch
    from torch import nn

    from counting.probes.common import ActivationEditor, ActivationRecorder
    from counting.probes.common.spec import HookSpec

    class Block(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.o_proj = nn.Linear(hidden, hidden)

        def forward(self, x):
            return self.o_proj(x)

    class Tiny(nn.Module):
        def __init__(self, hidden=8, depth=3):
            super().__init__()
            self.layers = nn.ModuleList(Block(hidden) for _ in range(depth))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = Tiny()
    x = torch.ones(1, 4, 8)

    hooks = [
        HookSpec("resid_post.layer1", "layers.1", "forward", 8),
        HookSpec("attn_out.layer2", "layers.2.o_proj", "forward_pre", 8),
    ]
    with ActivationRecorder(model, hooks) as rec:
        out = model(x)
        check("forward hook captured", "resid_post.layer1" in rec.cache)
        check("forward_pre hook captured", "attn_out.layer2" in rec.cache)
        check("captured shape", tuple(rec.cache["resid_post.layer1"].shape) == (1, 4, 8))
    check("hooks removed on exit", len(rec._handles) == 0)

    baseline = model(x)
    check("model unchanged after recording", torch.allclose(baseline, out))

    # An edit at layer 1 must change the output; removing it must restore it.
    with ActivationEditor(model) as editor:
        editor.edit(hooks[0], lambda t: t + 100.0)
        edited = model(x)
    check("edit changed the output", not torch.allclose(edited, baseline))
    check("output restored after the editor closed", torch.allclose(model(x), baseline))

    try:
        with ActivationEditor(model) as editor:
            editor.edit(hooks[1], lambda t: t)
        check("refuses to edit a forward_pre site", False)
    except ValueError:
        check("refuses to edit a forward_pre site", True)

    try:
        ActivationRecorder(model, [HookSpec("bogus", "layers.99", "forward")]).register()
        check("names a missing module", False)
    except KeyError as exc:
        check("names a missing module", "layers.99" in str(exc))


def test_specs() -> None:
    print("probe specs")
    for path in sorted(Path("configs/probes").glob("*.yaml")):
        spec = ProbeSpec.load(path)
        check(f"{path.name} parses", len(spec.hooks) > 0, f"{len(spec.hooks)} hooks")
        check(f"{path.name} has residual hooks", len(spec.residual_hooks) >= 3)
        check(f"{path.name} layers parse", all(h.layer is not None for h in spec.hooks))
        for hook in spec.sae_hooks:
            check(f"{path.name} sae path set", bool(hook.sae))

    gemma = ProbeSpec.load("configs/probes/gemma3-27b-it.yaml")
    check("gemma uses the VLM loader", gemma.loader == "conditional_generation")
    check("gemma attn hooks are forward_pre",
          all(h.mode == "forward_pre" for h in gemma.hooks_in_family("attn_out")))
    qwen = ProbeSpec.load("configs/probes/qwen3.5-35b-a3b.yaml")
    check("qwen has no SAEs", qwen.sae_hooks == [])
    check("qwen marks full-attention layers", qwen.is_full_attention(31))
    check("qwen excludes other layers", not qwen.is_full_attention(30))


def test_inspect_helpers() -> None:
    """The module-path guessing in `counting-probe inspect`.

    Exercised against synthetic module-name lists so it is covered without
    downloading a checkpoint. The transformers loading around it is stock API.
    """
    print("inspect helpers")
    from counting.probes.cli import _default_layers, _layer_prefix, _mlp_module

    qwen = [f"model.layers.{i}" for i in range(24)]
    qwen += [f"model.layers.{i}.self_attn.o_proj" for i in range(24)]
    qwen += [f"model.layers.{i}.mlp" for i in range(24)]
    check("finds a plain causal-LM layer path", _layer_prefix(qwen) == "model.layers",
          str(_layer_prefix(qwen)))

    # Gemma 3 nests the text tower; the vision tower also has a `.layers.N`, and
    # the language one is the one we want.
    gemma = [f"model.language_model.layers.{i}" for i in range(62)]
    gemma += [f"model.vision_tower.encoder.layers.{i}" for i in range(27)]
    gemma += [f"model.language_model.layers.{i}.post_feedforward_layernorm" for i in range(62)]
    check("prefers the language tower over the vision tower",
          _layer_prefix(gemma) == "model.language_model.layers", str(_layer_prefix(gemma)))

    check("no layer tree -> None", _layer_prefix(["embed", "lm_head"]) is None)

    check("mlp_out prefers post_feedforward_layernorm",
          _mlp_module(gemma, "model.language_model.layers", 31)
          == "model.language_model.layers.31.post_feedforward_layernorm")
    check("mlp_out falls back to the mlp module",
          _mlp_module(qwen, "model.layers", 5) == "model.layers.5.mlp")
    check("mlp_out returns None when neither exists",
          _mlp_module(["model.layers.0"], "model.layers", 0) is None)

    layers = _default_layers(list(range(62)))
    check("default layers span the stack", len(layers) == 4 and layers[-1] == 61, str(layers))
    check("default layers ascend", layers == sorted(layers), str(layers))
    check("default layers handle a short stack", len(_default_layers([0, 1, 2])) >= 1)
    check("default layers handle an empty stack", _default_layers([]) == [])


def main() -> int:
    test_direction_fit()
    test_fit_mask_matters()
    test_correlations()
    test_position_window()
    test_resampling()
    test_interventions()
    test_hooks_on_a_real_module()
    test_specs()
    test_inspect_helpers()
    print("\nall probe checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
