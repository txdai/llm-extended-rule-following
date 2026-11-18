"""Probe command line.

    counting-probe list
    counting-probe show configs/probes/gemma3-27b-it.yaml
    counting-probe inspect Qwen/Qwen3.5-35B-A3B --layers 15 23 31 39
    counting-probe check configs/probes/gemma3-27b-it.yaml

`inspect` is the one that saves real time when adding a model: it loads the
checkpoint's module tree and prints the residual-stream, attention-output, and
MLP-output paths for the layers you name, in the exact shape a probe config
wants. Paste the output into configs/probes/<model>.yaml and you are done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import CONFIG_ROOT, ConfigError
from .common import ProbeSpec

PROBE_CONFIG_DIR = CONFIG_ROOT / "probes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="counting-probe", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the probe configs in configs/probes/")

    show = sub.add_parser("show", help="print a resolved probe spec")
    show.add_argument("config", type=Path)

    inspect = sub.add_parser(
        "inspect", help="print candidate hook module paths for a checkpoint"
    )
    inspect.add_argument("checkpoint", help="HuggingFace id or local path")
    inspect.add_argument(
        "--layers", type=int, nargs="+", help="layers to emit hook entries for"
    )
    inspect.add_argument(
        "--all-modules", action="store_true", help="dump every named module instead"
    )
    inspect.add_argument(
        "--loader",
        choices=("causal_lm", "conditional_generation"),
        default="causal_lm",
    )

    check = sub.add_parser(
        "check", help="load the checkpoint and verify every hook module resolves"
    )
    check.add_argument("config", type=Path)

    return parser


def cmd_list() -> int:
    configs = sorted(PROBE_CONFIG_DIR.glob("*.yaml"))
    if not configs:
        print(f"no probe configs under {PROBE_CONFIG_DIR}")
        return 0
    for path in configs:
        try:
            spec = ProbeSpec.load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name:<28} !! {type(exc).__name__}: {exc}")
            continue
        sae = f", {len(spec.sae_hooks)} with SAEs" if spec.sae_hooks else ""
        print(f"  {path.name:<28} {spec.checkpoint}  ({len(spec.hooks)} hooks{sae})")
    return 0


def cmd_show(path: Path) -> int:
    spec = ProbeSpec.load(path)
    print(f"name        {spec.name}")
    print(f"label       {spec.label}")
    print(f"checkpoint  {spec.checkpoint}")
    print(f"loader      {spec.loader} / {spec.tokenizer}")
    print(f"dtype       {spec.dtype}   device: {spec.device}")
    if spec.sae_root:
        print(f"sae_root    {spec.sae_root}")
    if spec.full_attention_layers:
        print(f"full attn   {spec.full_attention_layers}")
    print("\nhooks")
    for hook in spec.hooks:
        sae = f"  sae={hook.sae}" if hook.sae else ""
        print(f"  {hook.label:<22} {hook.mode:<12} {hook.module}{sae}")
    return 0


def cmd_inspect(checkpoint: str, layers: list[int] | None, all_modules: bool, loader: str) -> int:
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    except (ImportError, ValueError) as exc:
        _report_transformers_import_failure(exc)
        return 1

    print(f"loading {checkpoint} on CPU (meta weights where possible)...", file=sys.stderr)
    config = AutoConfig.from_pretrained(checkpoint)
    with torch.device("meta"):
        cls = AutoModelForImageTextToText if loader == "conditional_generation" else AutoModelForCausalLM
        # transformers 5 renamed _from_config -> from_config.
        factory = getattr(cls, "from_config", None) or cls._from_config
        model = factory(config)

    names = [name for name, _ in model.named_modules() if name]
    if all_modules:
        for name in names:
            print(name)
        return 0

    prefix = _layer_prefix(names)
    if prefix is None:
        print(
            "could not find a '<...>.layers.N' module tree; "
            "re-run with --all-modules and pick the paths by hand",
            file=sys.stderr,
        )
        return 1

    available = sorted({int(n.split(f"{prefix}.")[1].split(".")[0]) for n in names if _is_layer(n, prefix)})
    layers = layers or _default_layers(available)
    hidden = getattr(config, "hidden_size", None) or getattr(
        getattr(config, "text_config", None), "hidden_size", None
    )

    print(f"# {checkpoint}: {len(available)} layers, hidden_size={hidden}")
    print("# paste under `hooks:` in configs/probes/<model>.yaml\n")
    print("  hooks:")
    for layer in layers:
        print(f"    resid_post.layer{layer}:")
        print(f"      module: {prefix}.{layer}")
        print("      mode: forward")
        print(f"      hidden_size: {hidden}")
    for layer in layers[-2:]:
        if f"{prefix}.{layer}.self_attn.o_proj" in names:
            print(f"    attn_out.layer{layer}:")
            print(f"      module: {prefix}.{layer}.self_attn.o_proj")
            print("      mode: forward_pre   # reads o_proj's input")
            print("      hidden_size: null   # num_heads * head_dim; check the model config")
    for layer in layers[-2:]:
        mlp = _mlp_module(names, prefix, layer)
        if mlp:
            print(f"    mlp_out.layer{layer}:")
            print(f"      module: {mlp}")
            print("      mode: forward")
            print(f"      hidden_size: {hidden}")
    return 0


def _is_layer(name: str, prefix: str) -> bool:
    tail = name.split(f"{prefix}.", 1)
    return len(tail) == 2 and tail[1].split(".")[0].isdigit()


def _layer_prefix(names: list[str]) -> str | None:
    """Find the '<...>.layers' path, preferring the language tower on VLMs."""
    candidates = sorted({n.rsplit(".", 1)[0] for n in names if n.rsplit(".", 1)[-1].isdigit()})
    layer_paths = [c for c in candidates if c.endswith("layers")]
    if not layer_paths:
        return None
    for path in layer_paths:
        if "language_model" in path:
            return path
    return min(layer_paths, key=len)


def _mlp_module(names: list[str], prefix: str, layer: int) -> str | None:
    # post_feedforward_layernorm is the right read point where it exists (Gemma 3);
    # otherwise the mlp module output is the equivalent site.
    for suffix in ("post_feedforward_layernorm", "mlp"):
        candidate = f"{prefix}.{layer}.{suffix}"
        if candidate in names:
            return candidate
    return None


def _default_layers(available: list[int]) -> list[int]:
    """Four layers spread through the stack: early, two middle, late."""
    if not available:
        return []
    n = len(available)
    return sorted({available[i * (n - 1) // 4] for i in range(1, 4)} | {available[-1]})


def _report_transformers_import_failure(exc: Exception) -> None:
    """Explain an import failure in terms the caller can act on.

    A binary-incompatibility ValueError here is almost always a stale
    scikit-learn compiled against NumPy 1.x: transformers imports
    sklearn.metrics.roc_curve from its generation utilities, so the failure
    surfaces as a torch/transformers problem rather than an sklearn one.
    """
    message = str(exc)
    if "numpy.dtype size changed" in message or "binary incompatibility" in message:
        print(
            "transformers could not be imported because a compiled dependency was\n"
            "built against a different NumPy. This is an environment problem, not a\n"
            "problem with the probe config. Usually:\n"
            "\n"
            "    pip install -U scikit-learn\n"
            "\n"
            f"underlying error: {exc}",
            file=sys.stderr,
        )
        return
    print(
        f"this command needs the probe extras: pip install -e '.[probes]'\n"
        f"underlying error: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )


def cmd_check(path: Path) -> int:
    spec = ProbeSpec.load(path)
    try:
        from .common import ActivationRecorder, load_model
    except (ImportError, ValueError) as exc:
        _report_transformers_import_failure(exc)
        return 1

    print(f"loading {spec.checkpoint} ...")
    model, _tokenizer, device = load_model(spec)
    print(f"loaded on {device}")

    module_map = dict(model.named_modules())
    ok = True
    for hook in spec.hooks:
        present = hook.module in module_map
        ok &= present
        print(f"  {'ok  ' if present else 'MISS'} {hook.label:<22} {hook.module}")
    if ok:
        with ActivationRecorder(model, spec.hooks):
            print("\nall hooks registered cleanly")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return cmd_list()
        if args.command == "show":
            return cmd_show(args.config)
        if args.command == "inspect":
            return cmd_inspect(args.checkpoint, args.layers, args.all_modules, args.loader)
        if args.command == "check":
            return cmd_check(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
