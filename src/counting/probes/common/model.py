"""Loading a checkpoint and reading activations out of it.

`torch` and `transformers` are imported lazily so the rest of the package stays
usable on a machine with neither installed -- only the probe path needs them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

from .spec import HookSpec, ProbeSpec

if TYPE_CHECKING:  # pragma: no cover
    import torch


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str) -> "torch.dtype":
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def load_model(spec: ProbeSpec, device: str | None = None) -> tuple[Any, Any, str]:
    """Return (model, tokenizer_or_processor, device) for a probe spec."""
    import torch  # noqa: F401 - imported for side effects on device resolution

    device = device or resolve_device(spec.device)
    kwargs: dict[str, Any] = {"dtype": resolve_dtype(spec.dtype)}
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation

    if spec.loader == "mlx":
        from .mlx_backend import load_mlx

        return load_mlx(spec.checkpoint)

    if spec.loader == "conditional_generation":
        # Covers every ForConditionalGeneration VLM in the slate (Gemma 3/4,
        # Muse-Glimmer) without hardcoding a class per family.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(spec.checkpoint, **kwargs)
        tokenizer = AutoProcessor.from_pretrained(spec.checkpoint)
    elif spec.loader == "causal_lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(spec.checkpoint, **kwargs)
        tokenizer = (
            AutoTokenizer.from_pretrained(spec.checkpoint)
            if spec.tokenizer == "auto"
            else _load_processor(spec.checkpoint)
        )
    else:
        raise ValueError(
            f"unknown loader {spec.loader!r}; expected 'causal_lm', 'conditional_generation', or 'mlx'"
        )

    return model.eval().to(device), tokenizer, device


def _load_processor(checkpoint: str) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(checkpoint)


class ActivationRecorder:
    """Captures activations at the sites a ProbeSpec names.

    Use as a context manager; hooks are removed on exit even if the forward pass
    raises, which matters because a leaked hook silently corrupts every later run
    in the same process.

        with ActivationRecorder(model, spec.residual_hooks) as rec:
            model(**batch)
            resid = rec.cache["resid_post.layer31"]
    """

    def __init__(self, model: Any, hooks: Iterable[HookSpec]) -> None:
        self.module_map = dict(model.named_modules())
        self.hooks = list(hooks)
        self.cache: dict[str, "torch.Tensor"] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationRecorder":
        self.register()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- hook plumbing -----------------------------------------------------

    @staticmethod
    def _tensor(value: object) -> "torch.Tensor":
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"could not extract a tensor from hook value of type {type(value)!r}")

    def _forward_hook(self, label: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: tuple, output: object) -> None:
            self.cache[label] = self._tensor(output)

        return hook

    def _forward_pre_hook(self, label: str) -> Callable[..., None]:
        def hook(_module: Any, inputs: tuple) -> None:
            self.cache[label] = self._tensor(inputs)

        return hook

    def register(self) -> None:
        for spec in self.hooks:
            if spec.module not in self.module_map:
                raise KeyError(
                    f"module {spec.module!r} for hook {spec.label!r} is not in this model; "
                    "run `counting-probe inspect` to list the available module paths"
                )
            module = self.module_map[spec.module]
            if spec.mode == "forward_pre":
                handle = module.register_forward_pre_hook(self._forward_pre_hook(spec.label))
            else:
                handle = module.register_forward_hook(self._forward_hook(spec.label))
            self._handles.append(handle)

    def clear(self) -> None:
        self.cache.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.cache.clear()


class ActivationEditor:
    """Rewrites activations mid-forward-pass -- steering, patching, clamping.

    An edit is `fn(tensor) -> tensor` installed at one hook site. This is the
    single primitive behind all four intervention profiles in the paper:

        steering        add alpha * direction at the final prompt token
        donor patch     replace states with a donor prompt's states
        linear patch    translate states along a fitted direction
        clamp           force the projection onto a direction to a target value

    Only `forward` hooks can be edited: a forward-pre edit would have to rewrite
    the module's inputs, which transformers does not let a pre-hook do portably.
    """

    def __init__(self, model: Any) -> None:
        self.module_map = dict(model.named_modules())
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationEditor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def edit(self, spec: HookSpec, fn: Callable[["torch.Tensor"], "torch.Tensor"]) -> None:
        if spec.mode != "forward":
            raise ValueError(
                f"hook {spec.label!r} is a {spec.mode} site and cannot be edited; "
                "edit a forward site such as a resid_post hook instead"
            )
        if spec.module not in self.module_map:
            raise KeyError(f"module {spec.module!r} for hook {spec.label!r} is not in this model")

        def hook(_module: Any, _inputs: tuple, output: object) -> object:
            import torch

            if isinstance(output, torch.Tensor):
                return fn(output)
            if isinstance(output, tuple):
                head, *rest = output
                return (fn(head), *rest)
            return output

        self._handles.append(self.module_map[spec.module].register_forward_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
