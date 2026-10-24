# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bridge Megatron's Linear CE contract to the FlashTrain Torch adapter."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


class HcuLinearCeExtensionError(RuntimeError):
    """Raised when the native runtime is unavailable or incomplete."""


_ARCH_RE = re.compile(r"^gfx[0-9]+$")
_BACKEND_ENV = "HCU_LINEAR_CE_BACKEND"
_LOADED_PATHS: set[Path] = set()
_FLASH_TRAIN_BINDING: Any | None = None
# The legacy custom-op loader still validates this set when an explicit
# torch_module is supplied by compatibility tests or older tooling.
_REQUIRED_OPS = ("forward", "backward", "forward_partials", "forward_finalize")


def _get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise HcuLinearCeExtensionError(
            "PyTorch is required to load the HCU Linear CE extension"
        ) from exc
    return torch


def _validate_arch(arch: str) -> str:
    normalized = str(arch).lower().split(":", 1)[0]
    if not _ARCH_RE.fullmatch(normalized):
        raise HcuLinearCeExtensionError(f"invalid HCU architecture: {arch!r}")
    return normalized


def _select_backend(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    backend = values.get(_BACKEND_ENV, "flashtrain").strip().lower()
    if backend in {"flashtrain", "flash_train"}:
        return "flashtrain"
    if backend == "legacy":
        return "legacy"
    raise HcuLinearCeExtensionError(
        f"invalid {_BACKEND_ENV}={backend!r}; expected 'flashtrain' or 'legacy'"
    )


def default_extension_path(
    arch: str, environ: Mapping[str, str] | None = None
) -> Path:
    values = os.environ if environ is None else environ
    override = values.get("HCU_LINEAR_CE_EXTENSION_PATH")
    if override:
        return Path(override).expanduser()

    normalized_arch = _validate_arch(arch)
    library_name = f"libhcu_linear_ce_{normalized_arch}.so"
    try:
        from hcu_linear_ce_artifacts import paths as artifact_paths
    except ImportError:
        return Path(__file__).with_name("lib") / library_name

    try:
        return Path(artifact_paths.extension_path(normalized_arch))
    except TypeError as exc:
        raise HcuLinearCeExtensionError(
            "the installed HCU Linear CE artifact package is obsolete; "
            "install an architecture-aware artifact wheel"
        ) from exc


def _registered_ops(torch_module: Any) -> Any | None:
    namespace = getattr(getattr(torch_module, "ops", None), "hcu_linear_ce", None)
    if namespace is None:
        return None
    if any(getattr(namespace, name, None) is None for name in _REQUIRED_OPS):
        return None
    return namespace


def require_ops(
    arch: str,
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    # The production path uses the official flash_train.torch adapter. Keep
    # the explicit torch_module path for loader-only compatibility tests and
    # older callers that still exercise the legacy custom-op loader.
    if torch_module is None and _select_backend(environ) == "flashtrain":
        global _FLASH_TRAIN_BINDING
        if _FLASH_TRAIN_BINDING is None:
            try:
                import flash_train.torch as ft_binding
            except ImportError as exc:
                raise HcuLinearCeExtensionError(
                    "flash_train.torch is required for the HCU Linear CE backend; "
                    "install flash-train with FTRAIN_BUILD_TORCH_BINDINGS=ON"
                ) from exc
            missing = [
                name
                for name in (
                    "linear_ce_forward_partials",
                    "linear_ce_forward_finalize",
                    "linear_ce_backward",
                )
                if not hasattr(ft_binding, name)
            ]
            if missing:
                raise HcuLinearCeExtensionError(
                    "flash_train.torch is missing Linear CE bindings: "
                    + ", ".join(missing)
                )
            _FLASH_TRAIN_BINDING = ft_binding
        return _FLASH_TRAIN_BINDING

    torch = _get_torch() if torch_module is None else torch_module
    registered = _registered_ops(torch)
    if registered is not None:
        return registered

    library_path = default_extension_path(arch, environ=environ).resolve()
    if not library_path.is_file():
        raise HcuLinearCeExtensionError(
            "HCU Linear CE extension is not installed or loaded. Build the native bridge, "
            "set HCU_LINEAR_CE_EXTENSION_PATH to the architecture-specific shared library, "
            f"and load it with torch.ops.load_library(). Expected path: {library_path}"
        )
    try:
        if library_path not in _LOADED_PATHS:
            torch.ops.load_library(str(library_path))
            _LOADED_PATHS.add(library_path)
    except (OSError, RuntimeError) as exc:
        raise HcuLinearCeExtensionError(
            f"failed to load HCU Linear CE extension {library_path}: {exc}"
        ) from exc

    registered = _registered_ops(torch)
    if registered is None:
        raise HcuLinearCeExtensionError(
            "loaded HCU Linear CE extension does not register "
            + ", ".join(f"torch.ops.hcu_linear_ce.{name}" for name in _REQUIRED_OPS)
        )
    return registered


def invoke_forward(
    hidden: Any,
    weight: Any,
    labels: Any,
    ignore_index: int,
    log_kernel: bool,
    arch: str,
) -> tuple[Any, Any, Any, Any]:
    binding = require_ops(arch)
    if hasattr(binding, "linear_ce_forward_partials"):
        if ignore_index != -100:
            raise NotImplementedError(
                "flash_train Linear CE supports ignore_index=-100 only"
            )
        lse_local, target_logit_local = binding.linear_ce_forward_partials(
            hidden, weight, labels, 0
        )
        outputs = binding.linear_ce_forward_finalize(
            lse_local, target_logit_local, labels
        )
    else:
        outputs = binding.forward(hidden, weight, labels, ignore_index, log_kernel)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise HcuLinearCeExtensionError(
            "Linear CE forward must return "
            "(loss, maximum, acc, num_valid_tokens)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def invoke_forward_partials(
    hidden: Any,
    weight: Any,
    labels: Any,
    shard_vocab_start: int,
    ignore_index: int,
    log_kernel: bool,
    arch: str,
) -> tuple[Any, Any]:
    """Return this rank's ``(lse_local, target_logit_local)`` over its shard."""
    binding = require_ops(arch)
    if hasattr(binding, "linear_ce_forward_partials"):
        if ignore_index != -100:
            raise NotImplementedError(
                "flash_train Linear CE supports ignore_index=-100 only"
            )
        outputs = binding.linear_ce_forward_partials(
            hidden, weight, labels, shard_vocab_start
        )
    else:
        outputs = binding.forward_partials(
            hidden, weight, labels, shard_vocab_start, ignore_index, log_kernel
        )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise HcuLinearCeExtensionError(
            "Linear CE forward_partials must return "
            "(lse_local, target_logit_local)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def invoke_forward_finalize(
    lse: Any,
    target_logit: Any,
    labels: Any,
    ignore_index: int,
    arch: str,
) -> tuple[Any, Any, Any, Any]:
    """Turn cross-rank-combined partials into the loss and saved tensors."""
    binding = require_ops(arch)
    if hasattr(binding, "linear_ce_forward_finalize"):
        if ignore_index != -100:
            raise NotImplementedError(
                "flash_train Linear CE supports ignore_index=-100 only"
            )
        outputs = binding.linear_ce_forward_finalize(lse, target_logit, labels)
    else:
        outputs = binding.forward_finalize(lse, target_logit, labels, ignore_index)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise HcuLinearCeExtensionError(
            "Linear CE forward_finalize must return "
            "(loss, maximum, acc, num_valid_tokens)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def invoke_backward(
    dlogprobs: Any,
    global_hidden: Any,
    weight: Any,
    labels: Any,
    maximum: Any,
    acc: Any,
    num_valid_tokens: Any,
    log_kernel: bool,
    arch: str,
    shard_vocab_start: int = 0,
    main_grad: Any = None,
) -> tuple[Any, Any]:
    # `main_grad` is the accumulation buffer of `weight`, in fp32 or bf16 —
    # the two element types Megatron can produce for this operator and the two
    # vocab_output.py accumulates into, through wgrad_gemm_accum_fp32 and
    # wgrad_gemm_accum_fp16 respectively. Its dtype is `grad_dtype`, which is
    # fp32 when gradients are reduced in fp32 (bf16 training auto-enables that)
    # and otherwise `param.dtype`, always bf16 here because the forward requires
    # bf16 hidden and weight. Anything else is rejected before this call.
    # When it is given, the native dW GEMM accumulates into it with beta=1 using
    # the buffer's own element type as its C matrix type, and never writes the
    # bf16 d_weight, so the second element of the returned pair is an
    # uninitialised full-shape placeholder. It keeps the [local_vocab, D] shape
    # because autograd checks the tangent returned for `weight` against the
    # forward input's shape; the arity stays at two so the autograd Function and
    # the PR2256-shaped contract are unaffected. Only DDP reads the flag that
    # suppresses the second accumulation, never the contents.
    binding = require_ops(arch)
    if hasattr(binding, "linear_ce_backward"):
        outputs = binding.linear_ce_backward(
            dlogprobs,
            global_hidden,
            weight,
            labels,
            maximum,
            acc,
            num_valid_tokens,
            shard_vocab_start,
            main_grad,
        )
    else:
        outputs = binding.backward(
            dlogprobs,
            global_hidden,
            weight,
            labels,
            maximum,
            acc,
            num_valid_tokens,
            log_kernel,
            shard_vocab_start,
            main_grad,
        )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise HcuLinearCeExtensionError(
            "Linear CE backward must return (d_hidden, d_weight)"
        )
    return tuple(outputs)  # type: ignore[return-value]


def reset_loader_state() -> None:
    _LOADED_PATHS.clear()
    global _FLASH_TRAIN_BINDING
    _FLASH_TRAIN_BINDING = None


__all__ = [
    "HcuLinearCeExtensionError",
    "default_extension_path",
    "invoke_backward",
    "invoke_forward",
    "invoke_forward_finalize",
    "invoke_forward_partials",
    "require_ops",
    "reset_loader_state",
]
