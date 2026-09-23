# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PR2256-compatible entry points for the HCU Linear CE native backend."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from . import extension
from .platform import require_hcu


def _get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required by the HCU Linear CE backend") from exc
    return torch


def _env_flag(name: str, environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = values.get(name, "0").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean value, got {values.get(name)!r}")


def _is_contiguous(tensor: Any) -> bool:
    return bool(tensor.is_contiguous())


def _get_distributed(torch: Any) -> Any:
    dist = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available():
        raise RuntimeError(
            "torch.distributed is required for tensor or sequence parallel Linear CE"
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "a tp_group was supplied but torch.distributed is not initialized"
        )
    return dist


def _group_rank_and_size(torch: Any, tp_group: Any) -> tuple[int, int]:
    """Return ``(rank, world_size)`` within ``tp_group``.

    ``tp_group is None`` is the PR2256 spelling of data parallel: rank 0 of a
    group of one, which is also the shape the native single-rank path expects.
    """
    if tp_group is None:
        return 0, 1
    dist = _get_distributed(torch)
    return int(dist.get_rank(group=tp_group)), int(dist.get_world_size(group=tp_group))


# One-off fail-closed checks are cached per (group, shape): they cost a
# collective, and repeating them every microbatch would put a synchronisation
# point in the steady-state training loop for no new information.
_VALIDATED_SHARDINGS: set[tuple[int, int, int, int]] = set()


def _validate_vocab_sharding(
    torch: Any, tp_group: Any, local_vocab: int, tp_world_size: int, num_tokens: int, dim: int
) -> None:
    """Reject shapes the vocabulary-parallel forward cannot serve.

    The fused forward is grafted onto an edge-free Tensile macro tile, so every
    extent it walks must be a multiple of 256. Under tensor parallelism the
    vocabulary extent it walks is the *local* shard, which is why the training
    job has to pad with ``--make-vocab-size-divisible-by 256``.
    """
    if local_vocab % 256 != 0:
        raise ValueError(
            f"local vocabulary size {local_vocab} is not a multiple of 256 "
            f"(tensor parallel size {tp_world_size}). The fused forward runs on an "
            "edge-free macro tile, so each rank's shard must be 256-aligned. Launch "
            "training with --make-vocab-size-divisible-by 256."
        )
    if num_tokens % 256 != 0:
        raise ValueError(
            f"token count {num_tokens} is not a multiple of 256, which the fused "
            "forward requires"
        )
    if dim % 256 != 0:
        raise ValueError(
            f"hidden dimension {dim} is not a multiple of 256, which the fused "
            "forward requires"
        )
    if tp_group is None or tp_world_size == 1:
        return

    key = (id(tp_group), local_vocab, tp_world_size, num_tokens)
    if key in _VALIDATED_SHARDINGS:
        return
    dist = _get_distributed(torch)
    device = torch.cuda.current_device()
    probe = torch.tensor([local_vocab, num_tokens], device=device, dtype=torch.int64)
    gathered = torch.empty(
        (tp_world_size, 2), device=device, dtype=torch.int64
    )
    dist.all_gather_into_tensor(gathered.view(-1), probe, group=tp_group)
    if not bool((gathered == probe.unsqueeze(0)).all().item()):
        raise ValueError(
            "every rank of the tensor parallel group must hold an equally sized "
            f"vocabulary shard and the same token count; this group reported "
            f"{gathered.tolist()}"
        )
    _VALIDATED_SHARDINGS.add(key)


def _combine_partials(
    torch: Any, tp_group: Any, tp_world_size: int, lse_local: Any, target_logit_local: Any
) -> tuple[Any, Any]:
    """Combine per-shard partials across the tensor parallel group.

    Upstream PR2256 spends three collectives here: an all-reduce of the running
    maximum, a rescale, then an all-reduce of the exponential sum and of the
    target logit. One all_gather of the already-folded
    ``lse = max + log(sum)`` carries the same information, and
    ``torch.logsumexp`` recombines it with its own max shift, so the result is
    the same max-shifted arithmetic in a single collective.
    """
    if tp_world_size == 1:
        # Not merely an optimisation: skipping the gather keeps the single-rank
        # result bit-identical to the data-parallel path instead of routing it
        # through a logsumexp over a length-one dimension.
        return lse_local, target_logit_local

    dist = _get_distributed(torch)
    num_tokens = lse_local.shape[0]
    local = torch.cat((lse_local, target_logit_local))
    gathered = torch.empty(
        (tp_world_size * local.shape[0],), device=local.device, dtype=local.dtype
    )
    dist.all_gather_into_tensor(gathered, local, group=tp_group)
    gathered = gathered.view(tp_world_size, 2 * num_tokens)
    lse = torch.logsumexp(gathered[:, :num_tokens], dim=0)
    # A rank that does not own a token's target column contributed a zero, so
    # the sum picks out the owner's logit.
    target_logit = gathered[:, num_tokens:].sum(dim=0)
    return lse.contiguous(), target_logit.contiguous()


def _validate_parallelism(tp_group: Any, sequence_parallel: bool) -> None:
    if sequence_parallel and tp_group is None:
        raise ValueError(
            "sequence_parallel=True requires a tp_group; PR2256 defines sequence "
            "parallelism as a variant of tensor parallelism"
        )


def _gather_sequence(
    torch: Any, tp_group: Any, tp_world_size: int, hidden: Any
) -> Any:
    """Concatenate the sequence-sharded hidden states along dim 0.

    Under sequence parallelism each rank holds ``N_global / tp_world_size``
    rows. The loss needs every token against the rank's vocabulary shard, so
    the hidden states are gathered once, before any computation, exactly where
    PR2256 places the gather. The gathered tensor is what the backward
    differentiates, which is why forward hands it back as ``global_hidden``.
    """
    if not _is_contiguous(hidden):
        raise ValueError("hidden must be contiguous for sequence parallelism")
    dist = _get_distributed(torch)
    shape = (int(hidden.shape[0]) * tp_world_size,) + tuple(
        int(extent) for extent in hidden.shape[1:]
    )
    gathered = torch.empty(shape, device=hidden.device, dtype=hidden.dtype)
    dist.all_gather_into_tensor(gathered, hidden, group=tp_group)
    return gathered


def _validate_and_flatten(hidden: Any, weight: Any, labels: Any, torch: Any) -> tuple[Any, Any]:
    if hidden.dim() not in (2, 3):
        raise ValueError(f"hidden must be 2D or 3D, got dim={hidden.dim()}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got dim={weight.dim()}")
    if labels.dim() != hidden.dim() - 1:
        raise ValueError(
            f"labels dim must be hidden.dim()-1, got hidden.dim={hidden.dim()}, "
            f"labels.dim={labels.dim()}"
        )
    if not all(_is_contiguous(tensor) for tensor in (hidden, weight, labels)):
        raise ValueError("hidden, weight, and labels must be contiguous")
    if not all(bool(getattr(tensor, "is_cuda", False)) for tensor in (hidden, weight, labels)):
        raise ValueError("hidden, weight, and labels must be HCU device tensors")
    if weight.device != hidden.device or labels.device != hidden.device:
        raise ValueError("hidden, weight, and labels must be on the same HCU device")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("hidden and weight must use BF16")
    if labels.dtype != torch.int64:
        raise ValueError("labels must use int64")
    if hidden.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"hidden D={hidden.shape[-1]} does not match weight D={weight.shape[1]}"
        )
    num_tokens = math.prod(hidden.shape[:-1])
    if labels.numel() != num_tokens:
        raise ValueError(
            f"labels contain {labels.numel()} elements but hidden has N={num_tokens} tokens"
        )
    return hidden.view(-1, hidden.shape[-1]), labels.view(-1)


def forward(
    hidden: Any,
    weight: Any,
    labels: Any,
    tp_group: Any = None,
    reduction: str = "mean",
    ignore_index: int = -100,
    sequence_parallel: bool = False,
) -> tuple[Any, Any, Any, Any, int, int, Any]:
    """Return the seven values consumed by PR2256 LinearCrossEntropy.forward."""
    _validate_parallelism(tp_group, sequence_parallel)
    if reduction != "mean" or ignore_index != -100:
        raise NotImplementedError(
            "HCU Linear CE backend supports reduction='mean' and ignore_index=-100 only"
        )
    torch = _get_torch()
    platform = require_hcu(torch_module=torch)
    tp_rank, tp_world_size = _group_rank_and_size(torch, tp_group)
    # Sequence parallelism is tensor parallelism with the tokens arriving in
    # shards; gathering them first makes the rest of the forward identical.
    global_hidden = (
        _gather_sequence(torch, tp_group, tp_world_size, hidden)
        if sequence_parallel
        else hidden
    )
    hidden_view, labels_view = _validate_and_flatten(
        global_hidden, weight, labels, torch
    )
    local_vocab = int(weight.shape[0])
    _validate_vocab_sharding(
        torch,
        tp_group,
        local_vocab,
        tp_world_size,
        int(hidden_view.shape[0]),
        int(hidden_view.shape[1]),
    )
    log_kernel = _env_flag("HCU_LINEAR_CE_LOG_KERNEL")
    # The shard this rank owns is [tp_rank * local_vocab, (tp_rank + 1) * local_vocab):
    # PR2256 hands every rank an equally sized contiguous slice of the vocabulary.
    shard_vocab_start = tp_rank * local_vocab
    lse_local, target_logit_local = extension.invoke_forward_partials(
        hidden_view,
        weight,
        labels_view,
        shard_vocab_start,
        ignore_index,
        log_kernel,
        platform.arch,
    )
    lse, target_logit = _combine_partials(
        torch, tp_group, tp_world_size, lse_local, target_logit_local
    )
    loss, maximum, acc, num_valid_tokens = extension.invoke_forward_finalize(
        lse, target_logit, labels_view, ignore_index, platform.arch
    )
    # Every rank sees the same labels and the same combined partials, so loss,
    # acc and num_valid_tokens are identical across the group, as PR2256 requires.
    return loss, maximum, acc, num_valid_tokens, tp_rank, tp_world_size, global_hidden


def backward(
    dlogprobs: Any,
    global_hidden: Any,
    weight: Any,
    labels: Any,
    maximum: Any,
    acc: Any,
    num_valid_tokens: Any,
    reduction: str,
    ignore_index: int,
    tp_group: Any,
    tp_rank: int,
    tp_world_size: int,
    sequence_parallel: bool,
    main_grad: Any = None,
) -> tuple[Any, Any]:
    """Return ``(d_hidden, d_weight)`` to PR2256 LinearCrossEntropy.backward.

    ``main_grad`` is ``weight.main_grad`` when the caller has decided that the
    dW GEMM may accumulate into it directly. In that case the second returned
    element is an uninitialised full-shape placeholder: the native kernel has
    already folded the gradient into ``main_grad`` with ``beta=1``, and DDP is
    told to skip its own ``main_grad.add_(grad)``. This is the same division of
    labour the non-fused ``vocab_output.py`` uses. The caller owns the decision
    — and the ``grad_added_to_main_grad`` flag that goes with it — because only
    it can see ``ctx``.
    """
    _validate_parallelism(tp_group, sequence_parallel)
    torch = _get_torch()
    expected_rank, expected_world_size = _group_rank_and_size(torch, tp_group)
    if tp_rank != expected_rank or tp_world_size != expected_world_size:
        raise ValueError(
            f"tp_rank/tp_world_size ({tp_rank}, {tp_world_size}) disagree with the "
            f"supplied group ({expected_rank}, {expected_world_size})"
        )
    if reduction != "mean" or ignore_index != -100:
        raise NotImplementedError(
            "HCU Linear CE backend supports reduction='mean' and ignore_index=-100 only"
        )
    platform = require_hcu(torch_module=torch)
    hidden_view, labels_view = _validate_and_flatten(global_hidden, weight, labels, torch)
    for name, tensor in (
        ("dlogprobs", dlogprobs),
        ("maximum", maximum),
        ("acc", acc),
        ("num_valid_tokens", num_valid_tokens),
    ):
        if not bool(getattr(tensor, "is_cuda", False)) or tensor.device != global_hidden.device:
            raise ValueError(f"{name} must be on the same HCU device as global_hidden")
        if not _is_contiguous(tensor):
            raise ValueError(f"{name} must be contiguous")
    num_tokens = int(hidden_view.shape[0])
    if dlogprobs.dim() != 0:
        raise ValueError("dlogprobs must be a scalar for reduction='mean'")
    if tuple(maximum.shape) != (num_tokens,) or tuple(acc.shape) != (num_tokens,):
        raise ValueError("maximum and acc must each have shape (N,)")
    if num_valid_tokens.dim() != 0 or num_valid_tokens.dtype != torch.int64:
        raise ValueError("num_valid_tokens must be a scalar int64 tensor")
    if main_grad is not None:
        if not bool(getattr(main_grad, "is_cuda", False)) or main_grad.device != weight.device:
            raise ValueError("main_grad must be on the same HCU device as weight")
        if main_grad.dtype != torch.float32:
            raise ValueError("main_grad must be float32")
        if tuple(main_grad.shape) != tuple(weight.shape):
            raise ValueError("main_grad must have the same shape as weight")
        if not _is_contiguous(main_grad):
            raise ValueError("main_grad must be contiguous")
    d_hidden, d_weight = extension.invoke_backward(
        dlogprobs,
        hidden_view,
        weight,
        labels_view,
        maximum,
        acc,
        num_valid_tokens,
        _env_flag("HCU_LINEAR_CE_LOG_KERNEL"),
        platform.arch,
        tp_rank * int(weight.shape[0]),
        main_grad,
    )
    if tp_world_size > 1:
        # Each rank differentiated only its own vocabulary slice, so the hidden
        # gradient is a partial sum. This is the same single collective the
        # non-fused column-parallel path performs, moved here unchanged, and it
        # sits outside the native pipeline on purpose: the backward autotuner
        # replays that pipeline once per candidate, and a collective inside it
        # would desynchronise the group.
        dist = _get_distributed(torch)
        dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=tp_group)
    d_hidden = d_hidden.view(*global_hidden.shape)
    if sequence_parallel:
        # The caller passed in a sequence shard and expects a gradient of the
        # same shape. PR2256 slices the all-reduced gradient rather than
        # replacing the all_reduce with a reduce_scatter, so the collective
        # count stays the same as the tensor-parallel path.
        rows = int(global_hidden.shape[0]) // tp_world_size
        d_hidden = d_hidden[tp_rank * rows : (tp_rank + 1) * rows].clone()
    # d_weight needs no communication at all: the rows this rank produced are
    # exactly the rows of the shard it owns.
    return d_hidden, d_weight


__all__ = ["backward", "forward"]
