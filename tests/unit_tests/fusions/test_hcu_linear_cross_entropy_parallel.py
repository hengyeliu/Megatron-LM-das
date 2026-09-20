# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Multi-HCU equivalence tests for tensor-parallel fused Linear CE.

Run under torchrun, one process per HCU::

    torchrun --nproc_per_node=2 \\
        tests/unit_tests/fusions/test_hcu_linear_cross_entropy_parallel.py

The reference is the non-fused path: one dense ``hidden @ weight.T`` over the
whole vocabulary in fp32 with ``torch.nn.functional.cross_entropy`` on top,
differentiated by autograd. Every rank builds the same reference from the same
seed, so the comparison is against unsharded arithmetic, not against another
rank.

A second check reproduces the upstream PR2256 cross-rank reduction --- an
all-reduce of the running maximum, a rescale, then an all-reduce of the
exponential sum and of the target logit --- and compares it per token against
the single all_gather this backend uses instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATOR_ROOT = REPO_ROOT / "hcu_megatron" / "core" / "fusions" / "linear_cross_entropy"
PACKAGE_NAME = "_hcu_linear_ce_parallel_under_test"


def _load_operator_package() -> tuple[object, object]:
    """Import the backend package directly, without requiring Megatron."""
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(OPERATOR_ROOT)]
    sys.modules[PACKAGE_NAME] = package
    modules = {}
    for name in ("platform", "extension", "entry"):
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE_NAME}.{name}", OPERATOR_ROOT / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{PACKAGE_NAME}.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["entry"], modules["extension"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--v", type=int, default=32768, help="global vocabulary size")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ignore-fraction", type=float, default=0.1)
    parser.add_argument("--loss-tolerance", type=float, default=5e-3)
    parser.add_argument("--grad-tolerance", type=float, default=2e-2)
    return parser.parse_args()


def make_inputs(args: argparse.Namespace, device: torch.device):
    """Identical on every rank: the TP contract says hidden and labels match."""
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden = torch.randn(
        (args.n, args.d), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    weight = torch.randn(
        (args.v, args.d), generator=generator, device=device, dtype=torch.float32
    ).mul_(0.02).to(torch.bfloat16)
    labels = torch.randint(
        0, args.v, (args.n,), generator=generator, device=device, dtype=torch.long
    )
    ignored = torch.rand((args.n,), generator=generator, device=device)
    labels = torch.where(ignored < args.ignore_fraction, torch.full_like(labels, -100), labels)
    return hidden, weight, labels


def reference(hidden, weight, labels):
    hidden_ref = hidden.float().detach().requires_grad_(True)
    weight_ref = weight.float().detach().requires_grad_(True)
    logits = hidden_ref @ weight_ref.transpose(0, 1)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
    loss.backward()
    return loss.detach(), hidden_ref.grad, weight_ref.grad


def upstream_style_combination(lse_local, target_logit_local, group):
    """The three-collective reduction PR2256 performs, for comparison only."""
    maximum = lse_local.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    # lse_local = max_local + log(sum_local); rescale the local sum onto the
    # group-wide maximum before adding the sums together.
    accumulator = torch.exp(lse_local - maximum)
    dist.all_reduce(accumulator, op=dist.ReduceOp.SUM, group=group)
    target_logit = target_logit_local.clone()
    dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=group)
    return maximum + torch.log(accumulator), target_logit


def report(rank: int, name: str, got, want, tolerance: float) -> bool:
    difference = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1e-6)
    relative = difference / scale
    ok = relative <= tolerance
    if rank == 0:
        print(
            f"{'TP_PASS' if ok else 'TP_FAIL'} tensor={name} max_abs={difference:.6g} "
            f"relative={relative:.6g} tolerance={tolerance}"
        )
    return ok


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    external = dist.is_initialized()
    if not external:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    entry, extension = _load_operator_package()

    if args.v % world_size != 0:
        raise SystemExit(f"global vocabulary {args.v} is not divisible by {world_size}")
    local_vocab = args.v // world_size

    hidden, weight, labels = make_inputs(args, device)
    weight_local = weight[rank * local_vocab : (rank + 1) * local_vocab].contiguous()
    group = dist.group.WORLD

    loss, maximum, acc, num_valid, tp_rank, tp_world, global_hidden = entry.forward(
        hidden, weight_local, labels, tp_group=group
    )
    dloss = torch.ones((), device=device, dtype=torch.float32)
    d_hidden, d_weight = entry.backward(
        dloss, global_hidden, weight_local, labels, maximum, acc, num_valid,
        "mean", -100, group, tp_rank, tp_world, False,
    )

    reference_loss, reference_dhidden, reference_dweight = reference(hidden, weight, labels)
    ok = True
    ok &= report(rank, "loss", loss, reference_loss, args.loss_tolerance)
    ok &= report(rank, "d_hidden", d_hidden, reference_dhidden, args.grad_tolerance)
    ok &= report(
        rank,
        "d_weight",
        d_weight,
        reference_dweight[rank * local_vocab : (rank + 1) * local_vocab],
        args.grad_tolerance,
    )

    # Every rank must agree on the loss: PR2256 requires identical logprobs
    # across the tensor-parallel group.
    gathered_loss = torch.empty((world_size,), device=device, dtype=torch.float32)
    dist.all_gather_into_tensor(gathered_loss, loss.reshape(1), group=group)
    agree = bool((gathered_loss == gathered_loss[0]).all().item())
    if rank == 0:
        print(f"{'TP_PASS' if agree else 'TP_FAIL'} tensor=loss_agreement_across_ranks")
    ok &= agree

    # Guardrail for the single-all_gather deviation.
    lse_local, target_logit_local = extension.invoke_forward_partials(
        hidden, weight_local, labels, rank * local_vocab, -100, False, "gfx936"
    )
    upstream_lse, upstream_target = upstream_style_combination(
        lse_local, target_logit_local, group
    )
    ours_lse, ours_target = entry._combine_partials(
        torch, group, world_size, lse_local, target_logit_local
    )
    ok &= report(rank, "lse_vs_three_allreduce", ours_lse, upstream_lse, 1e-5)
    ok &= report(rank, "target_logit_vs_three_allreduce", ours_target, upstream_target, 1e-6)

    # Sequence parallel: the same vocabulary shard, but this rank only supplies
    # (and only receives a gradient for) its slice of the tokens.
    if args.n % world_size != 0:
        raise SystemExit(f"token count {args.n} is not divisible by {world_size}")
    local_tokens = args.n // world_size
    token_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
    hidden_shard = hidden[token_slice].contiguous()
    (
        sp_loss, sp_maximum, sp_acc, sp_valid, sp_rank, sp_world, sp_global_hidden,
    ) = entry.forward(
        hidden_shard, weight_local, labels, tp_group=group, sequence_parallel=True
    )
    sp_dhidden, sp_dweight = entry.backward(
        dloss, sp_global_hidden, weight_local, labels, sp_maximum, sp_acc, sp_valid,
        "mean", -100, group, sp_rank, sp_world, True,
    )
    ok &= report(rank, "sp_loss", sp_loss, reference_loss, args.loss_tolerance)
    ok &= report(
        rank, "sp_d_hidden", sp_dhidden, reference_dhidden[token_slice], args.grad_tolerance
    )
    ok &= report(
        rank,
        "sp_d_weight",
        sp_dweight,
        reference_dweight[rank * local_vocab : (rank + 1) * local_vocab],
        args.grad_tolerance,
    )
    if tuple(sp_dhidden.shape) != tuple(hidden_shard.shape):
        if rank == 0:
            print(
                f"TP_FAIL tensor=sp_d_hidden_shape got={tuple(sp_dhidden.shape)} "
                f"want={tuple(hidden_shard.shape)}"
            )
        ok = False

    # Every token ignored: the group must agree that there is no loss to take,
    # and no rank may produce a gradient out of an empty batch.
    ignored_labels = torch.full_like(labels, -100)
    empty_loss, empty_max, empty_acc, empty_valid, _, _, _ = entry.forward(
        hidden, weight_local, ignored_labels, tp_group=group
    )
    empty_dhidden, empty_dweight = entry.backward(
        dloss, hidden, weight_local, ignored_labels, empty_max, empty_acc, empty_valid,
        "mean", -100, group, rank, world_size, False,
    )
    all_ignore_ok = (
        bool(torch.isnan(empty_loss).item())
        and int(empty_valid.item()) == 0
        and not bool(empty_dhidden.any().item())
        and not bool(empty_dweight.any().item())
    )
    gathered_flag = torch.empty((world_size,), device=device, dtype=torch.int64)
    dist.all_gather_into_tensor(
        gathered_flag,
        torch.tensor([1 if all_ignore_ok else 0], device=device, dtype=torch.int64),
        group=group,
    )
    all_ignore_ok = bool((gathered_flag == 1).all().item())
    if rank == 0:
        print(
            f"{'TP_PASS' if all_ignore_ok else 'TP_FAIL'} "
            "tensor=all_ignore_agrees_across_ranks"
        )
    ok &= all_ignore_ok

    verdict = torch.tensor([1 if ok else 0], device=device, dtype=torch.int64)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        print(
            f"{'TP_EQUIVALENCE_PASS' if verdict.item() else 'TP_EQUIVALENCE_FAIL'} "
            f"world_size={world_size} n={args.n} d={args.d} v={args.v} "
            f"local_vocab={local_vocab}"
        )
    if not external:
        dist.destroy_process_group()
    return 0 if verdict.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
