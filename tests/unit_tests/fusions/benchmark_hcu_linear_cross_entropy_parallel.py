# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Multi-HCU timing and communication accounting for tensor-parallel Linear CE.

Answers three questions from the TP/DP/SP plan:

* C7 -- how long the first fused iteration takes. The backward autotunes on the
  first call for a given shape, which makes one rank a straggler while the rest
  of the group waits in the next collective. This reports the first-iteration
  cost next to the steady-state cost so it can be compared against the RCCL
  watchdog timeout.
* C8 -- how many collectives each path issues. The fused path must not add
  communication: it should trade the non-fused path's three cross-entropy
  all-reduces for one all_gather, and keep the same single hidden-gradient
  all-reduce.
* Phase D input -- steady-state fused latency at the shard size this group
  produces, so the decision to request new rocBLAS forward sources is made on
  measurement rather than on the shape alone.

The baseline is the non-fused vocabulary-parallel path written out in torch:
a local ``hidden @ weight.T`` followed by the max/sum/target reduction that
``vocab_parallel_cross_entropy`` performs, differentiated by autograd.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATOR_ROOT = REPO_ROOT / "hcu_megatron" / "core" / "fusions" / "linear_cross_entropy"
PACKAGE_NAME = "_hcu_linear_ce_benchmark"


def _load_entry():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(OPERATOR_ROOT)]
    sys.modules[PACKAGE_NAME] = package
    module = None
    for name in ("platform", "extension", "entry"):
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE_NAME}.{name}", OPERATOR_ROOT / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{PACKAGE_NAME}.{name}"] = module
        spec.loader.exec_module(module)
    return module


class CollectiveCounter:
    """Count collectives by wrapping the entry points the paths actually call."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._originals: dict[str, object] = {}

    def __enter__(self) -> "CollectiveCounter":
        for name in ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor"):
            original = getattr(dist, name)
            self._originals[name] = original

            def wrapper(*args, __name=name, __original=original, **kwargs):
                self.counts[__name] = self.counts.get(__name, 0) + 1
                return __original(*args, **kwargs)

            setattr(dist, name, wrapper)
        return self

    def __exit__(self, *exc_info) -> None:
        for name, original in self._originals.items():
            setattr(dist, name, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--v", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def baseline_step(hidden, weight_local, labels, vocab_start, vocab_end, group):
    """vocab_parallel_cross_entropy over a local ColumnParallelLinear output."""
    hidden_grad_source = hidden.detach().requires_grad_(True)
    weight_grad_source = weight_local.detach().requires_grad_(True)
    logits = (hidden_grad_source @ weight_grad_source.transpose(0, 1)).float()
    maximum = logits.max(dim=-1).values
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    shifted = logits - maximum.unsqueeze(1)
    outside = (labels < vocab_start) | (labels >= vocab_end)
    local_labels = torch.where(outside, torch.zeros_like(labels), labels - vocab_start)
    target = shifted.gather(1, local_labels.unsqueeze(1)).squeeze(1)
    target = torch.where(outside, torch.zeros_like(target), target)
    dist.all_reduce(target, op=dist.ReduceOp.SUM, group=group)
    total = shifted.exp().sum(dim=-1)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    per_token = torch.log(total) - target
    valid = labels != -100
    loss = (per_token * valid).sum() / valid.sum()
    loss.backward()
    d_hidden = hidden_grad_source.grad
    # The column-parallel linear owns this reduction in the non-fused path.
    dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=group)
    return loss.detach(), d_hidden, weight_grad_source.grad


def timed(function, group) -> float:
    dist.barrier(group=group)
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    group = dist.group.WORLD
    entry = _load_entry()

    local_vocab = args.v // world_size
    vocab_start = rank * local_vocab
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden = torch.randn(
        (args.n, args.d), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    weight_local = torch.randn(
        (local_vocab, args.d), generator=generator, device=device, dtype=torch.float32
    ).mul_(0.02).to(torch.bfloat16)
    labels = torch.randint(
        0, args.v, (args.n,), generator=generator, device=device, dtype=torch.long
    )
    dloss = torch.ones((), device=device, dtype=torch.float32)

    def fused_step():
        loss, maximum, acc, valid, tp_rank, tp_world, global_hidden = entry.forward(
            hidden, weight_local, labels, tp_group=group
        )
        entry.backward(
            dloss, global_hidden, weight_local, labels, maximum, acc, valid,
            "mean", -100, group, tp_rank, tp_world, False,
        )

    # C7: the first iteration carries the backward autotune.
    first = timed(fused_step, group)
    first_across_ranks = torch.tensor([first], device=device, dtype=torch.float64)
    dist.all_reduce(first_across_ranks, op=dist.ReduceOp.MAX, group=group)

    for _ in range(args.warmup):
        fused_step()
    fused_times = [timed(fused_step, group) for _ in range(args.iterations)]

    def baseline():
        baseline_step(hidden, weight_local, labels, vocab_start, vocab_start + local_vocab, group)

    for _ in range(args.warmup):
        baseline()
    baseline_times = [timed(baseline, group) for _ in range(args.iterations)]

    # C8: count the collectives one iteration of each path issues.
    with CollectiveCounter() as fused_counter:
        fused_step()
    with CollectiveCounter() as baseline_counter:
        baseline()

    torch.cuda.reset_peak_memory_stats()
    fused_step()
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats()
    baseline()
    torch.cuda.synchronize()
    baseline_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if rank == 0:
        fused_median = statistics.median(fused_times)
        baseline_median = statistics.median(baseline_times)
        print(
            f"TP_TIMING world_size={world_size} n={args.n} d={args.d} v={args.v} "
            f"local_vocab={local_vocab}"
        )
        print(
            f"TP_TIMING first_iteration_ms={first_across_ranks.item():.3f} "
            f"(max across ranks, includes backward autotune)"
        )
        print(
            f"TP_TIMING fused_median_ms={fused_median:.3f} "
            f"baseline_median_ms={baseline_median:.3f} "
            f"speedup={baseline_median / fused_median:.3f}x"
        )
        print(
            f"TP_TIMING fused_peak_mib={fused_peak:.1f} "
            f"baseline_peak_mib={baseline_peak:.1f}"
        )
        print(f"TP_COLLECTIVES fused={fused_counter.counts}")
        print(f"TP_COLLECTIVES baseline={baseline_counter.counts}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
