#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Benchmark CuTeDSL bwd_dhu against C++ CUTLASS/CuTe and FLA Triton.

Scope:
  - fixed-length only
  - SM90, K=V=128, chunk_size=64
  - no g/gk
  - optional h0/dht enabled by default

Example:
  python benchmarks/bench_chunk_delta_h_bwd_dsl.py --B 1 --H 64 --T 4096 16384
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu as cpp_bwd_dhu
from cula.ops.chunk_delta_h_bwd_dsl import _select_bv
from cula.ops.chunk_delta_h_bwd_dsl import chunk_gated_delta_rule_bwd_dhu as dsl_bwd_dhu

BT = 64
K = V = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"
WARMUP = 10
N_ITERS = 100
NCU_MODE = False


def time_kernel(fn, warmup: int | None = None, iters: int | None = None) -> float:
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if iters is None:
        iters = 1 if NCU_MODE else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def accuracy_stats(ref: torch.Tensor, out: torch.Tensor) -> tuple[float, float]:
    diff = (ref.float() - out.float()).abs()
    return diff.max().item(), diff.mean().item()


def make_inputs(B: int, T: int, H: int, use_h0: bool, use_dht: bool, seed: int):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()
    k = (torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()
    w = (torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()
    do = (torch.randn(B, T, H, V, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()
    dv = (torch.randn(B, T, H, V, device=DEVICE, dtype=DTYPE) * 0.1).contiguous()
    h0 = (torch.randn(B, H, K, V, device=DEVICE, dtype=torch.float32) * 0.01).contiguous() if use_h0 else None
    dht = (torch.randn(B, H, K, V, device=DEVICE, dtype=torch.float32) * 0.1).contiguous() if use_dht else None
    return q, k, w, do, dv, h0, dht


def run_one(args, T: int):
    torch.cuda.empty_cache()
    q, k, w, do, dv, h0, dht = make_inputs(args.B, T, args.H, args.h0, args.dht, args.seed)
    common = dict(q=q, k=k, w=w, do=do, dv=dv, h0=h0, dht=dht, scale=args.scale, chunk_size=BT)

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(**common)
    cpp_dh, cpp_dh0, cpp_dv2 = cpp_bwd_dhu(**common)
    dsl_dh, dsl_dh0, dsl_dv2 = dsl_bwd_dhu(**common)
    torch.cuda.synchronize()

    dh_max, dh_mean = accuracy_stats(ref_dh, dsl_dh)
    dv_max, dv_mean = accuracy_stats(ref_dv2, dsl_dv2)
    if h0 is not None:
        dh0_max, dh0_mean = accuracy_stats(ref_dh0, dsl_dh0)
    else:
        dh0_max = dh0_mean = 0.0
    cpp_max = max(
        accuracy_stats(cpp_dh, dsl_dh)[0],
        accuracy_stats(cpp_dv2, dsl_dv2)[0],
        accuracy_stats(cpp_dh0, dsl_dh0)[0] if h0 is not None else 0.0,
    )

    fla_ms = time_kernel(lambda: fla_bwd_dhu(**common), args.warmup, args.iters)
    cpp_ms = time_kernel(lambda: cpp_bwd_dhu(**common), args.warmup, args.iters)
    dsl_ms = time_kernel(lambda: dsl_bwd_dhu(**common), args.warmup, args.iters)

    return {
        "T": T,
        "BV": _select_bv(args.B, args.H, V, torch.cuda.get_device_properties(DEVICE).multi_processor_count),
        "fla_ms": fla_ms,
        "cpp_ms": cpp_ms,
        "dsl_ms": dsl_ms,
        "dsl_vs_cpp": cpp_ms / dsl_ms if dsl_ms > 0 else float("inf"),
        "dh_max": dh_max,
        "dh_mean": dh_mean,
        "dv_max": dv_max,
        "dv_mean": dv_mean,
        "dh0_max": dh0_max,
        "dh0_mean": dh0_mean,
        "cpp_max": cpp_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--T", type=int, nargs="+", default=[4096, 4096 * 4])
    parser.add_argument("--scale", type=float, default=0.125)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-h0", dest="h0", action="store_false")
    parser.add_argument("--no-dht", dest="dht", action="store_false")
    parser.add_argument("--ncu", action="store_true")
    parser.set_defaults(h0=True, dht=True)
    args = parser.parse_args()

    if args.ncu:
        args.warmup = 1
        args.iters = 1
    global NCU_MODE
    NCU_MODE = args.ncu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("chunk_gated_delta_rule_bwd_dhu: CuTeDSL SM90 vs C++ CUTLASS/CuTe vs FLA Triton")
    print(f"B={args.B} H={args.H} K=V=128 h0={args.h0} dht={args.dht} warmup={args.warmup} iters={args.iters}")
    print(
        f"{'T':>8} {'BV':>4} {'dh max':>10} {'dv2 max':>10} {'dh0 max':>10} {'cpp max':>10} "
        f"{'FLA ms':>10} {'C++ ms':>10} {'DSL ms':>10} {'C++/DSL':>9}"
    )
    print("-" * 104)

    for T in args.T:
        if T % BT != 0:
            raise ValueError(f"T must be a multiple of 64, got {T}")
        r = run_one(args, T)
        print(
            f"{r['T']:8d} {r['BV']:4d} {r['dh_max']:10.4e} {r['dv_max']:10.4e} {r['dh0_max']:10.4e} "
            f"{r['cpp_max']:10.4e} {r['fla_ms']:10.4f} {r['cpp_ms']:10.4f} {r['dsl_ms']:10.4f} {r['dsl_vs_cpp']:9.3f}"
        )


if __name__ == "__main__":
    main()
