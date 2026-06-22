#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark C++ bwd_dhu64 against FLA Triton.

Scope:
  - fixed-length only
  - K in {64, 128}, V=64, chunk_size=64
  - no g/gk
  - optional h0/dht enabled by default

Example:
  python benchmarks/bench_chunk_delta_h_bwd_dhu64.py --B 1 --H 8 --K 128 --T 4096 16384
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu as cula_bwd_dhu


def time_kernel(fn, warmup: int, iters: int) -> float:
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


def accuracy_stats(ref: torch.Tensor, out: torch.Tensor) -> tuple[float, float, float]:
    diff = (ref.float() - out.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = ref.float().abs().max().item()
    rel_max = max_abs / denom if denom > 0 else 0.0
    return max_abs, mean_abs, rel_max


def make_inputs(B: int, T: int, H: int, K: int, use_h0: bool, use_dht: bool, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    V = 64
    q = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    k = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    w = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    do = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    dv = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    h0 = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01).contiguous() if use_h0 else None
    dht = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.1).contiguous() if use_dht else None
    return q, k, w, do, dv, h0, dht


def run_one(args, T: int):
    q, k, w, do, dv, h0, dht = make_inputs(args.B, T, args.H, args.K, args.h0, args.dht, args.seed)
    common = dict(q=q, k=k, w=w, do=do, dv=dv, h0=h0, dht=dht, scale=args.scale, chunk_size=64)

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(**common)
    out_dh, out_dh0, out_dv2 = cula_bwd_dhu(**common)
    torch.cuda.synchronize()

    dh_max, dh_mean, dh_rel = accuracy_stats(ref_dh, out_dh)
    dv_max, dv_mean, dv_rel = accuracy_stats(ref_dv2, out_dv2)
    if h0 is not None:
        dh0_max, dh0_mean, dh0_rel = accuracy_stats(ref_dh0, out_dh0)
    else:
        dh0_max = dh0_mean = dh0_rel = 0.0

    def run_fla():
        return fla_bwd_dhu(**common)

    def run_cula():
        return cula_bwd_dhu(**common)

    fla_ms = time_kernel(run_fla, args.warmup, args.iters)
    cula_ms = time_kernel(run_cula, args.warmup, args.iters)
    speedup = fla_ms / cula_ms if cula_ms > 0 else float("inf")

    return {
        "B": args.B,
        "T": T,
        "H": args.H,
        "fla_ms": fla_ms,
        "cula_ms": cula_ms,
        "speedup": speedup,
        "dh_max": dh_max,
        "dh_mean": dh_mean,
        "dh_rel": dh_rel,
        "dv_max": dv_max,
        "dv_mean": dv_mean,
        "dv_rel": dv_rel,
        "dh0_max": dh0_max,
        "dh0_mean": dh0_mean,
        "dh0_rel": dh0_rel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--K", type=int, choices=(64, 128), default=64)
    parser.add_argument("--T", type=int, nargs="+", default=[4096, 4096 * 4])
    parser.add_argument("--scale", type=float, default=0.125)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-h0", dest="h0", action="store_false")
    parser.add_argument("--no-dht", dest="dht", action="store_false")
    parser.add_argument("--ncu", action="store_true")
    parser.set_defaults(h0=True, dht=True)
    args = parser.parse_args()

    if args.ncu:
        args.warmup = 1
        args.iters = 1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("chunk_gated_delta_rule_bwd_dhu64: C++ CUTLASS/CuTe path vs FLA Triton")
    print(f"B={args.B} H={args.H} K={args.K} V=64 h0={args.h0} dht={args.dht} warmup={args.warmup} iters={args.iters}")
    print(f"{'T':>8} {'FLA ms':>10} {'C++ ms':>10} {'speedup':>8} {'dh max':>10} {'dv2 max':>10} {'dh0 max':>10}")
    print("-" * 82)

    for T in args.T:
        if T % 64 != 0:
            raise ValueError(f"T must be a multiple of 64, got {T}")
        r = run_one(args, T)
        print(
            f"{r['T']:8d} {r['fla_ms']:10.4f} {r['cula_ms']:10.4f} {r['speedup']:8.3f} "
            f"{r['dh_max']:10.4e} {r['dv_max']:10.4e} {r['dh0_max']:10.4e}"
        )


if __name__ == "__main__":
    main()
