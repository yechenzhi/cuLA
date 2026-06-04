#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dual-consumer SM90 #mma convert-layout probe.

This bench isolates the exact pattern needed by bwd_dhu:

    dv = dot(a, b).to(bf16)          # tensor<64x64xbf16, #mma>
    store(dv)                        # #mma -> convert_layout -> #blocked -> tt.store
    wdv = dot(w, dv)                 # same #mma value -> local_alloc #shared -> WGMMA

It dumps Triton's real TTGIR/PTX/SASS so CuTe/CUTLASS candidates can be matched
against the compiled lowering instead of the high-level Python source.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl

from cula.utils import assert_hopper

M = 64
N = 64
K = 64


@triton.jit
def triton_dual_consumer_kernel(a, b, w, out_dv, out_wdv, stride_ob: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    a_tile = tl.load(a + offs_m[:, None] * 64 + offs_k[None, :])
    b_tile = tl.load(b + offs_k[:, None] * 64 + offs_n[None, :])
    acc = tl.dot(a_tile, b_tile)
    dv = acc.to(tl.bfloat16)
    tl.store(out_dv + pid * stride_ob + offs_m[:, None] * 64 + offs_n[None, :], dv)

    w_tile = tl.load(w + offs_m[:, None] * 64 + offs_k[None, :])
    wdv = tl.dot(w_tile, dv)
    tl.store(out_wdv + pid * stride_ob + offs_m[:, None] * 64 + offs_n[None, :], wdv.to(tl.bfloat16))


def _bench(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def _sass_summary(sass: str, res: str, n_regs: int | None = None) -> dict[str, int | None]:
    summary: dict[str, int | None] = {"REG": n_regs, "STACK": None, "SHARED": None, "LOCAL": None}
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    if match:
        summary.update(dict(zip(["REG", "STACK", "SHARED", "LOCAL"], map(int, match.groups()))))
    for op in [
        "HGMMA",
        "STSM",
        "LDSM",
        "LDS",
        "STS.128",
        "STS",
        "STG.E.128",
        "STG.E.U16",
        "BAR.SYNC",
        "DEPBAR",
        "WARPGROUP",
        "LDL",
        "STL",
    ]:
        summary[op] = sass.count(op)
    return summary


def _extract_store_windows(sass: str) -> str:
    lines = sass.splitlines()
    hits = [idx for idx, line in enumerate(lines) if "STG.E.128" in line or "STSM.16" in line or "LDSM.16" in line]
    chunks: list[str] = []
    seen: set[tuple[int, int]] = set()
    for hit in hits:
        lo = max(0, hit - 8)
        hi = min(len(lines), hit + 9)
        key = (lo, hi)
        if key in seen:
            continue
        seen.add(key)
        chunks.append("\n".join(lines[lo:hi]))
    return "\n\n---\n\n".join(chunks)


def run_triton(out_dir: pathlib.Path, blocks: int, warmup: int, repeat: int) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    a = (torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    b = torch.eye(K, N, device="cuda", dtype=torch.bfloat16)
    w = torch.eye(M, K, device="cuda", dtype=torch.bfloat16)
    out_dv = torch.empty((blocks, M, N), device="cuda", dtype=torch.bfloat16)
    out_wdv = torch.empty_like(out_dv)

    def launch():
        triton_dual_consumer_kernel[(blocks,)](
            a,
            b,
            w,
            out_dv,
            out_wdv,
            M * N,
            num_warps=4,
            num_stages=3,
        )

    launch()
    torch.cuda.synchronize()
    compiled = next(reversed(triton_dual_consumer_kernel.device_caches[0][0].values()))
    name = "triton_dual_consumer"
    cubin = out_dir / f"{name}.cubin"
    cubin.write_bytes(compiled.asm["cubin"])
    for suffix in ["ttir", "ttgir", "ptx", "llir", "source"]:
        if suffix in compiled.asm:
            (out_dir / f"{name}_{suffix}.txt").write_text(compiled.asm[suffix])

    res = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(cubin)], check=False, capture_output=True, text=True
    ).stdout
    sass = subprocess.run(["cuobjdump", "--dump-sass", str(cubin)], check=False, capture_output=True, text=True).stdout
    (out_dir / f"{name}_resource.txt").write_text(res)
    (out_dir / f"{name}.sass").write_text(sass)
    (out_dir / f"{name}_store_windows.sass").write_text(_extract_store_windows(sass))

    ttgir = compiled.asm.get("ttgir", "")
    ms = _bench(launch, warmup, repeat)
    row: dict[str, object] = {
        "family": "triton",
        "variant": name,
        "blocks": blocks,
        "ms": ms,
        "ns_per_cta": ms * 1e6 / blocks,
        "dv_max_diff": (out_dv[0] - a).abs().max().item(),
        "wdv_max_diff": (out_wdv[0] - a).abs().max().item(),
        "ttgir_convert_layout": ttgir.count("ttg.convert_layout"),
        "ttgir_local_alloc": ttgir.count("ttg.local_alloc"),
        "ttgir_warp_group_dot": ttgir.count("warp_group_dot"),
        "has_dual_consumer_pattern": (
            "arith.truncf" in ttgir
            and "ttg.local_alloc %" in ttgir
            and "ttg.convert_layout %" in ttgir
            and "tt.store" in ttgir
        ),
        **_sass_summary(sass, res, getattr(compiled, "n_regs", None)),
    }
    print("TRITON_DUAL", json.dumps(row, sort_keys=True), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("benchmarks/profiles/dual_consumer_convert_layout")
    )
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    assert_hopper()
    run_triton(args.out_dir, args.blocks, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
