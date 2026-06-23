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

"""C++/CUTLASS-CuTe chunk_delta_h backward dH/dU helper.

This module intentionally exposes the first production targets:
K in {64, 128}, V=64 or K=128/V=128, chunk_size=64, fixed-length layout, no g/gk, no transposed state.
The return order matches FLA's ``chunk_gated_delta_rule_bwd_dhu``:
``(dh, dh0, dv2)``.
"""

import torch

import cula.cudac as cula_cuda


def _check_same_shape(name: str, x: torch.Tensor, ref: torch.Tensor) -> None:
    if x.shape != ref.shape:
        raise ValueError(f"{name}.shape must match q.shape, got {tuple(x.shape)} vs {tuple(ref.shape)}")


def _check_btv_shape(name: str, x: torch.Tensor, B: int, T: int, H: int, V: int) -> None:
    if x.shape != (B, T, H, V):
        raise ValueError(f"{name}.shape must be {(B, T, H, V)}, got {tuple(x.shape)}")


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    del chunk_indices, use_exp2

    if g is not None:
        raise NotImplementedError("C++ bwd_dhu64 currently supports no g gate only")
    if gk is not None:
        raise NotImplementedError("C++ bwd_dhu64 currently supports no gk gate only")
    if cu_seqlens is not None:
        raise NotImplementedError("C++ bwd_dhu64 currently supports fixed-length tensors only")
    if transpose_state_layout:
        raise NotImplementedError("C++ bwd_dhu64 currently supports [B, NT, H, K, V] state layout only")
    if chunk_size != 64:
        raise ValueError(f"C++ bwd_dhu64 requires chunk_size=64, got {chunk_size}")
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    K = q.shape[-1]
    V = do.shape[-1]
    if K not in (64, 128) or V not in (64, 128) or (V == 128 and K != 128):
        raise ValueError(f"C++ bwd_dhu64 requires K in {{64, 128}} with V=64, or K=128/V=128, got K={K}, V={V}")
    if q.shape[1] % 64 != 0:
        raise ValueError(f"C++ bwd_dhu64 requires T to be a multiple of 64, got T={q.shape[1]}")
    if q.dtype is not torch.bfloat16:
        raise TypeError(f"q must be bfloat16, got {q.dtype}")
    if not hasattr(cula_cuda, "chunk_gated_delta_rule_bwd_dhu64_cuda"):
        raise RuntimeError("chunk_gated_delta_rule_bwd_dhu64_cuda is only built when CULA_SM90A_ENABLED is enabled")

    _check_same_shape("k", k, q)
    _check_same_shape("w", w, q)
    B, T, H, K = q.shape
    _check_btv_shape("do", do, B, T, H, V)
    _check_btv_shape("dv", dv, B, T, H, V)

    for name, x in (("q", q), ("k", k), ("w", w), ("do", do), ("dv", dv)):
        if x.dtype is not torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {x.dtype}")
        if not x.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    expected_state_shape = (B, H, K, V)
    if h0 is not None:
        if h0.shape != expected_state_shape:
            raise ValueError(f"h0 must have shape {expected_state_shape}, got {tuple(h0.shape)}")
        if h0.dtype is not torch.float32:
            raise TypeError(f"h0 must be float32, got {h0.dtype}")
        if not h0.is_cuda or not h0.is_contiguous():
            raise ValueError("h0 must be a contiguous CUDA tensor")
    if dht is not None:
        if dht.shape != expected_state_shape:
            raise ValueError(f"dht must have shape {expected_state_shape}, got {tuple(dht.shape)}")
        if dht.dtype is not torch.float32:
            raise TypeError(f"dht must be float32, got {dht.dtype}")
        if not dht.is_cuda or not dht.is_contiguous():
            raise ValueError("dht must be a contiguous CUDA tensor")

    return cula_cuda.chunk_gated_delta_rule_bwd_dhu64_cuda(
        q,
        k,
        w,
        do,
        dv,
        h0,
        dht,
        1.0 if scale is None else float(scale),
    )


__all__ = ["chunk_gated_delta_rule_bwd_dhu"]
