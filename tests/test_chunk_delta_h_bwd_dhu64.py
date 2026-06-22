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

import pytest
import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu as cula_bwd_dhu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_inputs(B: int, T: int, H: int, *, use_h0: bool, use_dht: bool, seed: int = 42):
    torch.manual_seed(seed)
    device = "cuda"
    K = V = 64

    q = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    k = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    w = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    do = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    dv = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    h0 = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01).contiguous() if use_h0 else None
    dht = (torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.1).contiguous() if use_dht else None
    return q, k, w, do, dv, h0, dht


@pytest.mark.parametrize("B,H,T", [(1, 1, 64), (2, 3, 256)])
@pytest.mark.parametrize("use_h0,use_dht", [(False, False), (False, True), (True, False), (True, True)])
def test_bwd_dhu64_against_fla(B: int, H: int, T: int, use_h0: bool, use_dht: bool):
    q, k, w, do, dv, h0, dht = _make_inputs(B, T, H, use_h0=use_h0, use_dht=use_dht)
    scale = 0.125

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
    )
    our_dh, our_dh0, our_dv2 = cula_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(our_dh.float(), ref_dh.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(our_dv2.float(), ref_dv2.float(), atol=2e-2, rtol=2e-2)
    if use_h0:
        assert our_dh0 is not None
        torch.testing.assert_close(our_dh0.float(), ref_dh0.float(), atol=3e-2, rtol=3e-2)
    else:
        assert our_dh0 is None


def test_bwd_dhu64_long_t_against_fla():
    q, k, w, do, dv, h0, dht = _make_inputs(1, 4096, 1, use_h0=True, use_dht=True, seed=7)
    scale = 0.125

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, h0=h0, dht=dht, scale=scale, chunk_size=64)
    our_dh, our_dh0, our_dv2 = cula_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, h0=h0, dht=dht, scale=scale, chunk_size=64)
    torch.cuda.synchronize()

    torch.testing.assert_close(our_dh.float(), ref_dh.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(our_dv2.float(), ref_dv2.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(our_dh0.float(), ref_dh0.float(), atol=5e-2, rtol=5e-2)


def test_bwd_dhu64_rejects_unsupported_options():
    q, k, w, do, dv, h0, dht = _make_inputs(1, 64, 1, use_h0=True, use_dht=True)
    gk = torch.zeros_like(q, dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="gk"):
        cula_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, gk=gk, h0=h0, dht=dht)
    with pytest.raises(NotImplementedError, match="fixed-length"):
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32, device=q.device)
        cula_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, h0=h0, dht=dht, cu_seqlens=cu_seqlens)
