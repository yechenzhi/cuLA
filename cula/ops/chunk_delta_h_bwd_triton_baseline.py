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

"""
Experimental SM90 CuTe DSL baseline for FLA's Triton bwd_dhu kernel.

This file is intentionally independent from chunk_delta_h_bwd.py.  The first
variant targets the narrow Triton-alignment surface:

- SM90, bf16 inputs/outputs
- K=V=128, BT=64
- BV=64; Triton may autotune smaller BV for some low-head shapes, but this
  baseline keeps the SM90 WGMMA M-mode at 64.
- non-varlen only
- USE_G=False and USE_GK=False
- no store warp, no TMA store pipeline, no cluster multicast

The core loop follows the Triton dataflow:

    store carried dh
    dv2 = K @ dh + dv
    store dv2
    dh = dh + scale * do^T @ q - dv2^T @ w

dh and dv2 are kept as register fragments until the store point.  The store
path uses a simple per-CTA shared scratch tile, matching the Triton lowering's
shared staging without using the existing SM90 store pipeline.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32

from cula.utils import assert_hopper

BT = 64
DEFAULT_BV = 64
BK = 128
NUM_THREADS = 128


class ChunkDeltaRuleBwdDHUTritonBaselineSm90:
    def __init__(
        self,
        num_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        use_dht: bool,
        use_dh0: bool,
        bv: int,
        scale: float,
    ):
        if head_dim_k != BK or head_dim_v != BK:
            raise NotImplementedError(
                f"SM90 Triton-baseline bwd_dhu only supports K=V=128, got K={head_dim_k}, V={head_dim_v}."
            )
        if bv != DEFAULT_BV:
            raise NotImplementedError(
                f"SM90 Triton-baseline bwd_dhu currently keeps the WGMMA M-mode at BV={DEFAULT_BV}, got {bv}."
            )
        self.H = num_heads
        self.K = head_dim_k
        self.V = head_dim_v
        self.use_dht = use_dht
        self.use_dh0 = use_dh0
        self.scale = scale

        self.BT = BT
        self.BV = bv
        self.BK = BK
        self.num_threads = NUM_THREADS
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 128

        self.kdh_mma_tiler = (self.BV, self.BT, self.BK)
        self.update_mma_tiler = (self.BV, self.BK, self.BT)
        self.atom_layout_mnk = (1, 1, 1)

    @staticmethod
    def _c_layout_as_a_layout(c_layout: cute.Layout, a_shape) -> cute.Layout:
        return cute.make_layout(
            (a_shape, c_layout.shape[1], (c_layout.shape[2], cute.size(c_layout, mode=[0]) // cute.size(a_shape))),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (c_layout.stride[2], cute.size(a_shape, mode=[2]) * c_layout.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_operand_a(self, acc: cute.Tensor, tiled_mma: cute.TiledMma, dtype: cute.Numeric):
        a_layout = self._c_layout_as_a_layout(acc.layout, tiled_mma.tv_layout_A.shape[1])
        operand = cute.make_rmem_tensor(a_layout, dtype=dtype)
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        w: cute.Tensor,
        dht: cute.Tensor,
        dh0: cute.Tensor,
        do: cute.Tensor,
        dh: cute.Tensor,
        dv: cute.Tensor,
        dv2: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32],
        stream: cuda.CUstream,
    ):
        B, T, NT = problem_size

        kdh_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.kdh_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )
        update_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.update_mma_tiler[:2],
        )
        wdv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.update_mma_tiler[:2],
        )

        k_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            self.kdh_mma_tiler,
            self.io_dtype,
            1,
        )
        do_smem_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        q_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        w_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        dv_stage_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            1,
        )
        state_stage_layout = cute.make_layout((self.BV, self.BK), stride=(self.BK, 1))

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout)],
                self.buffer_align_bytes,
            ]
            sDo: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(do_smem_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout)],
                self.buffer_align_bytes,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout)],
                self.buffer_align_bytes,
            ]
            dvStage: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dv_stage_layout)],
                self.buffer_align_bytes,
            ]
            stateStage: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(state_stage_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            q,
            k,
            w,
            dht,
            dh0,
            do,
            dh,
            dv,
            dv2,
            problem_size,
            kdh_tiled_mma,
            update_tiled_mma,
            wdv_tiled_mma,
            k_smem_layout,
            do_smem_layout,
            q_smem_layout,
            w_smem_layout,
            dv_stage_layout,
            state_stage_layout,
        ).launch(
            grid=[cute.ceil_div(self.V, self.BV), B * self.H, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        w: cute.Tensor,
        dht: cute.Tensor,
        dh0: cute.Tensor,
        do: cute.Tensor,
        dh: cute.Tensor,
        dv: cute.Tensor,
        dv2: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32],
        kdh_tiled_mma: cute.TiledMma,
        update_tiled_mma: cute.TiledMma,
        wdv_tiled_mma: cute.TiledMma,
        k_smem_layout: cute.ComposedLayout,
        do_smem_layout: cute.ComposedLayout,
        q_smem_layout: cute.ComposedLayout,
        w_smem_layout: cute.ComposedLayout,
        dv_stage_layout: cute.ComposedLayout,
        state_stage_layout: cute.Layout,
    ):
        _, T, NT = problem_size
        tidx, _, _ = cute.arch.thread_idx()
        v_tile_idx, bh_idx, _ = cute.arch.block_idx()
        bidx = bh_idx // self.H
        hidx = bh_idx - bidx * self.H
        v_base = v_tile_idx * self.BV

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sDo = storage.sDo.get_tensor(do_smem_layout.outer, swizzle=do_smem_layout.inner)
        sQ = storage.sQ.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sW = storage.sW.get_tensor(w_smem_layout.outer, swizzle=w_smem_layout.inner)
        dvStage = storage.dvStage.get_tensor(dv_stage_layout.outer, swizzle=dv_stage_layout.inner)
        stateStage = storage.stateStage.get_tensor(state_stage_layout)

        local_tidx = tidx
        kdh_thr_mma = kdh_tiled_mma.get_slice(local_tidx)
        update_thr_mma = update_tiled_mma.get_slice(local_tidx)
        wdv_thr_mma = wdv_tiled_mma.get_slice(local_tidx)

        tKsB = kdh_thr_mma.partition_B(sK)
        tKrB = kdh_thr_mma.make_fragment_B(tKsB)
        tDoA = update_thr_mma.partition_A(sDo)
        tDoR = update_thr_mma.make_fragment_A(tDoA)
        tQsB = update_thr_mma.partition_B(sQ)
        tQrB = update_thr_mma.make_fragment_B(tQsB)
        tDvA = wdv_thr_mma.partition_A(dvStage)
        tDvR = wdv_thr_mma.make_fragment_A(tDvA)
        tWsB = wdv_thr_mma.partition_B(sW)
        tWrB = wdv_thr_mma.make_fragment_B(tWsB)

        c_dv = cute.make_identity_tensor((self.BV, self.BT))
        tCcDv = kdh_thr_mma.partition_C(c_dv)
        acc_dv = kdh_thr_mma.make_fragment_C(kdh_thr_mma.partition_shape_C((self.BV, self.BT)))

        c_state = cute.make_identity_tensor((self.BV, self.BK))
        tCcState = update_thr_mma.partition_C(c_state)
        state_shape = update_thr_mma.partition_shape_C((self.BV, self.BK))
        r_state = update_thr_mma.make_fragment_C(state_shape)
        acc_qdo = update_thr_mma.make_fragment_C(state_shape)
        acc_wdv = wdv_thr_mma.make_fragment_C(state_shape)

        for ei in cutlass.range(cute.size(r_state), unroll_full=True):
            v_rel, k_rel = tCcState[ei]
            value = Float32(0.0)
            if cutlass.const_expr(self.use_dht):
                value = dht[bidx, hidx, k_rel, v_base + v_rel].to(self.acc_dtype)
            r_state[ei] = value

        for chunk_rev in cutlass.range(0, NT, unroll=0):
            chunk_idx = NT - 1 - chunk_rev
            chunk_start = chunk_idx * self.BT

            # 1. Store the carried dh before this chunk updates it.
            for ei in cutlass.range(cute.size(r_state), unroll_full=True):
                v_rel, k_rel = tCcState[ei]
                stateStage[v_rel, k_rel] = r_state[ei].to(self.io_dtype)

            cute.arch.barrier()

            linear_state = tidx
            while linear_state < self.BV * self.BK:
                v_rel = linear_state // self.BK
                k_rel = linear_state - v_rel * self.BK
                dh[bidx, chunk_idx, hidx, k_rel, v_base + v_rel] = stateStage[v_rel, k_rel]
                linear_state += self.num_threads

            # Load the current chunk's WGMMA operands.  This is intentionally a
            # simple cooperative load, not a warp-specialized/TMA pipeline.
            linear_k = tidx
            while linear_k < self.BT * self.BK:
                t_rel = linear_k // self.BK
                k_rel = linear_k - t_rel * self.BK
                t_abs = chunk_start + t_rel
                k_val = self.io_dtype(0.0)
                q_val = self.io_dtype(0.0)
                w_val = self.io_dtype(0.0)
                if t_abs < T:
                    k_val = k[bidx, t_abs, hidx, k_rel]
                    q_val = q[bidx, t_abs, hidx, k_rel]
                    w_val = w[bidx, t_abs, hidx, k_rel]
                sK[t_rel, k_rel, 0] = k_val
                sQ[k_rel, t_rel, 0] = q_val
                sW[k_rel, t_rel, 0] = w_val
                linear_k += self.num_threads

            linear_do = tidx
            while linear_do < self.BV * self.BT:
                v_rel = linear_do // self.BT
                t_rel = linear_do - v_rel * self.BT
                t_abs = chunk_start + t_rel
                do_val = self.io_dtype(0.0)
                if t_abs < T:
                    do_val = do[bidx, t_abs, hidx, v_base + v_rel]
                sDo[v_rel, t_rel, 0] = do_val
                linear_do += self.num_threads

            cute.arch.barrier()

            # 2. acc_dv = K @ dh.  Internally this computes dh^T @ K^T as a
            # VxT tile so the carried state can feed WGMMA as an RMEM operand.
            acc_dv.fill(0.0)
            r_state_bf16 = cute.make_rmem_tensor_like(r_state, self.io_dtype)
            r_state_bf16.store(r_state.load().to(self.io_dtype))
            state_op = self.make_acc_operand_a(r_state_bf16, kdh_tiled_mma, self.io_dtype)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tKrB, mode=[2]), unroll_full=True):
                kdh_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(kdh_tiled_mma, acc_dv, state_op[None, None, kp], tKrB[None, None, kp, 0], acc_dv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # 3/4. dv2 = acc_dv + dv, then materialize it in shared memory.
            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                v_rel, t_rel = tCcDv[ei]
                t_abs = chunk_start + t_rel
                out = Float32(0.0)
                if t_abs < T:
                    out = acc_dv[ei] + dv[bidx, t_abs, hidx, v_base + v_rel].to(self.acc_dtype)
                dvStage[v_rel, t_rel, 0] = out.to(self.io_dtype)

            cute.arch.barrier()

            linear_dv = tidx
            while linear_dv < self.BV * self.BT:
                v_rel = linear_dv // self.BT
                t_rel = linear_dv - v_rel * self.BT
                t_abs = chunk_start + t_rel
                if t_abs < T:
                    dv2[bidx, t_abs, hidx, v_base + v_rel] = dvStage[v_rel, t_rel, 0]
                linear_dv += self.num_threads

            # 5. WDV = W^T @ dv2, using the same shared tile that feeds the store.
            acc_wdv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tWrB, mode=[2]), unroll_full=True):
                wdv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(wdv_tiled_mma, acc_wdv, tDvR[None, None, kp, 0], tWrB[None, None, kp, 0], acc_wdv)
            cute.nvgpu.warpgroup.commit_group()

            # 6. QDO = Q^T @ do.
            acc_qdo.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(update_tiled_mma, acc_qdo, tDoR[None, None, kp, 0], tQrB[None, None, kp, 0], acc_qdo)
            cute.nvgpu.warpgroup.commit_group()

            cute.nvgpu.warpgroup.wait_group(1)
            cute.nvgpu.warpgroup.wait_group(0)

            # 7. dh = dh + scale * QDO - WDV.
            for ei in cutlass.range(cute.size(r_state), unroll_full=True):
                r_state[ei] = r_state[ei] + acc_qdo[ei] * Float32(self.scale) - acc_wdv[ei]

            cute.arch.barrier()

        if cutlass.const_expr(self.use_dh0):
            for ei in cutlass.range(cute.size(r_state), unroll_full=True):
                v_rel, k_rel = tCcState[ei]
                dh0[bidx, hidx, k_rel, v_base + v_rel] = r_state[ei]


@functools.lru_cache(maxsize=32)
def _compile_bwd_dhu_triton_baseline_sm90(
    H: int,
    K: int,
    V: int,
    use_dht: bool,
    use_dh0: bool,
    bv: int,
    scale: float,
):
    kernel = ChunkDeltaRuleBwdDHUTritonBaselineSm90(
        num_heads=H,
        head_dim_k=K,
        head_dim_v=V,
        use_dht=use_dht,
        use_dh0=use_dh0,
        bv=bv,
        scale=scale,
    )

    sym_b = cute.sym_int()
    sym_t = cute.sym_int()
    sym_nt = cute.sym_int()
    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    do_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dv_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dv2_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dht_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, H, K, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dh0_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, H, K, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    dh_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_nt, H, K, V),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=128,
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel,
        q_fake,
        k_fake,
        w_fake,
        dht_fake,
        dh0_fake,
        do_fake,
        dh_fake,
        dv_fake,
        dv2_fake,
        (Int32(1), Int32(1), Int32(1)),
        stream_fake,
        options="--enable-tvm-ffi",
    )


def chunk_gated_delta_rule_bwd_dhu_sm90_triton_baseline(
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
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = BT,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """FLA-compatible entry point for the experimental Triton-like baseline."""
    del chunk_indices, chunk_offsets, use_exp2
    assert_hopper(q.device)
    if chunk_size != BT:
        raise NotImplementedError(f"Triton-baseline bwd_dhu only supports chunk_size={BT}.")
    if cu_seqlens is not None:
        raise NotImplementedError("Triton-baseline bwd_dhu currently supports non-varlen inputs only.")
    if g is not None or gk is not None:
        raise NotImplementedError("Triton-baseline bwd_dhu currently supports USE_G=False and USE_GK=False only.")
    if transpose_state_layout:
        raise NotImplementedError("Triton-baseline bwd_dhu currently supports transpose_state_layout=False only.")

    B, T, H, K = q.shape
    V = do.shape[-1]
    if K != BK or V != BK:
        raise NotImplementedError(f"Triton-baseline bwd_dhu only supports K=V=128, got K={K}, V={V}.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("q, k, and w must be bfloat16.")
    if do.dtype != torch.bfloat16 or dv.dtype != torch.bfloat16:
        raise TypeError("do and dv must be bfloat16.")
    if not q.is_contiguous() or not k.is_contiguous() or not w.is_contiguous():
        raise ValueError("q, k, and w must be contiguous.")
    if not do.is_contiguous() or not dv.is_contiguous():
        raise ValueError("do and dv must be contiguous.")
    if dht is not None and (dht.dtype != torch.float32 or not dht.is_contiguous()):
        raise ValueError("dht must be contiguous float32.")
    if h0 is not None and (h0.dtype != torch.float32 or not h0.is_contiguous()):
        raise ValueError("h0 must be contiguous float32 when requesting dh0.")

    state_shape = (B, H, K, V)
    if dht is not None and tuple(dht.shape) != state_shape:
        raise ValueError(f"dht must have shape {state_shape}, got {tuple(dht.shape)}.")
    if h0 is not None and tuple(h0.shape) != state_shape:
        raise ValueError(f"h0 must have shape {state_shape}, got {tuple(h0.shape)}.")

    NT = (T + BT - 1) // BT
    scale_value = K**-0.5 if scale is None else float(scale)
    bv = DEFAULT_BV
    dh = q.new_empty(B, NT, H, K, V)
    dv2 = torch.empty_like(dv)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None

    dht_arg = dht if dht is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    dh0_arg = dh0 if dh0 is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    compiled = _compile_bwd_dhu_triton_baseline_sm90(H, K, V, dht is not None, h0 is not None, bv, scale_value)
    compiled(q, k, w, dht_arg, dh0_arg, do, dh, dv, dv2, (Int32(B), Int32(T), Int32(NT)))
    return dh, dh0, dv2
