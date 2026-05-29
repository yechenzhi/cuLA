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

The carried state is held as two K-split register fragments in the WGMMA-friendly
[BV, 64] orientation.  The dh store uses a small StMatrix-based shared
local_alloc and 128-bit global stores; K@dh and WDV still consume register
operands directly.  This keeps the file independent from the existing SM90
store-warp/TMA pipeline while matching the Triton store lowering more closely.
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
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32

from cula.utils import assert_hopper

BT = 64
DEFAULT_BV = 64
BK = 128
BK_HALF = 64
NUM_THREADS = 128
NUM_STAGES = 3


class ChunkDeltaRuleBwdDHUTritonBaselineSm90:
    def __init__(
        self,
        num_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        use_dht: bool,
        use_dh0: bool,
        full_chunks: bool,
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
        self.full_chunks = full_chunks
        self.scale = scale

        self.BT = BT
        self.BV = bv
        self.BK = BK
        self.num_threads = NUM_THREADS
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 128

        self.kdh_mma_tiler = (self.BV, self.BT, BK_HALF)
        self.update_mma_tiler = (self.BV, BK_HALF, self.BT)
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
    def copy_chunk_operands_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_m0_thr,
        gK_bh_kt: cute.Tensor,
        gQ_bh_kt: cute.Tensor,
        gW_bh_kt: cute.Tensor,
        gDo_bh_vt: cute.Tensor,
        gDv_bh_vt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        sQ1: cute.Tensor,
        sQ2: cute.Tensor,
        sW1: cute.Tensor,
        sW2: cute.Tensor,
        sDo: cute.Tensor,
        dvStage: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, self.BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, self.BT), (1, chunk_idx))
        gQ1_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, self.BT), (0, chunk_idx))
        gQ2_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, self.BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, self.BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, self.BT), (1, chunk_idx))
        gDo_tile = cute.local_tile(gDo_bh_vt, (self.BV, self.BT), (v_tile_idx, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (self.BV, self.BT), (v_tile_idx, chunk_idx))

        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gK1_tile),
            copy_m0_thr.partition_D(sK1_kt[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gK2_tile),
            copy_m0_thr.partition_D(sK2_kt[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gQ1_tile),
            copy_m0_thr.partition_D(sQ1[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gQ2_tile),
            copy_m0_thr.partition_D(sQ2[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gW1_tile),
            copy_m0_thr.partition_D(sW1[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gW2_tile),
            copy_m0_thr.partition_D(sW2[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gDo_tile),
            copy_m0_thr.partition_D(sDo[(None, None, stage)]),
        )
        cute.copy(
            copy_contig_m0,
            copy_m0_thr.partition_S(gDv_tile),
            copy_m0_thr.partition_D(dvStage[(None, None, stage)]),
        )
        cute.arch.cp_async_commit_group()

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
            warpgroup.OperandSource.RMEM,
        )

        k_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR,
            self.kdh_mma_tiler,
            self.io_dtype,
            NUM_STAGES,
        )
        do_smem_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            NUM_STAGES,
        )
        q_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            NUM_STAGES,
        )
        w_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            NUM_STAGES,
        )
        dv_stage_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.COL_MAJOR,
            self.update_mma_tiler,
            self.io_dtype,
            NUM_STAGES,
        )
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        thread_layout = cute.make_layout((self.num_threads // 32, 32), stride=(32, 1))
        copy_contig_m0 = cute.make_tiled_copy_tv(copy_atom, thread_layout, cute.make_layout((8, 1)))
        state_epi_layout = sm90_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, BK_HALF),
            2,
        )
        state_store_atom = sm90_utils.get_smem_store_op(utils.LayoutEnum.COL_MAJOR, self.io_dtype, self.acc_dtype)
        state_tiled_copy = cute.make_tiled_copy_C(state_store_atom, update_tiled_mma)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        store_elems = 128 // self.io_dtype.width
        store_vk_thr_layout = cute.make_ordered_layout(
            (self.BV // store_elems, self.num_threads // (self.BV // store_elems)),
            order=(0, 1),
        )
        store_vk_val_layout = cute.make_layout((store_elems, 1))
        store_tiled_copy_vk = cute.make_tiled_copy_tv(store_atom, store_vk_thr_layout, store_vk_val_layout)

        @cute.struct
        class SharedStorage:
            sK1: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout)],
                self.buffer_align_bytes,
            ]
            sK2: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout)],
                self.buffer_align_bytes,
            ]
            sDo: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(do_smem_layout)],
                self.buffer_align_bytes,
            ]
            sQ1: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout)],
                self.buffer_align_bytes,
            ]
            sQ2: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout)],
                self.buffer_align_bytes,
            ]
            sW1: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout)],
                self.buffer_align_bytes,
            ]
            sW2: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout)],
                self.buffer_align_bytes,
            ]
            dvStage: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dv_stage_layout)],
                self.buffer_align_bytes,
            ]
            stateStage: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(state_epi_layout)],
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
            copy_contig_m0,
            state_tiled_copy,
            store_tiled_copy_vk,
            state_epi_layout,
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
        copy_contig_m0: cute.TiledCopy,
        state_tiled_copy: cute.TiledCopy,
        store_tiled_copy_vk: cute.TiledCopy,
        state_epi_layout: cute.ComposedLayout,
    ):
        _, T, NT = problem_size
        tidx, _, _ = cute.arch.thread_idx()
        v_tile_idx, bh_idx, _ = cute.arch.block_idx()
        bidx = bh_idx // self.H
        hidx = bh_idx - bidx * self.H
        v_base = v_tile_idx * self.BV

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK1 = storage.sK1.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sK2 = storage.sK2.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sDo = storage.sDo.get_tensor(do_smem_layout.outer, swizzle=do_smem_layout.inner)
        sQ1 = storage.sQ1.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sQ2 = storage.sQ2.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sW1 = storage.sW1.get_tensor(w_smem_layout.outer, swizzle=w_smem_layout.inner)
        sW2 = storage.sW2.get_tensor(w_smem_layout.outer, swizzle=w_smem_layout.inner)
        dvStage = storage.dvStage.get_tensor(dv_stage_layout.outer, swizzle=dv_stage_layout.inner)
        stateStage = storage.stateStage.get_tensor(state_epi_layout.outer, swizzle=state_epi_layout.inner)

        local_tidx = tidx
        copy_m0_thr = copy_contig_m0.get_slice(local_tidx)
        state_r2s_thr_copy = state_tiled_copy.get_slice(local_tidx)
        store_vk_thr_copy = store_tiled_copy_vk.get_slice(local_tidx)
        kdh_thr_mma = kdh_tiled_mma.get_slice(local_tidx)
        update_thr_mma = update_tiled_mma.get_slice(local_tidx)
        wdv_thr_mma = wdv_tiled_mma.get_slice(local_tidx)

        tK1B = kdh_thr_mma.partition_B(sK1)
        tK1R = kdh_thr_mma.make_fragment_B(tK1B)
        tK2B = kdh_thr_mma.partition_B(sK2)
        tK2R = kdh_thr_mma.make_fragment_B(tK2B)

        tDoA = update_thr_mma.partition_A(sDo)
        tDoR = update_thr_mma.make_fragment_A(tDoA)
        tQ1B = update_thr_mma.partition_B(sQ1)
        tQ1R = update_thr_mma.make_fragment_B(tQ1B)
        tQ2B = update_thr_mma.partition_B(sQ2)
        tQ2R = update_thr_mma.make_fragment_B(tQ2B)
        tW1B = wdv_thr_mma.partition_B(sW1)
        tW1R = wdv_thr_mma.make_fragment_B(tW1B)
        tW2B = wdv_thr_mma.partition_B(sW2)
        tW2R = wdv_thr_mma.make_fragment_B(tW2B)

        c_dv = cute.make_identity_tensor((self.BV, self.BT))
        tCcDv = kdh_thr_mma.partition_C(c_dv)
        acc_dv = kdh_thr_mma.make_fragment_C(kdh_thr_mma.partition_shape_C((self.BV, self.BT)))
        dv2_a_layout = self._c_layout_as_a_layout(acc_dv.layout, kdh_tiled_mma.tv_layout_A.shape[1])
        dv2_op = cute.make_rmem_tensor(dv2_a_layout, dtype=self.io_dtype)
        dv2_op_as_acc = cute.make_tensor(dv2_op.iterator, acc_dv.layout)

        c_state = cute.make_identity_tensor((self.BV, BK_HALF))
        tCcState = update_thr_mma.partition_C(c_state)
        state_shape = update_thr_mma.partition_shape_C((self.BV, BK_HALF))
        r_state1 = update_thr_mma.make_fragment_C(state_shape)
        r_state2 = update_thr_mma.make_fragment_C(state_shape)
        acc_qdo1 = update_thr_mma.make_fragment_C(state_shape)
        acc_qdo2 = update_thr_mma.make_fragment_C(state_shape)
        acc_wdv1 = wdv_thr_mma.make_fragment_C(state_shape)
        acc_wdv2 = wdv_thr_mma.make_fragment_C(state_shape)

        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
            v_rel, k_rel = tCcState[ei]
            value1 = Float32(0.0)
            value2 = Float32(0.0)
            if cutlass.const_expr(self.use_dht):
                value1 = dht[bidx, hidx, k_rel, v_base + v_rel].to(self.acc_dtype)
                value2 = dht[bidx, hidx, k_rel + BK_HALF, v_base + v_rel].to(self.acc_dtype)
            r_state1[ei] = value1
            r_state2[ei] = value2

        gK_bh = k[(bidx, None, hidx, None)]
        gQ_bh_tv = q[(bidx, None, hidx, None)]
        gW_bh_tv = w[(bidx, None, hidx, None)]
        gDo_bh_tv = do[(bidx, None, hidx, None)]
        gDv_bh_tv = dv[(bidx, None, hidx, None)]
        gK_bh_kt = cute.make_tensor(gK_bh.iterator, cute.select(gK_bh.layout, mode=[1, 0]))
        gQ_bh_kt = cute.make_tensor(gQ_bh_tv.iterator, cute.select(gQ_bh_tv.layout, mode=[1, 0]))
        gW_bh_kt = cute.make_tensor(gW_bh_tv.iterator, cute.select(gW_bh_tv.layout, mode=[1, 0]))
        gDo_bh_vt = cute.make_tensor(gDo_bh_tv.iterator, cute.select(gDo_bh_tv.layout, mode=[1, 0]))
        gDv_bh_vt = cute.make_tensor(gDv_bh_tv.iterator, cute.select(gDv_bh_tv.layout, mode=[1, 0]))
        sK1_kt = cute.make_tensor(sK1.iterator, cute.select(sK1.layout, mode=[1, 0, 2]))
        sK2_kt = cute.make_tensor(sK2.iterator, cute.select(sK2.layout, mode=[1, 0, 2]))

        if cutlass.const_expr(self.full_chunks):
            full_sequence = True
        else:
            full_sequence = T == NT * self.BT
        if full_sequence:
            if NT > 0:
                self.copy_chunk_operands_async(
                    copy_contig_m0,
                    copy_m0_thr,
                    gK_bh_kt,
                    gQ_bh_kt,
                    gW_bh_kt,
                    gDo_bh_vt,
                    gDv_bh_vt,
                    sK1_kt,
                    sK2_kt,
                    sQ1,
                    sQ2,
                    sW1,
                    sW2,
                    sDo,
                    dvStage,
                    NT - 1,
                    v_tile_idx,
                    Int32(0),
                )
            if NT > 1:
                self.copy_chunk_operands_async(
                    copy_contig_m0,
                    copy_m0_thr,
                    gK_bh_kt,
                    gQ_bh_kt,
                    gW_bh_kt,
                    gDo_bh_vt,
                    gDv_bh_vt,
                    sK1_kt,
                    sK2_kt,
                    sQ1,
                    sQ2,
                    sW1,
                    sW2,
                    sDo,
                    dvStage,
                    NT - 2,
                    v_tile_idx,
                    Int32(1),
                )

        for chunk_rev in cutlass.range(0, NT, unroll=0):
            chunk_idx = NT - 1 - chunk_rev
            chunk_start = chunk_idx * self.BT
            full_chunk = chunk_start + self.BT <= T
            current_stage = chunk_rev % NUM_STAGES

            if not full_sequence:
                current_stage = Int32(0)
                if full_chunk:
                    self.copy_chunk_operands_async(
                        copy_contig_m0,
                        copy_m0_thr,
                        gK_bh_kt,
                        gQ_bh_kt,
                        gW_bh_kt,
                        gDo_bh_vt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sQ1,
                        sQ2,
                        sW1,
                        sW2,
                        sDo,
                        dvStage,
                        chunk_idx,
                        v_tile_idx,
                        current_stage,
                    )

            # 1. Store the carried dh before this chunk updates it.
            tSrState = state_r2s_thr_copy.retile(r_state1)
            tSrState_bf16 = cute.make_rmem_tensor_like(tSrState, self.io_dtype)
            tSrState_bf16.store(tSrState.load().to(self.io_dtype))
            cute.copy(
                state_tiled_copy,
                tSrState_bf16,
                state_r2s_thr_copy.partition_D(stateStage[(None, None, 0)]),
            )
            tSrState = state_r2s_thr_copy.retile(r_state2)
            tSrState_bf16 = cute.make_rmem_tensor_like(tSrState, self.io_dtype)
            tSrState_bf16.store(tSrState.load().to(self.io_dtype))
            cute.copy(
                state_tiled_copy,
                tSrState_bf16,
                state_r2s_thr_copy.partition_D(stateStage[(None, None, 1)]),
            )

            cute.arch.barrier()

            gDh_kv = dh[(bidx, chunk_idx, hidx, None, None)]
            gDh_vk = cute.make_tensor(gDh_kv.iterator, cute.select(gDh_kv.layout, mode=[1, 0]))
            gDh1_tile = cute.local_tile(gDh_vk, (self.BV, BK_HALF), (v_tile_idx, 0))
            gDh2_tile = cute.local_tile(gDh_vk, (self.BV, BK_HALF), (v_tile_idx, 1))
            tSsDh = store_vk_thr_copy.partition_S(stateStage[(None, None, 0)])
            tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
            cute.autovec_copy(tSsDh, tSrDh)
            cute.copy(store_tiled_copy_vk, tSrDh, store_vk_thr_copy.partition_D(gDh1_tile))
            tSsDh = store_vk_thr_copy.partition_S(stateStage[(None, None, 1)])
            tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
            cute.autovec_copy(tSsDh, tSrDh)
            cute.copy(store_tiled_copy_vk, tSrDh, store_vk_thr_copy.partition_D(gDh2_tile))

            if full_sequence or full_chunk:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            if full_sequence:
                next_rev = chunk_rev + NUM_STAGES - 1
                if next_rev < NT:
                    self.copy_chunk_operands_async(
                        copy_contig_m0,
                        copy_m0_thr,
                        gK_bh_kt,
                        gQ_bh_kt,
                        gW_bh_kt,
                        gDo_bh_vt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sQ1,
                        sQ2,
                        sW1,
                        sW2,
                        sDo,
                        dvStage,
                        NT - 1 - next_rev,
                        v_tile_idx,
                        next_rev % NUM_STAGES,
                    )

            if not full_chunk:
                linear_k = tidx
                while linear_k < self.BT * BK_HALF:
                    t_rel = linear_k // BK_HALF
                    k_rel = linear_k - t_rel * BK_HALF
                    t_abs = chunk_start + t_rel
                    k_val1 = self.io_dtype(0.0)
                    k_val2 = self.io_dtype(0.0)
                    q_val1 = self.io_dtype(0.0)
                    q_val2 = self.io_dtype(0.0)
                    w_val1 = self.io_dtype(0.0)
                    w_val2 = self.io_dtype(0.0)
                    if t_abs < T:
                        k_val1 = k[bidx, t_abs, hidx, k_rel]
                        k_val2 = k[bidx, t_abs, hidx, k_rel + BK_HALF]
                        q_val1 = q[bidx, t_abs, hidx, k_rel]
                        q_val2 = q[bidx, t_abs, hidx, k_rel + BK_HALF]
                        w_val1 = w[bidx, t_abs, hidx, k_rel]
                        w_val2 = w[bidx, t_abs, hidx, k_rel + BK_HALF]
                    sK1[t_rel, k_rel, 0] = k_val1
                    sK2[t_rel, k_rel, 0] = k_val2
                    sQ1[k_rel, t_rel, 0] = q_val1
                    sQ2[k_rel, t_rel, 0] = q_val2
                    sW1[k_rel, t_rel, 0] = w_val1
                    sW2[k_rel, t_rel, 0] = w_val2
                    linear_k += self.num_threads

                linear_do = tidx
                while linear_do < self.BV * self.BT:
                    t_rel = linear_do // self.BV
                    v_rel = linear_do - t_rel * self.BV
                    t_abs = chunk_start + t_rel
                    do_val = self.io_dtype(0.0)
                    if t_abs < T:
                        do_val = do[bidx, t_abs, hidx, v_base + v_rel]
                    sDo[v_rel, t_rel, 0] = do_val
                    linear_do += self.num_threads

                linear_dv = tidx
                while linear_dv < self.BV * self.BT:
                    t_rel = linear_dv // self.BV
                    v_rel = linear_dv - t_rel * self.BV
                    t_abs = chunk_start + t_rel
                    dv_val = self.io_dtype(0.0)
                    if t_abs < T:
                        dv_val = dv[bidx, t_abs, hidx, v_base + v_rel]
                    dvStage[v_rel, t_rel, 0] = dv_val
                    linear_dv += self.num_threads

                cute.arch.barrier()

            # 2. acc_dv = K @ dh, split into two 64-wide K blocks like Triton.
            acc_dv.fill(0.0)
            state_a_layout = self._c_layout_as_a_layout(r_state1.layout, kdh_tiled_mma.tv_layout_A.shape[1])
            state1_op = cute.make_rmem_tensor(state_a_layout, dtype=self.io_dtype)
            state2_op = cute.make_rmem_tensor(state_a_layout, dtype=self.io_dtype)
            state1_op_as_acc = cute.make_tensor(state1_op.iterator, r_state1.layout)
            state2_op_as_acc = cute.make_tensor(state2_op.iterator, r_state2.layout)
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                state1_op_as_acc[ei] = r_state1[ei].to(self.io_dtype)
                state2_op_as_acc[ei] = r_state2[ei].to(self.io_dtype)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tK1R, mode=[2]), unroll_full=True):
                kdh_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(kdh_tiled_mma, acc_dv, state1_op[None, None, kp], tK1R[None, None, kp, current_stage], acc_dv)
            for kp in cutlass.range(cute.size(tK2R, mode=[2]), unroll_full=True):
                kdh_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                cute.gemm(kdh_tiled_mma, acc_dv, state2_op[None, None, kp], tK2R[None, None, kp, current_stage], acc_dv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # 3/4. dv2 = acc_dv + dv, then materialize it in shared memory.
            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                v_rel, t_rel = tCcDv[ei]
                t_abs = chunk_start + t_rel
                out = Float32(0.0)
                if cutlass.const_expr(self.full_chunks):
                    if t_rel < self.BT:
                        out = acc_dv[ei] + dvStage[v_rel, t_rel, current_stage].to(self.acc_dtype)
                else:
                    if t_abs < T:
                        out = acc_dv[ei] + dvStage[v_rel, t_rel, current_stage].to(self.acc_dtype)
                dv2_out = out.to(self.io_dtype)
                dv2_op_as_acc[ei] = dv2_out
                if cutlass.const_expr(self.full_chunks):
                    if t_rel < self.BT:
                        dv2[bidx, t_abs, hidx, v_base + v_rel] = dv2_out
                else:
                    if t_abs < T:
                        dv2[bidx, t_abs, hidx, v_base + v_rel] = dv2_out

            # 5/6. WDV and QDO are independent once dv2 is staged.  Queue the
            # four WGMMA groups first, then wait once before consuming them.
            acc_wdv1.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tW1R, mode=[2]), unroll_full=True):
                wdv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    wdv_tiled_mma,
                    acc_wdv1,
                    dv2_op[None, None, kp],
                    tW1R[None, None, kp, current_stage],
                    acc_wdv1,
                )
            cute.nvgpu.warpgroup.commit_group()

            acc_qdo1.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tQ1R, mode=[2]), unroll_full=True):
                update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    update_tiled_mma,
                    acc_qdo1,
                    tDoR[None, None, kp, current_stage],
                    tQ1R[None, None, kp, current_stage],
                    acc_qdo1,
                )
            cute.nvgpu.warpgroup.commit_group()

            acc_wdv2.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tW2R, mode=[2]), unroll_full=True):
                wdv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    wdv_tiled_mma,
                    acc_wdv2,
                    dv2_op[None, None, kp],
                    tW2R[None, None, kp, current_stage],
                    acc_wdv2,
                )
            cute.nvgpu.warpgroup.commit_group()

            acc_qdo2.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tQ2R, mode=[2]), unroll_full=True):
                update_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(
                    update_tiled_mma,
                    acc_qdo2,
                    tDoR[None, None, kp, current_stage],
                    tQ2R[None, None, kp, current_stage],
                    acc_qdo2,
                )
            cute.nvgpu.warpgroup.commit_group()

            cute.nvgpu.warpgroup.wait_group(0)

            # 7. dh = dh + scale * QDO - WDV.
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                delta1 = acc_qdo1[ei] * Float32(self.scale) - acc_wdv1[ei]
                delta2 = acc_qdo2[ei] * Float32(self.scale) - acc_wdv2[ei]
                r_state1[ei] = r_state1[ei] + delta1
                r_state2[ei] = r_state2[ei] + delta2

        if cutlass.const_expr(self.use_dh0):
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                v_rel, k_rel = tCcState[ei]
                dh0[bidx, hidx, k_rel, v_base + v_rel] = r_state1[ei]
                dh0[bidx, hidx, k_rel + BK_HALF, v_base + v_rel] = r_state2[ei]


@functools.lru_cache(maxsize=32)
def _compile_bwd_dhu_triton_baseline_sm90(
    H: int,
    K: int,
    V: int,
    use_dht: bool,
    use_dh0: bool,
    full_chunks: bool,
    bv: int,
    scale: float,
):
    kernel = ChunkDeltaRuleBwdDHUTritonBaselineSm90(
        num_heads=H,
        head_dim_k=K,
        head_dim_v=V,
        use_dht=use_dht,
        use_dh0=use_dh0,
        full_chunks=full_chunks,
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
    compiled = _compile_bwd_dhu_triton_baseline_sm90(
        H,
        K,
        V,
        dht is not None,
        h0 is not None,
        T % BT == 0,
        bv,
        scale_value,
    )
    compiled(q, k, w, dht_arg, dh0_arg, do, dh, dv, dv2, (Int32(B), Int32(T), Int32(NT)))
    return dh, dh0, dv2
