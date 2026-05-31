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

"""Experimental SM90 bwd_dhu baseline in Triton's [K,V] orientation.

This file is independent from ``chunk_delta_h_bwd.py`` and from the older
``[V,K]`` Triton baseline.  The first variant intentionally supports only:

* SM90 bf16
* K=V=128, BT=64, BV=64
* non-varlen, full chunks only
* USE_G=False and USE_GK=False

Each CTA owns one ``(B, H, Vtile)`` tile and carries the state as two
``[64, BV]`` register fragments, covering the full ``[K=128, Vtile]`` state.
The dot orientations match the standalone microkernels:

    KDH: K[T,K] @ dh[K,V] -> dv2[T,V]
    QDO: Q.T[K,T] @ do[T,V] -> qdo[K,V]
    WDV: W.T[K,T] @ dv2[T,V] -> wdv[K,V]
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T as _T

from cula.utils import assert_hopper

BT = 64
BV = 64
BK = 128
BK_HALF = 64
NUM_THREADS = 128


@cutlass.dsl_user_op
def _dep_zero_f32(dep, x, *, loc=None, ip=None):
    result = _llvm.inline_asm(
        _T.f32(),
        [dep.value, x.value],
        "sub.rn.f32 $0, $2, $2; add.rn.f32 $0, $0, $1;",
        "=f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


class ChunkDeltaRuleBwdDHUTritonOrientationSm90:
    def __init__(
        self,
        num_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        use_dht: bool,
        use_dh0: bool,
        scale: float,
        vectorize_dh_store: bool = False,
        vectorize_dv2_store: bool = False,
        load_pipeline_stages: int = 1,
    ):
        if head_dim_k != BK or head_dim_v != BK:
            raise NotImplementedError(f"Triton-orientation bwd_dhu only supports K=V=128, got K={head_dim_k}, V={head_dim_v}.")
        self.H = num_heads
        self.K = head_dim_k
        self.V = head_dim_v
        self.use_dht = use_dht
        self.use_dh0 = use_dh0
        self.scale = scale
        self.vectorize_dh_store = vectorize_dh_store
        self.vectorize_dv2_store = vectorize_dv2_store
        if load_pipeline_stages not in (1, 2):
            raise NotImplementedError(f"Only 1-stage or 2-stage load pipeline is supported, got {load_pipeline_stages}.")
        self.load_pipeline_stages = load_pipeline_stages
        self.num_threads = NUM_THREADS
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 128
        self.mma_tiler = (BT, BV, BK_HALF)
        self.atom_layout_mnk = (1, 1, 1)

    @cute.jit
    def copy_chunk_operands_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
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
        sDv: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gQ1_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gQ2_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gDo_tile = cute.local_tile(gDo_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))

        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ1_tile), copy_thr.partition_D(sQ1[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ2_tile), copy_thr.partition_D(sQ2[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDo_tile), copy_thr.partition_D(sDo[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, stage)]))
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

        kdh_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.mma_tiler[:2],
        )
        update_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.mma_tiler[:2],
        )

        load_stages = self.load_pipeline_stages
        k_layout = sm90_utils.make_smem_layout_a(utils.LayoutEnum.ROW_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        dh_read_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, 1)
        dh_store_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR, self.mma_tiler, self.io_dtype, 1)
        q_layout = sm90_utils.make_smem_layout_a(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        do_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        w_layout = sm90_utils.make_smem_layout_a(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        dv2_read_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, 1)
        dv2_store_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR, self.mma_tiler, self.io_dtype, 1)

        row_store_atom = sm90_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            self.io_dtype,
            self.acc_dtype,
        )
        dh_tiled_copy = cute.make_tiled_copy_C(row_store_atom, kdh_mma)
        dv2_tiled_copy = cute.make_tiled_copy_C(row_store_atom, kdh_mma)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        store_elems = 128 // self.io_dtype.width
        store_thr_dim_m = self.num_threads // (BV // store_elems)
        store_thr_dim_n = BV // store_elems
        store_thr_layout = cute.make_ordered_layout(
            (store_thr_dim_m, store_thr_dim_n),
            order=(1, 0),
        )
        store_val_layout = cute.make_layout((1, store_elems))
        store_tiled_copy_v = cute.make_tiled_copy_tv(store_atom, store_thr_layout, store_val_layout)
        load_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        load_thr_layout = cute.make_layout((self.num_threads // 32, 32), stride=(32, 1))
        load_tiled_copy = cute.make_tiled_copy_tv(load_atom, load_thr_layout, cute.make_layout((8, 1)))

        if cutlass.const_expr(self.load_pipeline_stages == 2):

            @cute.struct
            class SharedStorage:
                sK1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(k_layout)],
                    self.buffer_align_bytes,
                ]
                sK2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(k_layout)],
                    self.buffer_align_bytes,
                ]
                sDh1: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dh_read_layout), cute.cosize(dh_store_layout)),
                    ],
                    self.buffer_align_bytes,
                ]
                sDh2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dh_read_layout), cute.cosize(dh_store_layout)),
                    ],
                    self.buffer_align_bytes,
                ]
                sQ1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(q_layout)],
                    self.buffer_align_bytes,
                ]
                sQ2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(q_layout)],
                    self.buffer_align_bytes,
                ]
                sW1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(w_layout)],
                    self.buffer_align_bytes,
                ]
                sW2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(w_layout)],
                    self.buffer_align_bytes,
                ]
                sDo: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(do_layout)],
                    self.buffer_align_bytes,
                ]
                sDv: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(do_layout)],
                    self.buffer_align_bytes,
                ]
                sDv2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dv2_read_layout), cute.cosize(dv2_store_layout)),
                    ],
                    self.buffer_align_bytes,
                ]
        else:

            @cute.struct
            class SharedStorage:
                sK1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(k_layout)],
                    self.buffer_align_bytes,
                ]
                sK2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(k_layout)],
                    self.buffer_align_bytes,
                ]
                sDh1: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dh_read_layout), cute.cosize(dh_store_layout)),
                    ],
                    self.buffer_align_bytes,
                ]
                sDh2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dh_read_layout), cute.cosize(dh_store_layout)),
                    ],
                    self.buffer_align_bytes,
                ]
                sQ1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(q_layout)],
                    self.buffer_align_bytes,
                ]
                sQ2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(q_layout)],
                    self.buffer_align_bytes,
                ]
                sW1: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(w_layout)],
                    self.buffer_align_bytes,
                ]
                sW2: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(w_layout)],
                    self.buffer_align_bytes,
                ]
                sDo: cute.struct.Align[
                    cute.struct.MemRange[self.io_dtype, cute.cosize(do_layout)],
                    self.buffer_align_bytes,
                ]
                sDv2: cute.struct.Align[
                    cute.struct.MemRange[
                        self.io_dtype,
                        max(cute.cosize(dv2_read_layout), cute.cosize(dv2_store_layout)),
                    ],
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
            kdh_mma,
            update_mma,
            k_layout,
            dh_read_layout,
            dh_store_layout,
            q_layout,
            do_layout,
            w_layout,
            dv2_read_layout,
            dv2_store_layout,
            dh_tiled_copy,
            dv2_tiled_copy,
            store_tiled_copy_v,
            load_tiled_copy,
        ).launch(
            grid=[cute.ceil_div(self.V, BV), B * self.H, 1],
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
        kdh_mma: cute.TiledMma,
        update_mma: cute.TiledMma,
        k_layout: cute.ComposedLayout,
        dh_read_layout: cute.ComposedLayout,
        dh_store_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        do_layout: cute.ComposedLayout,
        w_layout: cute.ComposedLayout,
        dv2_read_layout: cute.ComposedLayout,
        dv2_store_layout: cute.ComposedLayout,
        dh_tiled_copy: cute.TiledCopy,
        dv2_tiled_copy: cute.TiledCopy,
        store_tiled_copy_v: cute.TiledCopy,
        load_tiled_copy: cute.TiledCopy,
    ):
        _, T, NT = problem_size
        tidx, _, _ = cute.arch.thread_idx()
        v_tile_idx, bh_idx, _ = cute.arch.block_idx()
        bidx = bh_idx // self.H
        hidx = bh_idx - bidx * self.H
        v_base = v_tile_idx * BV

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK1 = storage.sK1.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sK2 = storage.sK2.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sDh1 = storage.sDh1.get_tensor(dh_read_layout.outer, swizzle=dh_read_layout.inner)
        sDh2 = storage.sDh2.get_tensor(dh_read_layout.outer, swizzle=dh_read_layout.inner)
        sDh1Store = storage.sDh1.get_tensor(dh_store_layout.outer, swizzle=dh_store_layout.inner)
        sDh2Store = storage.sDh2.get_tensor(dh_store_layout.outer, swizzle=dh_store_layout.inner)
        sQ1 = storage.sQ1.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        sQ2 = storage.sQ2.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        sW1 = storage.sW1.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        sW2 = storage.sW2.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        sDo = storage.sDo.get_tensor(do_layout.outer, swizzle=do_layout.inner)
        if cutlass.const_expr(self.load_pipeline_stages == 2):
            sDv = storage.sDv.get_tensor(do_layout.outer, swizzle=do_layout.inner)
        sDv2 = storage.sDv2.get_tensor(dv2_read_layout.outer, swizzle=dv2_read_layout.inner)
        sDv2Store = storage.sDv2.get_tensor(dv2_store_layout.outer, swizzle=dv2_store_layout.inner)

        kdh_thr = kdh_mma.get_slice(tidx)
        update_thr = update_mma.get_slice(tidx)
        store_thr = store_tiled_copy_v.get_slice(tidx)
        load_thr = load_tiled_copy.get_slice(tidx)

        tK1A = kdh_thr.partition_A(sK1)
        tK1R = kdh_thr.make_fragment_A(tK1A)
        tK2A = kdh_thr.partition_A(sK2)
        tK2R = kdh_thr.make_fragment_A(tK2A)
        tDh1B = kdh_thr.partition_B(sDh1[(None, None, 0)])
        tDh1R = kdh_thr.make_fragment_B(tDh1B)
        tDh2B = kdh_thr.partition_B(sDh2[(None, None, 0)])
        tDh2R = kdh_thr.make_fragment_B(tDh2B)

        tQ1A = update_thr.partition_A(sQ1)
        tQ1R = update_thr.make_fragment_A(tQ1A)
        tQ2A = update_thr.partition_A(sQ2)
        tQ2R = update_thr.make_fragment_A(tQ2A)
        tW1A = update_thr.partition_A(sW1)
        tW1R = update_thr.make_fragment_A(tW1A)
        tW2A = update_thr.partition_A(sW2)
        tW2R = update_thr.make_fragment_A(tW2A)
        tDoB = update_thr.partition_B(sDo)
        tDoR = update_thr.make_fragment_B(tDoB)
        tDv2B = update_thr.partition_B(sDv2[(None, None, 0)])
        tDv2R = update_thr.make_fragment_B(tDv2B)

        c_tv = cute.make_identity_tensor((BT, BV))
        tCcTv = kdh_thr.partition_C(c_tv)
        c_kv = cute.make_identity_tensor((BK_HALF, BV))
        tCcKv = update_thr.partition_C(c_kv)
        acc_dv = kdh_thr.make_fragment_C(kdh_thr.partition_shape_C((BT, BV)))
        acc_wdv = update_thr.make_fragment_C(update_thr.partition_shape_C((BK_HALF, BV)))
        r_state1 = update_thr.make_fragment_C(update_thr.partition_shape_C((BK_HALF, BV)))
        r_state2 = update_thr.make_fragment_C(update_thr.partition_shape_C((BK_HALF, BV)))
        bf16_tmp = cute.make_rmem_tensor_like(acc_dv, self.io_dtype)

        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
            k_rel, v_rel = tCcKv[ei]
            value1 = Float32(0.0)
            value2 = Float32(0.0)
            if cutlass.const_expr(self.use_dht):
                value1 = dht[bidx, hidx, k_rel, v_base + v_rel]
                value2 = dht[bidx, hidx, k_rel + BK_HALF, v_base + v_rel]
            r_state1[ei] = value1
            r_state2[ei] = value2

        if cutlass.const_expr(self.load_pipeline_stages == 2):
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
            if NT > 0:
                self.copy_chunk_operands_async(
                    load_tiled_copy,
                    load_thr,
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
                    sDv,
                    NT - 1,
                    v_tile_idx,
                    Int32(0),
                )
            if NT > 1:
                self.copy_chunk_operands_async(
                    load_tiled_copy,
                    load_thr,
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
                    sDv,
                    NT - 2,
                    v_tile_idx,
                    Int32(1),
                )

        for chunk_rev in cutlass.range(0, NT, unroll=0):
            chunk_idx = NT - 1 - chunk_rev
            chunk_start = chunk_idx * BT
            current_stage = Int32(0)
            if cutlass.const_expr(self.load_pipeline_stages == 2):
                current_stage = chunk_rev % self.load_pipeline_stages

            # Store carried dh before this chunk updates it.
            if cutlass.const_expr(not self.vectorize_dh_store):
                for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                    k_rel, v_rel = tCcKv[ei]
                    dh[bidx, chunk_idx, hidx, k_rel, v_base + v_rel] = r_state1[ei].to(self.io_dtype)
                    dh[bidx, chunk_idx, hidx, k_rel + BK_HALF, v_base + v_rel] = r_state2[ei].to(self.io_dtype)

            if cutlass.const_expr(self.load_pipeline_stages == 1):
                linear = tidx
                while linear < BT * BK_HALF:
                    t_rel = linear // BK_HALF
                    k_rel = linear - t_rel * BK_HALF
                    t_abs = chunk_start + t_rel
                    sK1[t_rel, k_rel, 0] = k[bidx, t_abs, hidx, k_rel]
                    sK2[t_rel, k_rel, 0] = k[bidx, t_abs, hidx, k_rel + BK_HALF]
                    sQ1[k_rel, t_rel, 0] = q[bidx, t_abs, hidx, k_rel]
                    sQ2[k_rel, t_rel, 0] = q[bidx, t_abs, hidx, k_rel + BK_HALF]
                    sW1[k_rel, t_rel, 0] = w[bidx, t_abs, hidx, k_rel]
                    sW2[k_rel, t_rel, 0] = w[bidx, t_abs, hidx, k_rel + BK_HALF]
                    linear += self.num_threads

                linear_v = tidx
                while linear_v < BV * BT:
                    v_rel = linear_v // BT
                    t_rel = linear_v - v_rel * BT
                    t_abs = chunk_start + t_rel
                    sDo[v_rel, t_rel, 0] = do[bidx, t_abs, hidx, v_base + v_rel]
                    linear_v += self.num_threads

            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                bf16_tmp[ei] = r_state1[ei].to(self.io_dtype)
            copy_thr = dh_tiled_copy.get_slice(tidx)
            cute.copy(
                dh_tiled_copy,
                copy_thr.retile(bf16_tmp),
                copy_thr.partition_D(sDh1Store[(None, None, 0)]),
            )
            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                bf16_tmp[ei] = r_state2[ei].to(self.io_dtype)
            cute.copy(
                dh_tiled_copy,
                copy_thr.retile(bf16_tmp),
                copy_thr.partition_D(sDh2Store[(None, None, 0)]),
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier()

            if cutlass.const_expr(self.load_pipeline_stages == 2):
                if chunk_rev == 0:
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                next_chunk_idx = chunk_idx - 1
                if chunk_rev > 0 and next_chunk_idx >= 0:
                    self.copy_chunk_operands_async(
                        load_tiled_copy,
                        load_thr,
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
                        sDv,
                        next_chunk_idx,
                        v_tile_idx,
                        Int32(1) - current_stage,
                    )

            if cutlass.const_expr(self.vectorize_dh_store):
                dh_base_offset = (((bidx * NT + chunk_idx) * self.H + hidx) * BK + Int32(0)) * self.V + v_base
                dh_ptr = cute.make_ptr(
                    self.io_dtype,
                    (dh.iterator + dh_base_offset).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gDh1 = cute.make_tensor(
                    dh_ptr,
                    cute.make_layout(
                        (BK_HALF, BV),
                        stride=(self.V, 1),
                    ),
                )
                dh2_ptr = cute.make_ptr(
                    self.io_dtype,
                    (dh.iterator + dh_base_offset + BK_HALF * self.V).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gDh2 = cute.make_tensor(
                    dh2_ptr,
                    cute.make_layout(
                        (BK_HALF, BV),
                        stride=(self.V, 1),
                    ),
                )
                tSsDh = store_thr.partition_S(sDh1Store[(None, None, 0)])
                tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
                cute.autovec_copy(tSsDh, tSrDh)
                cute.copy(store_tiled_copy_v, tSrDh, store_thr.partition_D(gDh1))
                tSsDh = store_thr.partition_S(sDh2Store[(None, None, 0)])
                tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
                cute.autovec_copy(tSsDh, tSrDh)
                cute.copy(store_tiled_copy_v, tSrDh, store_thr.partition_D(gDh2))

            # dv2 = K1 @ dh1 + K2 @ dh2 + dv.
            acc_dv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDh1R, mode=[2]), unroll_full=True):
                kdh_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(kdh_mma, acc_dv, tK1R[None, None, kp, current_stage], tDh1R[None, None, kp], acc_dv)
                else:
                    cute.gemm(kdh_mma, acc_dv, tK1R[None, None, kp, 0], tDh1R[None, None, kp], acc_dv)
            for kp in cutlass.range(cute.size(tDh2R, mode=[2]), unroll_full=True):
                kdh_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(kdh_mma, acc_dv, tK2R[None, None, kp, current_stage], tDh2R[None, None, kp], acc_dv)
                else:
                    cute.gemm(kdh_mma, acc_dv, tK2R[None, None, kp, 0], tDh2R[None, None, kp], acc_dv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                t_rel, v_rel = tCcTv[ei]
                t_abs = chunk_start + t_rel
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    out = acc_dv[ei] + sDv[v_rel, t_rel, current_stage].to(self.acc_dtype)
                else:
                    out = acc_dv[ei] + dv[bidx, t_abs, hidx, v_base + v_rel].to(self.acc_dtype)
                out_bf16 = out.to(self.io_dtype)
                bf16_tmp[ei] = out_bf16
                if cutlass.const_expr(not self.vectorize_dv2_store):
                    dv2[bidx, t_abs, hidx, v_base + v_rel] = out_bf16

            copy_thr = dv2_tiled_copy.get_slice(tidx)
            cute.copy(
                dv2_tiled_copy,
                copy_thr.retile(bf16_tmp),
                copy_thr.partition_D(sDv2Store[(None, None, 0)]),
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier()

            if cutlass.const_expr(self.vectorize_dv2_store):
                dv2_base_offset = (bidx * T + chunk_start) * self.H * self.V + hidx * self.V + v_base
                dv2_ptr = cute.make_ptr(
                    self.io_dtype,
                    (dv2.iterator + dv2_base_offset).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gDv2 = cute.make_tensor(
                    dv2_ptr,
                    cute.make_layout(
                        (BT, BV),
                        stride=(self.H * self.V, 1),
                    ),
                )
                tSsDv2 = store_thr.partition_S(sDv2Store[(None, None, 0)])
                tSrDv2 = cute.make_fragment_like(tSsDv2, self.io_dtype)
                cute.autovec_copy(tSsDv2, tSrDv2)
                cute.copy(store_tiled_copy_v, tSrDv2, store_thr.partition_D(gDv2))

            # First K half: qdo1 = Q1.T @ do, wdv1 = W1.T @ dv2.
            acc_dv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(
                        update_mma, acc_dv, tQ1R[None, None, kp, current_stage], tDoR[None, None, kp, current_stage], acc_dv
                    )
                else:
                    cute.gemm(update_mma, acc_dv, tQ1R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
            cute.nvgpu.warpgroup.commit_group()

            acc_wdv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv)
                else:
                    cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            dep_zero = Float32(0.0)
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                r_state1[ei] = r_state1[ei] + delta
                dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])

            cute.arch.barrier()

            # Second K half: qdo2 = Q2.T @ do, wdv2 = W2.T @ dv2.
            acc_dv.fill(dep_zero)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(
                        update_mma, acc_dv, tQ2R[None, None, kp, current_stage], tDoR[None, None, kp, current_stage], acc_dv
                    )
                else:
                    cute.gemm(update_mma, acc_dv, tQ2R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
            cute.nvgpu.warpgroup.commit_group()

            acc_wdv.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    cute.gemm(update_mma, acc_wdv, tW2R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv)
                else:
                    cute.gemm(update_mma, acc_wdv, tW2R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                r_state2[ei] = r_state2[ei] + delta

            cute.arch.barrier()

        if cutlass.const_expr(self.use_dh0):
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                k_rel, v_rel = tCcKv[ei]
                dh0[bidx, hidx, k_rel, v_base + v_rel] = r_state1[ei]
                dh0[bidx, hidx, k_rel + BK_HALF, v_base + v_rel] = r_state2[ei]


@functools.lru_cache(maxsize=16)
def _compile_bwd_dhu_triton_orientation_sm90(
    H: int,
    K: int,
    V: int,
    use_dht: bool,
    use_dh0: bool,
    scale: float,
    vectorize_dh_store: bool = False,
    vectorize_dv2_store: bool = False,
    load_pipeline_stages: int = 1,
):
    kernel = ChunkDeltaRuleBwdDHUTritonOrientationSm90(
        num_heads=H,
        head_dim_k=K,
        head_dim_v=V,
        use_dht=use_dht,
        use_dh0=use_dh0,
        scale=scale,
        vectorize_dh_store=vectorize_dh_store,
        vectorize_dv2_store=vectorize_dv2_store,
        load_pipeline_stages=load_pipeline_stages,
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


def chunk_gated_delta_rule_bwd_dhu_sm90_triton_orientation(
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
    vectorize_dh_store: bool = False,
    vectorize_dv2_store: bool = False,
    load_pipeline_stages: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """FLA-compatible entry point for the Triton-orientation experiment."""
    del chunk_indices, chunk_offsets, use_exp2
    assert_hopper(q.device)
    if chunk_size != BT:
        raise NotImplementedError(f"Triton-orientation bwd_dhu only supports chunk_size={BT}.")
    if cu_seqlens is not None:
        raise NotImplementedError("Triton-orientation bwd_dhu currently supports non-varlen inputs only.")
    if g is not None or gk is not None:
        raise NotImplementedError("Triton-orientation bwd_dhu currently supports USE_G=False and USE_GK=False only.")
    if transpose_state_layout:
        raise NotImplementedError("Triton-orientation bwd_dhu currently supports transpose_state_layout=False only.")
    if load_pipeline_stages not in (1, 2):
        raise NotImplementedError(f"Only 1-stage or 2-stage load pipeline is supported, got {load_pipeline_stages}.")

    B, T, H, K = q.shape
    V = do.shape[-1]
    if K != BK or V != BK:
        raise NotImplementedError(f"Triton-orientation bwd_dhu only supports K=V=128, got K={K}, V={V}.")
    if T % BT != 0:
        raise NotImplementedError("Triton-orientation bwd_dhu first version requires full_chunks=True (T multiple of 64).")
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

    NT = T // BT
    scale_value = K**-0.5 if scale is None else float(scale)
    dh = q.new_empty(B, NT, H, K, V)
    dv2 = torch.empty_like(dv)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None

    dht_arg = dht if dht is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    dh0_arg = dh0 if dh0 is not None else torch.empty(state_shape, device=q.device, dtype=torch.float32)
    compiled = _compile_bwd_dhu_triton_orientation_sm90(
        H,
        K,
        V,
        dht is not None,
        h0 is not None,
        scale_value,
        bool(vectorize_dh_store),
        bool(vectorize_dv2_store),
        int(load_pipeline_stages),
    )
    compiled(q, k, w, dht_arg, dh0_arg, do, dh, dv, dv2, (Int32(B), Int32(T), Int32(NT)))
    return dh, dh0, dv2
