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
        dv_local_load: bool = False,
        do_dv_pipeline_stages: int | None = None,
        no_dh_store: bool = False,
        no_dv2_store: bool = False,
        no_qdo: bool = False,
        no_wdv: bool = False,
        kdh_dv_only: bool = False,
        schedule_variant: int = 0,
        late_dh_store: bool = False,
        update_variant: int = 0,
        early_dv2_store: bool = False,
        no_update_math: bool = False,
        dh_store_variant: int = 0,
        dh_store_pack_only: bool = False,
        kdh_skeleton_variant: int = 0,
        additive_variant: int = 0,
        wdv_skeleton_variant: int = 0,
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
        if load_pipeline_stages not in (1, 2, 3):
            raise NotImplementedError(
                f"Only 1-stage, 2-stage, or 3-stage load pipeline is supported, got {load_pipeline_stages}."
            )
        self.load_pipeline_stages = load_pipeline_stages
        self.do_dv_pipeline_stages = load_pipeline_stages if do_dv_pipeline_stages is None else do_dv_pipeline_stages
        if self.do_dv_pipeline_stages not in (1, 2, 3):
            raise NotImplementedError(
                f"Only 1-stage, 2-stage, or 3-stage do/dv pipeline is supported, got {self.do_dv_pipeline_stages}."
            )
        if self.do_dv_pipeline_stages > self.load_pipeline_stages:
            raise NotImplementedError("do/dv pipeline stages cannot exceed K/Q/W pipeline stages.")
        self.dv_local_load = dv_local_load
        if schedule_variant not in (0, 1, 2):
            raise NotImplementedError(f"Only schedule variants 0, 1, and 2 are supported, got {schedule_variant}.")
        if schedule_variant != 0 and (no_qdo or no_wdv or kdh_dv_only):
            raise NotImplementedError("benchmark decomposition flags are only supported with schedule_variant=0.")
        if update_variant not in (0, 1, 2, 3):
            raise NotImplementedError(f"Only update variants 0, 1, 2, and 3 are supported, got {update_variant}.")
        if schedule_variant != 0 and (update_variant != 0 or no_update_math):
            raise NotImplementedError("update lowering experiments are only supported with schedule_variant=0.")
        if dh_store_variant not in (0, 1, 2, 3):
            raise NotImplementedError(f"Only dh store variants 0, 1, 2, and 3 are supported, got {dh_store_variant}.")
        if late_dh_store and dh_store_variant != 0:
            raise NotImplementedError("legacy late_dh_store cannot be combined with dh_store_variant.")
        if kdh_skeleton_variant not in (0, 1, 2, 3, 4, 5):
            raise NotImplementedError(f"Only KDH skeleton variants 0..5 are supported, got {kdh_skeleton_variant}.")
        if additive_variant not in (0, 1, 2, 3, 4):
            raise NotImplementedError(f"Only additive reconstruction variants 0..4 are supported, got {additive_variant}.")
        if wdv_skeleton_variant not in (0, 1, 2, 3):
            raise NotImplementedError(f"Only WDV skeleton variants 0..3 are supported, got {wdv_skeleton_variant}.")
        if kdh_skeleton_variant != 0:
            kdh_dv_only = True
            no_qdo = True
            no_wdv = True
            no_dh_store = kdh_skeleton_variant not in (4, 5)
            no_dv2_store = kdh_skeleton_variant not in (3, 5)
        if wdv_skeleton_variant != 0:
            kdh_dv_only = False
            no_dh_store = True
            no_dv2_store = True
            no_qdo = True
            no_wdv = False
            no_update_math = True
        if additive_variant == 1:
            kdh_dv_only = False
            no_qdo = False
            no_wdv = True
            no_update_math = True
        elif additive_variant == 2:
            kdh_dv_only = False
            no_qdo = True
            no_wdv = False
            no_update_math = True
        elif additive_variant == 3:
            kdh_dv_only = False
            no_qdo = False
            no_wdv = False
            no_update_math = True
        elif additive_variant == 4:
            kdh_dv_only = False
            no_qdo = False
            no_wdv = False
            no_update_math = False
        self.no_dh_store = no_dh_store
        self.no_dv2_store = no_dv2_store
        self.no_qdo = no_qdo
        self.no_wdv = no_wdv
        self.kdh_dv_only = kdh_dv_only
        self.schedule_variant = schedule_variant
        self.late_dh_store = late_dh_store
        self.update_variant = update_variant
        self.early_dv2_store = early_dv2_store
        self.no_update_math = no_update_math
        self.dh_store_variant = dh_store_variant
        self.dh_store_pack_only = dh_store_pack_only
        self.kdh_skeleton_variant = kdh_skeleton_variant
        self.additive_variant = additive_variant
        self.wdv_skeleton_variant = wdv_skeleton_variant
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
        kqw_stage: Int32,
        do_dv_stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gQ1_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gQ2_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gDo_tile = cute.local_tile(gDo_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))

        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ1_tile), copy_thr.partition_D(sQ1[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ2_tile), copy_thr.partition_D(sQ2[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDo_tile), copy_thr.partition_D(sDo[(None, None, do_dv_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, do_dv_stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_kqw_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gK_bh_kt: cute.Tensor,
        gQ_bh_kt: cute.Tensor,
        gW_bh_kt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        sQ1: cute.Tensor,
        sQ2: cute.Tensor,
        sW1: cute.Tensor,
        sW2: cute.Tensor,
        chunk_idx: Int32,
        stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gQ1_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gQ2_tile = cute.local_tile(gQ_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))

        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ1_tile), copy_thr.partition_D(sQ1[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gQ2_tile), copy_thr.partition_D(sQ2[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_dodv_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gDo_bh_vt: cute.Tensor,
        gDv_bh_vt: cute.Tensor,
        sDo: cute.Tensor,
        sDv: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        stage: Int32,
    ):
        gDo_tile = cute.local_tile(gDo_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDo_tile), copy_thr.partition_D(sDo[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_k_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gK_bh_kt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        chunk_idx: Int32,
        stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_kdv_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gK_bh_kt: cute.Tensor,
        gDv_bh_vt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        sDv: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        k_stage: Int32,
        dv_stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, k_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, k_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, dv_stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_wdv_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gW_bh_kt: cute.Tensor,
        gDv_bh_vt: cute.Tensor,
        sW1: cute.Tensor,
        sW2: cute.Tensor,
        sDv: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        kqw_stage: Int32,
        dv_stage: Int32,
    ):
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, dv_stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_kw_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gK_bh_kt: cute.Tensor,
        gW_bh_kt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        sW1: cute.Tensor,
        sW2: cute.Tensor,
        chunk_idx: Int32,
        stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def copy_chunk_kwdv_async(
        self,
        copy_contig_m0: cute.TiledCopy,
        copy_thr,
        gK_bh_kt: cute.Tensor,
        gW_bh_kt: cute.Tensor,
        gDv_bh_vt: cute.Tensor,
        sK1_kt: cute.Tensor,
        sK2_kt: cute.Tensor,
        sW1: cute.Tensor,
        sW2: cute.Tensor,
        sDv: cute.Tensor,
        chunk_idx: Int32,
        v_tile_idx: Int32,
        kqw_stage: Int32,
        dv_stage: Int32,
    ):
        gK1_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gK2_tile = cute.local_tile(gK_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gW1_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (0, chunk_idx))
        gW2_tile = cute.local_tile(gW_bh_kt, (BK_HALF, BT), (1, chunk_idx))
        gDv_tile = cute.local_tile(gDv_bh_vt, (BV, BT), (v_tile_idx, chunk_idx))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK1_tile), copy_thr.partition_D(sK1_kt[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gK2_tile), copy_thr.partition_D(sK2_kt[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW1_tile), copy_thr.partition_D(sW1[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gW2_tile), copy_thr.partition_D(sW2[(None, None, kqw_stage)]))
        cute.copy(copy_contig_m0, copy_thr.partition_S(gDv_tile), copy_thr.partition_D(sDv[(None, None, dv_stage)]))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def store_dh_vectorized(
        self,
        store_tiled_copy_v: cute.TiledCopy,
        store_thr,
        dh: cute.Tensor,
        sDh1Store: cute.Tensor,
        sDh2Store: cute.Tensor,
        bidx: Int32,
        chunk_idx: Int32,
        hidx: Int32,
        NT: Int32,
        v_base: Int32,
    ):
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

    @cute.jit
    def pack_dh_vectorized_only(
        self,
        store_tiled_copy_v: cute.TiledCopy,
        store_thr,
        sDh1Store: cute.Tensor,
        sDh2Store: cute.Tensor,
    ) -> Float32:
        del store_tiled_copy_v
        dep_zero = Float32(0.0)
        tSsDh = store_thr.partition_S(sDh1Store[(None, None, 0)])
        tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
        cute.autovec_copy(tSsDh, tSrDh)
        for ei in cutlass.range(1, unroll_full=True):
            dep_zero = _dep_zero_f32(dep_zero, tSrDh[ei].to(self.acc_dtype))
        tSsDh = store_thr.partition_S(sDh2Store[(None, None, 0)])
        tSrDh = cute.make_fragment_like(tSsDh, self.io_dtype)
        cute.autovec_copy(tSsDh, tSrDh)
        for ei in cutlass.range(1, unroll_full=True):
            dep_zero = _dep_zero_f32(dep_zero, tSrDh[ei].to(self.acc_dtype))
        return dep_zero

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
        do_dv_stages = self.do_dv_pipeline_stages
        k_layout = sm90_utils.make_smem_layout_a(utils.LayoutEnum.ROW_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        dh_read_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, 1)
        dh_store_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR, self.mma_tiler, self.io_dtype, 1)
        q_layout = sm90_utils.make_smem_layout_a(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, load_stages)
        do_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.COL_MAJOR, self.mma_tiler, self.io_dtype, do_dv_stages)
        if cutlass.const_expr(self.dv_local_load):
            dv_stage_layout = sm90_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (BT, BV),
                do_dv_stages,
            )
        else:
            dv_stage_layout = do_layout
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
        dv_tiled_load = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.io_dtype,
            ),
            kdh_mma,
        )
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

        if cutlass.const_expr(self.load_pipeline_stages > 1):

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
                    cute.struct.MemRange[self.io_dtype, cute.cosize(dv_stage_layout)],
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
            dv_stage_layout,
            w_layout,
            dv2_read_layout,
            dv2_store_layout,
            dh_tiled_copy,
            dv2_tiled_copy,
            dv_tiled_load,
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
        dv_stage_layout: cute.ComposedLayout,
        w_layout: cute.ComposedLayout,
        dv2_read_layout: cute.ComposedLayout,
        dv2_store_layout: cute.ComposedLayout,
        dh_tiled_copy: cute.TiledCopy,
        dv2_tiled_copy: cute.TiledCopy,
        dv_tiled_load: cute.TiledCopy,
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
        if cutlass.const_expr(self.load_pipeline_stages > 1):
            sDv = storage.sDv.get_tensor(dv_stage_layout.outer, swizzle=dv_stage_layout.inner)
            if cutlass.const_expr(self.dv_local_load):
                sDvCopy = cute.make_tensor(sDv.iterator, cute.select(sDv.layout, mode=[1, 0, 2]))
            else:
                sDvCopy = sDv
        sDv2 = storage.sDv2.get_tensor(dv2_read_layout.outer, swizzle=dv2_read_layout.inner)
        sDv2Store = storage.sDv2.get_tensor(dv2_store_layout.outer, swizzle=dv2_store_layout.inner)

        kdh_thr = kdh_mma.get_slice(tidx)
        update_thr = update_mma.get_slice(tidx)
        store_thr = store_tiled_copy_v.get_slice(tidx)
        load_thr = load_tiled_copy.get_slice(tidx)
        if cutlass.const_expr(self.dv_local_load):
            dv_load_thr = dv_tiled_load.get_slice(tidx)

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
        if cutlass.const_expr(self.schedule_variant != 0):
            acc_wdv2 = update_thr.make_fragment_C(update_thr.partition_shape_C((BK_HALF, BV)))
        if cutlass.const_expr(self.update_variant == 1):
            update_tmp = update_thr.make_fragment_C(update_thr.partition_shape_C((BK_HALF, BV)))
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

        if cutlass.const_expr(self.load_pipeline_stages > 1):
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
                if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                    self.copy_chunk_wdv_async(
                        load_tiled_copy,
                        load_thr,
                        gW_bh_kt,
                        gDv_bh_vt,
                        sW1,
                        sW2,
                        sDvCopy,
                        NT - 1,
                        v_tile_idx,
                        Int32(0),
                        Int32(0),
                    )
                elif cutlass.const_expr(self.wdv_skeleton_variant == 2):
                    self.copy_chunk_kw_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gW_bh_kt,
                        sK1_kt,
                        sK2_kt,
                        sW1,
                        sW2,
                        NT - 1,
                        Int32(0),
                    )
                elif cutlass.const_expr(self.wdv_skeleton_variant == 3):
                    self.copy_chunk_kwdv_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gW_bh_kt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sW1,
                        sW2,
                        sDvCopy,
                        NT - 1,
                        v_tile_idx,
                        Int32(0),
                        Int32(0),
                    )
                elif cutlass.const_expr(self.kdh_skeleton_variant == 1):
                    self.copy_chunk_k_async(load_tiled_copy, load_thr, gK_bh_kt, sK1_kt, sK2_kt, NT - 1, Int32(0))
                elif cutlass.const_expr(self.kdh_skeleton_variant != 0):
                    self.copy_chunk_kdv_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sDvCopy,
                        NT - 1,
                        v_tile_idx,
                        Int32(0),
                        Int32(0),
                    )
                else:
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
                        sDvCopy,
                        NT - 1,
                        v_tile_idx,
                        Int32(0),
                        Int32(0),
                    )
            if NT > 1:
                if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                    self.copy_chunk_wdv_async(
                        load_tiled_copy,
                        load_thr,
                        gW_bh_kt,
                        gDv_bh_vt,
                        sW1,
                        sW2,
                        sDvCopy,
                        NT - 2,
                        v_tile_idx,
                        Int32(1),
                        Int32(1),
                    )
                elif cutlass.const_expr(self.wdv_skeleton_variant == 2):
                    self.copy_chunk_kw_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gW_bh_kt,
                        sK1_kt,
                        sK2_kt,
                        sW1,
                        sW2,
                        NT - 2,
                        Int32(1),
                    )
                elif cutlass.const_expr(self.wdv_skeleton_variant == 3):
                    self.copy_chunk_kwdv_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gW_bh_kt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sW1,
                        sW2,
                        sDvCopy,
                        NT - 2,
                        v_tile_idx,
                        Int32(1),
                        Int32(1),
                    )
                elif cutlass.const_expr(self.kdh_skeleton_variant == 1):
                    self.copy_chunk_k_async(load_tiled_copy, load_thr, gK_bh_kt, sK1_kt, sK2_kt, NT - 2, Int32(1))
                elif cutlass.const_expr(self.kdh_skeleton_variant != 0):
                    self.copy_chunk_kdv_async(
                        load_tiled_copy,
                        load_thr,
                        gK_bh_kt,
                        gDv_bh_vt,
                        sK1_kt,
                        sK2_kt,
                        sDvCopy,
                        NT - 2,
                        v_tile_idx,
                        Int32(1),
                        Int32(1),
                    )
                else:
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
                        sDvCopy,
                        NT - 2,
                        v_tile_idx,
                        Int32(1),
                        Int32(1),
                    )
            if cutlass.const_expr(self.load_pipeline_stages == 3):
                if NT > 2:
                    if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                        self.copy_chunk_wdv_async(
                            load_tiled_copy,
                            load_thr,
                            gW_bh_kt,
                            gDv_bh_vt,
                            sW1,
                            sW2,
                            sDvCopy,
                            NT - 3,
                            v_tile_idx,
                            Int32(2),
                            Int32(2),
                        )
                    elif cutlass.const_expr(self.wdv_skeleton_variant == 2):
                        self.copy_chunk_kw_async(
                            load_tiled_copy,
                            load_thr,
                            gK_bh_kt,
                            gW_bh_kt,
                            sK1_kt,
                            sK2_kt,
                            sW1,
                            sW2,
                            NT - 3,
                            Int32(2),
                        )
                    elif cutlass.const_expr(self.wdv_skeleton_variant == 3):
                        self.copy_chunk_kwdv_async(
                            load_tiled_copy,
                            load_thr,
                            gK_bh_kt,
                            gW_bh_kt,
                            gDv_bh_vt,
                            sK1_kt,
                            sK2_kt,
                            sW1,
                            sW2,
                            sDvCopy,
                            NT - 3,
                            v_tile_idx,
                            Int32(2),
                            Int32(2),
                        )
                    elif cutlass.const_expr(self.kdh_skeleton_variant == 1):
                        self.copy_chunk_k_async(load_tiled_copy, load_thr, gK_bh_kt, sK1_kt, sK2_kt, NT - 3, Int32(2))
                    elif cutlass.const_expr(self.kdh_skeleton_variant != 0):
                        self.copy_chunk_kdv_async(
                            load_tiled_copy,
                            load_thr,
                            gK_bh_kt,
                            gDv_bh_vt,
                            sK1_kt,
                            sK2_kt,
                            sDvCopy,
                            NT - 3,
                            v_tile_idx,
                            Int32(2),
                            Int32(2),
                        )
                    elif cutlass.const_expr(self.do_dv_pipeline_stages == 3):
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
                            sDvCopy,
                            NT - 3,
                            v_tile_idx,
                            Int32(2),
                            Int32(2),
                        )
                    else:
                        self.copy_chunk_kqw_async(
                            load_tiled_copy,
                            load_thr,
                            gK_bh_kt,
                            gQ_bh_kt,
                            gW_bh_kt,
                            sK1_kt,
                            sK2_kt,
                            sQ1,
                            sQ2,
                            sW1,
                            sW2,
                            NT - 3,
                            Int32(2),
                        )

        for chunk_rev in cutlass.range(0, NT, unroll=0):
            chunk_idx = NT - 1 - chunk_rev
            chunk_start = chunk_idx * BT
            current_stage = Int32(0)
            current_do_dv_stage = Int32(0)
            if cutlass.const_expr(self.load_pipeline_stages > 1):
                current_stage = chunk_rev % self.load_pipeline_stages
            if cutlass.const_expr(self.do_dv_pipeline_stages > 1):
                current_do_dv_stage = chunk_rev % self.do_dv_pipeline_stages

            # Store carried dh before this chunk updates it.
            if cutlass.const_expr(not self.vectorize_dh_store and not self.no_dh_store):
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
                    if cutlass.const_expr(self.kdh_skeleton_variant == 0):
                        sQ1[k_rel, t_rel, 0] = q[bidx, t_abs, hidx, k_rel]
                        sQ2[k_rel, t_rel, 0] = q[bidx, t_abs, hidx, k_rel + BK_HALF]
                        sW1[k_rel, t_rel, 0] = w[bidx, t_abs, hidx, k_rel]
                        sW2[k_rel, t_rel, 0] = w[bidx, t_abs, hidx, k_rel + BK_HALF]
                    linear += self.num_threads

                if cutlass.const_expr(self.kdh_skeleton_variant == 0):
                    linear_v = tidx
                    while linear_v < BV * BT:
                        v_rel = linear_v // BT
                        t_rel = linear_v - v_rel * BT
                        t_abs = chunk_start + t_rel
                        sDo[v_rel, t_rel, 0] = do[bidx, t_abs, hidx, v_base + v_rel]
                        linear_v += self.num_threads

            if cutlass.const_expr(self.wdv_skeleton_variant != 1):
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

            if cutlass.const_expr(self.load_pipeline_stages > 1):
                if cutlass.const_expr(self.load_pipeline_stages == 2):
                    if chunk_rev == 0 and NT > 1:
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                else:
                    if chunk_rev == 0 and NT > 2:
                        cute.arch.cp_async_wait_group(2)
                    elif NT - chunk_rev > 1:
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                next_chunk_idx = chunk_idx - 1
                next_stage = Int32(1) - current_stage
                if cutlass.const_expr(self.load_pipeline_stages == 3):
                    next_chunk_idx = chunk_idx - (self.load_pipeline_stages - 1)
                    next_stage = (chunk_rev - 1) % self.load_pipeline_stages
                should_prefetch = next_chunk_idx >= 0
                should_prefetch = should_prefetch and chunk_rev > 0
                if cutlass.const_expr(self.wdv_skeleton_variant != 0):
                    if should_prefetch:
                        if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                            self.copy_chunk_wdv_async(
                                load_tiled_copy,
                                load_thr,
                                gW_bh_kt,
                                gDv_bh_vt,
                                sW1,
                                sW2,
                                sDvCopy,
                                next_chunk_idx,
                                v_tile_idx,
                                next_stage,
                                next_stage,
                            )
                        elif cutlass.const_expr(self.wdv_skeleton_variant == 2):
                            self.copy_chunk_kw_async(
                                load_tiled_copy,
                                load_thr,
                                gK_bh_kt,
                                gW_bh_kt,
                                sK1_kt,
                                sK2_kt,
                                sW1,
                                sW2,
                                next_chunk_idx,
                                next_stage,
                            )
                        else:
                            self.copy_chunk_kwdv_async(
                                load_tiled_copy,
                                load_thr,
                                gK_bh_kt,
                                gW_bh_kt,
                                gDv_bh_vt,
                                sK1_kt,
                                sK2_kt,
                                sW1,
                                sW2,
                                sDvCopy,
                                next_chunk_idx,
                                v_tile_idx,
                                next_stage,
                                next_stage,
                            )
                elif cutlass.const_expr(self.kdh_skeleton_variant != 0):
                    if should_prefetch:
                        if cutlass.const_expr(self.kdh_skeleton_variant == 1):
                            self.copy_chunk_k_async(
                                load_tiled_copy,
                                load_thr,
                                gK_bh_kt,
                                sK1_kt,
                                sK2_kt,
                                next_chunk_idx,
                                next_stage,
                            )
                        else:
                            self.copy_chunk_kdv_async(
                                load_tiled_copy,
                                load_thr,
                                gK_bh_kt,
                                gDv_bh_vt,
                                sK1_kt,
                                sK2_kt,
                                sDvCopy,
                                next_chunk_idx,
                                v_tile_idx,
                                next_stage,
                                next_stage,
                            )
                elif cutlass.const_expr(self.load_pipeline_stages == 3 and self.do_dv_pipeline_stages == 2):
                    if chunk_rev > 0:
                        next_do_dv_chunk_idx = chunk_idx - (self.do_dv_pipeline_stages - 1)
                        next_do_dv_stage = (chunk_rev - 1) % self.do_dv_pipeline_stages
                        if next_do_dv_chunk_idx >= 0:
                            self.copy_chunk_dodv_async(
                                load_tiled_copy,
                                load_thr,
                                gDo_bh_vt,
                                gDv_bh_vt,
                                sDo,
                                sDvCopy,
                                next_do_dv_chunk_idx,
                                v_tile_idx,
                                next_do_dv_stage,
                            )
                        if next_chunk_idx >= 0:
                            self.copy_chunk_kqw_async(
                                load_tiled_copy,
                                load_thr,
                                gK_bh_kt,
                                gQ_bh_kt,
                                gW_bh_kt,
                                sK1_kt,
                                sK2_kt,
                                sQ1,
                                sQ2,
                                sW1,
                                sW2,
                                next_chunk_idx,
                                next_stage,
                            )
                else:
                    if should_prefetch:
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
                            sDvCopy,
                            next_chunk_idx,
                            v_tile_idx,
                            next_stage,
                            next_stage,
                        )

            dh_store_dep = Float32(0.0)
            if cutlass.const_expr(
                self.vectorize_dh_store
                and not self.late_dh_store
                and self.dh_store_variant in (0, 1)
                and (not self.no_dh_store or self.dh_store_pack_only)
            ):
                if cutlass.const_expr(self.dh_store_pack_only):
                    dh_store_dep = self.pack_dh_vectorized_only(
                        store_tiled_copy_v,
                        store_thr,
                        sDh1Store,
                        sDh2Store,
                    )
                elif cutlass.const_expr(not self.no_dh_store):
                    self.store_dh_vectorized(
                        store_tiled_copy_v,
                        store_thr,
                        dh,
                        sDh1Store,
                        sDh2Store,
                        bidx,
                        chunk_idx,
                        hidx,
                        NT,
                        v_base,
                    )
                if cutlass.const_expr(self.dh_store_variant == 1):
                    cute.arch.barrier()

            # dv2 = K1 @ dh1 + K2 @ dh2 + dv.
            if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                acc_dv.fill(0.0)
            else:
                acc_dv.fill(dh_store_dep)
                cute.nvgpu.warpgroup.fence()
                for kp in cutlass.range(cute.size(tDh1R, mode=[2]), unroll_full=True):
                    kdh_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    if cutlass.const_expr(self.load_pipeline_stages > 1):
                        cute.gemm(kdh_mma, acc_dv, tK1R[None, None, kp, current_stage], tDh1R[None, None, kp], acc_dv)
                    else:
                        cute.gemm(kdh_mma, acc_dv, tK1R[None, None, kp, 0], tDh1R[None, None, kp], acc_dv)
                for kp in cutlass.range(cute.size(tDh2R, mode=[2]), unroll_full=True):
                    kdh_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                    if cutlass.const_expr(self.load_pipeline_stages > 1):
                        cute.gemm(kdh_mma, acc_dv, tK2R[None, None, kp, current_stage], tDh2R[None, None, kp], acc_dv)
                    else:
                        cute.gemm(kdh_mma, acc_dv, tK2R[None, None, kp, 0], tDh2R[None, None, kp], acc_dv)
                cute.nvgpu.warpgroup.commit_group()
            if cutlass.const_expr(
                self.vectorize_dh_store
                and (self.late_dh_store or self.dh_store_variant == 2)
                and (not self.no_dh_store or self.dh_store_pack_only)
            ):
                if cutlass.const_expr(self.dh_store_variant == 2):
                    cute.arch.barrier()
                if cutlass.const_expr(self.dh_store_pack_only):
                    dh_store_dep = self.pack_dh_vectorized_only(
                        store_tiled_copy_v,
                        store_thr,
                        sDh1Store,
                        sDh2Store,
                    )
                elif cutlass.const_expr(not self.no_dh_store):
                    self.store_dh_vectorized(
                        store_tiled_copy_v,
                        store_thr,
                        dh,
                        sDh1Store,
                        sDh2Store,
                        bidx,
                        chunk_idx,
                        hidx,
                        NT,
                        v_base,
                    )
            if cutlass.const_expr(self.wdv_skeleton_variant != 1):
                cute.nvgpu.warpgroup.wait_group(0)

            if cutlass.const_expr(self.kdh_skeleton_variant == 1):
                for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                    bf16_tmp[ei] = acc_dv[ei].to(self.io_dtype)

            if cutlass.const_expr(self.kdh_skeleton_variant != 1 and self.load_pipeline_stages > 1 and self.dv_local_load):
                tDvR = dv_load_thr.retile(bf16_tmp)
                cute.copy(
                    dv_tiled_load,
                    dv_load_thr.partition_S(sDv[(None, None, current_do_dv_stage)]),
                    tDvR,
                )

            if cutlass.const_expr(self.kdh_skeleton_variant != 1):
                for ei in cutlass.range(cute.size(acc_dv), unroll_full=True):
                    t_rel, v_rel = tCcTv[ei]
                    t_abs = chunk_start + t_rel
                    if cutlass.const_expr(self.wdv_skeleton_variant == 2):
                        out = acc_dv[ei]
                    elif cutlass.const_expr(self.load_pipeline_stages > 1):
                        if cutlass.const_expr(self.dv_local_load):
                            out = acc_dv[ei] + bf16_tmp[ei].to(self.acc_dtype)
                        else:
                            out = acc_dv[ei] + sDv[v_rel, t_rel, current_do_dv_stage].to(self.acc_dtype)
                    else:
                        out = acc_dv[ei] + dv[bidx, t_abs, hidx, v_base + v_rel].to(self.acc_dtype)
                    out_bf16 = out.to(self.io_dtype)
                    bf16_tmp[ei] = out_bf16
                    if cutlass.const_expr(not self.vectorize_dv2_store and not self.no_dv2_store):
                        dv2[bidx, t_abs, hidx, v_base + v_rel] = out_bf16

            if cutlass.const_expr((not self.no_dv2_store) or (not self.kdh_dv_only) or self.kdh_skeleton_variant in (1, 2, 4)):
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

                if cutlass.const_expr(self.vectorize_dv2_store and not self.no_dv2_store):
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
                    if cutlass.const_expr(self.early_dv2_store):
                        cute.arch.barrier()

            if cutlass.const_expr(
                self.vectorize_dh_store and self.dh_store_variant == 3 and (not self.no_dh_store or self.dh_store_pack_only)
            ):
                cute.arch.barrier()
                if cutlass.const_expr(self.dh_store_pack_only):
                    dh_store_dep = self.pack_dh_vectorized_only(
                        store_tiled_copy_v,
                        store_thr,
                        sDh1Store,
                        sDh2Store,
                    )
                elif cutlass.const_expr(not self.no_dh_store):
                    self.store_dh_vectorized(
                        store_tiled_copy_v,
                        store_thr,
                        dh,
                        sDh1Store,
                        sDh2Store,
                        bidx,
                        chunk_idx,
                        hidx,
                        NT,
                        v_base,
                    )

            if cutlass.const_expr(not self.kdh_dv_only):
                if cutlass.const_expr(self.schedule_variant == 0):
                    # S0: current half-by-half schedule.
                    if cutlass.const_expr(not self.no_qdo):
                        acc_dv.fill(0.0)
                        cute.nvgpu.warpgroup.fence()
                        for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                            update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                            if cutlass.const_expr(self.load_pipeline_stages > 1):
                                cute.gemm(
                                    update_mma,
                                    acc_dv,
                                    tQ1R[None, None, kp, current_stage],
                                    tDoR[None, None, kp, current_do_dv_stage],
                                    acc_dv,
                                )
                            else:
                                cute.gemm(update_mma, acc_dv, tQ1R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                        cute.nvgpu.warpgroup.commit_group()
                    else:
                        acc_dv.fill(0.0)

                    dep_zero = Float32(0.0)
                    if cutlass.const_expr(not self.no_wdv):
                        acc_wdv.fill(0.0)
                        cute.nvgpu.warpgroup.fence()
                        for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                            update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                            if cutlass.const_expr(self.load_pipeline_stages > 1):
                                cute.gemm(
                                    update_mma, acc_wdv, tW1R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv
                                )
                            else:
                                cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
                        cute.nvgpu.warpgroup.commit_group()
                    else:
                        acc_wdv.fill(0.0)
                    if cutlass.const_expr(not (self.no_qdo and self.no_wdv)):
                        cute.nvgpu.warpgroup.wait_group(0)

                    if cutlass.const_expr(self.no_update_math):
                        if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                            for ei in cutlass.range(8, unroll_full=True):
                                dep_zero = _dep_zero_f32(dep_zero, acc_wdv[ei])
                            r_state1[0] = _dep_zero_f32(r_state1[0], dep_zero)
                        else:
                            for ei in cutlass.range(1, unroll_full=True):
                                if cutlass.const_expr(not self.no_qdo):
                                    dep_zero = _dep_zero_f32(dep_zero, acc_dv[ei])
                                if cutlass.const_expr(not self.no_wdv):
                                    dep_zero = _dep_zero_f32(dep_zero, acc_wdv[ei])
                                r_state1[ei] = _dep_zero_f32(r_state1[ei], dep_zero)
                        if cutlass.const_expr(self.wdv_skeleton_variant != 0):
                            if tidx == 0:
                                dv2[bidx, chunk_start, hidx, v_base] = dep_zero.to(self.io_dtype)
                    elif cutlass.const_expr(self.update_variant == 1):
                        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                            update_tmp[ei] = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                            r_state1[ei] = r_state1[ei] + update_tmp[ei]
                            dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])
                    elif cutlass.const_expr(self.update_variant == 2):
                        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                            delta = acc_dv[ei] * Float32(self.scale) + (-acc_wdv[ei])
                            r_state1[ei] = r_state1[ei] + delta
                            dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])
                    elif cutlass.const_expr(self.update_variant == 3):
                        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                            r_state1[ei] = r_state1[ei] + (acc_dv[ei] * Float32(self.scale) - acc_wdv[ei])
                            dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])
                    else:
                        for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                            delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                            r_state1[ei] = r_state1[ei] + delta
                            dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])

                    if cutlass.const_expr(not self.no_qdo):
                        acc_dv.fill(dep_zero)
                        cute.nvgpu.warpgroup.fence()
                        for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                            update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                            if cutlass.const_expr(self.load_pipeline_stages > 1):
                                cute.gemm(
                                    update_mma,
                                    acc_dv,
                                    tQ2R[None, None, kp, current_stage],
                                    tDoR[None, None, kp, current_do_dv_stage],
                                    acc_dv,
                                )
                            else:
                                cute.gemm(update_mma, acc_dv, tQ2R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                        cute.nvgpu.warpgroup.commit_group()
                    else:
                        acc_dv.fill(dep_zero)

                    dep_zero = Float32(0.0)
                    if cutlass.const_expr(not self.no_wdv):
                        acc_wdv.fill(0.0)
                        cute.nvgpu.warpgroup.fence()
                        for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                            update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                            if cutlass.const_expr(self.load_pipeline_stages > 1):
                                cute.gemm(
                                    update_mma, acc_wdv, tW2R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv
                                )
                            else:
                                cute.gemm(update_mma, acc_wdv, tW2R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
                        cute.nvgpu.warpgroup.commit_group()
                    else:
                        acc_wdv.fill(0.0)
                    if cutlass.const_expr(not (self.no_qdo and self.no_wdv)):
                        cute.nvgpu.warpgroup.wait_group(0)

                    if cutlass.const_expr(self.no_update_math):
                        if cutlass.const_expr(self.wdv_skeleton_variant == 1):
                            for ei in cutlass.range(8, unroll_full=True):
                                dep_zero = _dep_zero_f32(dep_zero, acc_wdv[ei])
                            r_state2[0] = _dep_zero_f32(r_state2[0], dep_zero)
                        else:
                            for ei in cutlass.range(1, unroll_full=True):
                                if cutlass.const_expr(not self.no_qdo):
                                    dep_zero = _dep_zero_f32(dep_zero, acc_dv[ei])
                                if cutlass.const_expr(not self.no_wdv):
                                    dep_zero = _dep_zero_f32(dep_zero, acc_wdv[ei])
                                r_state2[ei] = _dep_zero_f32(r_state2[ei], dep_zero)
                        if cutlass.const_expr(self.wdv_skeleton_variant != 0):
                            if tidx == 0:
                                dv2[bidx, chunk_start, hidx, v_base] = dep_zero.to(self.io_dtype)
                    elif cutlass.const_expr(self.update_variant == 1):
                        for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                            update_tmp[ei] = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                        for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                            r_state2[ei] = r_state2[ei] + update_tmp[ei]
                    elif cutlass.const_expr(self.update_variant == 2):
                        for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                            delta = acc_dv[ei] * Float32(self.scale) + (-acc_wdv[ei])
                            r_state2[ei] = r_state2[ei] + delta
                    elif cutlass.const_expr(self.update_variant == 3):
                        for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                            r_state2[ei] = r_state2[ei] + (acc_dv[ei] * Float32(self.scale) - acc_wdv[ei])
                    else:
                        for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                            delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                            r_state2[ei] = r_state2[ei] + delta

                elif cutlass.const_expr(self.schedule_variant == 1):
                    # S1: issue WDV1+WDV2 first, then QDO/update per half.
                    acc_wdv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv)
                        else:
                            cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
                    cute.nvgpu.warpgroup.commit_group()

                    acc_wdv2.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma, acc_wdv2, tW2R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv2
                            )
                        else:
                            cute.gemm(update_mma, acc_wdv2, tW2R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv2)
                    cute.nvgpu.warpgroup.commit_group()

                    acc_dv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma,
                                acc_dv,
                                tQ1R[None, None, kp, current_stage],
                                tDoR[None, None, kp, current_do_dv_stage],
                                acc_dv,
                            )
                        else:
                            cute.gemm(update_mma, acc_dv, tQ1R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    dep_zero = Float32(0.0)
                    for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                        delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                        r_state1[ei] = r_state1[ei] + delta
                        dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])

                    cute.arch.barrier()

                    acc_dv.fill(dep_zero)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma,
                                acc_dv,
                                tQ2R[None, None, kp, current_stage],
                                tDoR[None, None, kp, current_do_dv_stage],
                                acc_dv,
                            )
                        else:
                            cute.gemm(update_mma, acc_dv, tQ2R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                        delta = acc_dv[ei] * Float32(self.scale) - acc_wdv2[ei]
                        r_state2[ei] = r_state2[ei] + delta

                    cute.arch.barrier()
                else:
                    # S2: issue WDV1+QDO1+WDV2, update state1, then QDO2/update state2.
                    acc_wdv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv)
                        else:
                            cute.gemm(update_mma, acc_wdv, tW1R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv)
                    cute.nvgpu.warpgroup.commit_group()

                    acc_dv.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma,
                                acc_dv,
                                tQ1R[None, None, kp, current_stage],
                                tDoR[None, None, kp, current_do_dv_stage],
                                acc_dv,
                            )
                        else:
                            cute.gemm(update_mma, acc_dv, tQ1R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                    cute.nvgpu.warpgroup.commit_group()

                    acc_wdv2.fill(0.0)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDv2R, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma, acc_wdv2, tW2R[None, None, kp, current_stage], tDv2R[None, None, kp], acc_wdv2
                            )
                        else:
                            cute.gemm(update_mma, acc_wdv2, tW2R[None, None, kp, 0], tDv2R[None, None, kp], acc_wdv2)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    dep_zero = Float32(0.0)
                    for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                        delta = acc_dv[ei] * Float32(self.scale) - acc_wdv[ei]
                        r_state1[ei] = r_state1[ei] + delta
                        dep_zero = _dep_zero_f32(dep_zero, r_state1[ei])

                    cute.arch.barrier()

                    acc_dv.fill(dep_zero)
                    cute.nvgpu.warpgroup.fence()
                    for kp in cutlass.range(cute.size(tDoR, mode=[2]), unroll_full=True):
                        update_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
                        if cutlass.const_expr(self.load_pipeline_stages > 1):
                            cute.gemm(
                                update_mma,
                                acc_dv,
                                tQ2R[None, None, kp, current_stage],
                                tDoR[None, None, kp, current_do_dv_stage],
                                acc_dv,
                            )
                        else:
                            cute.gemm(update_mma, acc_dv, tQ2R[None, None, kp, 0], tDoR[None, None, kp, 0], acc_dv)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    for ei in cutlass.range(cute.size(r_state2), unroll_full=True):
                        delta = acc_dv[ei] * Float32(self.scale) - acc_wdv2[ei]
                        r_state2[ei] = r_state2[ei] + delta

                    cute.arch.barrier()

        if cutlass.const_expr(self.use_dh0):
            for ei in cutlass.range(cute.size(r_state1), unroll_full=True):
                k_rel, v_rel = tCcKv[ei]
                dh0[bidx, hidx, k_rel, v_base + v_rel] = r_state1[ei]
                dh0[bidx, hidx, k_rel + BK_HALF, v_base + v_rel] = r_state2[ei]


@functools.lru_cache(maxsize=256)
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
    dv_local_load: bool = False,
    do_dv_pipeline_stages: int | None = None,
    no_dh_store: bool = False,
    no_dv2_store: bool = False,
    no_qdo: bool = False,
    no_wdv: bool = False,
    kdh_dv_only: bool = False,
    schedule_variant: int = 0,
    late_dh_store: bool = False,
    update_variant: int = 0,
    early_dv2_store: bool = False,
    no_update_math: bool = False,
    dh_store_variant: int = 0,
    dh_store_pack_only: bool = False,
    kdh_skeleton_variant: int = 0,
    additive_variant: int = 0,
    wdv_skeleton_variant: int = 0,
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
        dv_local_load=dv_local_load,
        do_dv_pipeline_stages=do_dv_pipeline_stages,
        no_dh_store=no_dh_store,
        no_dv2_store=no_dv2_store,
        no_qdo=no_qdo,
        no_wdv=no_wdv,
        kdh_dv_only=kdh_dv_only,
        schedule_variant=schedule_variant,
        late_dh_store=late_dh_store,
        update_variant=update_variant,
        early_dv2_store=early_dv2_store,
        no_update_math=no_update_math,
        dh_store_variant=dh_store_variant,
        dh_store_pack_only=dh_store_pack_only,
        kdh_skeleton_variant=kdh_skeleton_variant,
        additive_variant=additive_variant,
        wdv_skeleton_variant=wdv_skeleton_variant,
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
    dv_local_load: bool = False,
    do_dv_pipeline_stages: int | None = None,
    no_dh_store: bool = False,
    no_dv2_store: bool = False,
    no_qdo: bool = False,
    no_wdv: bool = False,
    kdh_dv_only: bool = False,
    schedule_variant: int = 0,
    late_dh_store: bool = False,
    update_variant: int = 0,
    early_dv2_store: bool = False,
    no_update_math: bool = False,
    dh_store_variant: int = 0,
    dh_store_pack_only: bool = False,
    kdh_skeleton_variant: int = 0,
    additive_variant: int = 0,
    wdv_skeleton_variant: int = 0,
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
    if load_pipeline_stages not in (1, 2, 3):
        raise NotImplementedError(f"Only 1-stage, 2-stage, or 3-stage load pipeline is supported, got {load_pipeline_stages}.")
    if do_dv_pipeline_stages is None:
        do_dv_pipeline_stages = load_pipeline_stages
    if do_dv_pipeline_stages not in (1, 2, 3):
        raise NotImplementedError(
            f"Only 1-stage, 2-stage, or 3-stage do/dv pipeline is supported, got {do_dv_pipeline_stages}."
        )
    if do_dv_pipeline_stages > load_pipeline_stages:
        raise NotImplementedError("do/dv pipeline stages cannot exceed K/Q/W pipeline stages.")
    if schedule_variant not in (0, 1, 2):
        raise NotImplementedError(f"Only schedule variants 0, 1, and 2 are supported, got {schedule_variant}.")
    if schedule_variant != 0 and (no_qdo or no_wdv or kdh_dv_only):
        raise NotImplementedError("benchmark decomposition flags are only supported with schedule_variant=0.")
    if update_variant not in (0, 1, 2, 3):
        raise NotImplementedError(f"Only update variants 0, 1, 2, and 3 are supported, got {update_variant}.")
    if schedule_variant != 0 and (update_variant != 0 or no_update_math):
        raise NotImplementedError("update lowering experiments are only supported with schedule_variant=0.")
    if dh_store_variant not in (0, 1, 2, 3):
        raise NotImplementedError(f"Only dh store variants 0, 1, 2, and 3 are supported, got {dh_store_variant}.")
    if late_dh_store and dh_store_variant != 0:
        raise NotImplementedError("legacy late_dh_store cannot be combined with dh_store_variant.")
    if kdh_skeleton_variant not in (0, 1, 2, 3, 4, 5):
        raise NotImplementedError(f"Only KDH skeleton variants 0..5 are supported, got {kdh_skeleton_variant}.")
    if additive_variant not in (0, 1, 2, 3, 4):
        raise NotImplementedError(f"Only additive reconstruction variants 0..4 are supported, got {additive_variant}.")
    if wdv_skeleton_variant not in (0, 1, 2, 3):
        raise NotImplementedError(f"Only WDV skeleton variants 0..3 are supported, got {wdv_skeleton_variant}.")

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
        bool(dv_local_load),
        int(do_dv_pipeline_stages),
        bool(no_dh_store),
        bool(no_dv2_store),
        bool(no_qdo),
        bool(no_wdv),
        bool(kdh_dv_only),
        int(schedule_variant),
        bool(late_dh_store),
        int(update_variant),
        bool(early_dv2_store),
        bool(no_update_math),
        int(dh_store_variant),
        bool(dh_store_pack_only),
        int(kdh_skeleton_variant),
        int(additive_variant),
        int(wdv_skeleton_variant),
    )
    compiled(q, k, w, dht_arg, dh0_arg, do, dh, dv, dv2, (Int32(B), Int32(T), Int32(NT)))
    return dh, dh0, dv2
