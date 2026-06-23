# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CuTeDSL SM90 chunk_delta_h backward dH/dU helper.

Initial target:
  - fixed length only
  - SM90, K=V=128, chunk_size=64
  - no g/gk, no varlen

The public return order matches FLA's ``chunk_gated_delta_rule_bwd_dhu``:
``(dh, dh0, dv2)``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T as _T

from cula.utils import USE_FAST_MATH, assert_hopper

BT = 64
BK = 64
K_DIM = 128
V_DIM = 128

_kernel_cache: dict[tuple[int, int, int, int, bool], object] = {}


@cutlass.dsl_user_op
def _pack_bf16x2_f32(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    result = _llvm.inline_asm(
        _T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cutlass.dsl_user_op
def _store_bridge_64x32_bf16(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    p4: Int32,
    p5: Int32,
    p6: Int32,
    p7: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
            Int32(p0).ir_value(loc=loc, ip=ip),
            Int32(p1).ir_value(loc=loc, ip=ip),
            Int32(p2).ir_value(loc=loc, ip=ip),
            Int32(p3).ir_value(loc=loc, ip=ip),
            Int32(p4).ir_value(loc=loc, ip=ip),
            Int32(p5).ir_value(loc=loc, ip=ip),
            Int32(p6).ir_value(loc=loc, ip=ip),
            Int32(p7).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
        .reg .u32 r16, r17;
        .reg .u32 s0, l0, l1;
        .reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;
        .reg .u32 row, col;
        .reg .u64 g0, g1, row_bytes, col_bytes, step_bytes;
        mov.u32 r5, $1;
        and.b32 r6, r5, 24;
        shl.b32 r6, r6, 7;
        and.b32 r7, r5, 3;
        shl.b32 r7, r7, 5;
        or.b32 r8, r6, r7;
        and.b32 r9, r5, 96;
        shl.b32 r9, r9, 3;
        shl.b32 r10, r5, 2;
        and.b32 r10, r10, 112;
        or.b32 r11, r9, r10;
        xor.b32 r12, r8, r11;
        add.u32 s0, $2, r12;
        shr.u32 row, r5, 2;
        and.b32 row, row, 31;
        and.b32 col, r5, 3;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        mul.wide.u32 step_bytes, $4, 32;
        add.u64 g1, g0, step_bytes;
        st.shared.v4.b32 [s0], {$5, $7, $9, $11};
        st.shared.v4.b32 [s0+128], {$6, $8, $10, $12};
        fence.proxy.async.shared::cta;
        bar.sync 0;
        and.b32 r13, r5, 6;
        shl.b32 r13, r13, 9;
        and.b32 r14, r5, 7;
        shl.b32 r14, r14, 4;
        or.b32 r15, r13, r14;
        and.b32 r16, r5, 120;
        shl.b32 r16, r16, 2;
        xor.b32 r17, r15, r16;
        add.u32 l0, $2, r17;
        add.u32 l1, l0, 512;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];
        st.global.v4.b32 [g0], {o0, o1, o2, o3};
        st.global.v4.b32 [g1], {o4, o5, o6, o7};
        }""",
        "=r,r,r,l,r,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _store_bridge_64x64_postbar_bf16(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    p4: Int32,
    p5: Int32,
    p6: Int32,
    p7: Int32,
    p8: Int32,
    p9: Int32,
    p10: Int32,
    p11: Int32,
    p12: Int32,
    p13: Int32,
    p14: Int32,
    p15: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
            Int32(p0).ir_value(loc=loc, ip=ip),
            Int32(p1).ir_value(loc=loc, ip=ip),
            Int32(p2).ir_value(loc=loc, ip=ip),
            Int32(p3).ir_value(loc=loc, ip=ip),
            Int32(p4).ir_value(loc=loc, ip=ip),
            Int32(p5).ir_value(loc=loc, ip=ip),
            Int32(p6).ir_value(loc=loc, ip=ip),
            Int32(p7).ir_value(loc=loc, ip=ip),
            Int32(p8).ir_value(loc=loc, ip=ip),
            Int32(p9).ir_value(loc=loc, ip=ip),
            Int32(p10).ir_value(loc=loc, ip=ip),
            Int32(p11).ir_value(loc=loc, ip=ip),
            Int32(p12).ir_value(loc=loc, ip=ip),
            Int32(p13).ir_value(loc=loc, ip=ip),
            Int32(p14).ir_value(loc=loc, ip=ip),
            Int32(p15).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
        .reg .u32 r16, r17;
        .reg .u32 s0, s1, l0, l1, l2, l3;
        .reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;
        .reg .u32 o8, o9, o10, o11, o12, o13, o14, o15;
        .reg .u32 row, col;
        .reg .u64 g0, g1, g2, g3, row_bytes, col_bytes, step_bytes;
        mov.u32 r5, $1;
        and.b32 r6, r5, 24;
        shl.b32 r6, r6, 7;
        and.b32 r7, r5, 3;
        shl.b32 r7, r7, 5;
        or.b32 r8, r6, r7;
        and.b32 r9, r5, 96;
        shl.b32 r9, r9, 3;
        shl.b32 r10, r5, 2;
        and.b32 r10, r10, 112;
        or.b32 r11, r9, r10;
        xor.b32 r12, r8, r11;
        add.u32 s0, $2, r12;
        add.u32 s1, s0, 4096;
        shr.u32 row, r5, 2;
        and.b32 row, row, 31;
        and.b32 col, r5, 3;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        mul.wide.u32 step_bytes, $4, 32;
        add.u64 g1, g0, step_bytes;
        add.u64 g2, g0, 64;
        add.u64 g3, g1, 64;
        st.shared.v4.b32 [s0], {$5, $7, $9, $11};
        st.shared.v4.b32 [s0+128], {$6, $8, $10, $12};
        st.shared.v4.b32 [s1], {$13, $15, $17, $19};
        st.shared.v4.b32 [s1+128], {$14, $16, $18, $20};
        fence.proxy.async.shared::cta;
        bar.sync 0;
        and.b32 r13, r5, 6;
        shl.b32 r13, r13, 9;
        and.b32 r14, r5, 7;
        shl.b32 r14, r14, 4;
        or.b32 r15, r13, r14;
        and.b32 r16, r5, 120;
        shl.b32 r16, r16, 2;
        xor.b32 r17, r15, r16;
        add.u32 l0, $2, r17;
        add.u32 l1, l0, 512;
        add.u32 l2, l0, 4096;
        add.u32 l3, l2, 512;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o8, o9, o10, o11}, [l2];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o12, o13, o14, o15}, [l3];
        st.global.v4.b32 [g0], {o0, o1, o2, o3};
        st.global.v4.b32 [g1], {o4, o5, o6, o7};
        st.global.v4.b32 [g2], {o8, o9, o10, o11};
        st.global.v4.b32 [g3], {o12, o13, o14, o15};
        bar.sync 0;
        }""",
        "=r,r,r,l,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _store_dh0_float4_64x32(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, vec, row, col, smem_off, col_bytes;
        .reg .u32 a0, a1, a2, a3;
        .reg .u64 g0, row_bytes, col_bytes64;
        mov.u32 t, $1;

        mov.u32 vec, t;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 128;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 256;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 384;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _store_dh0_float4_64x64(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, vec, row, col, smem_off, col_bytes;
        .reg .u32 a0, a1, a2, a3;
        .reg .u64 g0, row_bytes, col_bytes64;
        mov.u32 t, $1;

        mov.u32 vec, t;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 128;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 256;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 384;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 512;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 640;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 768;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};

        add.u32 vec, t, 896;
        shr.u32 row, vec, 4;
        and.b32 col, vec, 15;
        shl.b32 col, col, 2;
        shl.b32 smem_off, vec, 4;
        mul.wide.u32 row_bytes, row, $4;
        shl.b32 col_bytes, col, 2;
        cvt.u64.u32 col_bytes64, col_bytes;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes64;
        add.u32 smem_off, $2, smem_off;
        ld.shared.v4.b32 {a0, a1, a2, a3}, [smem_off];
        st.global.v4.b32 [g0], {a0, a1, a2, a3};
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _cp_async_mn_sw128_64x64(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, vec, row, col, linear_bytes, swz, smem_off, s0;
        .reg .u64 g0, row_bytes, col_bytes;
        mov.u32 t, $1;

        mov.u32 vec, t;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 vec, t, 128;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 vec, t, 256;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 vec, t, 384;
        shr.u32 row, vec, 3;
        and.b32 col, vec, 7;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _cp_async_k_sw128_64x64(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, row0, row, col, linear_bytes, swz, smem_off, s0;
        .reg .u64 g0, row_bytes, col_bytes;
        mov.u32 t, $1;
        shr.u32 row0, t, 3;
        and.b32 col, t, 7;
        shl.b32 col, col, 3;

        mov.u32 row, row0;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 row, row0, 16;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 row, row0, 32;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 row, row0, 48;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 7;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 896;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _cp_async_mn_sw64_32x64(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, vec, row, col, linear_bytes, swz, smem_off, s0;
        .reg .u64 g0, row_bytes, col_bytes;
        mov.u32 t, $1;

        mov.u32 vec, t;
        shr.u32 row, vec, 2;
        and.b32 col, vec, 3;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 6;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 384;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;

        add.u32 vec, t, 128;
        shr.u32 row, vec, 2;
        and.b32 col, vec, 3;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        shl.b32 linear_bytes, row, 6;
        shl.b32 smem_off, col, 1;
        add.u32 linear_bytes, linear_bytes, smem_off;
        and.b32 swz, linear_bytes, 384;
        shr.u32 swz, swz, 3;
        xor.b32 smem_off, linear_bytes, swz;
        add.u32 s0, $2, smem_off;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _cp_async_dv_bridge_64x32(
    tidx: Int32,
    smem_base: Int32,
    gmem_base,
    row_stride_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    _llvm.inline_asm(
        _T.i32(),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            gmem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(row_stride_bytes).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, r0, r1, r2, r3, r4, s0;
        .reg .u32 row, col;
        .reg .u64 g0, g1, row_bytes, col_bytes, step_bytes;
        mov.u32 t, $1;
        shr.u32 row, t, 2;
        and.b32 row, row, 31;
        and.b32 col, t, 3;
        shl.b32 col, col, 3;
        mul.wide.u32 row_bytes, row, $4;
        mul.wide.u32 col_bytes, col, 2;
        add.u64 g0, $3, row_bytes;
        add.u64 g0, g0, col_bytes;
        mul.wide.u32 step_bytes, $4, 32;
        add.u64 g1, g0, step_bytes;
        and.b32 r0, t, 24;
        shl.b32 r0, r0, 7;
        and.b32 r1, t, 3;
        shl.b32 r1, r1, 5;
        or.b32 r2, r0, r1;
        and.b32 r3, t, 124;
        shl.b32 r3, r3, 2;
        xor.b32 r4, r2, r3;
        add.u32 s0, $2, r4;
        cp.async.cg.shared.global.L2::128B [s0], [g0], 16;
        cp.async.cg.shared.global.L2::128B [s0+512], [g1], 16;
        }""",
        "=r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _load_acc_64x32_bf16_from_smem_bridge(tidx: Int32, smem_base: Int32, *, loc=None, ip=None):
    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 16),
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .u32 t, r0, r1, r2, r3, r4, r5, r6, r7;
        .reg .u32 l0, l1;
        .reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;
        .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
        .reg .b16 h8, h9, h10, h11, h12, h13, h14, h15;
        mov.u32 t, $16;
        and.b32 r0, t, 6;
        shl.b32 r0, r0, 9;
        and.b32 r1, t, 15;
        shl.b32 r1, r1, 4;
        and.b32 r2, t, 96;
        shl.b32 r2, r2, 3;
        and.b32 r3, t, 16;
        shl.b32 r3, r3, 1;
        or.b32 r4, r1, r2;
        xor.b32 r5, r4, r3;
        or.b32 r6, r5, r0;
        add.u32 l0, $17, r6;
        xor.b32 r7, r6, 64;
        add.u32 l1, $17, r7;
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];
        ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];
        mov.b32 {h0, h1}, o0;
        mov.b32 {h2, h3}, o1;
        mov.b32 {h4, h5}, o2;
        mov.b32 {h6, h7}, o3;
        mov.b32 {h8, h9}, o4;
        mov.b32 {h10, h11}, o5;
        mov.b32 {h12, h13}, o6;
        mov.b32 {h14, h15}, o7;
        cvt.f32.bf16 $0, h0;
        cvt.f32.bf16 $1, h1;
        cvt.f32.bf16 $2, h2;
        cvt.f32.bf16 $3, h3;
        cvt.f32.bf16 $4, h4;
        cvt.f32.bf16 $5, h5;
        cvt.f32.bf16 $6, h6;
        cvt.f32.bf16 $7, h7;
        cvt.f32.bf16 $8, h8;
        cvt.f32.bf16 $9, h9;
        cvt.f32.bf16 $10, h10;
        cvt.f32.bf16 $11, h11;
        cvt.f32.bf16 $12, h12;
        cvt.f32.bf16 $13, h13;
        cvt.f32.bf16 $14, h14;
        cvt.f32.bf16 $15, h15;
        }""",
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(_llvm.extractvalue(_T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [1], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [2], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [3], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [4], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [5], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [6], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [7], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [8], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [9], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [10], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [11], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [12], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [13], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [14], loc=loc, ip=ip)),
        Float32(_llvm.extractvalue(_T.f32(), out, [15], loc=loc, ip=ip)),
    )


def _make_smem_layout(kind: warpgroup.SmemLayoutAtomKind, dtype: type[cutlass.Numeric], shape: tuple[int, ...], order):
    atom = warpgroup.make_smem_layout_atom(kind, dtype)
    return cute.tile_to_shape(atom, shape, order=order)


class ChunkDeltaBwdDhuSm90:
    def __init__(
        self,
        bv: int,
        h_static: int,
        use_fast_math: bool = True,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        assert bv in (32, 64)
        assert 1 <= h_static <= 64
        assert_hopper()
        self.BV = bv
        self.H_static = h_static
        self.use_fast_math = use_fast_math
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.threads_per_cta = 128
        self.buffer_align_bytes = 1024
        self.mma_tiler_kdh = (BT, bv, BK)
        self.mma_tiler_update = (BK, bv, BT)

    @staticmethod
    @cute.jit
    def _gemm_sm90_loop(
        tiled_mma: cute.TiledMma,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        accumulate_first: cutlass.Constexpr,
    ):
        for k_block_idx in cutlass.range_constexpr(cute.size(a, mode=[2])):
            if cutlass.const_expr(accumulate_first):
                tiled_mma.set(warpgroup.Field.ACCUMULATE, cutlass.Boolean(True))
            else:
                tiled_mma.set(warpgroup.Field.ACCUMULATE, cutlass.Boolean(k_block_idx != 0))
            cute.gemm(
                tiled_mma,
                c,
                a[None, None, k_block_idx],
                b[None, None, k_block_idx],
                c,
            )

    @staticmethod
    @cute.jit
    def _copy_acc_to_bf16(acc: cute.Tensor, dtype: cutlass.Constexpr):
        out = cute.make_rmem_tensor_like(acc, dtype)
        out.store(acc.load().to(dtype))
        return out

    @cute.jit
    def _copy_do_tile(self, gDo: cute.Tensor, sDo: cute.Tensor, tidx: Int32):
        vecs_per_row = self.BV // 8
        vec_iters = self.BV // 16
        for copy_iter in cutlass.range_constexpr(vec_iters):
            vec_idx = tidx + copy_iter * self.threads_per_cta
            t_rel = vec_idx // vecs_per_row
            v_vec = vec_idx - t_rel * vecs_per_row
            v_rel = v_vec * 8
            for j in cutlass.range_constexpr(8):
                sDo[v_rel + j, t_rel, 0] = gDo[v_rel + j, t_rel]

    @cute.jit
    def _store_dh0_tile(
        self,
        acc: cute.Tensor,
        tC: cute.Tensor,
        sDh0: cute.Tensor,
        gDh0: cute.Tensor,
        row_stride_bytes: Int32,
        tidx: Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(acc)):
            coord = tC[i]
            row = coord[0]
            col = coord[1]
            sDh0[row, col] = acc[i]
        cute.arch.sync_threads()

        if cutlass.const_expr(self.BV == 64):
            _store_dh0_float4_64x64(tidx, sDh0.iterator.toint(), gDh0.iterator, row_stride_bytes)
        else:
            _store_dh0_float4_64x32(tidx, sDh0.iterator.toint(), gDh0.iterator, row_stride_bytes)
        cute.arch.sync_threads()

    @cute.jit
    def _r2s_acc_store_bridge(
        self,
        acc: cute.Tensor,
        r2s: cute.TiledCopy,
        r2s_thr: cute.TiledCopy,
        sTile: cute.Tensor,
        bridge_ptr: cute.Pointer,
        gmem_ptr: cute.Pointer,
        row_stride_bytes: Int32,
        tidx: Int32,
    ):
        bf16 = self._copy_acc_to_bf16(acc, self.io_dtype)
        cute.copy(r2s, r2s.retile(bf16), r2s_thr.partition_D(sTile)[None, None, None, 0])

        smem_base = bridge_ptr.toint()
        _store_bridge_64x32_bf16(
            tidx,
            smem_base,
            gmem_ptr,
            row_stride_bytes,
            _pack_bf16x2_f32(acc[0], acc[1]),
            _pack_bf16x2_f32(acc[2], acc[3]),
            _pack_bf16x2_f32(acc[4], acc[5]),
            _pack_bf16x2_f32(acc[6], acc[7]),
            _pack_bf16x2_f32(acc[8], acc[9]),
            _pack_bf16x2_f32(acc[10], acc[11]),
            _pack_bf16x2_f32(acc[12], acc[13]),
            _pack_bf16x2_f32(acc[14], acc[15]),
        )
        if cutlass.const_expr(self.BV == 64):
            _store_bridge_64x32_bf16(
                tidx,
                smem_base + BT * 32 * 2,
                gmem_ptr + 32,
                row_stride_bytes,
                _pack_bf16x2_f32(acc[16], acc[17]),
                _pack_bf16x2_f32(acc[18], acc[19]),
                _pack_bf16x2_f32(acc[20], acc[21]),
                _pack_bf16x2_f32(acc[22], acc[23]),
                _pack_bf16x2_f32(acc[24], acc[25]),
                _pack_bf16x2_f32(acc[26], acc[27]),
                _pack_bf16x2_f32(acc[28], acc[29]),
                _pack_bf16x2_f32(acc[30], acc[31]),
            )

    @cute.jit
    def _r2s_acc_store_bridge_64x64_postbar(
        self,
        acc: cute.Tensor,
        r2s: cute.TiledCopy,
        r2s_thr: cute.TiledCopy,
        sTile: cute.Tensor,
        bridge_ptr: cute.Pointer,
        gmem_ptr: cute.Pointer,
        row_stride_bytes: Int32,
        tidx: Int32,
    ):
        bf16 = self._copy_acc_to_bf16(acc, self.io_dtype)
        cute.copy(r2s, r2s.retile(bf16), r2s_thr.partition_D(sTile)[None, None, None, 0])

        _store_bridge_64x64_postbar_bf16(
            tidx,
            bridge_ptr.toint(),
            gmem_ptr,
            row_stride_bytes,
            _pack_bf16x2_f32(acc[0], acc[1]),
            _pack_bf16x2_f32(acc[2], acc[3]),
            _pack_bf16x2_f32(acc[4], acc[5]),
            _pack_bf16x2_f32(acc[6], acc[7]),
            _pack_bf16x2_f32(acc[8], acc[9]),
            _pack_bf16x2_f32(acc[10], acc[11]),
            _pack_bf16x2_f32(acc[12], acc[13]),
            _pack_bf16x2_f32(acc[14], acc[15]),
            _pack_bf16x2_f32(acc[16], acc[17]),
            _pack_bf16x2_f32(acc[18], acc[19]),
            _pack_bf16x2_f32(acc[20], acc[21]),
            _pack_bf16x2_f32(acc[22], acc[23]),
            _pack_bf16x2_f32(acc[24], acc[25]),
            _pack_bf16x2_f32(acc[26], acc[27]),
            _pack_bf16x2_f32(acc[28], acc[29]),
            _pack_bf16x2_f32(acc[30], acc[31]),
        )

    @cute.jit
    def _cp_async_dv_bridge_64x64(
        self,
        tidx: Int32,
        smem_ptr: cute.Pointer,
        gmem_ptr: cute.Pointer,
        row_stride_bytes: Int32,
    ):
        smem_base = smem_ptr.toint()
        _cp_async_dv_bridge_64x32(tidx, smem_base, gmem_ptr, row_stride_bytes)
        _cp_async_dv_bridge_64x32(tidx, smem_base + BT * 32 * 2, gmem_ptr + 32, row_stride_bytes)

    @cute.jit
    def _add_dv_bridge_64x64_from_smem(
        self,
        acc: cute.Tensor,
        smem_ptr: cute.Pointer,
        tidx: Int32,
    ):
        vals0 = _load_acc_64x32_bf16_from_smem_bridge(tidx, smem_ptr.toint())
        for i in cutlass.range_constexpr(16):
            acc[i] += vals0[i]

        vals1 = _load_acc_64x32_bf16_from_smem_bridge(tidx, smem_ptr.toint() + BT * 32 * 2)
        for i in cutlass.range_constexpr(16):
            acc[i + 16] += vals1[i]

    @cute.jit
    def _add_dv_bridge_64x32_from_smem(
        self,
        acc: cute.Tensor,
        smem_ptr: cute.Pointer,
        tidx: Int32,
    ):
        vals = _load_acc_64x32_bf16_from_smem_bridge(tidx, smem_ptr.toint())
        for i in cutlass.range_constexpr(16):
            acc[i] += vals[i]

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        do_in: cute.Tensor,
        dv_in: cute.Tensor,
        dh_in: cute.Tensor,
        dh0_in: cute.Tensor,
        dv2_in: cute.Tensor,
        dht_in: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        scale: Float32,
        has_dht: Int32,
        store_dh0: Int32,
        stream,
    ):
        B, T, H, K, V = problem_size

        # SM90 GMMA setup.  The operand modes mirror the C++ reference:
        #   KDH:    K(T,K) [K-major] x dH(V,K) [MN-major] -> dV2(T,V)
        #   update: Q/W(K,T) [MN-major] x dO/dV2(V,T) [MN-major] -> dH(K,V)
        kdh_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            self.acc_dtype,
            (1, 1, 1),
            self.mma_tiler_kdh[:2],
        )
        update_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            self.acc_dtype,
            (1, 1, 1),
            self.mma_tiler_update[:2],
        )

        if cutlass.const_expr(self.BV == 64):
            s_k_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.K_SW128, self.io_dtype, (BT, BK, 1), (0, 1, 2))
            s_qw_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.MN_SW128, self.io_dtype, (BK, BT, 1), (0, 1, 2))
            s_vt_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.MN_SW128, self.io_dtype, (self.BV, BT, 1), (0, 1, 2))
            s_state_store_layout = _make_smem_layout(
                warpgroup.SmemLayoutAtomKind.K_SW128, self.io_dtype, (BK, self.BV, 1), (0, 1, 2)
            )
            s_state_read_layout = _make_smem_layout(
                warpgroup.SmemLayoutAtomKind.MN_SW128, self.io_dtype, (self.BV, BK, 1), (0, 1, 2)
            )
        else:
            # Matches the C++ generic K=128,VTile=32 path:
            # K/Q/W stay 64x64 SW128 tiles; do/dh/dv2 use 64x32 SW64 tiles.
            s_k_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.K_SW128, self.io_dtype, (BT, BK, 1), (0, 1, 2))
            s_qw_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.MN_SW128, self.io_dtype, (BK, BT, 1), (0, 1, 2))
            s_vt_layout = _make_smem_layout(warpgroup.SmemLayoutAtomKind.MN_SW64, self.io_dtype, (self.BV, BT, 1), (0, 1, 2))
            s_state_store_layout = _make_smem_layout(
                warpgroup.SmemLayoutAtomKind.K_SW64, self.io_dtype, (BK, self.BV, 1), (0, 1, 2)
            )
            s_state_read_layout = _make_smem_layout(
                warpgroup.SmemLayoutAtomKind.MN_SW64, self.io_dtype, (self.BV, BK, 1), (0, 1, 2)
            )

        atom_cp = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype)
        copy_k = cute.make_tiled_copy_tv(
            atom_cp,
            cute.make_layout((16, 8), stride=(8, 1)),
            cute.make_layout((1, 8)),
        )
        copy_mn = cute.make_tiled_copy_tv(
            atom_cp,
            cute.make_layout((8, 16), stride=(16, 1)),
            cute.make_layout((8, 1)),
        )

        atom_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.io_dtype,
        )
        r2s_kdh = cute.make_tiled_copy_C(atom_r2s, kdh_mma)
        r2s_update = cute.make_tiled_copy_C(atom_r2s, update_mma)

        self.kernel(
            q_in,
            k_in,
            w_in,
            do_in,
            dv_in,
            dh_in,
            dh0_in,
            dv2_in,
            dht_in,
            kdh_mma,
            update_mma,
            copy_k,
            copy_mn,
            r2s_kdh,
            r2s_update,
            s_k_layout,
            s_qw_layout,
            s_vt_layout,
            s_state_store_layout,
            s_state_read_layout,
            problem_size,
            scale,
            has_dht,
            store_dh0,
        ).launch(
            grid=(V // self.BV, B * H, 1),
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        do_in: cute.Tensor,
        dv_in: cute.Tensor,
        dh_in: cute.Tensor,
        dh0_in: cute.Tensor,
        dv2_in: cute.Tensor,
        dht_in: cute.Tensor,
        kdh_mma: cute.TiledMma,
        update_mma: cute.TiledMma,
        copy_k: cute.TiledCopy,
        copy_mn: cute.TiledCopy,
        r2s_kdh: cute.TiledCopy,
        r2s_update: cute.TiledCopy,
        s_k_layout: cute.ComposedLayout,
        s_qw_layout: cute.ComposedLayout,
        s_vt_layout: cute.ComposedLayout,
        s_state_store_layout: cute.ComposedLayout,
        s_state_read_layout: cute.ComposedLayout,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        scale: Float32,
        has_dht: Int32,
        store_dh0: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        v_tile, bh, _ = cute.arch.block_idx()
        B, T, H, K, V = problem_size
        NT = T // BT
        if cutlass.const_expr(self.H_static == 64):
            b = bh >> 6
            h = bh & 63
        elif cutlass.const_expr(self.H_static == 32):
            b = bh >> 5
            h = bh & 31
        elif cutlass.const_expr(self.H_static == 16):
            b = bh >> 4
            h = bh & 15
        elif cutlass.const_expr(self.H_static == 8):
            b = bh >> 3
            h = bh & 7
        elif cutlass.const_expr(self.H_static == 1):
            b = bh
            h = Int32(0)
        else:
            b = bh // H
            h = bh - b * H
        v_base = v_tile * self.BV
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        w_ptr = w_in.iterator
        do_ptr = do_in.iterator
        dv_ptr = dv_in.iterator
        dh_ptr = dh_in.iterator
        dh0_ptr = dh0_in.iterator
        dv2_ptr = dv2_in.iterator
        dht_ptr = dht_in.iterator

        smem = cutlass.utils.SmemAllocator()
        sK0Raw = smem.allocate_tensor(self.io_dtype, s_k_layout.outer, self.buffer_align_bytes)
        sK1Raw = smem.allocate_tensor(self.io_dtype, s_k_layout.outer, self.buffer_align_bytes)
        sQ0Raw = smem.allocate_tensor(self.io_dtype, s_qw_layout.outer, self.buffer_align_bytes)
        sQ1Raw = smem.allocate_tensor(self.io_dtype, s_qw_layout.outer, self.buffer_align_bytes)
        sW0Raw = smem.allocate_tensor(self.io_dtype, s_qw_layout.outer, self.buffer_align_bytes)
        sW1Raw = smem.allocate_tensor(self.io_dtype, s_qw_layout.outer, self.buffer_align_bytes)
        sDoRaw = smem.allocate_tensor(self.io_dtype, s_vt_layout.outer, self.buffer_align_bytes)
        sDv2Raw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        sDh0Raw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        sDh1Raw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        sDvPrefRaw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        sDhBridgeRaw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        if cutlass.const_expr(self.BV == 32):
            sDh1BridgeRaw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
            sDv2BridgeRaw = smem.allocate_tensor(self.io_dtype, s_state_store_layout.outer, self.buffer_align_bytes)
        else:
            sDh1BridgeRaw = sDhBridgeRaw
        sDh0FloatLayout = cute.make_layout((BK, self.BV), stride=(self.BV, 1))
        sDh0Float = smem.allocate_tensor(self.acc_dtype, sDh0FloatLayout, 16)

        sK0 = cute.make_tensor(
            cute.recast_ptr(sK0Raw.iterator, swizzle_=s_k_layout.inner, dtype=self.io_dtype), s_k_layout.outer
        )
        sK1 = cute.make_tensor(
            cute.recast_ptr(sK1Raw.iterator, swizzle_=s_k_layout.inner, dtype=self.io_dtype), s_k_layout.outer
        )
        sQ0 = cute.make_tensor(
            cute.recast_ptr(sQ0Raw.iterator, swizzle_=s_qw_layout.inner, dtype=self.io_dtype), s_qw_layout.outer
        )
        sQ1 = cute.make_tensor(
            cute.recast_ptr(sQ1Raw.iterator, swizzle_=s_qw_layout.inner, dtype=self.io_dtype), s_qw_layout.outer
        )
        sW0 = cute.make_tensor(
            cute.recast_ptr(sW0Raw.iterator, swizzle_=s_qw_layout.inner, dtype=self.io_dtype), s_qw_layout.outer
        )
        sW1 = cute.make_tensor(
            cute.recast_ptr(sW1Raw.iterator, swizzle_=s_qw_layout.inner, dtype=self.io_dtype), s_qw_layout.outer
        )
        sDo = cute.make_tensor(
            cute.recast_ptr(sDoRaw.iterator, swizzle_=s_vt_layout.inner, dtype=self.io_dtype), s_vt_layout.outer
        )
        sDv2Store = cute.make_tensor(
            cute.recast_ptr(sDv2Raw.iterator, swizzle_=s_state_store_layout.inner, dtype=self.io_dtype),
            s_state_store_layout.outer,
        )
        sDv2Read = cute.make_tensor(
            cute.recast_ptr(sDv2Raw.iterator, swizzle_=s_state_read_layout.inner, dtype=self.io_dtype),
            s_state_read_layout.outer,
        )
        sDh0Store = cute.make_tensor(
            cute.recast_ptr(sDh0Raw.iterator, swizzle_=s_state_store_layout.inner, dtype=self.io_dtype),
            s_state_store_layout.outer,
        )
        sDh0Read = cute.make_tensor(
            cute.recast_ptr(sDh0Raw.iterator, swizzle_=s_state_read_layout.inner, dtype=self.io_dtype),
            s_state_read_layout.outer,
        )
        sDh1Store = cute.make_tensor(
            cute.recast_ptr(sDh1Raw.iterator, swizzle_=s_state_store_layout.inner, dtype=self.io_dtype),
            s_state_store_layout.outer,
        )
        sDh1Read = cute.make_tensor(
            cute.recast_ptr(sDh1Raw.iterator, swizzle_=s_state_read_layout.inner, dtype=self.io_dtype),
            s_state_read_layout.outer,
        )

        kdh_thr = kdh_mma.get_slice(tidx)
        update_thr = update_mma.get_slice(tidx)
        r2s_kdh_thr = r2s_kdh.get_slice(tidx)
        r2s_update_thr = r2s_update.get_slice(tidx)

        c_update = cute.make_identity_tensor((BK, self.BV))
        tCUpdate = update_thr.partition_C(c_update)
        update_c_shape = update_thr.partition_shape_C((BK, self.BV))
        state0 = update_thr.make_fragment_C(update_c_shape)
        state1 = update_thr.make_fragment_C(update_c_shape)
        dht = cute.make_tensor(
            dht_ptr,
            cute.make_layout((K, V, (H, B)), stride=(V, 1, (K * V, H * K * V))),
        )

        if has_dht != 0:
            for i in cutlass.range_constexpr(cute.size(state0)):
                coord = tCUpdate[i]
                k_rel = coord[0]
                v_rel = coord[1]
                state0[i] = dht[k_rel, v_base + v_rel, (h, b)]
                state1[i] = dht[k_rel + BK, v_base + v_rel, (h, b)]
        else:
            state0.fill(0.0)
            state1.fill(0.0)

        for iter_idx in cutlass.range(NT):
            chunk = NT - Int32(1) - iter_idx
            t0 = chunk * BT
            base_k = ((b * T + t0) * H + h) * K
            base_v = ((b * T + t0) * H + h) * V + v_base

            if cutlass.const_expr(self.BV == 64):
                k_row_stride_bytes = H * K * 2
                v_row_stride_bytes = H * V * 2
                _cp_async_k_sw128_64x64(tidx, sK0Raw.iterator.toint(), k_ptr + base_k, k_row_stride_bytes)
                _cp_async_k_sw128_64x64(tidx, sK1Raw.iterator.toint(), k_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sW0Raw.iterator.toint(), w_ptr + base_k, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sQ0Raw.iterator.toint(), q_ptr + base_k, k_row_stride_bytes)
                cute.arch.cp_async_commit_group()
                _cp_async_mn_sw128_64x64(tidx, sW1Raw.iterator.toint(), w_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sQ1Raw.iterator.toint(), q_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sDoRaw.iterator.toint(), do_ptr + base_v, v_row_stride_bytes)
                self._cp_async_dv_bridge_64x64(
                    tidx,
                    sDvPrefRaw.iterator,
                    dv_ptr + base_v,
                    v_row_stride_bytes,
                )
                cute.arch.cp_async_commit_group()
            else:
                k_row_stride_bytes = H * K * 2
                v_row_stride_bytes = H * V * 2
                _cp_async_k_sw128_64x64(tidx, sK0Raw.iterator.toint(), k_ptr + base_k, k_row_stride_bytes)
                _cp_async_k_sw128_64x64(tidx, sK1Raw.iterator.toint(), k_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sW0Raw.iterator.toint(), w_ptr + base_k, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sQ0Raw.iterator.toint(), q_ptr + base_k, k_row_stride_bytes)
                cute.arch.cp_async_commit_group()
                _cp_async_mn_sw128_64x64(tidx, sW1Raw.iterator.toint(), w_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw128_64x64(tidx, sQ1Raw.iterator.toint(), q_ptr + base_k + BK, k_row_stride_bytes)
                _cp_async_mn_sw64_32x64(tidx, sDoRaw.iterator.toint(), do_ptr + base_v, v_row_stride_bytes)
                _cp_async_dv_bridge_64x32(tidx, sDvPrefRaw.iterator.toint(), dv_ptr + base_v, v_row_stride_bytes)
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.BV == 32):
                cute.arch.sync_threads()

            dh_base = (((b * NT + chunk) * H + h) * K) * V + v_base
            self._r2s_acc_store_bridge(
                state0,
                r2s_update,
                r2s_update_thr,
                sDh0Store,
                sDhBridgeRaw.iterator,
                dh_ptr + dh_base,
                V * 2,
                tidx,
            )
            self._r2s_acc_store_bridge(
                state1,
                r2s_update,
                r2s_update_thr,
                sDh1Store,
                sDh1BridgeRaw.iterator,
                dh_ptr + dh_base + BK * V,
                V * 2,
                tidx,
            )
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            tK0 = kdh_thr.partition_A(sK0)
            tK1 = kdh_thr.partition_A(sK1)
            tDh0 = kdh_thr.partition_B(sDh0Read)
            tDh1 = kdh_thr.partition_B(sDh1Read)
            tK0R = kdh_thr.make_fragment_A(tK0)
            tK1R = kdh_thr.make_fragment_A(tK1)
            tDh0R = kdh_thr.make_fragment_B(tDh0)
            tDh1R = kdh_thr.make_fragment_B(tDh1)
            kdh_c_shape = kdh_thr.partition_shape_C((BT, self.BV))
            acc_dv2 = kdh_thr.make_fragment_C(kdh_c_shape)
            acc_dv2.fill(0.0)

            cute.nvgpu.warpgroup.fence()
            self._gemm_sm90_loop(
                kdh_mma,
                tK0R[None, None, None, 0],
                tDh0R[None, None, None, 0],
                acc_dv2,
                False,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.nvgpu.warpgroup.fence()

            cute.nvgpu.warpgroup.fence()
            self._gemm_sm90_loop(
                kdh_mma,
                tK1R[None, None, None, 0],
                tDh1R[None, None, None, 0],
                acc_dv2,
                True,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.nvgpu.warpgroup.fence()

            if cutlass.const_expr(self.BV == 64):
                self._add_dv_bridge_64x64_from_smem(
                    acc_dv2,
                    sDvPrefRaw.iterator,
                    tidx,
                )
            else:
                self._add_dv_bridge_64x32_from_smem(
                    acc_dv2,
                    sDvPrefRaw.iterator,
                    tidx,
                )

            if cutlass.const_expr(self.BV == 64):
                self._r2s_acc_store_bridge_64x64_postbar(
                    acc_dv2,
                    r2s_kdh,
                    r2s_kdh_thr,
                    sDv2Store,
                    sDhBridgeRaw.iterator,
                    dv2_ptr + base_v,
                    H * V * 2,
                    tidx,
                )
            else:
                self._r2s_acc_store_bridge(
                    acc_dv2,
                    r2s_kdh,
                    r2s_kdh_thr,
                    sDv2Store,
                    sDv2BridgeRaw.iterator,
                    dv2_ptr + base_v,
                    H * V * 2,
                    tidx,
                )

            tQ0 = update_thr.partition_A(sQ0)
            tQ1 = update_thr.partition_A(sQ1)
            tW0 = update_thr.partition_A(sW0)
            tW1 = update_thr.partition_A(sW1)
            tDo = update_thr.partition_B(sDo)
            tDv2 = update_thr.partition_B(sDv2Read)
            tQ0R = update_thr.make_fragment_A(tQ0)
            tQ1R = update_thr.make_fragment_A(tQ1)
            tW0R = update_thr.make_fragment_A(tW0)
            tW1R = update_thr.make_fragment_A(tW1)
            tDoR = update_thr.make_fragment_B(tDo)
            tDv2R = update_thr.make_fragment_B(tDv2)

            acc_qdo0 = update_thr.make_fragment_C(update_c_shape)
            acc_qdo1 = update_thr.make_fragment_C(update_c_shape)
            acc_wdv0 = update_thr.make_fragment_C(update_c_shape)
            acc_wdv1 = update_thr.make_fragment_C(update_c_shape)
            acc_qdo0.fill(0.0)
            acc_qdo1.fill(0.0)
            acc_wdv0.fill(0.0)
            acc_wdv1.fill(0.0)

            cute.nvgpu.warpgroup.fence()
            self._gemm_sm90_loop(
                update_mma,
                tQ0R[None, None, None, 0],
                tDoR[None, None, None, 0],
                acc_qdo0,
                False,
            )
            self._gemm_sm90_loop(
                update_mma,
                tQ1R[None, None, None, 0],
                tDoR[None, None, None, 0],
                acc_qdo1,
                False,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.nvgpu.warpgroup.fence()

            cute.nvgpu.warpgroup.fence()
            self._gemm_sm90_loop(
                update_mma,
                tW0R[None, None, None, 0],
                tDv2R[None, None, None, 0],
                acc_wdv0,
                False,
            )
            self._gemm_sm90_loop(
                update_mma,
                tW1R[None, None, None, 0],
                tDv2R[None, None, None, 0],
                acc_wdv1,
                False,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.nvgpu.warpgroup.fence()

            for i in cutlass.range_constexpr(cute.size(state0)):
                state0[i] += acc_qdo0[i] * scale - acc_wdv0[i]
                state1[i] += acc_qdo1[i] * scale - acc_wdv1[i]

        if store_dh0 != 0:
            dh0_base = ((b * H + h) * K) * V + v_base
            gDh00 = cute.make_tensor(dh0_ptr + dh0_base, cute.make_layout((BK, self.BV), stride=(V, 1)))
            gDh01 = cute.make_tensor(dh0_ptr + dh0_base + BK * V, cute.make_layout((BK, self.BV), stride=(V, 1)))
            self._store_dh0_tile(state0, tCUpdate, sDh0Float, gDh00, V * 4, tidx)
            self._store_dh0_tile(state1, tCUpdate, sDh0Float, gDh01, V * 4, tidx)


def _compile_variant(H: int, BV: int, use_fast_math: bool):
    kernel_obj = ChunkDeltaBwdDhuSm90(BV, H, use_fast_math=use_fast_math)
    sym_b = cute.sym_int()
    sym_t = cute.sym_int()

    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K_DIM), stride_order=(3, 2, 1, 0), assumed_align=128)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K_DIM), stride_order=(3, 2, 1, 0), assumed_align=128)
    w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_b, sym_t, H, K_DIM), stride_order=(3, 2, 1, 0), assumed_align=128)
    do_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, sym_t, H, V_DIM), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    dv_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, sym_t, H, V_DIM), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    nt_fake = cute.sym_int()
    dh_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, nt_fake, H, K_DIM, V_DIM),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=128,
    )
    dh0_fake = make_fake_compact_tensor(
        cutlass.Float32, (sym_b, H, K_DIM, V_DIM), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    dv2_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, sym_t, H, V_DIM), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    dht_fake = make_fake_compact_tensor(
        cutlass.Float32, (sym_b, H, K_DIM, V_DIM), stride_order=(3, 2, 1, 0), assumed_align=128
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        w_fake,
        do_fake,
        dv_fake,
        dh_fake,
        dh0_fake,
        dv2_fake,
        dht_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K_DIM), Int32(V_DIM)),
        Float32(1.0),
        Int32(0),
        Int32(0),
        stream_fake,
        options="--enable-tvm-ffi",
    )


def _select_bv(B: int, H: int, V: int = V_DIM, sm_count: int | None = None) -> int:
    if V != V_DIM:
        raise ValueError(f"CuTeDSL bwd_dhu requires V={V_DIM}, got V={V}")
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return 64 if sm_count <= B * H * (V // 64) else 32


def _get_compiled(H: int, BV: int):
    key = (H, K_DIM, V_DIM, BV, USE_FAST_MATH)
    if key not in _kernel_cache:
        _kernel_cache[key] = _compile_variant(H, BV, USE_FAST_MATH)
    return _kernel_cache[key]


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
        raise NotImplementedError("CuTeDSL bwd_dhu currently supports no g gate only")
    if gk is not None:
        raise NotImplementedError("CuTeDSL bwd_dhu currently supports no gk gate only")
    if cu_seqlens is not None:
        raise NotImplementedError("CuTeDSL bwd_dhu currently supports fixed-length tensors only")
    if transpose_state_layout:
        raise NotImplementedError("CuTeDSL bwd_dhu currently supports [B, NT, H, K, V] state layout only")
    if chunk_size != BT:
        raise ValueError(f"CuTeDSL bwd_dhu requires chunk_size={BT}, got {chunk_size}")
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    B, T, H, K = q.shape
    V = do.shape[-1]
    if K != K_DIM or V != V_DIM:
        raise ValueError(f"CuTeDSL bwd_dhu requires K=V=128, got K={K}, V={V}")
    if T % BT != 0:
        raise ValueError(f"CuTeDSL bwd_dhu requires T to be a multiple of {BT}, got {T}")

    _check_same_shape("k", k, q)
    _check_same_shape("w", w, q)
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
        if h0.dtype is not torch.float32 or not h0.is_cuda or not h0.is_contiguous():
            raise ValueError("h0 must be a contiguous CUDA float32 tensor")
    if dht is not None:
        if dht.shape != expected_state_shape:
            raise ValueError(f"dht must have shape {expected_state_shape}, got {tuple(dht.shape)}")
        if dht.dtype is not torch.float32 or not dht.is_cuda or not dht.is_contiguous():
            raise ValueError("dht must be a contiguous CUDA float32 tensor")

    NT = T // BT
    dh = q.new_empty(B, NT, H, K, V)
    dh0 = (
        torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
        if h0 is not None
        else torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)
    )
    dv2 = torch.empty_like(dv)
    dht_arg = dht if dht is not None else torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)

    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    BV = _select_bv(B, H, V, sm_count)
    compiled = _get_compiled(H, BV)
    compiled(
        q,
        k,
        w,
        do,
        dv,
        dh,
        dh0,
        dv2,
        dht_arg,
        (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V)),
        Float32(1.0 if scale is None else float(scale)),
        Int32(1 if dht is not None else 0),
        Int32(1 if h0 is not None else 0),
    )
    return dh, (dh0 if h0 is not None else None), dv2


__all__ = ["chunk_gated_delta_rule_bwd_dhu"]
