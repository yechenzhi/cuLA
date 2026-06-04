#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SM90 #mma -> store conversion microkernels.

This isolates the layout conversion used by Triton's

    tensor<64x64xbf16, #mma> -> ttg.convert_layout -> #blocked -> tt.store

and compares it with the CuTe primitives currently available to this repo.
It intentionally does not import or modify the full bwd_dhu kernel.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T as _T
from torch.utils.cpp_extension import load_inline

from cula.utils import assert_hopper

M = 64
N = 64
K = 64
NUM_THREADS = 128


@cutlass.dsl_user_op
def _pack_bf16x2_f32(hi, lo, *, loc=None, ip=None):
    packed = _llvm.inline_asm(
        _T.i32(),
        [
            Float32(hi).ir_value(loc=loc, ip=ip),
            Float32(lo).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(packed)


@cutlass.dsl_user_op
def _store_tritonlike_packed_64x64_bf16(
    tidx,
    smem_base,
    gmem_base,
    row_stride_bytes,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    *,
    loc=None,
    ip=None,
):
    _llvm.inline_asm(
        None,
        [
            Int32(tidx).ir_value(loc=loc, ip=ip),
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Int64(gmem_base).ir_value(loc=loc, ip=ip),
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
        "{\n"
        ".reg .u32 t8, st_a, st_b, st_c, st_logical, st_addr;\n"
        ".reg .u32 row_base, row, col, out_col, out_col_bytes, out_row, tmp;\n"
        ".reg .u32 l0, l1, l2, l3;\n"
        ".reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;\n"
        ".reg .u32 o8, o9, o10, o11, o12, o13, o14, o15;\n"
        ".reg .u64 g0, g1, g2, g3, row_bytes, group_bytes;\n"
        "bar.sync 0;\n"
        "shl.b32 t8, $0, 3;\n"
        "shl.b32 st_a, $0, 6;\n"
        "and.b32 st_a, st_a, 960;\n"
        "shr.u32 st_b, $0, 1;\n"
        "and.b32 st_b, st_b, 8;\n"
        "shl.b32 st_c, $0, 5;\n"
        "and.b32 st_c, st_c, 3072;\n"
        "or.b32 st_logical, st_a, st_b;\n"
        "or.b32 st_logical, st_logical, st_c;\n"
        "shr.u32 st_addr, st_logical, 2;\n"
        "and.b32 st_addr, st_addr, 1008;\n"
        "shl.b32 tmp, st_logical, 1;\n"
        "add.u32 st_addr, st_addr, tmp;\n"
        "add.u32 st_addr, st_addr, $1;\n"
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [st_addr], {$4, $5, $6, $7};\n"
        "add.u32 tmp, st_addr, 32;\n"
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [tmp], {$8, $9, $10, $11};\n"
        "add.u32 tmp, st_addr, 64;\n"
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [tmp], {$12, $13, $14, $15};\n"
        "add.u32 tmp, st_addr, 96;\n"
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [tmp], {$16, $17, $18, $19};\n"
        "bar.sync 0;\n"
        "and.b32 row_base, t8, 960;\n"
        "shr.u32 row, row_base, 6;\n"
        "and.b32 col, t8, 56;\n"
        "mul.wide.u32 row_bytes, row, $3;\n"
        "cvt.u64.u32 group_bytes, col;\n"
        "shl.b64 group_bytes, group_bytes, 1;\n"
        "add.u64 g0, $2, row_bytes;\n"
        "add.u64 g0, g0, group_bytes;\n"
        "mul.wide.u32 group_bytes, $3, 16;\n"
        "add.u64 g1, g0, group_bytes;\n"
        "mul.wide.u32 group_bytes, $3, 32;\n"
        "add.u64 g2, g0, group_bytes;\n"
        "mul.wide.u32 group_bytes, $3, 48;\n"
        "add.u64 g3, g0, group_bytes;\n"
        "and.b32 out_col, t8, 1016;\n"
        "shl.b32 out_col_bytes, out_col, 1;\n"
        "and.b32 out_row, $0, 120;\n"
        "shl.b32 out_row, out_row, 1;\n"
        "add.u32 l0, $1, out_row;\n"
        "add.u32 l0, l0, out_col_bytes;\n"
        "or.b32 tmp, out_col, 1024;\n"
        "shr.u32 tmp, tmp, 2;\n"
        "and.b32 tmp, tmp, 496;\n"
        "add.u32 tmp, tmp, out_col_bytes;\n"
        "add.u32 tmp, tmp, 2048;\n"
        "add.u32 l1, $1, tmp;\n"
        "or.b32 tmp, out_col, 2048;\n"
        "shr.u32 tmp, tmp, 2;\n"
        "and.b32 tmp, tmp, 752;\n"
        "add.u32 tmp, tmp, out_col_bytes;\n"
        "add.u32 tmp, tmp, 4096;\n"
        "add.u32 l2, $1, tmp;\n"
        "or.b32 tmp, out_col, 3072;\n"
        "shr.u32 tmp, tmp, 2;\n"
        "and.b32 tmp, tmp, 1008;\n"
        "add.u32 tmp, tmp, out_col_bytes;\n"
        "add.u32 tmp, tmp, 6144;\n"
        "add.u32 l3, $1, tmp;\n"
        "ld.shared.v4.b32 {o0, o1, o2, o3}, [l0];\n"
        "ld.shared.v4.b32 {o4, o5, o6, o7}, [l1];\n"
        "ld.shared.v4.b32 {o8, o9, o10, o11}, [l2];\n"
        "ld.shared.v4.b32 {o12, o13, o14, o15}, [l3];\n"
        "st.global.v4.b32 [g0], {o0, o1, o2, o3};\n"
        "st.global.v4.b32 [g1], {o4, o5, o6, o7};\n"
        "st.global.v4.b32 [g2], {o8, o9, o10, o11};\n"
        "st.global.v4.b32 [g3], {o12, o13, o14, o15};\n"
        "}\n",
        ",".join(["r", "r", "l", "r"] + ["r"] * 16),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


CUTLASS_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

using namespace cute;

static constexpr int kM = 64;
static constexpr int kN = 64;
static constexpr int kElements = kM * kN;
static constexpr int kThreads = 128;

using Element = cutlass::bfloat16_t;
using Acc = float;

__global__ __launch_bounds__(kThreads)
void cutlass_cpp_epilogue_kernel(Element const* __restrict__ src, Element* __restrict__ out) {
  int tid = int(threadIdx.x);
  int bid = int(blockIdx.x);

  using CLayout = Layout<Shape<_64, _64>, Stride<_64, _1>>;
  __shared__ alignas(128) Element smem[kElements];

  Tensor gSrc = make_tensor(make_gmem_ptr(src), CLayout{});
  Tensor gOut = make_tensor(make_gmem_ptr(out + bid * kElements), CLayout{});
  Tensor sC = make_tensor(make_smem_ptr(smem), CLayout{});

  auto tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(tid);
  Tensor tCgC = thr_mma.partition_C(gSrc);
  Tensor tCrAcc = thr_mma.make_fragment_C(tCgC);

  CUTE_UNROLL
  for (int i = 0; i < size(tCrAcc); ++i) {
    tCrAcc(i) = static_cast<Acc>(tCgC(i));
  }

  Tensor tCrOut = make_fragment_like<Element>(tCrAcc);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrOut); ++i) {
    tCrOut(i) = Element(tCrAcc(i));
  }

  auto r2s = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma);
  auto r2s_thr = r2s.get_slice(tid);
  copy(r2s, r2s_thr.retile_S(tCrOut), r2s_thr.partition_D(sC));

  __syncthreads();

  auto s2g = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  auto s2g_thr = s2g.get_slice(tid);
  Tensor tSs = s2g_thr.partition_S(sC);
  Tensor tSr = make_fragment_like(tSs);
  copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSs, tSr);
  copy(s2g, tSr, s2g_thr.partition_D(gOut));
}

__global__ __launch_bounds__(kThreads)
void cutlass_cpp_wgmma_kernel(
    Element const* __restrict__ a,
    Element const* __restrict__ b,
    Element* __restrict__ out) {
  int tid = int(threadIdx.x);
  int bid = int(blockIdx.x);

  using CLayout = Layout<Shape<_64, _64>, Stride<_64, _1>>;
  using SmemLayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Element>{},
      make_shape(Int<64>{}, Int<64>{}, Int<1>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Element>{},
      make_shape(Int<64>{}, Int<64>{}, Int<1>{})));

  __shared__ alignas(128) Element smem_a[cosize_v<SmemLayoutA>];
  __shared__ alignas(128) Element smem_b[cosize_v<SmemLayoutB>];
  __shared__ alignas(128) Element smem_c[kElements];

  Tensor sA = make_tensor(make_smem_ptr(smem_a), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem_b), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem_c), CLayout{});
  Tensor gOut = make_tensor(make_gmem_ptr(out + bid * kElements), CLayout{});

  for (int linear = tid; linear < kElements; linear += kThreads) {
    int row = linear / kN;
    int col = linear - row * kN;
    sA(row, col, 0) = a[row * kN + col];
    sB(row, col, 0) = b[col * kN + row];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  auto tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(tid);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCgC = thr_mma.partition_C(gOut);
  Tensor tCrAcc = thr_mma.make_fragment_C(tCgC);
  clear(tCrAcc);

  warpgroup_fence_operand(tCrAcc);
  warpgroup_arrive();
  gemm(tiled_mma, tCrA(_, _, _, _0{}), tCrB(_, _, _, _0{}), tCrAcc);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrAcc);

  Tensor tCrOut = make_fragment_like<Element>(tCrAcc);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrOut); ++i) {
    tCrOut(i) = Element(tCrAcc(i));
  }

  auto r2s = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma);
  auto r2s_thr = r2s.get_slice(tid);
  copy(r2s, r2s_thr.retile_S(tCrOut), r2s_thr.partition_D(sC));

  __syncthreads();

  auto s2g = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  auto s2g_thr = s2g.get_slice(tid);
  Tensor tSs = s2g_thr.partition_S(sC);
  Tensor tSr = make_fragment_like(tSs);
  copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSs, tSr);
  copy(s2g, tSr, s2g_thr.partition_D(gOut));
}

__global__ __launch_bounds__(kThreads)
void cutlass_cpp_wgmma_tiled_kernel(
    Element const* __restrict__ a,
    Element const* __restrict__ b,
    Element* __restrict__ out) {
  int tid = int(threadIdx.x);
  int bid = int(blockIdx.x);

  auto shape_mnk = make_shape(Int<64>{}, Int<64>{}, Int<64>{});
  auto cta_tiler = make_shape(Int<64>{}, Int<64>{}, Int<64>{});

  using CLayout = Layout<Shape<_64, _64>, Stride<_64, _1>>;
  using SmemLayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Element>{},
      make_shape(Int<64>{}, Int<64>{}, Int<1>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<Element>{},
      make_shape(Int<64>{}, Int<64>{}, Int<1>{})));

  __shared__ alignas(128) Element smem_a[cosize_v<SmemLayoutA>];
  __shared__ alignas(128) Element smem_b[cosize_v<SmemLayoutB>];

  Tensor mA = make_tensor(
      make_gmem_ptr(a),
      select<0, 2>(shape_mnk),
      make_stride(Int<64>{}, Int<1>{}));
  // Logical B is (N,K) so that B(n,k) = b[k,n], matching Triton's KxN input.
  Tensor mB = make_tensor(
      make_gmem_ptr(b),
      select<1, 2>(shape_mnk),
      make_stride(Int<1>{}, Int<64>{}));
  Tensor mC = make_tensor(
      make_gmem_ptr(out + bid * kElements),
      select<0, 1>(shape_mnk),
      make_stride(Int<64>{}, Int<1>{}));

  Tensor gA = local_tile(mA, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, make_coord(0, 0, _), Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, make_coord(0, 0, _), Step<_1, _1, X>{});

  Tensor sA = make_tensor(make_smem_ptr(smem_a), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem_b), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem_a), CLayout{});

  auto copyA = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  auto copyB = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_8, _16>>{},
      Layout<Shape<_8, _1>>{});

  auto thr_copy_a = copyA.get_slice(tid);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(as_position_independent_swizzle_tensor(sA));
  auto thr_copy_b = copyB.get_slice(tid);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(as_position_independent_swizzle_tensor(sB));

  copy(copyA, tAgA(_, _, _, _0{}), tAsA(_, _, _, _0{}));
  copy(copyB, tBgB(_, _, _, _0{}), tBsB(_, _, _, _0{}));
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  auto tiled_mma = make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{});
  auto thr_mma = tiled_mma.get_slice(tid);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrAcc = thr_mma.make_fragment_C(tCgC);
  clear(tCrAcc);

  auto tCrA0 = tCrA(_, _, Int<0>{}, _0{});
  auto tCrA1 = tCrA(_, _, Int<1>{}, _0{});
  auto tCrA2 = tCrA(_, _, Int<2>{}, _0{});
  auto tCrA3 = tCrA(_, _, Int<3>{}, _0{});
  auto tCrB0 = tCrB(_, _, Int<0>{}, _0{});
  auto tCrB1 = tCrB(_, _, Int<1>{}, _0{});
  auto tCrB2 = tCrB(_, _, Int<2>{}, _0{});
  auto tCrB3 = tCrB(_, _, Int<3>{}, _0{});
  uint64_t desc_a0 = tCrA0[0];
  uint64_t desc_a1 = tCrA1[0];
  uint64_t desc_a2 = tCrA2[0];
  uint64_t desc_a3 = tCrA3[0];
  uint64_t desc_b0 = tCrB0[0];
  uint64_t desc_b1 = tCrB1[0];
  uint64_t desc_b2 = tCrB2[0];
  uint64_t desc_b3 = tCrB3[0];

  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  uint32_t r8 = 0, r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0;
  uint32_t r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0;
  uint32_t r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0;

  warpgroup_fence_operand(tCrAcc);
  warpgroup_arrive();
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mov.pred p, -1;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32, %33, 0, 1, 1, 0, 1;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %34, %35, p, 1, 1, 0, 1;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %36, %37, p, 1, 1, 0, 1;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %38, %39, p, 1, 1, 0, 1;\n"
      "}\n"
      : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
        "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7),
        "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),
        "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15),
        "+r"(r16), "+r"(r17), "+r"(r18), "+r"(r19),
        "+r"(r20), "+r"(r21), "+r"(r22), "+r"(r23),
        "+r"(r24), "+r"(r25), "+r"(r26), "+r"(r27),
        "+r"(r28), "+r"(r29), "+r"(r30), "+r"(r31)
      : "l"(desc_a0), "l"(desc_b0), "l"(desc_a1), "l"(desc_b1),
        "l"(desc_a2), "l"(desc_b2), "l"(desc_a3), "l"(desc_b3));
  warpgroup_commit_batch();
  warpgroup_wait<0>();
#define CULA_MOV_ACC(I)                     \
  do {                                      \
    float tmp;                              \
    asm volatile("mov.b32 %0, %1;"          \
                 : "=f"(tmp)                \
                 : "r"(r##I));              \
    tCrAcc(I) = tmp;                        \
  } while (0)
  CULA_MOV_ACC(0);
  CULA_MOV_ACC(1);
  CULA_MOV_ACC(2);
  CULA_MOV_ACC(3);
  CULA_MOV_ACC(4);
  CULA_MOV_ACC(5);
  CULA_MOV_ACC(6);
  CULA_MOV_ACC(7);
  CULA_MOV_ACC(8);
  CULA_MOV_ACC(9);
  CULA_MOV_ACC(10);
  CULA_MOV_ACC(11);
  CULA_MOV_ACC(12);
  CULA_MOV_ACC(13);
  CULA_MOV_ACC(14);
  CULA_MOV_ACC(15);
  CULA_MOV_ACC(16);
  CULA_MOV_ACC(17);
  CULA_MOV_ACC(18);
  CULA_MOV_ACC(19);
  CULA_MOV_ACC(20);
  CULA_MOV_ACC(21);
  CULA_MOV_ACC(22);
  CULA_MOV_ACC(23);
  CULA_MOV_ACC(24);
  CULA_MOV_ACC(25);
  CULA_MOV_ACC(26);
  CULA_MOV_ACC(27);
  CULA_MOV_ACC(28);
  CULA_MOV_ACC(29);
  CULA_MOV_ACC(30);
  CULA_MOV_ACC(31);
#undef CULA_MOV_ACC
  warpgroup_fence_operand(tCrAcc);

  Tensor tCrOut = make_fragment_like<Element>(tCrAcc);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrOut); ++i) {
    tCrOut(i) = Element(tCrAcc(i));
  }

  auto r2s = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma);
  auto r2s_thr = r2s.get_slice(tid);
  copy(r2s, r2s_thr.retile_S(tCrOut), r2s_thr.partition_D(sC));

  __syncthreads();

  auto s2g = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  auto s2g_thr = s2g.get_slice(tid);
  Tensor tSs = s2g_thr.partition_S(sC);
  Tensor tSr = make_fragment_like(tSs);
  copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSs, tSr);
  copy(s2g, tSr, s2g_thr.partition_D(gC));
}

__global__ __launch_bounds__(kThreads)
void cutlass_cpp_tritonlike_kernel(
    Element const* __restrict__ a,
    Element const* __restrict__ b,
    Element* __restrict__ out) {
  int tid = int(threadIdx.x);
  int bid = int(blockIdx.x);

  __shared__ alignas(128) unsigned char smem[16 * 1024];
  uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  uint32_t t8 = uint32_t(tid) << 3;
  uint32_t row_col_base = t8 & 960u;
  uint32_t col_base = t8 & 56u;
  uint64_t g_base = (uint64_t(row_col_base) + uint64_t(col_base)) * 2ull;
  int warp_bit = ((tid >> 5) & 1) ? -1 : 0;
  uint32_t xor_warp = uint32_t(warp_bit) & 288u;
  uint32_t sw_src = (uint32_t(tid) & 24u) * 9u;
  uint32_t sw_high = t8 & 512u;

  auto swizzled_smem_offset = [&](uint32_t k_offset) {
    uint32_t logical = (sw_src ^ (col_base | k_offset)) ^ xor_warp;
    logical |= sw_high;
    return logical << 1;
  };

  uint32_t a0, a1, a2, a3, a4, a5, a6, a7;
  uint32_t a8, a9, a10, a11, a12, a13, a14, a15;
  uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
  uint32_t b8, b9, b10, b11, b12, b13, b14, b15;

  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "l"(reinterpret_cast<char const*>(a) + g_base));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7)
               : "l"(reinterpret_cast<char const*>(a) + g_base + 2048));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a8), "=r"(a9), "=r"(a10), "=r"(a11)
               : "l"(reinterpret_cast<char const*>(a) + g_base + 4096));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a12), "=r"(a13), "=r"(a14), "=r"(a15)
               : "l"(reinterpret_cast<char const*>(a) + g_base + 6144));

  uint32_t s0 = smem_base + swizzled_smem_offset(0);
  uint32_t s1 = smem_base + swizzled_smem_offset(1024);
  uint32_t s2 = smem_base + swizzled_smem_offset(2048);
  uint32_t s3 = smem_base + swizzled_smem_offset(3072);
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(s0), "r"(a0), "r"(a1), "r"(a2), "r"(a3));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(s1), "r"(a4), "r"(a5), "r"(a6), "r"(a7));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(s2), "r"(a8), "r"(a9), "r"(a10), "r"(a11));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(s3), "r"(a12), "r"(a13), "r"(a14), "r"(a15));

  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
               : "l"(reinterpret_cast<char const*>(b) + g_base));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(b4), "=r"(b5), "=r"(b6), "=r"(b7)
               : "l"(reinterpret_cast<char const*>(b) + g_base + 2048));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(b8), "=r"(b9), "=r"(b10), "=r"(b11)
               : "l"(reinterpret_cast<char const*>(b) + g_base + 4096));
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(b12), "=r"(b13), "=r"(b14), "=r"(b15)
               : "l"(reinterpret_cast<char const*>(b) + g_base + 6144));

  uint32_t sb0 = s0 + 8192;
  uint32_t sb1 = s1 + 8192;
  uint32_t sb2 = s2 + 8192;
  uint32_t sb3 = s3 + 8192;
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(sb0), "r"(b0), "r"(b1), "r"(b2), "r"(b3));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(sb1), "r"(b4), "r"(b5), "r"(b6), "r"(b7));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(sb2), "r"(b8), "r"(b9), "r"(b10), "r"(b11));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(sb3), "r"(b12), "r"(b13), "r"(b14), "r"(b15));

  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");

  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
  uint32_t r8 = 0, r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0;
  uint32_t r16 = 0, r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0;
  uint32_t r24 = 0, r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0;

#define CULA_ACC_ARGS                                                        \
  "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4), "+r"(r5),             \
      "+r"(r6), "+r"(r7), "+r"(r8), "+r"(r9), "+r"(r10), "+r"(r11),      \
      "+r"(r12), "+r"(r13), "+r"(r14), "+r"(r15), "+r"(r16), "+r"(r17),  \
      "+r"(r18), "+r"(r19), "+r"(r20), "+r"(r21), "+r"(r22), "+r"(r23),  \
      "+r"(r24), "+r"(r25), "+r"(r26), "+r"(r27), "+r"(r28), "+r"(r29),  \
      "+r"(r30), "+r"(r31)
#define CULA_WGMMA_ZERO(DA, DB)                                                \
  asm volatile(                                                                \
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                 \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"              \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "   \
      "%32, %33, 0, 1, 1, 0, 1;"                                              \
      : CULA_ACC_ARGS                                                          \
      : "l"(DA), "l"(DB))
#define CULA_WGMMA_ONE(DA, DB)                                                 \
  asm volatile(                                                                \
      "{ .reg .pred p; mov.pred p, -1; "                                      \
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                 \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"              \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "   \
      "%32, %33, p, 1, 1, 0, 1; }"                                           \
      : CULA_ACC_ARGS                                                          \
      : "l"(DA), "l"(DB))

  constexpr uint64_t kTritonDescFlags = 0x4000004002000000ull;
  auto make_desc = [&](uint32_t addr) {
    return (uint64_t((addr >> 4) & 0x3fffu) | kTritonDescFlags);
  };

  uint64_t da0 = make_desc(smem_base);
  uint64_t db0 = make_desc(smem_base + 8192);
  CULA_WGMMA_ZERO(da0, db0);
  uint64_t da1 = make_desc(smem_base + 32);
  uint64_t db1 = make_desc(smem_base + 10240);
  CULA_WGMMA_ONE(da1, db1);
  uint64_t da2 = make_desc(smem_base + 64);
  uint64_t db2 = make_desc(smem_base + 12288);
  CULA_WGMMA_ONE(da2, db2);
  uint64_t da3 = make_desc(smem_base + 96);
  uint64_t db3 = make_desc(smem_base + 14336);
  CULA_WGMMA_ONE(da3, db3);
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

#undef CULA_WGMMA_ONE
#undef CULA_WGMMA_ZERO
#undef CULA_ACC_ARGS

  uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
  uint32_t p8, p9, p10, p11, p12, p13, p14, p15;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p0) : "r"(r0), "r"(r1));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p1) : "r"(r2), "r"(r3));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p2) : "r"(r4), "r"(r5));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p3) : "r"(r6), "r"(r7));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p4) : "r"(r8), "r"(r9));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p5) : "r"(r10), "r"(r11));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p6) : "r"(r12), "r"(r13));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p7) : "r"(r14), "r"(r15));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p8) : "r"(r16), "r"(r17));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p9) : "r"(r18), "r"(r19));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p10) : "r"(r20), "r"(r21));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p11) : "r"(r22), "r"(r23));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p12) : "r"(r24), "r"(r25));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p13) : "r"(r26), "r"(r27));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p14) : "r"(r28), "r"(r29));
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p15) : "r"(r30), "r"(r31));

  __syncthreads();

  uint32_t st_a = (uint32_t(tid) << 6) & 960u;
  uint32_t st_b = (uint32_t(tid) >> 1) & 8u;
  uint32_t st_c = (uint32_t(tid) << 5) & 3072u;
  uint32_t st_logical = st_a | st_b | st_c;
  uint32_t st_addr = smem_base + ((st_logical >> 2) & 1008u) + (st_logical << 1);
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(st_addr), "r"(p0), "r"(p1), "r"(p2), "r"(p3));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(st_addr + 32), "r"(p4), "r"(p5), "r"(p6), "r"(p7));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(st_addr + 64), "r"(p8), "r"(p9), "r"(p10), "r"(p11));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(st_addr + 96), "r"(p12), "r"(p13), "r"(p14), "r"(p15));

  __syncthreads();

  uint32_t out_col = t8 & 1016u;
  uint32_t out_col_bytes = out_col << 1;
  uint32_t out_row = (uint32_t(tid) & 120u) << 1;
  uint32_t l0 = smem_base + out_row + out_col_bytes;
  uint32_t o0, o1, o2, o3, o4, o5, o6, o7;
  uint32_t o8, o9, o10, o11, o12, o13, o14, o15;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
               : "r"(l0));
  uint32_t l1 = smem_base + (((out_col | 1024u) >> 2) & 496u) + out_col_bytes + 2048u;
  uint32_t l2 = smem_base + (((out_col | 2048u) >> 2) & 752u) + out_col_bytes + 4096u;
  uint32_t l3 = smem_base + (((out_col | 3072u) >> 2) & 1008u) + out_col_bytes + 6144u;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(o4), "=r"(o5), "=r"(o6), "=r"(o7)
               : "r"(l1));
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(o8), "=r"(o9), "=r"(o10), "=r"(o11)
               : "r"(l2));
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(o12), "=r"(o13), "=r"(o14), "=r"(o15)
               : "r"(l3));

  char* out_base = reinterpret_cast<char*>(out) + (uint64_t(bid) * uint64_t(kElements) * 2ull) + g_base;
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_base), "r"(o0), "r"(o1), "r"(o2), "r"(o3));
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_base + 2048), "r"(o4), "r"(o5), "r"(o6), "r"(o7));
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_base + 4096), "r"(o8), "r"(o9), "r"(o10), "r"(o11));
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_base + 6144), "r"(o12), "r"(o13), "r"(o14), "r"(o15));
}

void launch_cutlass_cpp_epilogue(torch::Tensor src, torch::Tensor out) {
  TORCH_CHECK(src.is_cuda() && out.is_cuda(), "src/out must be CUDA tensors");
  TORCH_CHECK(src.scalar_type() == at::ScalarType::BFloat16, "src must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(src.numel() == kElements, "src must be 64x64");
  TORCH_CHECK(out.dim() == 3 && out.size(1) == kM && out.size(2) == kN,
              "out must have shape [blocks,64,64]");
  auto* src_ptr = reinterpret_cast<Element const*>(src.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<Element*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cutlass_cpp_epilogue_kernel<<<dim3(out.size(0)), dim3(kThreads), 0, stream>>>(src_ptr, out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cutlass_cpp_wgmma(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "a/b/out must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16, "a must be bf16");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::BFloat16, "b must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(a.numel() == kElements && b.numel() == kElements, "a/b must be 64x64");
  TORCH_CHECK(out.dim() == 3 && out.size(1) == kM && out.size(2) == kN,
              "out must have shape [blocks,64,64]");
  auto* a_ptr = reinterpret_cast<Element const*>(a.data_ptr<at::BFloat16>());
  auto* b_ptr = reinterpret_cast<Element const*>(b.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<Element*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cutlass_cpp_wgmma_kernel<<<dim3(out.size(0)), dim3(kThreads), 0, stream>>>(a_ptr, b_ptr, out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cutlass_cpp_wgmma_tiled(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "a/b/out must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16, "a must be bf16");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::BFloat16, "b must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(a.numel() == kElements && b.numel() == kElements, "a/b must be 64x64");
  TORCH_CHECK(out.dim() == 3 && out.size(1) == kM && out.size(2) == kN,
              "out must have shape [blocks,64,64]");
  auto* a_ptr = reinterpret_cast<Element const*>(a.data_ptr<at::BFloat16>());
  auto* b_ptr = reinterpret_cast<Element const*>(b.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<Element*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cutlass_cpp_wgmma_tiled_kernel<<<dim3(out.size(0)), dim3(kThreads), 0, stream>>>(a_ptr, b_ptr, out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cutlass_cpp_tritonlike(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "a/b/out must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16, "a must be bf16");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::BFloat16, "b must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(a.numel() == kElements && b.numel() == kElements, "a/b must be 64x64");
  TORCH_CHECK(out.dim() == 3 && out.size(1) == kM && out.size(2) == kN,
              "out must have shape [blocks,64,64]");
  auto* a_ptr = reinterpret_cast<Element const*>(a.data_ptr<at::BFloat16>());
  auto* b_ptr = reinterpret_cast<Element const*>(b.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<Element*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cutlass_cpp_tritonlike_kernel<<<dim3(out.size(0)), dim3(kThreads), 0, stream>>>(a_ptr, b_ptr, out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_cutlass_cpp_epilogue", &launch_cutlass_cpp_epilogue);
  m.def("launch_cutlass_cpp_wgmma", &launch_cutlass_cpp_wgmma);
  m.def("launch_cutlass_cpp_wgmma_tiled", &launch_cutlass_cpp_wgmma_tiled);
  m.def("launch_cutlass_cpp_tritonlike", &launch_cutlass_cpp_tritonlike);
}
"""


@triton.jit
def triton_mma_store_kernel(a, b, out, stride_ob: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)
    a_tile = tl.load(a + offs_m[:, None] * 64 + offs_k[None, :])
    b_tile = tl.load(b + offs_k[:, None] * 64 + offs_n[None, :])
    acc = tl.dot(a_tile, b_tile)
    tl.store(out + pid * stride_ob + offs_m[:, None] * 64 + offs_n[None, :], acc.to(tl.bfloat16))


def _sass_summary(sass: str, res: str, n_regs: int | None = None) -> dict[str, int | None]:
    summary: dict[str, int | None] = {"REG": n_regs, "STACK": None, "SHARED": None, "LOCAL": None}
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    if match:
        summary.update(dict(zip(["REG", "STACK", "SHARED", "LOCAL"], map(int, match.groups()))))
    for op in [
        "HGMMA",
        "LDGSTS",
        "STSM",
        "LDSM",
        "LDS",
        "STS",
        "STG.E.128",
        "STG.E.U16",
        "BAR.SYNC",
        "DEPBAR",
        "WARPGROUP",
        "MEMBAR",
        "FENCE",
        "LDL",
        "STL",
        "FADD",
        "FFMA",
        "FMUL",
        "SHFL",
        "IADD3",
        "IMAD",
        "LEA",
    ]:
        summary[op] = sass.count(op)
    return summary


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


def dump_triton(out_dir: pathlib.Path, blocks: int, warmup: int, repeat: int) -> dict[str, int | float | bool | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.eye(K, N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((blocks, M, N), device="cuda", dtype=torch.bfloat16)
    triton_mma_store_kernel[(blocks,)](a, b, out, M * N, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    compiled = next(reversed(triton_mma_store_kernel.device_caches[0][0].values()))

    name = "triton_mma_to_store"
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
    ref = a
    max_diff = (out[0] - ref).abs().max().item()
    ms = _bench(lambda: triton_mma_store_kernel[(blocks,)](a, b, out, M * N, num_warps=4, num_stages=3), warmup, repeat)
    ttgir = compiled.asm.get("ttgir", "")
    row = {
        "family": "triton",
        "variant": name,
        "blocks": blocks,
        "ms": ms,
        "ns_per_cta": ms * 1e6 / blocks,
        "max_diff": max_diff,
        "has_mma_to_blocked_store": "#mma" in ttgir and "ttg.convert_layout" in ttgir and "tt.store" in ttgir,
        "ttgir_convert_layout": ttgir.count("ttg.convert_layout"),
        "ttgir_local_alloc": ttgir.count("ttg.local_alloc"),
        **_sass_summary(sass, res, getattr(compiled, "n_regs", None)),
    }
    print("TRITON", json.dumps(row, sort_keys=True), flush=True)
    return row


@functools.lru_cache(maxsize=1)
def _compile_cutlass_cpp():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    root = pathlib.Path(__file__).resolve().parent.parent
    return load_inline(
        name="cula_convert_layout_cutlass_cpp_sm90",
        cpp_sources="",
        cuda_sources=CUTLASS_CPP_SOURCE,
        extra_include_paths=[
            str(root / "csrc" / "cutlass" / "include"),
            str(root / "csrc" / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        with_cuda=True,
        verbose=False,
    )


def _filter_cuobjdump_function(text: str, function_substr: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if "Function " in line and function_substr in line:
            start = idx
            break
    if start is None:
        return text
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if "Function " in lines[idx]:
            end = idx
            break
    return "\n".join(lines[start:end])


def dump_cutlass_cpp_resource(module, out_dir: pathlib.Path, name: str, function_substr: str) -> dict[str, int | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = pathlib.Path(module.__file__)
    res = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(so_path)], check=False, capture_output=True, text=True
    ).stdout
    sass = subprocess.run(["cuobjdump", "--dump-sass", str(so_path)], check=False, capture_output=True, text=True).stdout
    res_fn = _filter_cuobjdump_function(res, function_substr)
    sass_fn = _filter_cuobjdump_function(sass, function_substr)
    (out_dir / f"{name}_resource.txt").write_text(res_fn)
    (out_dir / f"{name}.sass").write_text(sass_fn)
    return _sass_summary(sass_fn, res_fn)


def run_cutlass_cpp_variant(name: str, out_dir: pathlib.Path, blocks: int, warmup: int, repeat: int) -> dict[str, object]:
    row: dict[str, object] = {"family": "cutlass_cpp", "variant": name, "blocks": blocks}
    if name not in ("cutlass_cpp_epilogue", "cutlass_cpp_wgmma", "cutlass_cpp_wgmma_tiled", "cutlass_cpp_tritonlike"):
        row.update({"status": "unsupported"})
        print("CUTLASS_CPP", json.dumps(row, sort_keys=True), flush=True)
        return row
    try:
        module = _compile_cutlass_cpp()
    except Exception as exc:
        row.update({"status": "compile_fail", "error": str(exc).splitlines()[-1]})
        print("CUTLASS_CPP", json.dumps(row, sort_keys=True), flush=True)
        return row

    function_substr = {
        "cutlass_cpp_epilogue": "cutlass_cpp_epilogue_kernel",
        "cutlass_cpp_wgmma": "cutlass_cpp_wgmma_kernel",
        "cutlass_cpp_wgmma_tiled": "cutlass_cpp_wgmma_tiled_kernel",
        "cutlass_cpp_tritonlike": "cutlass_cpp_tritonlike_kernel",
    }[name]
    row.update({"status": "compiled", **dump_cutlass_cpp_resource(module, out_dir, name, function_substr)})
    src = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    rhs = torch.eye(K, N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((blocks, M, N), device="cuda", dtype=torch.bfloat16)
    if name == "cutlass_cpp_epilogue":

        def fn():
            module.launch_cutlass_cpp_epilogue(src, out)
    elif name == "cutlass_cpp_wgmma":

        def fn():
            module.launch_cutlass_cpp_wgmma(src, rhs, out)
    elif name == "cutlass_cpp_wgmma_tiled":

        def fn():
            module.launch_cutlass_cpp_wgmma_tiled(src, rhs, out)
    else:

        def fn():
            module.launch_cutlass_cpp_tritonlike(src, rhs, out)

    fn()
    torch.cuda.synchronize()
    max_diff = (out[0] - src).abs().max().item()
    ms = _bench(fn, warmup, repeat)
    row.update({"max_diff": max_diff, "ms": ms, "ns_per_cta": ms * 1e6 / blocks})
    print("CUTLASS_CPP", json.dumps(row, sort_keys=True), flush=True)
    return row


@dataclass(frozen=True)
class CuteVariant:
    name: str


class CuteConvertStoreMicro:
    def __init__(self, variant: CuteVariant):
        self.variant = variant
        self.kernel_name = variant.name.split("_smem", 1)[0]
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.mma_tiler = (M, N, K)
        self.atom_layout_mnk = (1, 1, 1)
        self.num_threads = NUM_THREADS
        self.buffer_align_bytes = 128
        self.launch_smem_bytes = 0
        if "_smem96" in variant.name:
            self.launch_smem_bytes = 96 * 1024
        elif "_smem160" in variant.name:
            self.launch_smem_bytes = 160 * 1024
        elif "_smem200" in variant.name:
            self.launch_smem_bytes = 200 * 1024

    @cute.jit
    def __call__(self, src: cute.Tensor, rhs: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            self.mma_tiler[:2],
        )
        if cutlass.const_expr(self.kernel_name.startswith("shared_epi_o01")):
            store_smem_layout = sm90_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (M, N),
                1,
                smem_order=(0, 1, 2),
            )
        elif cutlass.const_expr(self.kernel_name.startswith("shared_epi_col")):
            store_smem_layout = sm90_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.COL_MAJOR,
                (M, N),
                1,
            )
        elif cutlass.const_expr(self.kernel_name.startswith("shared_epi")):
            store_smem_layout = sm90_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (M, N),
                1,
            )
        else:
            store_smem_layout = sm90_utils.make_smem_layout_b(
                utils.LayoutEnum.ROW_MAJOR,
                self.mma_tiler,
                self.io_dtype,
                1,
            )
        a_smem_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR,
            self.mma_tiler,
            self.io_dtype,
            1,
        )
        b_smem_layout = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.COL_MAJOR,
            self.mma_tiler,
            self.io_dtype,
            1,
        )
        row_store_atom = sm90_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype)
        stsm_tiled_copy = cute.make_tiled_copy_C(row_store_atom, tiled_mma)
        stsm_c_atom_tiled_copy = cute.make_tiled_copy_C_atom(row_store_atom, tiled_mma)
        stsm_s_tiled_copy = cute.make_tiled_copy_S(row_store_atom, stsm_c_atom_tiled_copy)
        ldm_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.io_dtype),
            tiled_mma,
        )
        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype, num_bits_per_copy=128)
        store_thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
        store_val_layout = cute.make_layout((1, 8))
        store_tiled_copy = cute.make_tiled_copy_tv(store_atom, store_thr_layout, store_val_layout)
        c_store_tiled_copy = cute.make_tiled_copy_C(store_atom, tiled_mma)
        c_atom_store_tiled_copy = cute.make_tiled_copy_C_atom(store_atom, tiled_mma)

        @cute.struct
        class SharedStorage:
            sTile: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(store_smem_layout)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_smem_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(b_smem_layout)],
                self.buffer_align_bytes,
            ]
            sRaw: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, 8192],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        launch_smem_bytes = self.shared_storage.size_in_bytes()
        if self.launch_smem_bytes > launch_smem_bytes:
            launch_smem_bytes = self.launch_smem_bytes

        self.kernel(
            src,
            rhs,
            out,
            tiled_mma,
            store_smem_layout,
            a_smem_layout,
            b_smem_layout,
            stsm_tiled_copy,
            stsm_s_tiled_copy,
            ldm_tiled_copy,
            store_tiled_copy,
            c_store_tiled_copy,
            c_atom_store_tiled_copy,
        ).launch(
            grid=[out.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, 1, 1],
            smem=launch_smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def store_mma_c_fragment_blocked_shared(
        self,
        stsm_tiled_copy: cute.TiledCopy,
        store_tiled_copy: cute.TiledCopy,
        stsm_thr,
        store_thr,
        src_bf16: cute.Tensor,
        sTile: cute.Tensor,
        gOut: cute.Tensor,
    ):
        """Local explicit primitive for #mma-C-fragment -> blocked global store.

        This mirrors Triton's lowered path for #mma -> #blocked store:
        accumulator registers are first written with STSM to CTA shared memory,
        then reloaded as a 128-bit blocked TV fragment and stored to global.
        """
        cute.copy(stsm_tiled_copy, stsm_thr.retile(src_bf16), stsm_thr.partition_D(sTile[(None, None, 0)]))
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier()
        tSs = store_thr.partition_S(sTile[(None, None, 0)])
        tSr = cute.make_fragment_like(tSs, self.io_dtype)
        cute.autovec_copy(tSs, tSr)
        cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))

    @cute.jit
    def store_acc_tritonlike_raw(
        self,
        tidx: Int32,
        acc: cute.Tensor,
        sRaw: cute.Tensor,
        gOut: cute.Tensor,
    ):
        p0 = _pack_bf16x2_f32(acc[1], acc[0])
        p1 = _pack_bf16x2_f32(acc[3], acc[2])
        p2 = _pack_bf16x2_f32(acc[5], acc[4])
        p3 = _pack_bf16x2_f32(acc[7], acc[6])
        p4 = _pack_bf16x2_f32(acc[9], acc[8])
        p5 = _pack_bf16x2_f32(acc[11], acc[10])
        p6 = _pack_bf16x2_f32(acc[13], acc[12])
        p7 = _pack_bf16x2_f32(acc[15], acc[14])
        p8 = _pack_bf16x2_f32(acc[17], acc[16])
        p9 = _pack_bf16x2_f32(acc[19], acc[18])
        p10 = _pack_bf16x2_f32(acc[21], acc[20])
        p11 = _pack_bf16x2_f32(acc[23], acc[22])
        p12 = _pack_bf16x2_f32(acc[25], acc[24])
        p13 = _pack_bf16x2_f32(acc[27], acc[26])
        p14 = _pack_bf16x2_f32(acc[29], acc[28])
        p15 = _pack_bf16x2_f32(acc[31], acc[30])
        _store_tritonlike_packed_64x64_bf16(
            tidx,
            sRaw.iterator.toint(),
            gOut.iterator.toint(),
            Int32(N * self.io_dtype.width // 8),
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
            p9,
            p10,
            p11,
            p12,
            p13,
            p14,
            p15,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        rhs: cute.Tensor,
        out: cute.Tensor,
        tiled_mma: cute.TiledMma,
        store_smem_layout: cute.ComposedLayout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        stsm_tiled_copy: cute.TiledCopy,
        stsm_s_tiled_copy: cute.TiledCopy,
        ldm_tiled_copy: cute.TiledCopy,
        store_tiled_copy: cute.TiledCopy,
        c_store_tiled_copy: cute.TiledCopy,
        c_atom_store_tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sTile = storage.sTile.get_tensor(store_smem_layout.outer, swizzle=store_smem_layout.inner)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sRaw = storage.sRaw.get_tensor(cute.make_layout((8192,)))
        thr_mma = tiled_mma.get_slice(tidx)
        store_thr = store_tiled_copy.get_slice(tidx)
        ldm_thr = ldm_tiled_copy.get_slice(tidx)
        c_store_thr = c_store_tiled_copy.get_slice(tidx)
        c_atom_store_thr = c_atom_store_tiled_copy.get_slice(tidx)
        gOut = out[(bid, None, None)]

        acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((M, N)))
        c = cute.make_identity_tensor((M, N))
        tCc = thr_mma.partition_C(c)
        src_bf16 = cute.make_rmem_tensor_like(acc, self.io_dtype)
        if cutlass.const_expr(
            self.kernel_name.endswith("_mma")
            or self.kernel_name.endswith("_mma_add0")
            or self.kernel_name.endswith("_mma_bar")
            or self.kernel_name == "mma_scalar_out"
        ):
            linear_a = tidx
            while linear_a < M * K:
                m_rel = linear_a // K
                k_rel = linear_a - m_rel * K
                sA[m_rel, k_rel, 0] = src[m_rel, k_rel]
                linear_a += self.num_threads
            linear_b = tidx
            while linear_b < N * K:
                n_rel = linear_b // K
                k_rel = linear_b - n_rel * K
                sB[n_rel, k_rel, 0] = rhs[k_rel, n_rel]
                linear_b += self.num_threads
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier()

            tA = thr_mma.partition_A(sA[(None, None, 0)])
            tAR = thr_mma.make_fragment_A(tA)
            tB = thr_mma.partition_B(sB[(None, None, 0)])
            tBR = thr_mma.make_fragment_B(tB)
            acc.fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for kp in cutlass.range(cute.size(tBR, mode=[2]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                cute.gemm(tiled_mma, acc, tAR[None, None, kp], tBR[None, None, kp], acc)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            if cutlass.const_expr(self.kernel_name.endswith("_mma_bar")):
                cute.arch.barrier()
            for ei in cutlass.range(cute.size(src_bf16), unroll_full=True):
                if cutlass.const_expr(self.kernel_name.endswith("_mma_add0")):
                    src_bf16[ei] = (acc[ei] + Float32(0.0)).to(self.io_dtype)
                else:
                    src_bf16[ei] = acc[ei].to(self.io_dtype)
        else:
            for ei in cutlass.range(cute.size(src_bf16), unroll_full=True):
                m, n = tCc[ei]
                src_bf16[ei] = src[m, n]

        if cutlass.const_expr(self.kernel_name == "mma_scalar_out"):
            for ei in cutlass.range(cute.size(acc), unroll_full=True):
                m_rel, n_rel = tCc[ei]
                gOut[m_rel, n_rel] = acc[ei].to(self.io_dtype)
        elif cutlass.const_expr(self.kernel_name == "tritonlike_raw_mma"):
            self.store_acc_tritonlike_raw(tidx, acc, sRaw, gOut)
        elif cutlass.const_expr(self.kernel_name in ("direct_tv", "direct_tv_mma")):
            reg_bridge = cute.make_tensor(
                src_bf16.iterator,
                cute.make_layout(((8, 1), 4, 1), stride=((1, 0), 8, 0)),
            )
            cute.copy(store_tiled_copy, reg_bridge, store_thr.partition_D(gOut))
        elif cutlass.const_expr(self.kernel_name in ("direct_copy_c", "direct_copy_c_mma")):
            reg_store = c_store_thr.retile(src_bf16)
            cute.copy(c_store_tiled_copy, reg_store, c_store_thr.partition_D(gOut))
        elif cutlass.const_expr(self.kernel_name in ("direct_copy_c_atom", "direct_copy_c_atom_mma")):
            reg_store = c_atom_store_thr.retile(src_bf16)
            gdst = c_atom_store_thr.partition_D(gOut)
            print(f"direct_copy_c_atom src_bf16_layout={cute.pretty_str(src_bf16.layout)}")
            print(f"direct_copy_c_atom reg_store_layout={cute.pretty_str(reg_store.layout)}")
            print(f"direct_copy_c_atom gdst_layout={cute.pretty_str(gdst.layout)}")
            cute.copy(c_atom_store_tiled_copy, reg_store, gdst)
        elif cutlass.const_expr(self.kernel_name in ("direct_copy_c_atom_bridge", "direct_copy_c_atom_bridge_mma")):
            gdst = c_atom_store_thr.partition_D(gOut)
            reg_bridge = cute.make_tensor(
                src_bf16.iterator,
                cute.make_layout(
                    (((2, 2, 2), 1), 1, 4),
                    stride=(((1, 2, 4), 0), 0, 8),
                ),
            )
            cute.copy(c_atom_store_tiled_copy, reg_bridge, gdst)
        elif cutlass.const_expr(self.kernel_name in ("direct_copy_c_bridge", "direct_copy_c_bridge_mma")):
            reg_bridge = cute.make_tensor(
                src_bf16.iterator,
                cute.make_layout((((2, 2, 2), 4), 1, 1), stride=(((1, 2, 4), 8), 0, 0)),
            )
            cute.copy(c_store_tiled_copy, reg_bridge, c_store_thr.partition_D(gOut))
        elif cutlass.const_expr(self.kernel_name == "map_offset"):
            reg = cute.make_rmem_tensor_like(src_bf16, self.io_dtype)
            for ei in cutlass.range(cute.size(reg), unroll_full=True):
                reg[ei] = Float32(ei).to(self.io_dtype)
            reg_bridge = cute.make_tensor(reg.iterator, cute.make_layout(((8, 1), 4, 1), stride=((1, 0), 8, 0)))
            cute.copy(store_tiled_copy, reg_bridge, store_thr.partition_D(gOut))
        elif cutlass.const_expr(self.kernel_name == "map_tid"):
            reg = cute.make_rmem_tensor_like(src_bf16, self.io_dtype)
            for ei in cutlass.range(cute.size(reg), unroll_full=True):
                reg[ei] = Float32(tidx).to(self.io_dtype)
            reg_bridge = cute.make_tensor(reg.iterator, cute.make_layout(((8, 1), 4, 1), stride=((1, 0), 8, 0)))
            cute.copy(store_tiled_copy, reg_bridge, store_thr.partition_D(gOut))
        else:
            stsm_thr = stsm_tiled_copy.get_slice(tidx)
            if cutlass.const_expr(self.kernel_name == "shared"):
                self.store_mma_c_fragment_blocked_shared(
                    stsm_tiled_copy,
                    store_tiled_copy,
                    stsm_thr,
                    store_thr,
                    src_bf16,
                    sTile,
                    gOut,
                )
            elif cutlass.const_expr(self.kernel_name == "shared_acc_mma"):
                cute.copy(
                    stsm_tiled_copy,
                    stsm_thr.retile(acc),
                    stsm_thr.partition_D(sTile[(None, None, 0)]),
                )
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier()
                tSs = store_thr.partition_S(sTile[(None, None, 0)])
                tSr = cute.make_fragment_like(tSs, self.io_dtype)
                cute.autovec_copy(tSs, tSr)
                cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))
            elif cutlass.const_expr(self.kernel_name == "shared_cutlass_style"):
                stsm_s_thr = stsm_s_tiled_copy.get_slice(tidx)
                cute.copy(
                    stsm_s_tiled_copy,
                    stsm_s_thr.retile(src_bf16),
                    stsm_s_thr.partition_D(sTile[(None, None, 0)]),
                )
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier()
                tSs = store_thr.partition_S(sTile[(None, None, 0)])
                tSr = cute.make_fragment_like(tSs, self.io_dtype)
                cute.autovec_copy(tSs, tSr)
                cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))
            elif cutlass.const_expr(self.kernel_name == "shared_cutlass_buffered"):
                stsm_s_thr = stsm_s_tiled_copy.get_slice(tidx)
                tRS_sD = stsm_s_thr.partition_D(sTile[(None, None, 0)])
                tRS_rAcc = stsm_s_tiled_copy.retile(src_bf16)
                rD_shape = cute.shape(stsm_s_thr.partition_S(sTile[(None, None, 0)]))
                rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor_like(rD_layout, self.io_dtype)
                for epi_v in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                    tRS_rD[epi_v] = tRS_rAcc[epi_v]
                cute.copy(stsm_s_tiled_copy, tRS_rD, tRS_sD)
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier()
                tSs = store_thr.partition_S(sTile[(None, None, 0)])
                tSr = cute.make_fragment_like(tSs, self.io_dtype)
                cute.autovec_copy(tSs, tSr)
                cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))
            elif cutlass.const_expr(self.kernel_name == "shared_cutlass_buffered_acc_mma"):
                stsm_s_thr = stsm_s_tiled_copy.get_slice(tidx)
                tRS_sD = stsm_s_thr.partition_D(sTile[(None, None, 0)])
                tRS_rAcc = stsm_s_tiled_copy.retile(acc)
                rD_shape = cute.shape(stsm_s_thr.partition_S(sTile[(None, None, 0)]))
                rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD_acc = cute.make_rmem_tensor_like(rD_layout, self.acc_dtype)
                for epi_v in cutlass.range(cute.size(tRS_rD_acc), unroll_full=True):
                    tRS_rD_acc[epi_v] = tRS_rAcc[epi_v]
                tRS_rD_out = cute.make_rmem_tensor_like(rD_layout, self.io_dtype)
                tRS_rD_out.store(tRS_rD_acc.load().to(self.io_dtype))
                cute.copy(stsm_s_tiled_copy, tRS_rD_out, tRS_sD)
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier()
                tSs = store_thr.partition_S(sTile[(None, None, 0)])
                tSr = cute.make_fragment_like(tSs, self.io_dtype)
                cute.autovec_copy(tSs, tSr)
                cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))
            else:
                cute.copy(stsm_tiled_copy, stsm_thr.retile(src_bf16), stsm_thr.partition_D(sTile[(None, None, 0)]))
                if cutlass.const_expr(
                    not (self.kernel_name.startswith("shared_no_fence") or self.kernel_name.startswith("shared_no_sync"))
                ):
                    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                if cutlass.const_expr(
                    not (self.kernel_name.startswith("shared_no_barrier") or self.kernel_name.startswith("shared_no_sync"))
                ):
                    cute.arch.barrier()
                if cutlass.const_expr(self.kernel_name in ("ldm_bridge", "ldm_bridge_mma")):
                    reg_ldm = cute.make_rmem_tensor_like(src_bf16, self.io_dtype)
                    reg_ldm_cv = ldm_thr.retile(reg_ldm)
                    cute.copy(ldm_tiled_copy, ldm_thr.partition_S(sTile[(None, None, 0)]), reg_ldm_cv)
                    reg_bridge = cute.make_tensor(
                        reg_ldm.iterator,
                        cute.make_layout(((8, 1), 4, 1), stride=((1, 0), 8, 0)),
                    )
                    cute.copy(store_tiled_copy, reg_bridge, store_thr.partition_D(gOut))
                else:
                    tSs = store_thr.partition_S(sTile[(None, None, 0)])
                    tSr = cute.make_fragment_like(tSs, self.io_dtype)
                    cute.autovec_copy(tSs, tSr)
                    cute.copy(store_tiled_copy, tSr, store_thr.partition_D(gOut))


@functools.lru_cache(maxsize=128)
def _compile_cute(name: str):
    kernel = CuteConvertStoreMicro(CuteVariant(name))
    sym_g = cute.sym_int()
    src_fake = make_fake_compact_tensor(cutlass.BFloat16, (M, N), stride_order=(1, 0), assumed_align=128)
    rhs_fake = make_fake_compact_tensor(cutlass.BFloat16, (K, N), stride_order=(1, 0), assumed_align=128)
    out_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_g, M, N), stride_order=(2, 1, 0), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, src_fake, rhs_fake, out_fake, stream_fake, options="--enable-tvm-ffi")


def _extract_fatbin(ir_text: str) -> bytes:
    match = re.search(r'@kernels_binary\("(.*)"\) \{addr_space', ir_text, re.S)
    if match is None:
        raise RuntimeError("could not find embedded fatbin")
    s = match.group(1)
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\":
            if i + 2 < len(s) and all(c in "0123456789abcdefABCDEF" for c in s[i + 1 : i + 3]):
                out.append(int(s[i + 1 : i + 3], 16))
                i += 3
            else:
                out.append({"n": 10, "t": 9, "\\": 92, '"': 34}.get(s[i + 1], ord(s[i + 1])))
                i += 2
        else:
            out.extend(s[i].encode())
            i += 1
    return bytes(out)


def dump_cute_resource(compiled, out_dir: pathlib.Path, name: str) -> dict[str, int | None]:
    mlir = str(compiled.ir_module)
    mlir_path = out_dir / f"cute_{name}.mlir"
    fatbin = out_dir / f"cute_{name}.fatbin"
    mlir_path.write_text(mlir)
    fatbin.write_bytes(_extract_fatbin(mlir))
    res = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(fatbin)], check=False, capture_output=True, text=True
    ).stdout
    sass = subprocess.run(["cuobjdump", "--dump-sass", str(fatbin)], check=False, capture_output=True, text=True).stdout
    (out_dir / f"cute_{name}_resource.txt").write_text(res)
    (out_dir / f"cute_{name}.sass").write_text(sass)
    return _sass_summary(sass, res)


def run_cute_variant(name: str, out_dir: pathlib.Path, blocks: int, warmup: int, repeat: int) -> dict[str, object]:
    row: dict[str, object] = {"family": "cute", "variant": name, "blocks": blocks}
    try:
        compiled = _compile_cute(name)
    except Exception as exc:
        row.update({"status": "compile_fail", "error": str(exc).splitlines()[-1]})
        print("CUTE", json.dumps(row, sort_keys=True), flush=True)
        return row

    row.update({"status": "compiled", **dump_cute_resource(compiled, out_dir, name)})
    src = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    rhs = torch.eye(K, N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((blocks, M, N), device="cuda", dtype=torch.bfloat16)
    compiled(src, rhs, out)
    torch.cuda.synchronize()
    max_diff = (out[0] - src).abs().max().item()
    ms = _bench(lambda: compiled(src, rhs, out), warmup, repeat)
    row.update({"max_diff": max_diff, "ms": ms, "ns_per_cta": ms * 1e6 / blocks})
    print("CUTE", json.dumps(row, sort_keys=True), flush=True)
    return row


def run_mapping(out_dir: pathlib.Path) -> dict[str, object]:
    compiled_off = _compile_cute("map_offset")
    compiled_tid = _compile_cute("map_tid")
    compiled_direct = _compile_cute("direct_tv")
    zero = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)
    rhs = torch.eye(K, N, device="cuda", dtype=torch.bfloat16)
    out_off = torch.empty((1, M, N), device="cuda", dtype=torch.bfloat16)
    out_tid = torch.empty_like(out_off)
    compiled_off(zero, rhs, out_off)
    compiled_tid(zero, rhs, out_tid)
    rows_src = torch.arange(M, device="cuda", dtype=torch.bfloat16)[:, None].expand(M, N).contiguous()
    cols_src = torch.arange(N, device="cuda", dtype=torch.bfloat16)[None, :].expand(M, N).contiguous()
    out_r = torch.empty_like(out_off)
    out_c = torch.empty_like(out_off)
    compiled_direct(rows_src, rhs, out_r)
    compiled_direct(cols_src, rhs, out_c)
    torch.cuda.synchronize()
    off = out_off[0].float().cpu().round().to(torch.int64)
    tid = out_tid[0].float().cpu().round().to(torch.int64)
    sr = out_r[0].float().cpu().round().to(torch.int64)
    sc = out_c[0].float().cpu().round().to(torch.int64)
    src_map: dict[tuple[int, int], tuple[int, int]] = {}
    conflicts = 0
    for m in range(M):
        for n in range(N):
            key = (int(tid[m, n]), int(off[m, n]))
            val = (int(sr[m, n]), int(sc[m, n]))
            if key in src_map and src_map[key] != val:
                conflicts += 1
            src_map[key] = val
    missing = 0
    same_thread_hits = 0
    cross_warp_needed = 0
    cross_cta_warpgroup_needed = 0
    for m in range(M):
        for n in range(N):
            t = int(tid[m, n])
            matches = [o for o in range(32) if src_map.get((t, o)) == (m, n)]
            if matches:
                same_thread_hits += 1
            else:
                missing += 1
                producer = None
                for key, coord in src_map.items():
                    if coord == (m, n):
                        producer = key[0]
                        break
                if producer is not None:
                    if producer // 32 != t // 32:
                        cross_warp_needed += 1
                    if producer // 128 != t // 128:
                        cross_cta_warpgroup_needed += 1
    row = {
        "same_thread_hits": same_thread_hits,
        "missing_same_thread": missing,
        "src_map_size": len(src_map),
        "conflicts": conflicts,
        "cross_warp_needed": cross_warp_needed,
        "cross_cta_warpgroup_needed": cross_cta_warpgroup_needed,
    }
    (out_dir / "cute_direct_tv_mapping.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    print("MAPPING", json.dumps(row, sort_keys=True), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("benchmarks/profiles/convert_layout_microkernels")
    )
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--cutlass-cpp-variants",
        nargs="+",
        default=["cutlass_cpp_epilogue", "cutlass_cpp_wgmma", "cutlass_cpp_wgmma_tiled", "cutlass_cpp_tritonlike"],
    )
    parser.add_argument(
        "--cute-variants",
        nargs="+",
        default=[
            "direct_copy_c",
            "direct_copy_c_bridge",
            "direct_tv",
            "shared",
            "shared_no_fence",
            "shared_no_barrier",
            "shared_no_sync",
            "ldm_bridge",
        ],
    )
    args = parser.parse_args()
    assert_hopper()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rows.append(dump_triton(args.out_dir, args.blocks, args.warmup, args.repeat))
    for name in args.cutlass_cpp_variants:
        rows.append(run_cutlass_cpp_variant(name, args.out_dir, args.blocks, args.warmup, args.repeat))
    for name in args.cute_variants:
        rows.append(run_cute_variant(name, args.out_dir, args.blocks, args.warmup, args.repeat))
    mapping = run_mapping(args.out_dir)
    summary = {"rows": rows, "mapping": mapping}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
