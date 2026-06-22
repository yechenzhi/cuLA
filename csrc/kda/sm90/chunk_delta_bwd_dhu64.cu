// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <torch/python.h>

namespace cula::kda::sm90::bwd_dhu64 {

using namespace cute;

constexpr int kBT = 64;
constexpr int kBV = 32;
constexpr int kK = 64;
constexpr int kV = 64;
constexpr int kThreads = 128;
constexpr int kStages = 3;

using Element = cutlass::bfloat16_t;
using Acc = float;

template <int KDim>
constexpr int kKBlocks = KDim / kK;

template <int KDim>
constexpr int kDvPrefetchStages = KDim > kK ? 2 : 0;

template <int KDim>
constexpr int kPrefetchDistance = KDim > kK ? 2 : kStages;

template <int KDim>
constexpr int kSmemElements = kStages * (kBT * kBV + kKBlocks<KDim> * 3 * kBT * kK) +
                              kDvPrefetchStages<KDim> * kBT * kBV + (kKBlocks<KDim> + 3) * kBT * kBV;

template <int KDim>
constexpr int kSmemBytes = kSmemElements<KDim> * int(sizeof(Element));

template <typename TiledMma, typename TiledCopy, typename ThrCopy, typename AccFrag, typename STile>
CUTE_DEVICE void
r2s_acc(
    TiledMma const& tiled_mma, TiledCopy const& r2s, ThrCopy const& r2s_thr, AccFrag const& acc, STile const& sTile) {
    (void)tiled_mma;
    Tensor bf16 = make_fragment_like<Element>(acc);
    CUTE_UNROLL
    for (int i = 0; i < size(bf16); ++i) {
        bf16(i) = Element(acc(i));
    }
    copy(r2s, r2s_thr.retile_S(bf16), r2s_thr.partition_D(sTile(_, _, _0{})));
}

__device__ __forceinline__ uint32_t
pack_bf16x2(float lo, float hi) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ void
store_bridge_64x32_bf16(
    int tidx,
    uint32_t smem_base,
    uint64_t gmem_base,
    uint32_t row_stride_bytes,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t p4,
    uint32_t p5,
    uint32_t p6,
    uint32_t p7) {
    asm volatile(
        "{\n"
        ".reg .u32 r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;\n"
        ".reg .u32 r16, r17;\n"
        ".reg .u32 s0, l0, l1;\n"
        ".reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;\n"
        ".reg .u32 row, col;\n"
        ".reg .u64 g0, g1, row_bytes, col_bytes, step_bytes;\n"
        "mov.u32 r5, %0;\n"
        "and.b32 r6, r5, 24;\n"
        "shl.b32 r6, r6, 7;\n"
        "and.b32 r7, r5, 3;\n"
        "shl.b32 r7, r7, 5;\n"
        "or.b32 r8, r6, r7;\n"
        "and.b32 r9, r5, 96;\n"
        "shl.b32 r9, r9, 3;\n"
        "shl.b32 r10, r5, 2;\n"
        "and.b32 r10, r10, 112;\n"
        "or.b32 r11, r9, r10;\n"
        "xor.b32 r12, r8, r11;\n"
        "add.u32 s0, %1, r12;\n"
        "shr.u32 row, r5, 2;\n"
        "and.b32 row, row, 31;\n"
        "and.b32 col, r5, 3;\n"
        "shl.b32 col, col, 3;\n"
        "mul.wide.u32 row_bytes, row, %3;\n"
        "mul.wide.u32 col_bytes, col, 2;\n"
        "add.u64 g0, %2, row_bytes;\n"
        "add.u64 g0, g0, col_bytes;\n"
        "mul.wide.u32 step_bytes, %3, 32;\n"
        "add.u64 g1, g0, step_bytes;\n"
        "st.shared.v4.b32 [s0], {%4, %6, %8, %10};\n"
        "st.shared.v4.b32 [s0+128], {%5, %7, %9, %11};\n"
        "fence.proxy.async.shared::cta;\n"
        "bar.sync 0;\n"
        "and.b32 r13, r5, 6;\n"
        "shl.b32 r13, r13, 9;\n"
        "and.b32 r14, r5, 7;\n"
        "shl.b32 r14, r14, 4;\n"
        "or.b32 r15, r13, r14;\n"
        "and.b32 r16, r5, 120;\n"
        "shl.b32 r16, r16, 2;\n"
        "xor.b32 r17, r15, r16;\n"
        "add.u32 l0, %1, r17;\n"
        "add.u32 l1, l0, 512;\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];\n"
        "st.global.v4.b32 [g0], {o0, o1, o2, o3};\n"
        "st.global.v4.b32 [g1], {o4, o5, o6, o7};\n"
        "}\n"
        :
        : "r"(tidx),
          "r"(smem_base),
          "l"(gmem_base),
          "r"(row_stride_bytes),
          "r"(p0),
          "r"(p1),
          "r"(p2),
          "r"(p3),
          "r"(p4),
          "r"(p5),
          "r"(p6),
          "r"(p7)
        : "memory");
}

template <typename AccFrag>
CUTE_DEVICE void
store_acc_64x32_bf16(int tid, Element* smem_store, Element* gmem, uint32_t row_stride_bytes, AccFrag const& acc) {
    uint32_t p0 = pack_bf16x2(acc(0), acc(1));
    uint32_t p1 = pack_bf16x2(acc(2), acc(3));
    uint32_t p2 = pack_bf16x2(acc(4), acc(5));
    uint32_t p3 = pack_bf16x2(acc(6), acc(7));
    uint32_t p4 = pack_bf16x2(acc(8), acc(9));
    uint32_t p5 = pack_bf16x2(acc(10), acc(11));
    uint32_t p6 = pack_bf16x2(acc(12), acc(13));
    uint32_t p7 = pack_bf16x2(acc(14), acc(15));
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_store));
    store_bridge_64x32_bf16(
        tid, smem_addr, reinterpret_cast<uint64_t>(gmem), row_stride_bytes, p0, p1, p2, p3, p4, p5, p6, p7);
}

template <typename STile>
CUTE_DEVICE void
cp_async_do_tile(int tid, Element const* __restrict__ do_base, STile const& sDo, int H) {
    CUTE_UNROLL
    for (int iter = 0; iter < 2; ++iter) {
        int vec_idx = tid + iter * kThreads;
        int t_rel = vec_idx >> 2;
        int v_rel = (vec_idx & 3) * 8;
        auto const* src = reinterpret_cast<uint128_t const*>(do_base + int64_t(t_rel) * H * kV + v_rel);
        auto* dst = reinterpret_cast<uint128_t*>(&sDo(v_rel, t_rel, _0{}));
        SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(*src, *dst);
    }
}

CUTE_DEVICE void
cp_async_dv_bridge_tile(int tid, Element const* __restrict__ dv_base, Element* smem_stage, int H) {
    int row = (tid >> 2) & 31;
    int col = (tid & 3) * 8;
    int store_offset = (((tid & 24) << 7) | ((tid & 3) << 5)) ^ ((tid & 124) << 2);
    auto const* src0 = reinterpret_cast<uint128_t const*>(dv_base + int64_t(row) * H * kV + col);
    auto const* src1 = reinterpret_cast<uint128_t const*>(dv_base + int64_t(row + 32) * H * kV + col);
    char* smem_bytes = reinterpret_cast<char*>(smem_stage);
    auto* dst0 = reinterpret_cast<uint128_t*>(smem_bytes + store_offset);
    auto* dst1 = reinterpret_cast<uint128_t*>(smem_bytes + store_offset + 512);
    SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(*src0, *dst0);
    SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>::copy(*src1, *dst1);
}

template <typename AccFrag>
CUTE_DEVICE void
load_acc_64x32_bf16(int tid, Element* smem_load, Element const* gmem, uint32_t row_stride_bytes, AccFrag& acc) {
    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_load));
    asm volatile(
        "{\n"
        ".reg .u32 t, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;\n"
        ".reg .u32 s0, l0, l1;\n"
        ".reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;\n"
        ".reg .u32 a0, a1, a2, a3, b0, b1, b2, b3;\n"
        ".reg .u32 row, col;\n"
        ".reg .u64 g0, g1, row_bytes, col_bytes, step_bytes;\n"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
        ".reg .b16 h8, h9, h10, h11, h12, h13, h14, h15;\n"
        "mov.u32 t, %16;\n"
        "shr.u32 row, t, 2;\n"
        "and.b32 row, row, 31;\n"
        "and.b32 col, t, 3;\n"
        "shl.b32 col, col, 3;\n"
        "mul.wide.u32 row_bytes, row, %19;\n"
        "mul.wide.u32 col_bytes, col, 2;\n"
        "add.u64 g0, %18, row_bytes;\n"
        "add.u64 g0, g0, col_bytes;\n"
        "mul.wide.u32 step_bytes, %19, 32;\n"
        "add.u64 g1, g0, step_bytes;\n"
        "ld.global.v4.b32 {a0, a1, a2, a3}, [g0];\n"
        "ld.global.v4.b32 {b0, b1, b2, b3}, [g1];\n"
        "and.b32 r0, t, 24;\n"
        "shl.b32 r0, r0, 7;\n"
        "and.b32 r1, t, 3;\n"
        "shl.b32 r1, r1, 5;\n"
        "or.b32 r2, r0, r1;\n"
        "and.b32 r3, t, 124;\n"
        "shl.b32 r3, r3, 2;\n"
        "xor.b32 r4, r2, r3;\n"
        "add.u32 s0, %17, r4;\n"
        "st.shared.v4.b32 [s0], {a0, a1, a2, a3};\n"
        "st.shared.v4.b32 [s0+512], {b0, b1, b2, b3};\n"
        "bar.sync 0;\n"
        "and.b32 r0, t, 6;\n"
        "shl.b32 r0, r0, 9;\n"
        "and.b32 r1, t, 15;\n"
        "shl.b32 r1, r1, 4;\n"
        "and.b32 r2, t, 96;\n"
        "shl.b32 r2, r2, 3;\n"
        "and.b32 r3, t, 16;\n"
        "shl.b32 r3, r3, 1;\n"
        "or.b32 r4, r1, r2;\n"
        "xor.b32 r5, r4, r3;\n"
        "or.b32 r6, r5, r0;\n"
        "add.u32 l0, %17, r6;\n"
        "xor.b32 r7, r6, 64;\n"
        "add.u32 l1, %17, r7;\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];\n"
        "mov.b32 {h0, h1}, o0;\n"
        "mov.b32 {h2, h3}, o1;\n"
        "mov.b32 {h4, h5}, o2;\n"
        "mov.b32 {h6, h7}, o3;\n"
        "mov.b32 {h8, h9}, o4;\n"
        "mov.b32 {h10, h11}, o5;\n"
        "mov.b32 {h12, h13}, o6;\n"
        "mov.b32 {h14, h15}, o7;\n"
        "cvt.f32.bf16 %0, h0;\n"
        "cvt.f32.bf16 %1, h1;\n"
        "cvt.f32.bf16 %2, h2;\n"
        "cvt.f32.bf16 %3, h3;\n"
        "cvt.f32.bf16 %4, h4;\n"
        "cvt.f32.bf16 %5, h5;\n"
        "cvt.f32.bf16 %6, h6;\n"
        "cvt.f32.bf16 %7, h7;\n"
        "cvt.f32.bf16 %8, h8;\n"
        "cvt.f32.bf16 %9, h9;\n"
        "cvt.f32.bf16 %10, h10;\n"
        "cvt.f32.bf16 %11, h11;\n"
        "cvt.f32.bf16 %12, h12;\n"
        "cvt.f32.bf16 %13, h13;\n"
        "cvt.f32.bf16 %14, h14;\n"
        "cvt.f32.bf16 %15, h15;\n"
        "}\n"
        : "=f"(f0),
          "=f"(f1),
          "=f"(f2),
          "=f"(f3),
          "=f"(f4),
          "=f"(f5),
          "=f"(f6),
          "=f"(f7),
          "=f"(f8),
          "=f"(f9),
          "=f"(f10),
          "=f"(f11),
          "=f"(f12),
          "=f"(f13),
          "=f"(f14),
          "=f"(f15)
        : "r"(tid), "r"(smem_addr), "l"(reinterpret_cast<uint64_t>(gmem)), "r"(row_stride_bytes)
        : "memory");
    acc(0) = f0;
    acc(1) = f1;
    acc(2) = f2;
    acc(3) = f3;
    acc(4) = f4;
    acc(5) = f5;
    acc(6) = f6;
    acc(7) = f7;
    acc(8) = f8;
    acc(9) = f9;
    acc(10) = f10;
    acc(11) = f11;
    acc(12) = f12;
    acc(13) = f13;
    acc(14) = f14;
    acc(15) = f15;
}

template <typename AccFrag>
CUTE_DEVICE void
load_acc_64x32_bf16_from_smem(int tid, Element* smem_load, AccFrag& acc) {
    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_load));
    asm volatile(
        "{\n"
        ".reg .u32 t, r0, r1, r2, r3, r4, r5, r6, r7;\n"
        ".reg .u32 l0, l1;\n"
        ".reg .u32 o0, o1, o2, o3, o4, o5, o6, o7;\n"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n"
        ".reg .b16 h8, h9, h10, h11, h12, h13, h14, h15;\n"
        "mov.u32 t, %16;\n"
        "and.b32 r0, t, 6;\n"
        "shl.b32 r0, r0, 9;\n"
        "and.b32 r1, t, 15;\n"
        "shl.b32 r1, r1, 4;\n"
        "and.b32 r2, t, 96;\n"
        "shl.b32 r2, r2, 3;\n"
        "and.b32 r3, t, 16;\n"
        "shl.b32 r3, r3, 1;\n"
        "or.b32 r4, r1, r2;\n"
        "xor.b32 r5, r4, r3;\n"
        "or.b32 r6, r5, r0;\n"
        "add.u32 l0, %17, r6;\n"
        "xor.b32 r7, r6, 64;\n"
        "add.u32 l1, %17, r7;\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o0, o1, o2, o3}, [l0];\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {o4, o5, o6, o7}, [l1];\n"
        "mov.b32 {h0, h1}, o0;\n"
        "mov.b32 {h2, h3}, o1;\n"
        "mov.b32 {h4, h5}, o2;\n"
        "mov.b32 {h6, h7}, o3;\n"
        "mov.b32 {h8, h9}, o4;\n"
        "mov.b32 {h10, h11}, o5;\n"
        "mov.b32 {h12, h13}, o6;\n"
        "mov.b32 {h14, h15}, o7;\n"
        "cvt.f32.bf16 %0, h0;\n"
        "cvt.f32.bf16 %1, h1;\n"
        "cvt.f32.bf16 %2, h2;\n"
        "cvt.f32.bf16 %3, h3;\n"
        "cvt.f32.bf16 %4, h4;\n"
        "cvt.f32.bf16 %5, h5;\n"
        "cvt.f32.bf16 %6, h6;\n"
        "cvt.f32.bf16 %7, h7;\n"
        "cvt.f32.bf16 %8, h8;\n"
        "cvt.f32.bf16 %9, h9;\n"
        "cvt.f32.bf16 %10, h10;\n"
        "cvt.f32.bf16 %11, h11;\n"
        "cvt.f32.bf16 %12, h12;\n"
        "cvt.f32.bf16 %13, h13;\n"
        "cvt.f32.bf16 %14, h14;\n"
        "cvt.f32.bf16 %15, h15;\n"
        "}\n"
        : "=f"(f0),
          "=f"(f1),
          "=f"(f2),
          "=f"(f3),
          "=f"(f4),
          "=f"(f5),
          "=f"(f6),
          "=f"(f7),
          "=f"(f8),
          "=f"(f9),
          "=f"(f10),
          "=f"(f11),
          "=f"(f12),
          "=f"(f13),
          "=f"(f14),
          "=f"(f15)
        : "r"(tid), "r"(smem_addr)
        : "memory");
    acc(0) = f0;
    acc(1) = f1;
    acc(2) = f2;
    acc(3) = f3;
    acc(4) = f4;
    acc(5) = f5;
    acc(6) = f6;
    acc(7) = f7;
    acc(8) = f8;
    acc(9) = f9;
    acc(10) = f10;
    acc(11) = f11;
    acc(12) = f12;
    acc(13) = f13;
    acc(14) = f14;
    acc(15) = f15;
}

template <int KDim>
__global__
__launch_bounds__(kThreads, 1) void chunk_delta_bwd_dhu64_sm90_kernel(
    Element const* __restrict__ q,
    Element const* __restrict__ k,
    Element const* __restrict__ w,
    Element const* __restrict__ do_,
    Element const* __restrict__ dv,
    float const* __restrict__ dht,
    Element* __restrict__ dh,
    float* __restrict__ dh0,
    Element* __restrict__ dv2,
    int B,
    int T,
    int H,
    int NT,
    float scale,
    bool has_dht,
    bool store_dh0) {
    (void)B;
    int tid = int(threadIdx.x);
    int v_tile = int(blockIdx.x);
    int bh = int(blockIdx.y);
    int b = bh / H;
    int h = bh - b * H;
    int v_base = v_tile * kBV;

    auto shape_mnk = make_shape(Int<64>{}, Int<32>{}, Int<64>{});
    auto cta_tiler = make_shape(Int<64>{}, Int<32>{}, Int<64>{});

    using SmemLayoutK64 =
        decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, make_shape(Int<64>{}, Int<64>{}, Int<1>{})));
    using SmemLayoutMN64 =
        decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, make_shape(Int<64>{}, Int<64>{}, Int<1>{})));
    using SmemLayoutK32 =
        decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, make_shape(Int<64>{}, Int<32>{}, Int<1>{})));
    using SmemLayoutMN32Read =
        decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{}, make_shape(Int<32>{}, Int<64>{}, Int<1>{})));

    constexpr int kSmemDoOffset = 0;
    constexpr int kSmemKOffset = kSmemDoOffset + kStages * cosize_v<SmemLayoutMN32Read>;
    constexpr int kSmemWOffset = kSmemKOffset + kKBlocks<KDim> * kStages * cosize_v<SmemLayoutK64>;
    constexpr int kSmemQOffset = kSmemWOffset + kKBlocks<KDim> * kStages * cosize_v<SmemLayoutMN64>;
    constexpr int kSmemDvPrefOffset = kSmemQOffset + kKBlocks<KDim> * kStages * cosize_v<SmemLayoutMN64>;
    constexpr int kSmemDhOffset = kSmemDvPrefOffset + kDvPrefetchStages<KDim> * cosize_v<SmemLayoutMN32Read>;
    constexpr int kSmemDv2Offset = kSmemDhOffset + kKBlocks<KDim> * cosize_v<SmemLayoutK32>;
    constexpr int kSmemDhBridgeOffset = kSmemDv2Offset + cosize_v<SmemLayoutK32>;
    constexpr int kSmemDv2BridgeOffset = kSmemDhBridgeOffset + cosize_v<SmemLayoutK32>;
    constexpr int kSmemTotal = kSmemDv2BridgeOffset + cosize_v<SmemLayoutK32>;
    static_assert(kSmemTotal == kSmemElements<KDim>);

    extern __shared__ __align__(128) unsigned char smem_bytes[];
    Element* smem_raw = reinterpret_cast<Element*>(smem_bytes);

    Element* smem_do = smem_raw + kSmemDoOffset;
    Element* smem_k = smem_raw + kSmemKOffset;
    Element* smem_w = smem_raw + kSmemWOffset;
    Element* smem_q = smem_raw + kSmemQOffset;
    Element* smem_dv_pref = smem_raw + kSmemDvPrefOffset;
    Element* smem_dh = smem_raw + kSmemDhOffset;
    Element* smem_dv2 = smem_raw + kSmemDv2Offset;
    Element* smem_dh_bridge = smem_raw + kSmemDhBridgeOffset;
    Element* smem_dv2_bridge = smem_raw + kSmemDv2BridgeOffset;

    Tensor sDhStore = make_tensor(make_smem_ptr(smem_dh), SmemLayoutK32{});
    Tensor sDhRead = make_tensor(make_smem_ptr(smem_dh), SmemLayoutMN32Read{});
    Tensor sDhStore1 = make_tensor(make_smem_ptr(smem_dh + cosize_v<SmemLayoutK32>), SmemLayoutK32{});
    Tensor sDhRead1 = make_tensor(make_smem_ptr(smem_dh + cosize_v<SmemLayoutK32>), SmemLayoutMN32Read{});
    Tensor sDv2Store = make_tensor(make_smem_ptr(smem_dv2), SmemLayoutK32{});
    Tensor sDv2Read = make_tensor(make_smem_ptr(smem_dv2), SmemLayoutMN32Read{});

    auto copy_k = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});
    auto copy_mn = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{}, Layout<Shape<_8, _16>>{}, Layout<Shape<_8, _1>>{});
    auto copy_k_thr = copy_k.get_slice(tid);
    auto copy_mn_thr = copy_mn.get_slice(tid);

    auto kdh_mma = make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{});
    auto update_mma = make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>{});
    auto kdh_thr = kdh_mma.get_slice(tid);
    auto update_thr = update_mma.get_slice(tid);

    auto r2s_kdh = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, kdh_mma);
    auto r2s_update = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, update_mma);
    auto r2s_kdh_thr = r2s_kdh.get_slice(tid);
    auto r2s_update_thr = r2s_update.get_slice(tid);

    auto c_kv = make_identity_tensor(make_shape(Int<64>{}, Int<32>{}));
    auto tCUpdate = update_thr.partition_C(c_kv);
    Tensor state = update_thr.make_fragment_C(tCUpdate);
    Tensor state1 = update_thr.make_fragment_C(tCUpdate);
    if (has_dht) {
        CUTE_UNROLL
        for (int i = 0; i < size(state); ++i) {
            auto coord = tCUpdate(i);
            int k_rel = int(get<0>(coord));
            int v_rel = int(get<1>(coord));
            int64_t off = ((int64_t(b) * H + h) * KDim + k_rel) * kV + v_base + v_rel;
            state(i) = dht[off];
            if constexpr (KDim > kK) {
                state1(i) = dht[off + int64_t(kK) * kV];
            }
        }
    } else {
        clear(state);
        if constexpr (KDim > kK) {
            clear(state1);
        }
    }

    auto prefetch_chunk = [&](int chunk, int stage) {
        int t0 = chunk * kBT;
        int64_t base_t = (int64_t(b) * T + t0) * H * KDim + int64_t(h) * KDim;
        int64_t base_v = (int64_t(b) * T + t0) * H * kV + int64_t(h) * kV + v_base;

        Tensor sDoStage =
            make_tensor(make_smem_ptr(smem_do + stage * cosize_v<SmemLayoutMN32Read>), SmemLayoutMN32Read{});
        Tensor sKStage =
            make_tensor(make_smem_ptr(smem_k + stage * kKBlocks<KDim> * cosize_v<SmemLayoutK64>), SmemLayoutK64{});
        Tensor sWStage =
            make_tensor(make_smem_ptr(smem_w + stage * kKBlocks<KDim> * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
        Tensor sQStage =
            make_tensor(make_smem_ptr(smem_q + stage * kKBlocks<KDim> * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});

        Tensor mK = make_tensor(make_gmem_ptr(k + base_t), select<0, 2>(shape_mnk), make_stride(H * KDim, Int<1>{}));
        Tensor gK = local_tile(mK, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(
            copy_k,
            copy_k_thr.partition_S(gK),
            copy_k_thr.partition_D(as_position_independent_swizzle_tensor(sKStage)));

        if constexpr (KDim > kK) {
            Tensor sKStage1 = make_tensor(
                make_smem_ptr(smem_k + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutK64>), SmemLayoutK64{});
            Tensor mK1 =
                make_tensor(make_gmem_ptr(k + base_t + kK), select<0, 2>(shape_mnk), make_stride(H * KDim, Int<1>{}));
            Tensor gK1 = local_tile(mK1, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
            copy(
                copy_k,
                copy_k_thr.partition_S(gK1),
                copy_k_thr.partition_D(as_position_independent_swizzle_tensor(sKStage1)));
        }

        Tensor mW = make_tensor(make_gmem_ptr(w + base_t), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * KDim));
        Tensor gW = local_tile(mW, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(
            copy_mn,
            copy_mn_thr.partition_S(gW),
            copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sWStage)));

        Tensor mQ = make_tensor(make_gmem_ptr(q + base_t), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * KDim));
        Tensor gQ = local_tile(mQ, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(
            copy_mn,
            copy_mn_thr.partition_S(gQ),
            copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sQStage)));

        if constexpr (KDim > kK) {
            cp_async_fence();

            Tensor sWStage1 = make_tensor(
                make_smem_ptr(smem_w + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
            Tensor mW1 =
                make_tensor(make_gmem_ptr(w + base_t + kK), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * KDim));
            Tensor gW1 = local_tile(mW1, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
            copy(
                copy_mn,
                copy_mn_thr.partition_S(gW1),
                copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sWStage1)));

            Tensor sQStage1 = make_tensor(
                make_smem_ptr(smem_q + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
            Tensor mQ1 =
                make_tensor(make_gmem_ptr(q + base_t + kK), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * KDim));
            Tensor gQ1 = local_tile(mQ1, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
            copy(
                copy_mn,
                copy_mn_thr.partition_S(gQ1),
                copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sQStage1)));
        }

        cp_async_do_tile(tid, do_ + base_v, sDoStage, H);
        if constexpr (KDim > kK) {
            int dv_stage = (NT - 1 - chunk) & 1;
            cp_async_dv_bridge_tile(tid, dv + base_v, smem_dv_pref + dv_stage * cosize_v<SmemLayoutMN32Read>, H);
        }
        cp_async_fence();
    };

    int pending_groups = 0;
    if constexpr (KDim > kK) {
        int initial_prefetch = NT < kPrefetchDistance<KDim> ? NT : kPrefetchDistance<KDim>;
        for (int pref = 0; pref < initial_prefetch; ++pref) {
            prefetch_chunk(NT - 1 - pref, pref % kStages);
            pending_groups += kKBlocks<KDim>;
        }
    } else {
        int initial_prefetch = NT < kStages ? NT : kStages;
        for (int pref = 0; pref < initial_prefetch; ++pref) {
            prefetch_chunk(NT - 1 - pref, pref);
            ++pending_groups;
        }
    }

    for (int iter = 0; iter < NT; ++iter) {
        int chunk = NT - 1 - iter;
        int stage = iter % kStages;
        int t0 = chunk * kBT;
        int64_t base_v = (int64_t(b) * T + t0) * H * kV + int64_t(h) * kV + v_base;

        Tensor sDoStage =
            make_tensor(make_smem_ptr(smem_do + stage * cosize_v<SmemLayoutMN32Read>), SmemLayoutMN32Read{});
        Tensor sKStage =
            make_tensor(make_smem_ptr(smem_k + stage * kKBlocks<KDim> * cosize_v<SmemLayoutK64>), SmemLayoutK64{});
        Tensor sWStage =
            make_tensor(make_smem_ptr(smem_w + stage * kKBlocks<KDim> * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
        Tensor sQStage =
            make_tensor(make_smem_ptr(smem_q + stage * kKBlocks<KDim> * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
        Tensor sKStage1 = make_tensor(
            make_smem_ptr(smem_k + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutK64>), SmemLayoutK64{});
        Tensor sWStage1 = make_tensor(
            make_smem_ptr(smem_w + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});
        Tensor sQStage1 = make_tensor(
            make_smem_ptr(smem_q + (stage * kKBlocks<KDim> + 1) * cosize_v<SmemLayoutMN64>), SmemLayoutMN64{});

        int64_t dh_base = (((int64_t(b) * NT + chunk) * H + h) * KDim) * kV + v_base;
        // FLA/Triton stores dh before applying this chunk's update.  The
        // global-store layout is separate from the WGMMA-B shared tile.
        r2s_acc(update_mma, r2s_update, r2s_update_thr, state, sDhStore);
        cutlass::arch::fence_view_async_shared();
        store_acc_64x32_bf16(tid, smem_dh_bridge, dh + dh_base, uint32_t(kV * sizeof(Element)), state);
        if constexpr (KDim > kK) {
            r2s_acc(update_mma, r2s_update, r2s_update_thr, state1, sDhStore1);
            cutlass::arch::fence_view_async_shared();
            store_acc_64x32_bf16(
                tid, smem_dh_bridge, dh + dh_base + int64_t(kK) * kV, uint32_t(kV * sizeof(Element)), state1);
        }

        if constexpr (KDim > kK) {
            if (pending_groups >= 6) {
                cp_async_wait<4>();
            } else if (pending_groups == 4) {
                cp_async_wait<2>();
            } else {
                cp_async_wait<0>();
            }
            pending_groups -= 2;
        } else {
            if (pending_groups >= kStages) {
                cp_async_wait<kStages - 1>();
            } else if (pending_groups == 2) {
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            --pending_groups;
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        Tensor tK = kdh_thr.partition_A(sKStage);
        Tensor tKR = kdh_thr.make_fragment_A(tK);
        Tensor tDh = kdh_thr.partition_B(sDhRead);
        Tensor tDhR = kdh_thr.make_fragment_B(tDh);
        Tensor tK1 = kdh_thr.partition_A(sKStage1);
        Tensor tK1R = kdh_thr.make_fragment_A(tK1);
        Tensor tDh1 = kdh_thr.partition_B(sDhRead1);
        Tensor tDh1R = kdh_thr.make_fragment_B(tDh1);
        auto c_tv = make_identity_tensor(make_shape(Int<64>{}, Int<32>{}));
        auto tCKdh = kdh_thr.partition_C(c_tv);
        Tensor acc_dv = kdh_thr.make_fragment_C(tCKdh);
        if constexpr (KDim > kK) {
            clear(acc_dv);

            warpgroup_fence_operand(acc_dv);
            warpgroup_arrive();
            gemm(kdh_mma, tKR(_, _, _, _0{}), tDhR(_, _, _, _0{}), acc_dv);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_dv);

            warpgroup_arrive();
            gemm(kdh_mma, tK1R(_, _, _, _0{}), tDh1R(_, _, _, _0{}), acc_dv);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_dv);

            Tensor acc_dv_in = kdh_thr.make_fragment_C(tCKdh);
            int dv_stage = iter & 1;
            load_acc_64x32_bf16_from_smem(tid, smem_dv_pref + dv_stage * cosize_v<SmemLayoutMN32Read>, acc_dv_in);
            CUTE_UNROLL
            for (int i = 0; i < size(acc_dv); ++i) {
                acc_dv(i) += acc_dv_in(i);
            }
        } else {
            load_acc_64x32_bf16(tid, smem_dh_bridge, dv + base_v, uint32_t(H * kV * sizeof(Element)), acc_dv);

            warpgroup_fence_operand(acc_dv);
            warpgroup_arrive();
            gemm(kdh_mma, tKR(_, _, _, _0{}), tDhR(_, _, _, _0{}), acc_dv);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_dv);
        }
        cutlass::arch::fence_view_async_shared();

        r2s_acc(kdh_mma, r2s_kdh, r2s_kdh_thr, acc_dv, sDv2Store);
        cutlass::arch::fence_view_async_shared();
        store_acc_64x32_bf16(tid, smem_dv2_bridge, dv2 + base_v, uint32_t(H * kV * sizeof(Element)), acc_dv);

        Tensor tQ = update_thr.partition_A(sQStage);
        Tensor tQR = update_thr.make_fragment_A(tQ);
        Tensor tDo = update_thr.partition_B(sDoStage);
        Tensor tDoR = update_thr.make_fragment_B(tDo);
        Tensor tDv2 = update_thr.partition_B(sDv2Read);
        Tensor tDv2R = update_thr.make_fragment_B(tDv2);

        if constexpr (KDim > kK) {
            Tensor tQ1 = update_thr.partition_A(sQStage1);
            Tensor tQ1R = update_thr.make_fragment_A(tQ1);
            Tensor tW = update_thr.partition_A(sWStage);
            Tensor tWR = update_thr.make_fragment_A(tW);
            Tensor tW1 = update_thr.partition_A(sWStage1);
            Tensor tW1R = update_thr.make_fragment_A(tW1);
            Tensor acc_qdo = update_thr.make_fragment_C(tCUpdate);
            Tensor acc_wdv = update_thr.make_fragment_C(tCUpdate);
            Tensor acc_qdo1 = update_thr.make_fragment_C(tCUpdate);
            Tensor acc_wdv1 = update_thr.make_fragment_C(tCUpdate);
            clear(acc_qdo);
            clear(acc_wdv);
            clear(acc_qdo1);
            clear(acc_wdv1);

            warpgroup_fence_operand(acc_qdo);
            warpgroup_fence_operand(acc_qdo1);
            warpgroup_arrive();
            gemm(update_mma, tQR(_, _, _, _0{}), tDoR(_, _, _, _0{}), acc_qdo);
            gemm(update_mma, tQ1R(_, _, _, _0{}), tDoR(_, _, _, _0{}), acc_qdo1);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_qdo);
            warpgroup_fence_operand(acc_qdo1);
            cutlass::arch::fence_view_async_shared();

            warpgroup_fence_operand(acc_wdv);
            warpgroup_fence_operand(acc_wdv1);
            warpgroup_arrive();
            gemm(update_mma, tWR(_, _, _, _0{}), tDv2R(_, _, _, _0{}), acc_wdv);
            gemm(update_mma, tW1R(_, _, _, _0{}), tDv2R(_, _, _, _0{}), acc_wdv1);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_wdv);
            warpgroup_fence_operand(acc_wdv1);

            CUTE_UNROLL
            for (int i = 0; i < size(state); ++i) {
                state(i) += acc_qdo(i) * scale - acc_wdv(i);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(state1); ++i) {
                state1(i) += acc_qdo1(i) * scale - acc_wdv1(i);
            }
        } else {
            Tensor acc_qdo = update_thr.make_fragment_C(tCUpdate);
            Tensor acc_wdv = update_thr.make_fragment_C(tCUpdate);
            clear(acc_qdo);
            clear(acc_wdv);

            warpgroup_fence_operand(acc_qdo);
            warpgroup_arrive();
            gemm(update_mma, tQR(_, _, _, _0{}), tDoR(_, _, _, _0{}), acc_qdo);
            warpgroup_commit_batch();

            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_qdo);
            cutlass::arch::fence_view_async_shared();

            Tensor tW = update_thr.partition_A(sWStage);
            Tensor tWR = update_thr.make_fragment_A(tW);

            warpgroup_fence_operand(acc_wdv);
            warpgroup_arrive();
            gemm(update_mma, tWR(_, _, _, _0{}), tDv2R(_, _, _, _0{}), acc_wdv);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_wdv);

            CUTE_UNROLL
            for (int i = 0; i < size(state); ++i) {
                state(i) += acc_qdo(i) * scale - acc_wdv(i);
            }
        }

        int prefetch_iter;
        if constexpr (KDim > kK) {
            prefetch_iter = iter + kPrefetchDistance<KDim>;
        } else {
            prefetch_iter = iter + kStages;
        }
        if (prefetch_iter < NT) {
            if constexpr (KDim > kK) {
                prefetch_chunk(NT - 1 - prefetch_iter, prefetch_iter % kStages);
            } else {
                prefetch_chunk(NT - 1 - prefetch_iter, stage);
            }
            pending_groups += kKBlocks<KDim>;
        }
    }

    if (store_dh0) {
        CUTE_UNROLL
        for (int i = 0; i < size(state); ++i) {
            auto coord = tCUpdate(i);
            int k_rel = int(get<0>(coord));
            int v_rel = int(get<1>(coord));
            int64_t off = ((int64_t(b) * H + h) * KDim + k_rel) * kV + v_base + v_rel;
            dh0[off] = state(i);
            if constexpr (KDim > kK) {
                dh0[off + int64_t(kK) * kV] = state1(i);
            }
        }
    }
}

void
check_bf16_4d(char const* name, at::Tensor const& x, int64_t B, int64_t T, int64_t H, int64_t D) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.dim() == 4, name, " must be rank-4 [B, T, H, D]");
    TORCH_CHECK(
        x.size(0) == B && x.size(1) == T && x.size(2) == H && x.size(3) == D,
        name,
        " shape mismatch, expected [",
        B,
        ", ",
        T,
        ", ",
        H,
        ", ",
        D,
        "], got ",
        x.sizes());
}

}  // namespace cula::kda::sm90::bwd_dhu64

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
ChunkGatedDeltaRuleBwdDhu64(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor w,
    torch::Tensor dO,
    torch::Tensor dV,
    std::optional<torch::Tensor> h0,
    std::optional<torch::Tensor> dht,
    double scale) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(q.device());
    using namespace cula::kda::sm90::bwd_dhu64;

    TORCH_CHECK(q.dim() == 4, "q must be rank-4 [B, T, H, K]");
    int64_t B = q.size(0);
    int64_t T = q.size(1);
    int64_t H = q.size(2);
    int64_t K = q.size(3);
    TORCH_CHECK(K == 64 || K == 128, "only K=64 or K=128 is supported, got K=", K);
    TORCH_CHECK(dO.dim() == 4 && dO.size(3) == kV, "only V=64 is supported");
    TORCH_CHECK(T % kBT == 0, "T must be a multiple of 64, got T=", T);

    check_bf16_4d("q", q, B, T, H, K);
    check_bf16_4d("k", k, B, T, H, K);
    check_bf16_4d("w", w, B, T, H, K);
    check_bf16_4d("do", dO, B, T, H, kV);
    check_bf16_4d("dv", dV, B, T, H, kV);

    if (h0.has_value()) {
        TORCH_CHECK(h0->is_cuda(), "h0 must be CUDA when provided");
        TORCH_CHECK(h0->scalar_type() == at::kFloat, "h0 must be float32 when provided");
        TORCH_CHECK(h0->is_contiguous(), "h0 must be contiguous when provided");
        TORCH_CHECK(h0->sizes() == at::IntArrayRef({B, H, K, kV}), "h0 shape must be [B, H, K, 64], got ", h0->sizes());
    }
    if (dht.has_value()) {
        TORCH_CHECK(dht->is_cuda(), "dht must be CUDA when provided");
        TORCH_CHECK(dht->scalar_type() == at::kFloat, "dht must be float32 when provided");
        TORCH_CHECK(dht->is_contiguous(), "dht must be contiguous when provided");
        TORCH_CHECK(
            dht->sizes() == at::IntArrayRef({B, H, K, kV}), "dht shape must be [B, H, K, 64], got ", dht->sizes());
    }

    int64_t NT = T / kBT;
    auto dh = torch::empty({B, NT, H, K, kV}, q.options());
    auto dv2 = torch::empty_like(dV);
    std::optional<torch::Tensor> dh0 = std::nullopt;
    if (h0.has_value()) {
        dh0 = torch::empty({B, H, K, kV}, h0->options().dtype(at::kFloat));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    static std::once_flag attr_once64;
    static std::once_flag attr_once128;
    if (K == 64) {
        std::call_once(attr_once64, [] {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                chunk_delta_bwd_dhu64_sm90_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes<64>));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                chunk_delta_bwd_dhu64_sm90_kernel<64>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared));
        });
        chunk_delta_bwd_dhu64_sm90_kernel<64><<<dim3(kV / kBV, B * H), dim3(kThreads), kSmemBytes<64>, stream>>>(
            reinterpret_cast<Element const*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(w.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(dO.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(dV.data_ptr<at::BFloat16>()),
            dht.has_value() ? dht->data_ptr<float>() : nullptr,
            reinterpret_cast<Element*>(dh.data_ptr<at::BFloat16>()),
            dh0.has_value() ? dh0->data_ptr<float>() : nullptr,
            reinterpret_cast<Element*>(dv2.data_ptr<at::BFloat16>()),
            static_cast<int>(B),
            static_cast<int>(T),
            static_cast<int>(H),
            static_cast<int>(NT),
            static_cast<float>(scale),
            dht.has_value(),
            dh0.has_value());
    } else {
        std::call_once(attr_once128, [] {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                chunk_delta_bwd_dhu64_sm90_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes<128>));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                chunk_delta_bwd_dhu64_sm90_kernel<128>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared));
        });
        chunk_delta_bwd_dhu64_sm90_kernel<128><<<dim3(kV / kBV, B * H), dim3(kThreads), kSmemBytes<128>, stream>>>(
            reinterpret_cast<Element const*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(w.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(dO.data_ptr<at::BFloat16>()),
            reinterpret_cast<Element const*>(dV.data_ptr<at::BFloat16>()),
            dht.has_value() ? dht->data_ptr<float>() : nullptr,
            reinterpret_cast<Element*>(dh.data_ptr<at::BFloat16>()),
            dh0.has_value() ? dh0->data_ptr<float>() : nullptr,
            reinterpret_cast<Element*>(dv2.data_ptr<at::BFloat16>()),
            static_cast<int>(B),
            static_cast<int>(T),
            static_cast<int>(H),
            static_cast<int>(NT),
            static_cast<float>(scale),
            dht.has_value(),
            dh0.has_value());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dh, dh0, dv2};
}
