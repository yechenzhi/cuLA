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

using Element = cutlass::bfloat16_t;
using Acc = float;

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
        "bar.sync 0;\n"
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
    constexpr int kSmemActOffset = kSmemDoOffset + cosize_v<SmemLayoutMN32Read>;
    constexpr int kSmemWOffset = kSmemActOffset + cosize_v<SmemLayoutMN64>;
    constexpr int kSmemKOffset = kSmemWOffset + cosize_v<SmemLayoutMN64>;
    constexpr int kSmemDhOffset = kSmemKOffset + cosize_v<SmemLayoutK64>;
    constexpr int kSmemDv2Offset = kSmemDhOffset + cosize_v<SmemLayoutK32>;
    constexpr int kSmemTotal = kSmemDv2Offset + cosize_v<SmemLayoutK32>;

    __shared__ alignas(128) Element smem_raw[kSmemTotal];

    Element* smem_do = smem_raw + kSmemDoOffset;
    Element* smem_act = smem_raw + kSmemActOffset;
    Element* smem_w = smem_raw + kSmemWOffset;
    Element* smem_k = smem_raw + kSmemKOffset;
    Element* smem_dh = smem_raw + kSmemDhOffset;
    Element* smem_dv2 = smem_raw + kSmemDv2Offset;

    Tensor sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK64{});
    Tensor sDo = make_tensor(make_smem_ptr(smem_do), SmemLayoutMN32Read{});
    Tensor sAct = make_tensor(make_smem_ptr(smem_act), SmemLayoutMN64{});
    Tensor sW = make_tensor(make_smem_ptr(smem_w), SmemLayoutMN64{});
    Tensor sDhStore = make_tensor(make_smem_ptr(smem_dh), SmemLayoutK32{});
    Tensor sDhRead = make_tensor(make_smem_ptr(smem_dh), SmemLayoutMN32Read{});
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
    if (has_dht) {
        CUTE_UNROLL
        for (int i = 0; i < size(state); ++i) {
            auto coord = tCUpdate(i);
            int k_rel = int(get<0>(coord));
            int v_rel = int(get<1>(coord));
            int64_t off = ((int64_t(b) * H + h) * kK + k_rel) * kV + v_base + v_rel;
            state(i) = dht[off];
        }
    } else {
        clear(state);
    }

    for (int chunk = NT - 1; chunk >= 0; --chunk) {
        int t0 = chunk * kBT;
        int64_t base_t = (int64_t(b) * T + t0) * H * kK + int64_t(h) * kK;
        int64_t base_v = (int64_t(b) * T + t0) * H * kV + int64_t(h) * kV + v_base;

        Tensor mK = make_tensor(make_gmem_ptr(k + base_t), select<0, 2>(shape_mnk), make_stride(H * kK, Int<1>{}));
        Tensor gK = local_tile(mK, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(copy_k, copy_k_thr.partition_S(gK), copy_k_thr.partition_D(as_position_independent_swizzle_tensor(sK)));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        int64_t dh_base = (((int64_t(b) * NT + chunk) * H + h) * kK) * kV + v_base;
        // FLA/Triton stores dh before applying this chunk's update.  The
        // global-store layout is separate from the WGMMA-B shared tile.
        r2s_acc(update_mma, r2s_update, r2s_update_thr, state, sDhStore);
        cutlass::arch::fence_view_async_shared();
        store_acc_64x32_bf16(tid, smem_do, dh + dh_base, uint32_t(kV * sizeof(Element)), state);

        Tensor tK = kdh_thr.partition_A(sK);
        Tensor tKR = kdh_thr.make_fragment_A(tK);
        Tensor tDh = kdh_thr.partition_B(sDhRead);
        Tensor tDhR = kdh_thr.make_fragment_B(tDh);
        auto c_tv = make_identity_tensor(make_shape(Int<64>{}, Int<32>{}));
        auto tCKdh = kdh_thr.partition_C(c_tv);
        Tensor acc_dv = kdh_thr.make_fragment_C(tCKdh);
        CUTE_UNROLL
        for (int i = 0; i < size(acc_dv); ++i) {
            auto coord = tCKdh(i);
            int t_rel = int(get<0>(coord));
            int v_rel = int(get<1>(coord));
            acc_dv(i) = float(dv[base_v + int64_t(t_rel) * H * kV + v_rel]);
        }

        warpgroup_fence_operand(acc_dv);
        warpgroup_arrive();
        gemm(kdh_mma, tKR(_, _, _, _0{}), tDhR(_, _, _, _0{}), acc_dv);
        warpgroup_commit_batch();

        Tensor mQ = make_tensor(make_gmem_ptr(q + base_t), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * kK));
        Tensor gQ = local_tile(mQ, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(
            copy_mn,
            copy_mn_thr.partition_S(gQ),
            copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sAct)));
        CUTE_UNROLL
        for (int idx = tid; idx < kBV * kBT; idx += kThreads) {
            int v_rel = idx % kBV;
            int t_rel = idx / kBV;
            sDo(v_rel, t_rel, _0{}) = do_[base_v + int64_t(t_rel) * H * kV + v_rel];
        }
        cp_async_fence();

        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_dv);
        cp_async_wait<0>();
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        r2s_acc(kdh_mma, r2s_kdh, r2s_kdh_thr, acc_dv, sDv2Store);
        cutlass::arch::fence_view_async_shared();
        store_acc_64x32_bf16(tid, smem_w, dv2 + base_v, uint32_t(H * kV * sizeof(Element)), acc_dv);

        Tensor tQ = update_thr.partition_A(sAct);
        Tensor tQR = update_thr.make_fragment_A(tQ);
        Tensor tDo = update_thr.partition_B(sDo);
        Tensor tDoR = update_thr.make_fragment_B(tDo);
        Tensor tDv2 = update_thr.partition_B(sDv2Read);
        Tensor tDv2R = update_thr.make_fragment_B(tDv2);

        Tensor acc_qdo = update_thr.make_fragment_C(tCUpdate);
        Tensor acc_wdv = update_thr.make_fragment_C(tCUpdate);
        clear(acc_qdo);
        clear(acc_wdv);

        warpgroup_fence_operand(acc_qdo);
        warpgroup_arrive();
        gemm(update_mma, tQR(_, _, _, _0{}), tDoR(_, _, _, _0{}), acc_qdo);
        warpgroup_commit_batch();

        Tensor mW = make_tensor(make_gmem_ptr(w + base_t), select<0, 2>(shape_mnk), make_stride(Int<1>{}, H * kK));
        Tensor gW = local_tile(mW, cta_tiler, make_coord(0, 0, _), Step<_1, X, _1>{});
        copy(copy_mn, copy_mn_thr.partition_S(gW), copy_mn_thr.partition_D(as_position_independent_swizzle_tensor(sW)));
        cp_async_fence();

        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qdo);
        cp_async_wait<0>();
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        Tensor tW = update_thr.partition_A(sW);
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
        __syncthreads();
    }

    if (store_dh0) {
        CUTE_UNROLL
        for (int i = 0; i < size(state); ++i) {
            auto coord = tCUpdate(i);
            int k_rel = int(get<0>(coord));
            int v_rel = int(get<1>(coord));
            int64_t off = ((int64_t(b) * H + h) * kK + k_rel) * kV + v_base + v_rel;
            dh0[off] = state(i);
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
    TORCH_CHECK(K == kK, "only K=64 is supported, got K=", K);
    TORCH_CHECK(dO.dim() == 4 && dO.size(3) == kV, "only V=64 is supported");
    TORCH_CHECK(T % kBT == 0, "T must be a multiple of 64, got T=", T);

    check_bf16_4d("q", q, B, T, H, kK);
    check_bf16_4d("k", k, B, T, H, kK);
    check_bf16_4d("w", w, B, T, H, kK);
    check_bf16_4d("do", dO, B, T, H, kV);
    check_bf16_4d("dv", dV, B, T, H, kV);

    if (h0.has_value()) {
        TORCH_CHECK(h0->is_cuda(), "h0 must be CUDA when provided");
        TORCH_CHECK(h0->scalar_type() == at::kFloat, "h0 must be float32 when provided");
        TORCH_CHECK(h0->is_contiguous(), "h0 must be contiguous when provided");
        TORCH_CHECK(
            h0->sizes() == at::IntArrayRef({B, H, kK, kV}), "h0 shape must be [B, H, 64, 64], got ", h0->sizes());
    }
    if (dht.has_value()) {
        TORCH_CHECK(dht->is_cuda(), "dht must be CUDA when provided");
        TORCH_CHECK(dht->scalar_type() == at::kFloat, "dht must be float32 when provided");
        TORCH_CHECK(dht->is_contiguous(), "dht must be contiguous when provided");
        TORCH_CHECK(
            dht->sizes() == at::IntArrayRef({B, H, kK, kV}), "dht shape must be [B, H, 64, 64], got ", dht->sizes());
    }

    int64_t NT = T / kBT;
    auto dh = torch::empty({B, NT, H, kK, kV}, q.options());
    auto dv2 = torch::empty_like(dV);
    std::optional<torch::Tensor> dh0 = std::nullopt;
    if (h0.has_value()) {
        dh0 = torch::empty({B, H, kK, kV}, h0->options().dtype(at::kFloat));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    chunk_delta_bwd_dhu64_sm90_kernel<<<dim3(kV / kBV, B * H), dim3(kThreads), 0, stream>>>(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dh, dh0, dv2};
}
