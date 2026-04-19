#pragma once

#include <immintrin.h>
#include <cstring>
#include <algorithm>

#include "matrix.hpp"
#include "packing.hpp"
#include "avx2_ops.hpp"
#include "kernel.hpp"

template<bool EnableBlocking, size_t MC, size_t NC, size_t KC>
struct blocking_policy {
    static constexpr bool enabled = EnableBlocking;
    static constexpr size_t mc = MC;
    static constexpr size_t nc = NC;
    static constexpr size_t kc = KC;
};

template<size_t MC, size_t NC, size_t KC>
using blocked = blocking_policy<true, MC, NC, KC>;
using unblocked = blocking_policy<false, 0, 0, 0>;

template<typename T, typename Kernel, typename PackingPolicy, size_t NR>
static inline void gemm_cleanup_block(
    const T* __restrict A_block, const T* __restrict B_block, T* __restrict C_block,
    size_t m, size_t n,
    size_t lda, size_t ldb, size_t ldc,
    size_t m_remain, size_t n_remain, size_t k_remain
)
{
    const T* B_blk = PackingPolicy::pack_B
        ? &B_block[(n / NR) * (k_remain * NR)]
        : &B_block[n];
    Kernel::tail(
        &A_block[m * lda],
        B_blk,
        &C_block[m * ldc + n],
        k_remain,
        lda, ldb, ldc,
        m_remain, n_remain
    );
}

template<typename T, typename Kernel, typename PackingPolicy = unpacked>
static inline void gemm_inner(
    const T* __restrict A_block,
    const T* __restrict B_block,
    T* __restrict C_block,
    size_t lda, size_t ldb, size_t ldc,
    size_t m_end, size_t n_end, size_t k_end)
{
    static constexpr size_t MR = Kernel::MR;
    static constexpr size_t NR = Kernel::NR;

    size_t m;
    for (m = 0; m + MR <= m_end; m += MR) {
        size_t n;
        // Fast path - main loop
        for (n = 0; n + NR <= n_end; n += NR) {
            const T* B_blk = PackingPolicy::pack_B
                ? &B_block[(n / NR) * (k_end * NR)]
                : &B_block[n];
            Kernel::apply(
                &A_block[m * lda],
                B_blk,
                &C_block[m * ldc + n],
                k_end,
                lda, ldb, ldc
            );
        }

        // clean-up n tail
        if (n < n_end) {
            gemm_cleanup_block<T, kernel::cleanup<T>, PackingPolicy, NR>(
                A_block, B_block, C_block, m, n, lda, ldb, ldc,
                MR, n_end - n, k_end);
        }
    }

    // clean-up m tail
    if (m < m_end) {
        size_t n;
        for (n = 0; n + NR <= n_end; n += NR) {
            gemm_cleanup_block<T, Kernel, PackingPolicy, NR>(
                A_block, B_block, C_block, m, n, lda, ldb, ldc,
                m_end - m, NR, k_end);
        }

        // clean-up mn tail
        if (n < n_end) {
            gemm_cleanup_block<T, kernel::cleanup<T>, PackingPolicy, NR>(
                A_block, B_block, C_block, m, n, lda, ldb, ldc,
                m_end - m, n_end - n, k_end);
        }
    }
}

template<
    typename T, typename Kernel, typename BlockingPolicy, typename PackingPolicy,
    typename LayoutA, typename LayoutB, typename LayoutC
>
static inline void gemm_packed(
    const T* __restrict A, const T* __restrict B, T* __restrict C,
    size_t M, size_t N, size_t K)
{
    static_assert(BlockingPolicy::enabled && PackingPolicy::enabled());
    static_assert(std::is_same_v<LayoutC, row_major_t>);

    static constexpr size_t MC = BlockingPolicy::mc;
    static constexpr size_t NC = BlockingPolicy::nc;
    static constexpr size_t KC = BlockingPolicy::kc;
    static constexpr size_t NR = Kernel::NR;

    PackingContext_A<T, MC, KC, PackingPolicy::pack_A, LayoutA> apc;
    PackingContext_B<T, KC, NC, NR, PackingPolicy::pack_B, LayoutB> bpc;

    for (size_t k0 = 0; k0 < K; k0 += KC)
    for (size_t n0 = 0; n0 < N; n0 += NC)
    {
        size_t n_max = std::min(n0 + NC, N) - n0;
        size_t k_max = std::min(k0 + KC, K) - k0;

        // Pack B if enabled
        const auto [B_block, ldb] = bpc.get_block(
            matrix_accessor<T, LayoutB>::ptr(B, K, N, k0, n0),
            K, N, k_max, n_max);

        for (size_t m0 = 0; m0 < M; m0 += MC)
        {
            size_t m_max = std::min(m0 + MC, M) - m0;

            // Pack A if enabled
            const auto [A_block, lda] = apc.get_block(
                matrix_accessor<T, LayoutA>::ptr(A, M, K, m0, k0),
                M, K, m_max, k_max);
            
            gemm_inner<T, Kernel, PackingPolicy>(
                A_block,
                B_block,
                &C[m0 * N + n0],
                lda, ldb, N,
                m_max, n_max, k_max
            );
        }
    }
}

template<typename T, typename Kernel, typename BlockingPolicy>
static inline void gemm_blocked(
    const T* __restrict A, const T* __restrict B, T* __restrict C,
    size_t M, size_t N, size_t K)
{
    static_assert(BlockingPolicy::enabled);

    static constexpr size_t MC = BlockingPolicy::mc;
    static constexpr size_t NC = BlockingPolicy::nc;
    static constexpr size_t KC = BlockingPolicy::kc;

    for (size_t m0 = 0; m0 < M; m0 += MC)
    for (size_t n0 = 0; n0 < N; n0 += NC)
    for (size_t k0 = 0; k0 < K; k0 += KC)
    {
        size_t k_max = std::min(k0 + KC, K) - k0;
        size_t n_max = std::min(n0 + NC, N) - n0;
        size_t m_max = std::min(m0 + MC, M) - m0;
        
        gemm_inner<T, Kernel>(
            &A[m0 * K + k0],
            &B[k0 * N + n0],
            &C[m0 * N + n0],
            K, N, N,
            m_max, n_max, k_max
        );
    }
}

template<typename T, typename Kernel, typename BlockingPolicy = unblocked,
    typename PackingPolicy = unpacked, typename LayoutA = row_major_t,
    typename LayoutB = row_major_t>
static inline void gemm(
    const T* __restrict A, const T* __restrict B, T* __restrict C,
    size_t M, size_t N, size_t K)
{
    using LayoutC = row_major_t;

    if constexpr (BlockingPolicy::enabled && PackingPolicy::enabled()) {
        gemm_packed<T, Kernel, BlockingPolicy, PackingPolicy,
                    LayoutA, LayoutB, LayoutC>(A, B, C, M, N, K);
    } else if constexpr (BlockingPolicy::enabled) {
        static_assert(std::is_same_v<LayoutA, row_major_t>
                      && std::is_same_v<LayoutC, row_major_t>
                      && std::is_same_v<LayoutB, row_major_t>);
        gemm_blocked<T, Kernel, BlockingPolicy>(A, B, C, M, N, K);
    } else {
        static_assert(std::is_same_v<LayoutA, row_major_t>
                      && std::is_same_v<LayoutC, row_major_t>
                      && std::is_same_v<LayoutB, row_major_t>);
        gemm_inner<T, Kernel>(A, B, C, K, N, N, M, N, K);
    }
}

template<typename T, typename BlockingPolicy, typename PackingPolicy,
    size_t MR, typename LayoutA = row_major_t, typename LayoutB = row_major_t>
static void gemm_avx2(
    const T* __restrict A, const T* __restrict B, T* __restrict C,
    size_t M, size_t N, size_t K)
{
    using Kernel = kernel::avx2<T, MR, PackingPolicy::pack_B>;
    gemm<T, Kernel, BlockingPolicy, PackingPolicy, LayoutA, LayoutB>(A, B, C, M, N, K);
}
