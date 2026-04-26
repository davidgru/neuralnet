#pragma once

#include "../common/avx2_ops.hpp"

namespace kernel {

// -------------------------
// 4x8 AVX2 microkernel
// -------------------------


// AA MxK block, BB KxN block, CC MxN block
// +--------+    +--------+    +--------+
// |        |    |     xx |    |        |
// |        |    |     xx |    |        |
// |    xx  |    |        |    |     xx |
// |    xx  |    |        |    |     xx |
// +--------+    +--------+    +--------+

// A at i:IR,k:KR  B at k:KR,j:JR      C at i:IR,j:JR
// Stride lda=K    packed           Stride ldc=N
template<typename T, size_t _MR, bool Aligned_B_Access>
struct avx2 {
    static_assert(_MR > 0, "MR must be greater than 0");
    static constexpr size_t MR = _MR;
    static constexpr size_t NR = avx2_ops<T>::lanes::value;
    static constexpr bool aligned_b_access = Aligned_B_Access;

    static void apply(const T* __restrict A, const T* __restrict B, T* __restrict C,
        size_t K, size_t SA, size_t SB, size_t SC, size_t m_end = MR)
    {
        using ops = avx2_ops<T>;
        using reg = typename ops::reg;

        reg c[MR];

        for (size_t m = 0; m < m_end; ++m) {
            c[m] = ops::load(&C[m * SC]);
        }

        for (size_t k = 0; k < K; ++k) {
            reg b = ops::template load<Aligned_B_Access>(&B[k * SB]);

            for (size_t m = 0; m < m_end; ++m) {
                reg a = ops::set1(A[m * SA + k]);
                c[m] = ops::fmadd(a, b, c[m]);
            }
        }

        for (size_t m = 0; m < m_end; ++m) {
            ops::store(&C[m * SC], c[m]);
        }
    }

    // unsafe
    static void tail(
        const T* __restrict A, const T* __restrict B, T* __restrict C,
        size_t K, size_t SA, size_t SB, size_t SC,
        size_t m_end, size_t n_end
    ) {
        apply(A, B, C, K, SA, SB, SC, m_end);
    }
};


template<typename T>
struct cleanup {
    static constexpr size_t MR = 1;
    static constexpr size_t NR = 1;

    static inline void apply(
        const T* __restrict A, const T* __restrict B, T* __restrict C,
        size_t K, size_t SA, size_t SB, size_t SC,
        size_t m_end = NR, size_t n_end = NR)
    {
        for (size_t m = 0; m < m_end; m++)
        for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < n_end; n++)
            C[m * SC + n] += A[m * SA + k] * B[k * SB + n];
    }

    static inline void tail(
        const T* __restrict A, const T* __restrict B, T* __restrict C,
        size_t K, size_t SA, size_t SB, size_t SC,
        size_t m_end, size_t n_end)
    {
        apply(A, B, C, K, SA, SB, SC, m_end, n_end);
    }
};

} // namespace kernel
