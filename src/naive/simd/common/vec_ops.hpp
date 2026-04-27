#pragma once

#include "avx2_ops.hpp"

template<typename T,
         bool Aligned_A = false,
         bool Aligned_B = false>
static inline void vec_copy(
    T* __restrict A,
    const T* __restrict B,
    size_t N)
{
    using ops = avx2_ops<T>;
    using reg = typename ops::reg;
    static constexpr size_t KR = ops::lanes::value;

    size_t i;
    for (i = 0; i + KR <= N; i += KR) {
        reg v = ops::template load<Aligned_A>(&B[i]);
        ops::template store<Aligned_B>(&A[i], v);
    }
    for (; i < N; ++i) {
        A[i] = B[i];
    }
}

template<typename T,
         bool Aligned_A = false,
         bool Aligned_B = false>
static inline void vec_add(
    T* __restrict A,
    const T* __restrict B,
    size_t N)
{
    using ops = avx2_ops<T>;
    using reg = typename ops::reg;
    static constexpr size_t KR = ops::lanes::value;

    size_t i;
    for (i = 0; i + KR <= N; i += KR) {
        reg a = ops::template load<Aligned_A>(&A[i]);
        reg b = ops::template load<Aligned_B>(&B[i]);
        a = ops::add(a, b);
        ops::template store<Aligned_A>(&A[i], a);
    }
    for (; i < N; ++i) {
        A[i] += B[i];
    }
}

template<typename T,
         bool Aligned_A = false>
static inline void vec_scalar_add(
    T* __restrict A,
    T b,
    size_t N)
{
    using ops = avx2_ops<T>;
    using reg = typename ops::reg;
    static constexpr size_t KR = ops::lanes::value;

    size_t i;
    reg bb = ops::set1(b);
    for (i = 0; i + KR <= N; i += KR) {
        reg v = ops::template load<Aligned_A>(&A[i]);
        v = ops::add(v, bb);
        ops::template store<Aligned_A>(&A[i], v);
    }
    for (; i < N; ++i) {
        A[i] += b;
    }
}

template<typename T,
         size_t BlockSize,
         bool Aligned_A = false>
static inline T vec_sum(const T* A, size_t N)
{
    using ops = avx2_ops<T>;
    using reg = typename ops::reg;
    static constexpr size_t KR = ops::lanes::value;
    static_assert(BlockSize % KR == 0);

    // Init accumulators to zero
    reg acc[BlockSize / KR];
    for (size_t k = 0; k < BlockSize / KR; ++k) {
        acc[k] = ops::zero();
    }

    size_t n;
    for (n = 0; n + BlockSize <= N; n += BlockSize)
    for (size_t k = 0; k < BlockSize / KR; ++k)
    {
        reg a = ops::template load<Aligned_A>(&A[n + k * KR]);
        acc[k] = ops::add(acc[k], a);
    }

    // horizontally sum accumulators into acc[0]
    for (size_t k = 1; k < BlockSize / KR; ++k) {
        acc[0] = ops::add(acc[0], acc[k]);
    }

    // horizontally sum acc[0]
    T sum = ops::hadd(acc[0]);
    
    // tail
    if constexpr ((BlockSize / 4) >= KR) {
        // kernel with smaller block size
        sum += vec_sum<T, BlockSize / 4, Aligned_A>(&A[n], N - n);
    } else if constexpr ((BlockSize / 2) >= KR) {
        // kernel with smaller block size
        sum += vec_sum<T, BlockSize / 2, Aligned_A>(&A[n], N - n);
    } else {
        // one-by-one
        for (; n < N; ++n) {
            sum += A[n];
        }
    }

    return sum;
}

template<typename T,
         bool Aligned = false>
static inline void vec_memset(
    T* __restrict A,
    T val,
    size_t N)
{
    using ops = avx2_ops<T>;
    using reg = typename ops::reg;
    static constexpr size_t KR = ops::lanes::value;

    size_t i;
    for (i = 0; i + KR <= N; i += KR) {
        reg v = ops::set1(val);
        ops::template store<Aligned>(&A[i], v);
    }
    for (; i < N; ++i) {
        A[i] = val;
    }
}
