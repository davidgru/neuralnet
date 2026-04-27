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
