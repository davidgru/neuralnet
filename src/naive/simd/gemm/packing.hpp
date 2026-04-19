#pragma once

#include <cstddef>

#include "matrix.hpp"

template<bool Pack_A, bool Pack_B>
struct packing_policy {
    static constexpr bool pack_A = Pack_A;
    static constexpr bool pack_B = Pack_B;
    static constexpr bool enabled() { return pack_A || pack_B; };
};

using unpacked = packing_policy<false, false>;
using pack_a = packing_policy<true, false>;
using pack_b = packing_policy<false, true>;
using pack_ab = packing_policy<true, true>;


template<typename T, typename Layout>
struct DummyPackingContext;

template<typename T>
struct DummyPackingContext<T, row_major_t>
{
    static inline std::pair<const T*, size_t> get_block(
        const T* __restrict M,
        size_t H, size_t W,
        size_t hc, size_t wc) {
        return std::make_pair(M, W);
    };
};

template<typename T>
struct DummyPackingContext<T, col_major_t>
{
    static inline std::pair<const T*, size_t> get_block(
        const T* __restrict M,
        size_t H, size_t W,
        size_t hc, size_t wc) {
        return std::make_pair(M, H);
    };
};

template<typename T, size_t MC, size_t KC, bool Pack, typename Layout>
struct PackingContext_A;

template<typename T, size_t MC, size_t KC, typename Layout>
struct PackingContext_A<T, MC, KC, false, Layout> : DummyPackingContext<T, Layout> {};

template<typename T, size_t MC, size_t KC, typename Layout>
struct PackingContext_A<T, MC, KC, true, Layout>
{
    T* Ap = nullptr;

    PackingContext_A() 
        : Ap(static_cast<T*>(std::aligned_alloc(
                std::hardware_destructive_interference_size,
                MC * KC * sizeof(T)
            ))) {};
    
    ~PackingContext_A() {
        std::free(Ap);
    }

    inline std::pair<const T*, size_t> get_block(
        const T* __restrict A,
        size_t M, size_t K,
        size_t mc, size_t kc)
    {
        for (size_t m = 0; m < mc; ++m) {
            for (size_t k = 0; k < kc; ++k) {
                Ap[m * KC + k] = matrix_accessor<T, Layout>::get(A, M, K, m, k);
            }
            for (size_t k = kc; k < KC; ++k) {
                Ap[m * KC + k] = T{0};
            }
        };
        return std::make_pair(Ap, KC);
    }

    PackingContext_A(const PackingContext_A&) = delete;
    PackingContext_A& operator=(const PackingContext_A&) = delete;
};

template<typename T, size_t KC, size_t NC, size_t NR, bool Pack, typename Layout>
struct PackingContext_B;

template<typename T, size_t KC, size_t NC, size_t NR, typename Layout>
struct PackingContext_B<T, KC, NC, NR, false, Layout> : DummyPackingContext<T, Layout> {};

template<typename T, size_t KC, size_t NC, size_t NR, typename Layout>
struct PackingContext_B<T, KC, NC, NR, true, Layout>
{
    T* Bp = nullptr;

    PackingContext_B() 
        : Bp(static_cast<T*>(std::aligned_alloc(
                std::hardware_destructive_interference_size,
                KC * NC * sizeof(T)
            ))) {};
    
    ~PackingContext_B() {
        std::free(Bp);
    }

    //
    // Pack row major B into (nc/NR x kc) micro-panels of size NR
    //
    inline std::pair<const T*, size_t> get_block(
        const T* __restrict B,
        size_t K, size_t N,
        size_t kc, size_t nc)
    {
        T* Bp_local = Bp;
        for (size_t n = 0; n < nc; n += NR)
        for (size_t k = 0; k < kc; ++k)
        {
            size_t width = std::min(NR, nc - n);

            // pack one row of the micro-panel
            for (size_t j = 0; j < width; ++j)
                Bp_local[j] = matrix_accessor<T, Layout>::get(B, K, N, k, n + j);
            for (size_t j = width; j < NR; ++j)
                Bp_local[j] = T{0};
            Bp_local += NR;
        }
        return std::make_pair(Bp, NR);
    }

    PackingContext_B(const PackingContext_B&) = delete;
    PackingContext_B& operator=(const PackingContext_B&) = delete;
};
