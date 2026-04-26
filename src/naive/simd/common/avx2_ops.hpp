#pragma once

#include <cstdint>
#include <type_traits>

#include <immintrin.h>

template<typename T>
struct avx2_ops;

template<>
struct avx2_ops<float> {
    using reg = __m256;
    using lanes = std::integral_constant<size_t, 8>;

    template<bool aligned = false>
    static reg load(const float* p) {
        if constexpr (aligned) {
            return _mm256_load_ps(p);
        } else {
            return _mm256_loadu_ps(p);
        }
    }

    template<bool aligned = false>
    static void store(float* p, reg v) {
        if constexpr (aligned) {
            _mm256_store_ps(p, v);
        } else {
            _mm256_storeu_ps(p, v);    
        }
    }
    
    static reg set1(float v) { return _mm256_set1_ps(v); }
    static reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b, c); };
    static reg add(reg a, reg b) { return _mm256_add_ps(a, b); };
};

template<>
struct avx2_ops<double> {
    using reg = __m256d;
    using lanes = std::integral_constant<size_t, 4>;

    template<bool aligned = false>
    static reg load(const double* p) {
        if constexpr (aligned) {
            return _mm256_load_pd(p);
        } else {
            return _mm256_loadu_pd(p);
        }
    }

    template<bool aligned = false>
    static void store(double* p, reg v) {
        if constexpr (aligned) {
            _mm256_store_pd(p, v);
        } else {
            _mm256_storeu_pd(p, v);    
        }
    }
    
    static reg set1(double v) { return _mm256_set1_pd(v); }
    static reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b, c); };
    static reg add(reg a, reg b) { return _mm256_add_pd(a, b); };
};

template<typename T>
struct avx2_ops_i32 {
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>);

    using reg = __m256i;
    using lanes = std::integral_constant<size_t, 8>;

    template<bool aligned = false>
    static reg load(const T* p) {
        if constexpr (aligned) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));;
        } else {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));;
        }
    }

    template<bool aligned = false>
    static void store(T* p, reg v) {
        if constexpr (aligned) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);;
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);;    
        }
    }
    
    static reg set1(T v) { return _mm256_set1_epi32(v); }
    static reg fmadd(reg a, reg b, reg c) {
        return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b));
    };
    static reg add(reg a, reg b) { return _mm256_add_epi32(a, b); };
};

template<>
struct avx2_ops<int32_t> : avx2_ops_i32<int32_t> {};

template<>
struct avx2_ops<uint32_t> : avx2_ops_i32<uint32_t> {};

template<typename T>
struct avx2_ops_i16 {
    static_assert(std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>);

    using reg = __m256i;
    using lanes = std::integral_constant<size_t, 16>;

    template<bool aligned = false>
    static reg load(const T* p) {
        if constexpr (aligned) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));;
        } else {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));;
        }
    }

    template<bool aligned = false>
    static void store(T* p, reg v) {
        if constexpr (aligned) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);;
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);;    
        }
    }

    static reg set1(T v) { return _mm256_set1_epi16(v); }
    static reg fmadd(reg a, reg b, reg c) {
        return _mm256_add_epi16(c, _mm256_mullo_epi16(a, b));
    };
    static reg add(reg a, reg b) { return _mm256_add_epi16(a, b); };
};

template<>
struct avx2_ops<uint16_t> : avx2_ops_i16<uint16_t> {};

template<>
struct avx2_ops<int16_t> : avx2_ops_i16<int16_t> {};
