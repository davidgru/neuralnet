#pragma once

enum class matrix_layout {
    row_major,
    col_major
};

template<matrix_layout Layout>
struct matrix_layout_tag {
    static constexpr matrix_layout value = Layout;
};

using row_major_t = matrix_layout_tag<matrix_layout::row_major>;
using col_major_t = matrix_layout_tag<matrix_layout::col_major>;

template<typename T, typename Layout>
struct matrix_accessor;

template<typename T>
struct matrix_accessor<T, row_major_t> {
    static inline const T& get (const T* ptr, size_t h, size_t w, size_t i, size_t j) {
        return ptr[i * w + j];
    }

    static inline const T* ptr (const T* ptr, size_t h, size_t w, size_t i, size_t j) {
        return &ptr[i * w + j];
    }
};

template<typename T>
struct matrix_accessor<T, col_major_t> {
    static inline const T& get (const T* ptr, size_t h, size_t w, size_t i, size_t j) {
        return ptr[j * h + i];
    }

    static inline const T* ptr (const T* ptr, size_t h, size_t w, size_t i, size_t j) {
        return &ptr[j * h + i];
    }
};
