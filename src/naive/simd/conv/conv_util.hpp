#pragma once

#include <cstddef>
#include <stdexcept>

static inline size_t cod(
    size_t I, size_t K, size_t P, size_t S, size_t D, size_t CROP
) {
    return (I + 2 * P - D * (K - 1) - 1) / S + 1 - 2 * CROP;
}
