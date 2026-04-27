#pragma once

#include <cstddef>


#include "../common/vec_ops.hpp"
#include "../gemm/gemm.hpp"

#include "im2col.hpp"

template<typename T>
static inline void swap_N_C(
    const T* __restrict X,
    T* __restrict Y,
    size_t N, size_t C, size_t H, size_t W)
{
    // Y': (OC)x(N*OH*OW) = (flatten) OCxNxOHxOW
    // Y': OCxNxOHxOW => Y: NxOCxOHxOW (reshape)
    for (size_t c = 0; c < C; ++c)
    for (size_t n = 0; n < N; ++n)
    {
        vec_copy<T>(
            &Y[c * (N * H * W) + n * (H * W)],
            &X[n * (C * H * W) + c * (H * W)],
            H * W
        );
    }
}

template<typename T, auto Im2Col, auto GEMM>
void conv(
    const T* __restrict X, // NxICxIHxIW input
    const T* __restrict F, // OCxICxKHxKW filter
    T* __restrict Y, // NxOCxOHxOW output
    size_t N, size_t IC, size_t IH, size_t IW, // input dimensions
    size_t OC, size_t KH, size_t KW, // kernel dimensions
    size_t PH, size_t PW, // padding
    size_t SH, size_t SW, // string
    size_t DH, size_t DW,  // dilation
    size_t CROP_H, size_t CROP_W // skip output edges
) {
    const size_t OH = cod(IH, KH, PH, SH, DH, CROP_H);
    const size_t OW = cod(IW, KW, PW, SW, DW, CROP_W);

    // M used as Im2Col output Matrix and also as temp buffer for reshaping Y
    T* M = new T[std::max(
        (N * OH * OW) * (IC * KH * KW),
        N * OC * OH * OW
    )];

    // F: OCxICxKHxKW => F': OCx(IC*KH*KW) (implicit flatten)
    // X: NxICxIHxIW => M: (IC*KH*KW)x(N*OH*OW) (im2col)
    Im2Col(X, M, N, IC, IH, IW, KH, KW, PH, PW, SH, SW, DH, DW, CROP_H, CROP_W);

    // Y' = F * M
    // Y': OCx(N*OH*OW)
    GEMM(F, M, Y, OC, (N * OH * OW), (IC * KH * KW));

    // Y': OCx(N*OH*OW) => Y: NxOCxOHxOW (reshape)
    swap_N_C(Y, M, OC, N, OH, OW);
    vec_copy(Y, M, N * OC * OH * OW);

    delete[] M;
}
