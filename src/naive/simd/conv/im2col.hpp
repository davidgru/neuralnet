#pragma once

#include <cstddef>
#include <limits>

#include "../common/vec_ops.hpp"
#include "conv_util.hpp"

template<size_t BS = 1>
static inline std::pair<size_t, size_t> ose(
  ptrdiff_t k, ptrdiff_t IE, ptrdiff_t OE,
  ptrdiff_t P, ptrdiff_t S, ptrdiff_t D, ptrdiff_t CROP)
{
  // (oh + CROP_H) * SH + kh * DH >= PH
  // oh >= (PH - kh * DH) / SH - CROP_H
  // <- oh >= PH / SH - CROP_H
  // <- oh >= oh0 && OH0 >= PH / SH - CROP_H

  // oh < (IH + PH - kh * DH) / SH - CROP_H
  // <- oh < (IH + PH - KH * DH) / SH - CROP_H
  // <- oh < oh0 + OH0 && oh0 + OH0 < (IH + PH - KH * DH) / SH - CROP_H

  const ptrdiff_t PkD = P - k * D;
  const size_t os = std::max(ptrdiff_t{0}, (PkD + S - 1) / S - CROP);
  const size_t oe = std::min(OE, (IE + PkD + S - 1) / S - CROP);
  return {((os + BS - 1) / BS) * BS, (oe / BS) * BS};
}

template<size_t BS = 1>
static inline std::pair<size_t, size_t> osek(
  ptrdiff_t o, ptrdiff_t IE, ptrdiff_t KE,
  ptrdiff_t P, ptrdiff_t S, ptrdiff_t D, ptrdiff_t CROP)
{
  const ptrdiff_t PoCS = P - (o + CROP) * S;
  const size_t ks = std::max(ptrdiff_t{0}, (PoCS + D - 1) / D);
  const size_t ke = std::min(KE, (IE + PoCS + D - 1) / D);
  return {((ks + BS - 1) / BS) * BS, (ke / BS) * BS};
}

// fast copy function in case stride = 1
template<typename T>
static inline void im2col_copy_row_s1(
  T* __restrict M_row,
  const T* __restrict X_row,
  size_t N,
  size_t SW = 1 // ignored, assumed 1
) {
  vec_copy(M_row, X_row, N);
}

template<typename T>
static inline void im2col_copy_row_generic(
  T* __restrict M_row,
  const T* __restrict X_row,
  size_t N,
  size_t SW
) {
    for (size_t n = 0; n < N; ++n) {
        M_row[n] = X_row[n * SW];  
    }
}

template<bool EnableBlocking, size_t _N0, size_t _C0, size_t _KH0, size_t _KW0, size_t _OH0, size_t _OW0>
struct im2col_blocking_policy {
    static constexpr bool enabled = EnableBlocking;
    static constexpr size_t N0 = _N0 > 0 ? _N0 : std::numeric_limits<size_t>::max();
    static constexpr size_t C0 = _C0 > 0 ? _C0 : std::numeric_limits<size_t>::max();
    static constexpr size_t KH0 = _KH0 > 0 ? _KH0 : std::numeric_limits<size_t>::max();
    static constexpr size_t KW0 = _KW0 > 0 ? _KW0 : std::numeric_limits<size_t>::max();
    static constexpr size_t OH0 = _OH0 > 0 ? _OH0 : std::numeric_limits<size_t>::max();
    static constexpr size_t OW0 = _OW0 > 0 ? _OW0 : std::numeric_limits<size_t>::max();
};

template<size_t _N0, size_t _C0, size_t _KH0, size_t _KW0, size_t _OH0, size_t _OW0>
using im2col_blocked = im2col_blocking_policy<true, _N0, _C0, _KH0, _KW0, _OH0, _OW0>;
using im2col_unblocked = im2col_blocking_policy<true, 0, 0, 0, 0, 0, 0>;

template<typename T, auto RowCopyFunc, typename BlockingPolicy>
static inline void im2col_o(
    const T* __restrict X,
    T* __restrict M,
    size_t N, size_t IC, size_t IH, size_t IW,
    size_t KH, size_t KW,
    size_t PH, size_t PW,
    size_t SH, size_t SW,
    size_t DH, size_t DW,
    size_t CROP_H, size_t CROP_W
) {
    static constexpr size_t N0 = BlockingPolicy::N0;
    static constexpr size_t C0 = BlockingPolicy::C0;
    static constexpr size_t KH0 = BlockingPolicy::KH0;
    static constexpr size_t KW0 = BlockingPolicy::KW0;
    static constexpr size_t OH0 = BlockingPolicy::OH0;
    static constexpr size_t OW0 = BlockingPolicy::OW0;
  
    const size_t OH = cod(IH, KH, PH, SH, DH, CROP_H);
    const size_t OW = cod(IW, KW, PW, SW, DW, CROP_W);

    for (size_t  n0 = 0;  n0 <  N;  n0 +=  N0)
    for (size_t  c0 = 0;  c0 < IC;  c0 +=  C0)
    for (size_t kh0 = 0; kh0 < KH; kh0 += KH0)
    for (size_t kw0 = 0; kw0 < KW; kw0 += KW0)
    for (size_t oh0 = 0; oh0 < OH; oh0 += OH0)
    {
        const size_t n_max = std::min(n0 + N0, N);
        const size_t c_max = std::min(c0 + C0, IC);
        const size_t kh_max = std::min(kh0 + KH0, KH);
        const size_t kw_max = std::min(kw0 + KW0, KW);
        const size_t oh_max = std::min(oh0 + OH0, OH);

        for (size_t  n =  n0;  n <  n_max; ++n)
        for (size_t  c =  c0;  c <  c_max; ++c)
        for (size_t kh = kh0; kh < kh_max; ++kh)
        for (size_t kw = kw0; kw < kw_max; ++kw)
        {
          const auto [ohs, ohe] = ose(kh, IH, OH, PH, SH, DH, CROP_H);
          const auto [ows, owe] = ose(kw, IW, OW, PW, SW, DW, CROP_W);

          if (oh0 < ohs) {
              vec_memset(&M[
                  c * (KH * KW * N * OH * OW)
                + kh * (KW * N * OH * OW)
                + kw * (N * OH * OW)
                + n * (OH * OW)
                + oh0 * (OW)
              ], T{0}, (std::min(ohs, oh_max) - oh0) * OW);
          }
          for (size_t oh = std::max(oh0, ohs); oh < std::min(oh_max, ohe); ++oh)
          {
              const size_t xh = (oh + CROP_H) * SH + kh * DH - PH;
              const size_t xws = (ows + CROP_W) * SW + kw * DW - PW;
              
              vec_memset(&M[
                  c * (KH * KW * N * OH * OW)
                  + kh * (KW * N * OH * OW)
                + kw * (N * OH * OW)
                + n * (OH * OW)
                + oh * (OW)
              ], T{0}, ows);

              RowCopyFunc(
                &M[
                  c * (KH * KW * N * OH * OW)
                  + kh * (KW * N * OH * OW)
                  + kw * (N * OH * OW)
                  + n * (OH * OW)
                  + oh * (OW)
                  + ows
                ],
                &X[
                  n  * (IC * IH * IW)
                  + c * (IH * IW)
                  + xh * (IW)
                  + xws
                ],
                owe - ows, SW
              );

              vec_memset(&M[
                  c * (KH * KW * N * OH * OW)
                  + kh * (KW * N * OH * OW)
                  + kw * (N * OH * OW)
                  + n * (OH * OW)
                  + oh * (OW)
                  + owe
                ], T{0}, OW - owe);
            }

            if (oh_max >= ohe) {
                vec_memset(&M[
                    c * (KH * KW * N * OH * OW)
                  + kh * (KW * N * OH * OW)
                  + kw * (N * OH * OW)
                  + n * (OH * OW)
                  + std::max(oh0, ohe) * (OW)
                ], T{0}, (oh_max - std::max(oh0, ohe)) * OW);
            }
        }
    }
}

template<typename T, auto RowCopyFunc, typename BlockingPolicy>
static inline void im2col_k(
    const T* __restrict X,
    T* __restrict M,
    size_t N, size_t IC, size_t IH, size_t IW,
    size_t KH, size_t KW,
    size_t PH, size_t PW,
    size_t SH, size_t SW,
    size_t DH, size_t DW,
    size_t CROP_H, size_t CROP_W
) {
    static constexpr size_t N0 = BlockingPolicy::N0;
    static constexpr size_t C0 = BlockingPolicy::C0;
    static constexpr size_t KH0 = BlockingPolicy::KH0;
    static constexpr size_t KW0 = BlockingPolicy::KW0;
    static constexpr size_t OH0 = BlockingPolicy::OH0;
    static constexpr size_t OW0 = BlockingPolicy::OW0;

    const size_t OH = cod(IH, KH, PH, SH, DH, CROP_H);
    const size_t OW = cod(IW, KW, PW, SW, DW, CROP_W);

    for (size_t  n0 = 0;  n0 <  N;  n0 +=  N0)
    for (size_t  c0 = 0;  c0 < IC;  c0 +=  C0)
    for (size_t oh0 = 0; oh0 < OH; oh0 += OH0)
    for (size_t ow0 = 0; ow0 < OW; ow0 += OW0)
    for (size_t kh0 = 0; kh0 < KH; kh0 += KH0)
    {
        const size_t n_max = std::min(n0 + N0, N);
        const size_t c_max = std::min(c0 + C0, IC);
        const size_t oh_max = std::min(oh0 + OH0, OH);
        const size_t ow_max = std::min(ow0 + OW0, OW);
        const size_t kh_max = std::min(kh0 + KH0, KH);

        for (size_t  n =  n0;  n <  n_max; ++n)
        for (size_t  c =  c0;  c <  c_max; ++c)
        for (size_t oh = oh0; oh < oh_max; ++oh)
        for (size_t ow = ow0; ow < ow_max; ++ow)
        {
            const auto [khs, khe] = osek(oh, IH, KH, PH, SH, DH, CROP_H);
            const auto [kws, kwe] = osek(ow, IW, KW, PW, SW, DW, CROP_W);

            if (kh0 < khs) {
                vec_memset(&M[
                    n * (OH * OW * IC * KH * KW)
                  + oh * (OW * IC * KH * KW)
                  + ow * (IC * KH * KW)
                  +  c * (KH * KW)
                  + kh0 * (KW)
                ], T{0}, (std::min(khs, kh_max) - kh0) * KW);
            }
            for (size_t kh = std::max(kh0, khs); kh < std::min(kh_max, khe); ++kh)
            {
                const size_t xh = (oh + CROP_H) * SH + kh * DH - PH;
                const size_t xws = (ow + CROP_W) * SW + kws * DW - PW;
                
                vec_memset(&M[
                    n * (OH * OW * IC * KH * KW)
                  + oh * (OW * IC * KH * KW)
                  + ow * (IC * KH * KW)
                  +  c * (KH * KW)
                  + kh * (KW)
                ], T{0}, kws);

                RowCopyFunc(
                  &M[
                      n * (OH * OW * IC * KH * KW)
                    + oh * (OW * IC * KH * KW)
                    + ow * (IC * KH * KW)
                    +  c * (KH * KW)
                    + kh * (KW)
                    + kws
                  ],
                  &X[
                      n * (IC * IH * IW)
                    +  c * (IH * IW)
                    + xh * (IW)
                    + xws
                  ],
                  kwe - kws, DW
                );

                vec_memset(&M[
                    n * (OH * OW * IC * KH * KW)
                  + oh * (OW * IC * KH * KW)
                  + ow * (IC * KH * KW)
                  +  c * (KH * KW)
                  + kh * (KW)
                  + kwe
                ], T{0}, KW - kwe);
            }

            if (kh_max >= khe) {
                vec_memset(&M[
                    n * (OH * OW * IC * KH * KW)
                  + oh * (OW * IC * KH * KW)
                  + ow * (IC * KH * KW)
                  +  c * (KH * KW)
                  + std::max(kh0, khe) * (KW)
                ], T{0}, (kh_max - std::max(kh0, khe)) * KW);
            }
        }
    }
}

template<typename T, typename BlockingPolicy = im2col_unblocked>
static inline void im2col(
    const T* __restrict X, // NxICxIHxIW input
    T* __restrict M, // output matrix (IC*KH*KW)x(N*OH*OW)
    size_t N, size_t IC, size_t IH, size_t IW, // input dimensions
    size_t KH, size_t KW, // kernel dimensions
    size_t PH, size_t PW, // padding
    size_t SH, size_t SW, // string
    size_t DH, size_t DW,  // dilation
    size_t CROP_H, size_t CROP_W // skip output edges
) {
    if (SW == 1) {
        im2col_o<T,
            im2col_copy_row_s1<T>,
            BlockingPolicy>(
            X, M, N, IC, IH, IW, KH, KW,
            PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    } else {
        im2col_o<T,
            im2col_copy_row_generic<T>,
            BlockingPolicy>(
            X, M, N, IC, IH, IW, KH, KW,
            PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    }
}

template<typename T, typename BlockingPolicy = im2col_unblocked>
static inline void im2col_T(
    const T* __restrict X, // NxICxIHxIW input
    T* __restrict M, // output matrix (N*OH*OW)x(IC*KH*KW)
    size_t N, size_t IC, size_t IH, size_t IW, // input dimensions
    size_t KH, size_t KW, // kernel dimensions
    size_t PH, size_t PW, // padding
    size_t SH, size_t SW, // string
    size_t DH, size_t DW,  // dilation
    size_t CROP_H, size_t CROP_W // skip output edges
) {
    if (DW == 1) {
        im2col_k<T,
            im2col_copy_row_s1<T>,
            BlockingPolicy>(
            X, M, N, IC, IH, IW, KH, KW,
            PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    } else {
        im2col_k<T,
            im2col_copy_row_generic<T>,
            BlockingPolicy>(
            X, M, N, IC, IH, IW, KH, KW,
            PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    }
}
