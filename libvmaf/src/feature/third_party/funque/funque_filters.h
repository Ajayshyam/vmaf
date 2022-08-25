/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */
#ifndef FILTERS_FUNQUE_H_
#define FILTERS_FUNQUE_H_
#include <stddef.h>

#include "config.h"

typedef struct dwt2buffers {
    float *bands[4];
    int width;
    int height;
}dwt2buffers;

void spatial_filter(float *src, float *dst, int width, int height);

void funque_dwt2(float *src, dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);

void funque_vifdwt2_band0(float *src, float *band_a, ptrdiff_t dst_stride, int width, int height);

void normalize_bitdepth(float *src, float *dst, int scaler, ptrdiff_t dst_stride, int width, int height);

#endif /* FILTERS_FUNQUE_H_ */