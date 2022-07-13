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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#define MAX_ALIGN 32
#define ALIGN_FLOOR(x) ((x) - (x) % MAX_ALIGN)
#define ALIGN_CEIL(x)  ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define MAX_SAMPLE_VALUE_10BIT                      0x3FF
#define CLIP3(MinVal, MaxVal, a)        (((a)<(MinVal)) ? (MinVal) : (((a)>(MaxVal)) ? (MaxVal) :(a)))

#if FUNQUE_DOUBLE_DTYPE
typedef double funque_dtype;
#else
typedef float funque_dtype;
#endif

#define SPAT_FILTER_COEFF_SHIFT 16
#define SPAT_FILTER_INTER_SHIFT  9
#define SPAT_FILTER_OUT_SHIFT   16
typedef int16_t spat_fil_coeff_dtype;
typedef int16_t spat_fil_inter_dtype;
typedef int32_t spat_fil_accum_dtype;
typedef int16_t spat_fil_output_dtype;


#define DWT2_COEFF_UPSHIFT 7
#define DWT2_INTER_SHIFT   8  //Shifting to make the intermediate have Q16 format
#define DWT2_OUT_SHIFT     7  //Shifting to make the output have Q16 format
//#define DWT2_COEFF_UPSHIFT 15
//#define DWT2_INTER_SHIFT   16  //Shifting to make the intermediate have Q16 format
//#define DWT2_OUT_SHIFT     15  //Shifting to make the output have Q16 format

typedef int16_t dwt2_dtype;
typedef int32_t dwt2_accum_dtype;
typedef int16_t dwt2_inter_dtype;

/// Structures definitions
typedef struct dwt2buffers {
    funque_dtype *bands[4];
    int width;
    int height;
}dwt2buffers;

typedef struct i_dwt2buffers {
    dwt2_dtype *bands[4];
    int width;
    int height;
}i_dwt2buffers;

// typedef struct i_dwt2buffers_mod {
    // dwt2_dtype *bands0;
    // dwt2_dtype *bands1;
    // dwt2_dtype *bands2;
    // dwt2_dtype *bands3;
    // int width;
    // int height;
// }i_dwt2buffers_mod;

/// C Neon function declaration
void integer_funque_dwt2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);

/// Neon function declaration
void integer_funque_dwt2_neon( spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);
