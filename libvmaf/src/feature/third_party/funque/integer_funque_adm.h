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
#ifndef FEATURE_ADM_H_
#define FEATURE_ADM_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <stdlib.h>

#include <string.h>
#include <math.h>

#include "integer_funque_filters.h"

typedef int16_t adm_i16_dtype;
typedef int32_t adm_i32_dtype;
typedef uint16_t adm_u16_dtype;
typedef int64_t adm_i64_dtype;

typedef struct i_adm_buffers
{
  adm_i32_dtype *bands[4];
  int width;
  int height;
} i_adm_buffers;

typedef struct u_adm_buffers
{
    adm_u16_dtype *bands[4];
    int width;
    int height;
} u_adm_buffers;


#define ADM_CUBE_SHIFT 8
#define ADM_CUBE_SHIFT_ROUND (1 << (ADM_CUBE_SHIFT - 1))

#define K_INTEGRALIMG_ADM 3
#define ADM_CUBE_DIV pow(2,ADM_CUBE_SHIFT)

/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

#define SHIFT_ADM_DECOUPLE_FINAL 16

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif
#define COS_1DEG_SQ cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0)

int integer_compute_adm_funque(ModuleFunqueState m, i_dwt2buffers ref, i_dwt2buffers dist, double *adm_score, 
                               double *adm_score_num, double *adm_score_den, size_t width, size_t height, 
                               float border_size, int32_t* adm_div_lookup);
void integer_adm_decouple_c(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den);
void integer_adm_integralimg_numscore_c(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                                     int stride, int width, int height, 
                                     adm_i32_dtype *interim_x, float border_size, double *adm_score_num);
void div_lookup_generator(int32_t* adm_div_lookup);

static inline void adm_horz_integralsum(int k, size_t r_width_p1, 
                                   int64_t *num_sum, adm_i32_dtype *interim_x, 
                                   int32_t *x_pad, int xpad_i, int index, 
                                   i_dwt2buffers pyr_1)
{
    int32_t interim_sum = 0;
    adm_i32_dtype masking_threshold;
    adm_i32_dtype val, pyr_abs;
    int32_t masked_pyr;
    //Initialising first column value to 0
    int32_t sum = 0;

    int64_t num_cube1, num_cube2, num_cube3;
    /**
     * The horizontal accumulation similar to vertical accumulation
     * sum = prev_col_sum + interim_vertical_sum
     * The previous k col interim sum is not subtracted since it is not available here
     */
    for (int j=1; j<k+1; j++)
    {
        interim_sum = interim_sum + interim_x[j];
    }

    sum = interim_sum + x_pad[xpad_i];

    masking_threshold = sum;

    pyr_abs = abs((adm_i32_dtype)pyr_1.bands[1][index]) * 30;
    val = pyr_abs - masking_threshold;
    masked_pyr = (adm_i32_dtype)MAX(val, 0);
    num_cube1 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
    num_sum[0] += ((num_cube1 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
    
    pyr_abs = abs((adm_i32_dtype)pyr_1.bands[2][index]) * 30;
    val = pyr_abs - masking_threshold;
    masked_pyr = (adm_i32_dtype)MAX(val, 0);
    num_cube2 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
    num_sum[1] += ((num_cube2 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
    
    pyr_abs = abs((adm_i32_dtype)pyr_1.bands[3][index]) * 30;
    val = pyr_abs - masking_threshold;
    masked_pyr = (adm_i32_dtype)MAX(val, 0);
    num_cube3 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
    num_sum[2] += ((num_cube3 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63

    //The sum is needed only from the k+1 column, hence not computed here
    index++; 

    //Similar to prev loop, but previous k col interim metric sum is subtracted
    for (size_t j=k+1; j<r_width_p1; j++)
    {
        interim_sum = interim_sum + interim_x[j] - interim_x[j - k];
        sum = interim_sum + x_pad[xpad_i + j - k];

        {
            masking_threshold = sum;

            pyr_abs = abs((adm_i32_dtype)pyr_1.bands[1][index]) * 30;
            val = pyr_abs - masking_threshold;
            masked_pyr = (adm_i32_dtype)MAX(val, 0);
            num_cube1 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[0] += ((num_cube1 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
            
            pyr_abs = abs((adm_i32_dtype)pyr_1.bands[2][index]) * 30;
            val = pyr_abs - masking_threshold;
            masked_pyr = (adm_i32_dtype)MAX(val, 0);
            num_cube2 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[1] += ((num_cube2 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
            
            pyr_abs = abs((adm_i32_dtype)pyr_1.bands[3][index]) * 30;
            val = pyr_abs - masking_threshold;
            masked_pyr = (adm_i32_dtype)MAX(val, 0);
            num_cube3 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[2] += ((num_cube3 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
        }
        index++;
    }
}

#endif /* _FEATURE_ADM_H_ */