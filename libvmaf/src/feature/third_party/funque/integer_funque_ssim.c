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

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "integer_funque_filters.h"
#include "integer_funque_ssim.h"
#include "funque_ssim_options.h"
// #define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline int16_t get_best_i16_from_u64(uint64_t temp, int *power)
{
    assert(temp >= 0x20000);
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t) temp;
}

int integer_compute_ssim_funque(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;

    int width = ref->width;
    int height = ref->height;

    /**
     * C1 is constant is added to ref^2, dist^2, 
     *  - hence we have to multiply by pending_div^2
     * As per floating point,C1 is added to 2*(mx/win_dim)*(my/win_dim) & (mx/win_dim)*(mx/win_dim)+(my/win_dim)*(my/win_dim)
     * win_dim = 1 << n_levels, where n_levels = 1
     * Since win_dim division is avoided for mx & my, C1 is left shifted by 1
     */
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div*pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    /**
     * shifts are handled similar to C1
     * not shifted left because the other terms to which this is added undergoes equivalent right shift 
     */
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div*pending_div) >> (SSIM_INTER_VAR_SHIFTS+SSIM_INTER_CS_SHIFT-2)));
    
    ssim_inter_dtype var_x, var_y, cov_xy;
    ssim_inter_dtype map;
    ssim_accum_dtype map_num;
    ssim_accum_dtype map_den;
    int16_t i16_map_den;
    dwt2_dtype mx, my;
    ssim_inter_dtype var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype l_num, l_den, cs_num, cs_den;

#if ENABLE_MINK3POOL
    ssim_accum_dtype rowcube_1minus_map = 0;
    double accumcube_1minus_map = 0;
    const ssim_inter_dtype const_1 = 32768;  //div_Q_factor>>SSIM_SHIFT_DIV
#else
    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_map_sq = 0;
    ssim_accum_dtype map_sq_insum = 0;
#endif

    int index = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            index = i * width + j;

            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x  = 0;
            var_y  = 0;
            cov_xy = 0;

            for (int k = 1; k < 4; k++)
            {
                var_x  += ((ssim_inter_dtype)ref->bands[k][index]  * ref->bands[k][index]);
                var_y  += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index]  * dist->bands[k][index]);
            }
            var_x_band0  = (ssim_inter_dtype)mx * mx;
            var_y_band0  = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;

            /**
             * ref, dist in Q15 with range [-23875, 23875]
             * num_bands = 3(for accumulation) where max value requires 31 bits
             * Since additions are present in next stage right shifted by SSIM_INTER_VAR_SHIFTS
            */
            var_x  = (var_x  >> SSIM_INTER_VAR_SHIFTS);
            var_y  = (var_y  >> SSIM_INTER_VAR_SHIFTS);
            cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

            //l = (2*mx*my + C1) / (mx*mx + my*my + C1)
            // Splitting this into 2 variables l_num, l_den
            // l_num = (2*mx*my)>>1 + C1
            //This is because, 2*mx*my takes full 32 bits (mx holds 16bits-> 1sign 15bit for value)
            //After mul, mx*my takes 31bits including sign
            //Hence 2*mx*my takes full 32 bits, for addition with C1 right shifted by 1
            l_num = ((2>>SSIM_INTER_L_SHIFT)*cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0)>>SSIM_INTER_L_SHIFT) + C1);

            //cs = (2*cov_xy+C2)/(var_x+var_y+C2)
            //Similar to l, cs is split to cs_num cs_den
            //One extra shift is done here since more accumulation is happening across bands
            //Hence the extra left shift is avoided for C2 unlike C1
            cs_num = ((2>>SSIM_INTER_CS_SHIFT)*cov_xy+C2);
            cs_den = (((var_x+var_y)>>SSIM_INTER_CS_SHIFT)+C2);
            
            map_num = (ssim_accum_dtype)l_num * cs_num;
            map_den = (ssim_accum_dtype)l_den * cs_den;
            
            /**
             * l_den & cs_den are variance terms, hence they will always be +ve 
             * getting best 15bits and retaining one signed bit, using get_best_i16_from_u64
             * This is done to reuse ADM division LUT, which has LUT for values from -2^15 to 2^15
            */
            int power_val;
            i16_map_den = get_best_i16_from_u64((uint64_t) map_den, &power_val);
            /**
             * The actual equation of map is map_num/map_den
             * The division is done using LUT, results of div_lookup = 2^30/i16_map_den
             * map = map_num/map_den => map = map_num / (i16_map_den << power_val)
             * => map = (map_num >> power_val) / i16_map_den
             * => map = (map_num >> power_val) * (div_lookup[i16_map_den + 32768] >> 30) //since it has -ve vals in 1st half
             * => map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> 30
             * Shift by 30 might be very high even for 32 bits precision, hence shift only by 15 
            */
            map = ((map_num >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;
#if ENABLE_MINK3POOL
            ssim_accum_dtype const1_minus_map = const_1 - map;
            rowcube_1minus_map += const1_minus_map * const1_minus_map * const1_minus_map;
#else
            accum_map += map;
            map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype) map * map));
#endif
        }
#if ENABLE_MINK3POOL
        accumcube_1minus_map += (double) rowcube_1minus_map;
        rowcube_1minus_map = 0;
#endif
    }

#if ENABLE_MINK3POOL
    double ssim_val = 1 - cbrt(accumcube_1minus_map/(width*height))/const_1;
    *score = ssim_clip(ssim_val, 0, 1);
#else
    accum_map_sq = map_sq_insum / (height * width);

    double ssim_mean = (double)accum_map / (height * width);


    double ssim_std; 
    ssim_std = sqrt(MAX(0, ((double) accum_map_sq - ssim_mean*ssim_mean)));

    *score = (ssim_std / ssim_mean);
#endif
    ret = 0;

    return ret;
}