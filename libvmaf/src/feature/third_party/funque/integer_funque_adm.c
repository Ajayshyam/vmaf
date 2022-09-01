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
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "integer_funque_adm.h"
#include "mem.h"
#include "adm_tools.h"
#include "integer_funque_filters.h"

static const int32_t div_Q_factor = 1073741824; // 2^30

void div_lookup_generator(int32_t *adm_div_lookup)
{
    for (int i = 1; i <= 32768; ++i)
    {
        int32_t recip = (int32_t)(div_Q_factor / i);
        adm_div_lookup[32768 + i] = recip;
        adm_div_lookup[32768 - i] = 0 - recip;
    }
}

void integer_adm_integralimg_numscore_c(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                                     int stride, int width, int height, 
                                     adm_i32_dtype *interim_x, float border_size, double *adm_score_num)
{    
    int i, j, index;

    int64_t num_sum[3] = {0};
    double accum_num[3] = {0};
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/
    int x_reflect = (int)((k - stride) / 2) * ADM_REFLECT_PAD;
	int border_h = (border_size * height);
    int border_w = (border_size * width);
    // int loop_h, loop_w; 
    int dlm_width, dlm_height;

	int extra_sample_h = 0, extra_sample_w = 0;
	
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.

	*/	
#if ADM_REFLECT_PAD
    extra_sample_w = 0;
    extra_sample_h = 0;
#else
    extra_sample_w = 1;
    extra_sample_h = 1;
#endif

	border_h -= extra_sample_h;
	border_w -= extra_sample_w;

#if !ADM_REFLECT_PAD
    //If reflect pad is disabled & if border_size is 0, process 1 row,col pixels lesser
    border_h = MAX(1,border_h);
    border_w = MAX(1,border_w);
#endif
	
	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);
    
    int r_width = dlm_width + (2 * x_reflect);
    int r_height = dlm_height + (2 * x_reflect);
    int r_width_p1 = r_width + 1;
    int xpad_i;

    memset(interim_x, 0, r_width_p1 * sizeof(adm_i32_dtype));
    for (i=1; i<k+1; i++)
    {
        int src_offset = (i-1) * r_width;
        /**
         * In this loop the pixels are summated vertically and stored in interim buffer
         * The interim buffer is of size 1 row
         * inter_sum = prev_inter_sum + cur_pixel_val
         * 
         * where inter_sum will have vertical pixel sums, 
         * prev_inter_sum will have prev rows inter_sum and 
         * The previous k row metric val is not subtracted since it is not available here 
         */

        for (j=1; j<r_width_p1; j++)
        {
            interim_x[j] = interim_x[j] + x_pad[src_offset + j - 1];
        }
    }
    /**
     * The integral score is used from kxk offset of 2D array
     * Hence horizontal summation of 1st k rows are not used, hence that compuattion is avoided
     */

    // int row_offset = k * r_width_p1;
    xpad_i = r_width + 1;
    //When padding is disabled extra row, col would be available, 
    //which should not be used for score computation
    index = (extra_sample_h) * dlm_width + extra_sample_w;
    //The numerator score is not accumulated for the first row
    adm_horz_integralsum(k, r_width_p1, num_sum, interim_x, 
                            x_pad, xpad_i, index, pyr_1);

    accum_num[0] += num_sum[0];
    accum_num[1] += num_sum[1];
    accum_num[2] += num_sum[2];

    for (i=k+1; i<r_height+1; i++)
    {
        // row_offset = i * r_width_p1;
        int src_offset = (i-1) * r_width;
        int pre_k_src_offset = (i-1-k) * r_width;
        /**
         * This loop is similar to the loop across columns seen in 1st for loop
         * In this loop the pixels are summated vertically and stored in interim buffer
         * The interim buffer is of size 1 row
         * inter_sum = prev_inter_sum + cur_pixel_val - prev_k-row_pixel_val
         */
        for (j=1; j<r_width_p1; j++)
        {
            interim_x[j] = interim_x[j] + x_pad[src_offset + j - 1] - x_pad[pre_k_src_offset + j - 1];
        }
        xpad_i = (i+1-k)*(r_width) + 1;
        
        index = (i+extra_sample_h-k) * dlm_width + extra_sample_w;

        //horizontal summation & numerator score accumulation
        num_sum[0] = 0;
        num_sum[1] = 0;
        num_sum[2] = 0;

        adm_horz_integralsum(k, r_width_p1, num_sum, interim_x, 
                                x_pad, xpad_i, index, pyr_1);
        accum_num[0] += num_sum[0];
        accum_num[1] += num_sum[1];
        accum_num[2] += num_sum[2];
    }

    double num_band = 0;
    for(int band=1; band<4; band++)
    {
        num_band += powf(accum_num[band-1], 1.0/3.0);
    }
    *adm_score_num = num_band + 1e-4;
}

void integer_adm_decouple_c(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den)
{
    // const float cos_1deg_sq = COS_1DEG_SQ;

    size_t width = ref.width;
    size_t height = ref.height;
    int i, j, k, index, addIndex,restIndex;
    
    adm_i16_dtype tmp_val;
    int angle_flag;
    
    adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
    int border_h = (border_size * height);
    int border_w = (border_size * width);
	double den_sum[3] = {0};
    int64_t den_row_sum[3] = {0};
    int64_t col0_ref_cube[3] = {0};
    int loop_h, loop_w, dlm_width, dlm_height;
	int extra_sample_h = 0, extra_sample_w = 0;

	adm_i64_dtype den_cube[3] = {0};


	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	*/
#if ADM_REFLECT_PAD
    extra_sample_w = 0;
    extra_sample_h = 0;
#else
    extra_sample_w = 1;
    extra_sample_h = 1;

#endif
	
	border_h -= extra_sample_h;
	border_w -= extra_sample_w;

#if !ADM_REFLECT_PAD
    //If reflect pad is disabled & if border_size is 0, process 1 row,col pixels lesser
    border_h = MAX(1,border_h);
    border_w = MAX(1,border_w);
#endif

    loop_h = height - border_h;
    loop_w = width - border_w;
	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);

	//The width of i_dlm_add buffer will be extra only if padding is enabled
    int dlm_add_w = dlm_width  + (ADM_REFLECT_PAD << 1);
    // int dlm_add_h = dlm_height + (ADM_REFLECT_PAD << 1);

    for (i = border_h; i < loop_h; i++)
    {
        if(extra_sample_w)
        {
            for(k=1; k<4; k++)
            {
                int16_t ref_abs = abs(ref.bands[k][i*width + border_w]);
                col0_ref_cube[k-1] = (int64_t) ref_abs * ref_abs * ref_abs;
            }
        }
        for (j = border_w; j < loop_w; j++)
        {
            index = i * width + j;

            //If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;

			restIndex = (i - border_h) * (dlm_width) + j - border_w;
            ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
            o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
            t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);
            angle_flag = ((ot_dp >= 0) && (((adm_i64_dtype)ot_dp * ot_dp) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_sq * t_mag_sq)));
            i_dlm_add[addIndex] = 0;
            for (k = 1; k < 4; k++)
            {
                /**
                 * Division dist/ref is carried using lookup table adm_div_lookup and converted to multiplication
                 */
                adm_i32_dtype tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((adm_i64_dtype)adm_div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
                adm_u16_dtype kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
                /**
                 * kh is in Q15 type and ref.bands[k][index] is in Q16 type hence shifted by
                 * 15 to make result Q16
                 */
                tmp_val = (((adm_i32_dtype)kh * ref.bands[k][index]) + 16384) >> 15;
                
                i_dlm_rest.bands[k][restIndex] = angle_flag ? dist.bands[k][index] : tmp_val;
                /**
                 * Absolute is taken here for the difference value instead of 
                 * taking absolute of pyr_2 in integer_dlm_contrast_mask_one_way function
                 */
                i_dlm_add[addIndex] += (int32_t)abs(dist.bands[k][index] - i_dlm_rest.bands[k][restIndex]);

                //Accumulating denominator score to avoid load in next stage
                int16_t ref_abs = abs(ref.bands[k][index]);
                den_cube[k-1] = (adm_i64_dtype)ref_abs * ref_abs * ref_abs;
                
                den_row_sum[k-1] += den_cube[k-1];
            }
        }
        if(extra_sample_w)
        {
            for(k = 0; k < 3; k++)
            {
                den_row_sum[k] -= den_cube[k];
                den_row_sum[k] -= col0_ref_cube[k];
            }
        }
        if((i != border_h && i != (loop_h - 1)) || !extra_sample_h)
        {
            for(k=0; k<3; k++)
            {
                den_sum[k] += den_row_sum[k];
            }
        }
        den_row_sum[0] = 0;
        den_row_sum[1] = 0;
        den_row_sum[2] = 0;

        if(!extra_sample_w)
		{
			addIndex = (i + 1 - border_h) * (dlm_add_w);

			i_dlm_add[addIndex + 0] = i_dlm_add[addIndex + 2];
			i_dlm_add[addIndex + dlm_width + 1] = i_dlm_add[addIndex + dlm_width - 1];
		}			
    }

	if(!extra_sample_h)
	{
		int row2Idx = 2 * (dlm_add_w);
		int rowLast2Idx = (dlm_height - 1) * (dlm_add_w);
		int rowLastPadIdx = (dlm_height + 1) * (dlm_add_w);

		memcpy(&i_dlm_add[0], &i_dlm_add[row2Idx], sizeof(int32_t) * (dlm_add_w));

		memcpy(&i_dlm_add[rowLastPadIdx], &i_dlm_add[rowLast2Idx], sizeof(int32_t) * (dlm_width+2));
	}
    
    //Calculating denominator score
    double den_band = 0;
    for(k=0; k<3; k++)
    {
        double accum_den = (double) den_sum[k] / ADM_CUBE_DIV;
        den_band += powf((double)(accum_den), 1.0 / 3.0);
    }
    // compensation for the division by thirty in the numerator
    *adm_score_den = (den_band * 30) + 1e-4;

}


int integer_compute_adm_funque(ModuleFunqueState m, i_dwt2buffers i_ref, i_dwt2buffers i_dist, double *adm_score, double *adm_score_num, double *adm_score_den, size_t width, size_t height, float border_size, int32_t *adm_div_lookup)
{
    i_dwt2buffers i_dlm_rest;
    adm_i32_dtype *i_dlm_add, *interim_x;
	int border_h = (border_size * height);
    int border_w = (border_size * width);
	// int loop_h, loop_w;
    int dlm_width, dlm_height;

	int extra_sample_h = 0, extra_sample_w = 0;
	
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/	
	
	// add one sample on the boundary to account for integral image calculation
#if ADM_REFLECT_PAD
    extra_sample_w = 0;
    extra_sample_h = 0;
#else
    extra_sample_w = 1;
    extra_sample_h = 1;

#endif

	border_h -= extra_sample_h;
	border_w -= extra_sample_w;

#if !ADM_REFLECT_PAD
    //If reflect pad is disabled & if border_size is 0, process 1 row,col pixels lesser
    border_h = MAX(1,border_h);
    border_w = MAX(1,border_w);
#endif

    // loop_h = height - border_h;
    // loop_w = width - border_w;

	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);
	
    i_dlm_rest.bands[1] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);
    i_dlm_rest.bands[2] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);
    i_dlm_rest.bands[3] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);

    i_dlm_add = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * (dlm_height+ (ADM_REFLECT_PAD<<1)) * (dlm_width+(ADM_REFLECT_PAD<<1)));
    interim_x = (adm_i32_dtype *)malloc((width + K_INTEGRALIMG_ADM) * sizeof(adm_i32_dtype));

    // double row_num, accum_num = 0;

    m.integer_funque_adm_decouple(i_ref, i_dist, i_dlm_rest, i_dlm_add, adm_div_lookup, border_size, adm_score_den);

    m.integer_adm_integralimg_numscore(i_dlm_rest, i_dlm_add, K_INTEGRALIMG_ADM, 1, width, height, interim_x, border_size, adm_score_num);


    *adm_score = (*adm_score_num) / (*adm_score_den);

    for (int i = 1; i < 4; i++)
    {
        free(i_dlm_rest.bands[i]);
    }
    free(interim_x);

    free(i_dlm_add);
    int ret = 0;
    return ret;
}
