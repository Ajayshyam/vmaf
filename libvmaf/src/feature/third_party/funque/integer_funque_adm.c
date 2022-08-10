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

#define EXTRA_SAMPLE_BORDER 1
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

static inline int clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

void integer_reflect_pad_adm(const adm_u16_dtype *src, size_t width, size_t height, int reflect, adm_u16_dtype *dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;
    
    for (size_t i = reflect; i != (out_height - reflect); i++)
    {
        for (int j = 0; j != reflect; j++)
        {
          dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }
        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(adm_u16_dtype) * width);
    
        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }
    
    for (int i = 0; i != reflect; i++)
    {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(adm_u16_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(adm_u16_dtype) * out_width);
    }
}

static inline adm_horz_integralsum(int row_offset, int k, size_t r_width_p1, 
                                   int64_t *num_sum, adm_i32_dtype *interim_x, 
                                   int32_t *x_pad, int xpad_i, int index, 
                                   i_dwt2buffers pyr_1, int extra_sample_w)
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
    for (size_t j=1; j<k+1; j++)
    {
        interim_sum = interim_sum + interim_x[j];
    }

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
            masked_pyr = (adm_i32_dtype)clip(val, 0.0, val);
            num_cube1 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[0] += ((num_cube1 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
            
            pyr_abs = abs((adm_i32_dtype)pyr_1.bands[2][index]) * 30;
            val = pyr_abs - masking_threshold;
            masked_pyr = (adm_i32_dtype)clip(val, 0.0, val);
            num_cube2 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[1] += ((num_cube2 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
            
            pyr_abs = abs((adm_i32_dtype)pyr_1.bands[3][index]) * 30;
            val = pyr_abs - masking_threshold;
            masked_pyr = (adm_i32_dtype)clip(val, 0.0, val);
            num_cube3 = (int64_t) masked_pyr * masked_pyr * masked_pyr;
            num_sum[2] += ((num_cube3 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
        }
        index++;
    }
    //Removing the last column i.e. padded column cube value
    if(extra_sample_w)
    {
        num_sum[0] -= ((num_cube1 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT);
        num_sum[1] -= ((num_cube2 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT);
        num_sum[2] -= ((num_cube3 + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT);
    }
}

void integer_integral_image_adm_sums(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                                     int stride, int width, int height, 
                                     adm_i32_dtype *interim_x, float border_size, double *adm_score_num)
{    
    int i, j, index;
    adm_i32_dtype pyr_abs;
    int64_t num_sum[3] = {0};
    double accum_num[3] = {0};
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/
    int x_reflect = (int)((k - stride) / 2);
	int border_h = (border_size * height);
    int border_w = (border_size * width);
    int loop_h, loop_w, dlm_width, dlm_height;
	int extra_sample_h = 0, extra_sample_w = 0;
	
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/	
	
#if EXTRA_SAMPLE_BORDER
	// add one sample on the boundary to account for integral image calculation
	if(border_h)
		extra_sample_h = 1; 
	
	if(border_w)
		extra_sample_w = 1; 
#endif

	border_h -= extra_sample_h;
	border_w -= extra_sample_w;
	
    loop_h = height - border_h;
    loop_w = width - border_w;
	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);
    
    size_t r_width = dlm_width + (2 * x_reflect);
    size_t r_height = dlm_height + (2 * x_reflect);
    size_t r_width_p1 = r_width + 1;
    int xpad_i;

    memset(interim_x, 0, r_width_p1 * sizeof(adm_i32_dtype));
    for (size_t i=1; i<k+1; i++)
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
        for (size_t j=1; j<r_width_p1; j++)
        {
            interim_x[j] = interim_x[j] + x_pad[src_offset + j - 1];
        }
    }
    /**
     * The integral score is used from kxk offset of 2D array
     * Hence horizontal summation of 1st k rows are not used, hence that compuattion is avoided
     */
    int row_offset = k * r_width_p1;
    xpad_i = r_width + 1;
    index = 0;
    //The numerator score is not accumulated for the first row
    adm_horz_integralsum(row_offset, k, r_width_p1, num_sum, interim_x, 
                            x_pad, xpad_i, index, pyr_1, extra_sample_w);
    if(!extra_sample_h)
    {
        accum_num[0] += num_sum[0];
        accum_num[1] += num_sum[1];
        accum_num[2] += num_sum[2];
    }

    for (size_t i=k+1; i<r_height+1; i++)
    {
        row_offset = i * r_width_p1;
        int src_offset = (i-1) * r_width;
        int pre_k_src_offset = (i-1-k) * r_width;
        /**
         * This loop is similar to the loop across columns seen in 1st for loop
         * In this loop the pixels are summated vertically and stored in interim buffer
         * The interim buffer is of size 1 row
         * inter_sum = prev_inter_sum + cur_pixel_val - prev_k-row_pixel_val
         */
        for (size_t j=1; j<r_width_p1; j++)
        {
            interim_x[j] = interim_x[j] + x_pad[src_offset + j - 1] - x_pad[pre_k_src_offset + j - 1];
        }
        xpad_i = (i+1-k)*(r_width) + 1;
        index = (i-k) * dlm_width;
        //horizontal summation & numerator score accumulation
        num_sum[0] = 0;
        num_sum[1] = 0;
        num_sum[2] = 0;
        adm_horz_integralsum(row_offset, k, r_width_p1, num_sum, interim_x, 
                                x_pad, xpad_i, index, pyr_1, extra_sample_w);
        accum_num[0] += num_sum[0];
        accum_num[1] += num_sum[1];
        accum_num[2] += num_sum[2];
    }
    //Removing the row sum
    if(extra_sample_h)
    {
        accum_num[0] -= num_sum[0];
        accum_num[1] -= num_sum[1];
        accum_num[2] -= num_sum[2];
    }
    double num_band = 0;
    for(int band=1; band<4; band++)
    {
        num_band += powf(accum_num[band-1], 1.0/3.0);
    }
    *adm_score_num = num_band + 1e-4;
}

void integer_dlm_contrast_mask_one_way(i_dwt2buffers pyr_1, int32_t *pyr_2, size_t width, size_t height, float border_size, double *adm_score_num)
{
    int k;

    adm_i32_dtype *interim_x = (adm_i32_dtype *)malloc((width + K_INTEGRALIMG_ADM) * sizeof(adm_i32_dtype));

    integer_integral_image_adm_sums(pyr_1, pyr_2, K_INTEGRALIMG_ADM, 1, width, height, interim_x, border_size, adm_score_num);

    free(interim_x);
}

void integer_dlm_decouple(i_dwt2buffers ref, i_dwt2buffers dist, 
                          i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add, 
                          int32_t *adm_div_lookup, float border_size, double *adm_score_den)
{
    const float cos_1deg_sq = COS_1DEG_SQ;
    size_t width = ref.width;
    size_t height = ref.height;
    int i, j, k, index, addIndex,restIndex;
    
    adm_i16_dtype tmp_val;
    int angle_flag;
    
    adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
    int border_h = (border_size * height);
    int border_w = (border_size * width);
	int64_t den_sum[3] = {0};
    int64_t den_row_sum[3] = {0};
    int64_t col0_ref_cube[3] = {0};
    int loop_h, loop_w, dlm_width, dlm_height;
	int extra_sample_h = 0, extra_sample_w = 0;
	adm_i64_dtype den_cube[3];

	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/	
#if EXTRA_SAMPLE_BORDER	
	// add one sample on the boundary to account for integral image calculation
	if(border_h)
		extra_sample_h = 1; 
	
	if(border_w)
		extra_sample_w = 1; 
#endif
	
	border_h -= extra_sample_h;
	border_w -= extra_sample_w;
	
    loop_h = height - border_h;
    loop_w = width - border_w;
	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);
	
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
            addIndex = (i + 1 - border_h) * (dlm_width + 2) + j + 1 - border_w;
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
			addIndex = (i + 1 - border_h) * (dlm_width + 2);
			i_dlm_add[addIndex + 0] = i_dlm_add[addIndex + 2];
			i_dlm_add[addIndex + dlm_width + 1] = i_dlm_add[addIndex + dlm_width - 1];
		}			
    }

	if(!extra_sample_h)
	{
		int row2Idx = 2 * (dlm_width + 2);
		int rowLast2Idx = (dlm_height - 1) * (dlm_width + 2);
		int rowLastPadIdx = (dlm_height + 1) * (dlm_width + 2);

		memcpy(&i_dlm_add[0], &i_dlm_add[row2Idx], sizeof(int32_t) * (dlm_width + 2));
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

int integer_compute_adm_funque(i_dwt2buffers i_ref, i_dwt2buffers i_dist, double *adm_score, double *adm_score_num, double *adm_score_den, size_t width, size_t height, float border_size, int16_t shift_val, int32_t *adm_div_lookup)
{
    int i, j, k, index;
    adm_i64_dtype num_sum = 0, den_sum = 0;
    adm_i32_dtype ref_abs;
    adm_i64_dtype num_cube = 0, den_cube = 0;
    double num_band = 0, den_band = 0;
    i_dwt2buffers i_dlm_rest;
    adm_i32_dtype *i_dlm_add;
	int border_h = (border_size * height);
    int border_w = (border_size * width);
	int loop_h, loop_w, dlm_width, dlm_height;
	int extra_sample_h = 0, extra_sample_w = 0;
	
	/**
	DLM has the configurability of computing the metric only for the
	centre region. currently border_size defines the percentage of pixels to be avoided
	from all sides so that size of centre region is defined.
	
	*/	
	
	// add one sample on the boundary to account for integral image calculation
	if(border_h)
		extra_sample_h = 1;
	
	if(border_w)
		extra_sample_w = 1;
	
	border_h -= extra_sample_h;
	border_w -= extra_sample_w;
	
    loop_h = height - border_h;
    loop_w = width - border_w;
	
	dlm_height = height - (border_h << 1);
	dlm_width = width - (border_w << 1);
	
    i_dlm_rest.bands[1] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);
    i_dlm_rest.bands[2] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);
    i_dlm_rest.bands[3] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * dlm_height * dlm_width);
    i_dlm_add = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * (dlm_height+2) * (dlm_width+2));
    
    double row_num, accum_num = 0;

    integer_dlm_decouple(i_ref, i_dist, i_dlm_rest, i_dlm_add, adm_div_lookup, border_size, adm_score_den);
    
    integer_dlm_contrast_mask_one_way(i_dlm_rest, i_dlm_add, width, height, border_size, adm_score_num);

    *adm_score = (*adm_score_num) / (*adm_score_den);

    for (int i = 1; i < 4; i++)
    {
        free(i_dlm_rest.bands[i]);
    }
    free(i_dlm_add);
    int ret = 0;
    return ret;
}
