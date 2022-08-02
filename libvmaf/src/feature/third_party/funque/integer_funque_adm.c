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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "integer_funque_adm.h"
#include "mem.h"
#include "adm_tools.h"
#include "integer_funque_filters.h"

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

void integer_integral_image_adm_sums(i_dwt2buffers pyr_1, adm_u16_dtype *x, int k, int stride, i_adm_buffers masked_pyr, int width, int height, int band_index)
{
    adm_u16_dtype *x_pad;
    adm_i64_dtype *sum;
    adm_i64_dtype *temp_sum;
    int i, j, index;
    adm_i32_dtype pyr_abs;
    
    int x_reflect = (int)((k - stride) / 2);
    
    x_pad = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    
    integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);
    
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    size_t int_stride = r_width + 1;
    
    sum = (adm_i64_dtype *)malloc((r_width + 1) * (r_height + 1) * sizeof(adm_i64_dtype));
    temp_sum = (adm_i64_dtype *)malloc((r_width + 1) * (r_height + 1) * sizeof(adm_i64_dtype));
	/*
	** Setting the first row values to 0
	*/
    memset(sum, 0, int_stride * sizeof(adm_i64_dtype));

    for (size_t i = 1; i < (k + 1); i++)
    {
        temp_sum[i * int_stride] = 0; // Setting the first column value to 0
        for (size_t j = 1; j < (k + 1); j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)];
        }
        for (size_t j = k + 1; j < int_stride; j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)] - x_pad[(i - 1) * r_width + j - k - 1];
        }
        for (size_t j = 1; j < int_stride; j++)
        {
        	sum[i * int_stride + j] = temp_sum[i * int_stride + j] + sum[(i - 1) * int_stride + j];
        }
    }
  
    for (size_t i = (k + 1); i < (r_height + 1); i++)
    {
        temp_sum[i * int_stride] = 0; // Setting the first column value to 0
        for (size_t j = 1; j < (k + 1); j++)
        {
           temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)];
        }
        for (size_t j = k + 1; j < int_stride; j++)
        {
            temp_sum[i * int_stride + j] = temp_sum[i * int_stride + j - 1] + x_pad[(i - 1) * r_width + (j - 1)] - x_pad[(i - 1) * r_width + j - k - 1];
        }
        for (size_t j = 1; j < int_stride; j++)
        {
        	sum[i * int_stride + j] = temp_sum[i * int_stride + j] + sum[(i - 1) * int_stride + j] - temp_sum[(i - k) * int_stride + j];
        }
    }
    /*
	** For band 1 loop the pyr_1 value is multiplied by 
	** 30 to avaoid the precision loss that would happen 
	** due to the division by 30 of masking_threshold
	*/
    if(band_index == 1)
    {
        for (i = 0; i < height; i++)
        {
        	for (j = 0; j < width; j++)
        	{
        		adm_i32_dtype masking_threshold;
        		adm_i32_dtype val;
        		index = i * width + j;
        		masking_threshold = (adm_i32_dtype)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
        		pyr_abs = abs((adm_i32_dtype)pyr_1.bands[1][index]) * 30;
        		val = pyr_abs - masking_threshold;
        		masked_pyr.bands[1][index] = val;
        		pyr_abs = abs((adm_i32_dtype)pyr_1.bands[2][index]) * 30;
        		val = pyr_abs - masking_threshold;
        		masked_pyr.bands[2][index] = val;
        		pyr_abs = abs((adm_i32_dtype)pyr_1.bands[3][index]) * 30;
        		val = pyr_abs - masking_threshold;
        		masked_pyr.bands[3][index] = val;
        	}
        }
    }
	
    if(band_index == 2)
    {
	    for (i = 0; i < height; i++)
	    {
	    	for (j = 0; j < width; j++)
	    	{
	    		adm_i32_dtype masking_threshold;
	    		adm_i32_dtype val;
	    		index = i * width + j;
	    		masking_threshold = (adm_i32_dtype)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
	    		val = masked_pyr.bands[1][index] - masking_threshold;
	    		masked_pyr.bands[1][index] = val;
	    		val = masked_pyr.bands[2][index] - masking_threshold;
	    		masked_pyr.bands[2][index] = val;
	    		val = masked_pyr.bands[3][index] - masking_threshold;
	    		masked_pyr.bands[3][index] = val;
	    	}
	    }
    }
	/*
	** For band 3 loop the final value is clipped
	** to minimum of zero.
	*/
    if(band_index == 3)
    {
	    for (i = 0; i < height; i++)
	    {
	    	for (j = 0; j < width; j++)
	    	{
	    		adm_i32_dtype masking_threshold;
	    		adm_i32_dtype val;
	    		index = i * width + j;
	    		masking_threshold = (adm_i32_dtype)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
	    		val = masked_pyr.bands[1][index] - masking_threshold;
	    		masked_pyr.bands[1][index] = (adm_i32_dtype)clip(val, 0.0, val);
	    		val = masked_pyr.bands[2][index] - masking_threshold;
	    		masked_pyr.bands[2][index] = (adm_i32_dtype)clip(val, 0.0, val);
	    		val = masked_pyr.bands[3][index] - masking_threshold;
	    		masked_pyr.bands[3][index] = (adm_i32_dtype)clip(val, 0.0, val);
	    	}
	    }
    }
    
    free(temp_sum);
    free(sum);
    free(x_pad);
}

void integer_dlm_contrast_mask_one_way(i_dwt2buffers pyr_1, u_adm_buffers pyr_2, i_adm_buffers masked_pyr, size_t width, size_t height)
{
    int k;
    
    for (k = 1; k < 4; k++)
    {
        integer_integral_image_adm_sums(pyr_1, pyr_2.bands[k], 3, 1, masked_pyr, width, height, k);
    }
}

void integer_dlm_decouple(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, u_adm_buffers i_dlm_add, int32_t *adm_div_lookup)
{
    const float cos_1deg_sq = COS_1DEG_SQ;
    size_t width = ref.width;
    size_t height = ref.height;
    int i, j, k, index;
    
    adm_i16_dtype tmp_val;
    int angle_flag;
    
    adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
    
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            index = i * width + j;
            ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
            o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
            t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);
            angle_flag = ((ot_dp >= 0) && (((adm_i64_dtype)ot_dp * ot_dp) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_sq * t_mag_sq)));
            
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
                
                i_dlm_rest.bands[k][index] = angle_flag ? dist.bands[k][index] : tmp_val;
                /**
                 * Absolute is taken here for the difference value instead of 
                 * taking absolute of pyr_2 in integer_dlm_contrast_mask_one_way function
                 */
                i_dlm_add.bands[k][index] = abs(dist.bands[k][index] - i_dlm_rest.bands[k][index]);
            }
        }
    }
}

int integer_compute_adm_funque(i_dwt2buffers i_ref, i_dwt2buffers i_dist, double *adm_score, double *adm_score_num, double *adm_score_den, size_t width, size_t height, float border_size, int16_t shift_val, int32_t *adm_div_lookup)
{
    int i, j, k, index;
    adm_i64_dtype num_sum = 0, den_sum = 0;
    adm_i32_dtype ref_abs;
    adm_i64_dtype num_cube = 0, den_cube = 0;
    double num_band = 0, den_band = 0;
    i_dwt2buffers i_dlm_rest;
    u_adm_buffers i_dlm_add;
    i_adm_buffers i_pyr_rest;
    i_dlm_rest.bands[1] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
    i_dlm_rest.bands[2] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
    i_dlm_rest.bands[3] = (adm_i16_dtype *)malloc(sizeof(adm_i16_dtype) * height * width);
    i_dlm_add.bands[1]  = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
    i_dlm_add.bands[2]  = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
    i_dlm_add.bands[3]  = (adm_u16_dtype *)malloc(sizeof(adm_u16_dtype) * height * width);
    i_pyr_rest.bands[1] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);
    i_pyr_rest.bands[2] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);
    i_pyr_rest.bands[3] = (adm_i32_dtype *)malloc(sizeof(adm_i32_dtype) * height * width);

    integer_dlm_decouple(i_ref, i_dist, i_dlm_rest, i_dlm_add, adm_div_lookup);
    
    integer_dlm_contrast_mask_one_way(i_dlm_rest, i_dlm_add, i_pyr_rest, width, height);
    
    int border_h = (border_size * height);
    int border_w = (border_size * width);
    int loop_h = height - border_h;
    int loop_w = width - border_w;
    double row_num, row_den, accum_num = 0, accum_den = 0;

    for (k = 1; k < 4; k++)
    {
        for (i = border_h; i < loop_h; i++)
        {
            for (j = border_w; j < loop_w; j++)
            {
                index = i * width + j;
                num_cube = (adm_i64_dtype)i_pyr_rest.bands[k][index] * i_pyr_rest.bands[k][index] * i_pyr_rest.bands[k][index];
                num_sum += ((num_cube + ADM_CUBE_SHIFT_ROUND) >> ADM_CUBE_SHIFT); // reducing precision from 71 to 63
                ref_abs = abs((adm_i64_dtype)i_ref.bands[k][index]);
                den_cube = (adm_i64_dtype)ref_abs * ref_abs * ref_abs;
                den_sum += den_cube;
            }
            row_num = (double)num_sum;
            row_den = (double)den_sum;
            accum_num += row_num;
            accum_den += row_den;
            num_sum = 0;
            den_sum = 0;
        }
        accum_den = accum_den / ADM_CUBE_DIV;
        den_band += powf((double)(accum_den), 1.0 / 3.0);
        num_band += powf((double)(accum_num), 1.0 / 3.0);
        accum_num = 0;
        accum_den = 0;
    }
    
    *adm_score_num = num_band + 1e-4;
    // compensation for the division by thirty in the numerator
    *adm_score_den = (den_band * 30) + 1e-4;
    *adm_score = (*adm_score_num) / (*adm_score_den);
    
    for (int i = 1; i < 4; i++)
    {
        free(i_dlm_rest.bands[i]);
        free(i_dlm_add.bands[i]);
        free(i_pyr_rest.bands[i]);
    }

    int ret = 0;
    return ret;
}
