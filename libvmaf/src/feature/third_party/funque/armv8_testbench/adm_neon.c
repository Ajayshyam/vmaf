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
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <arm_neon.h>

typedef struct i32_adm_buffers
{
  int32_t *bands[4];
  int width;
  int height;
} i32_adm_buffers;

typedef struct u16_adm_buffers
{
    uint16_t *bands[4];
    int width;
    int height;
} u16_adm_buffers;

typedef struct i16_adm_buffers
{
    int16_t *bands[4];
    int width;
    int height;
} i16_adm_buffers;

static inline int clip(int value, int low, int high)
{
    return value < low ? low : (value > high ? high : value);
}

void integer_reflect_pad_adm(const uint16_t *src, size_t width, size_t height, int reflect, uint16_t *dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;
    
    for (size_t i = reflect; i != (out_height - reflect); i++)
    {
        for (int j = 0; j != reflect; j++)
        {
          dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }
        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(uint16_t) * width);
    
        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }
    
    for (int i = 0; i != reflect; i++)
    {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(uint16_t) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(uint16_t) * out_width);
    }
}

void integer_integral_image_adm_sums(i16_adm_buffers pyr_1, uint16_t *x, int k, int stride, i32_adm_buffers masked_pyr, int width, int height, int band_index)
{
    uint16_t *x_pad;
   
    int i, j, index;
    int32_t pyr_abs;
    
    int x_reflect = (int)((k - stride) / 2);
    
    x_pad = (uint16_t *)malloc(sizeof(uint16_t) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    
    integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);
    
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    size_t int_stride = r_width + 1;
    
    //  int64_t *sum;
    // int64_t *temp_sum;
    // sum = (int64_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int64_t));
    // temp_sum = (int64_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int64_t));

    int32_t *sum;
    int32_t *temp_sum;
    sum = (int32_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int32_t));
    temp_sum = (int32_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int32_t));

	/*
	** Setting the first row values to 0
	*/
    memset(sum, 0, int_stride * sizeof(int64_t));

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
        		int32_t masking_threshold;
        		int32_t val;
        		index = i * width + j;
        		masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
        		pyr_abs = abs((int32_t)pyr_1.bands[1][index]) * 30;
                
        		val = pyr_abs - masking_threshold;
                // printf("%d\n",val);
               
        		masked_pyr.bands[1][index] = val;
                 if(index == 100)
                    printf("%d index: %d\n", masked_pyr.bands[1][index], index);
        		pyr_abs = abs((int32_t)pyr_1.bands[2][index]) * 30;
        		val = pyr_abs - masking_threshold;
        		masked_pyr.bands[2][index] = val;
        		pyr_abs = abs((int32_t)pyr_1.bands[3][index]) * 30;
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
	    		int32_t masking_threshold;
	    		int32_t val;
	    		index = i * width + j;
	    		masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
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
	    		int32_t masking_threshold;
	    		int32_t val;
	    		index = i * width + j;
	    		masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
	    		val = masked_pyr.bands[1][index] - masking_threshold;
	    		masked_pyr.bands[1][index] = (int32_t)clip(val, 0.0, val);
	    		val = masked_pyr.bands[2][index] - masking_threshold;
	    		masked_pyr.bands[2][index] = (int32_t)clip(val, 0.0, val);
	    		val = masked_pyr.bands[3][index] - masking_threshold;
	    		masked_pyr.bands[3][index] = (int32_t)clip(val, 0.0, val);
                if(index == 100)
                    printf("%d index_3: %d\n", masked_pyr.bands[1][index], index);
	    	}
	    }
    }
    
    free(temp_sum);
    free(sum);
    free(x_pad);
}


void integer_integral_image_adm_sums_neon(i16_adm_buffers pyr_1, uint16_t *x, int k, int stride, i32_adm_buffers masked_pyr, int width, int height, int band_index)
{
    uint16_t *x_pad;
    // int64_t *sum;
    // int64_t *temp_sum;
    int i, j, index;
    int32_t pyr_abs;
    
    int x_reflect = (int)((k - stride) / 2);
    
    x_pad = (uint16_t *)malloc(sizeof(uint16_t) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    
    integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);
    
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);
    size_t int_stride = r_width + 1;
    
     //  int64_t *sum;
    // int64_t *temp_sum;
    // sum = (int64_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int64_t));
    // temp_sum = (int64_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int64_t));

    int32_t *sum;
    int32_t *temp_sum;
    sum = (int32_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int32_t));
    temp_sum = (int32_t *)malloc((r_width + 1) * (r_height + 1) * sizeof(int32_t));

	/*
	** Setting the first row values to 0
	*/
    memset(sum, 0, int_stride * sizeof(int64_t));

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
        	for (j = 0; j < width;)
        	{
        		// int32_t masking_threshold;
        		// int32_t val;
        		index = i * width + j;
        		// masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
        		// pyr_abs = abs((int32_t)pyr_1.bands[1][index]) * 30;
        		// val = pyr_abs - masking_threshold;
        		// masked_pyr.bands[1][index] = val;
        		// pyr_abs = abs((int32_t)pyr_1.bands[2][index]) * 30;
        		// val = pyr_abs - masking_threshold;
        		// masked_pyr.bands[2][index] = val;
        		// pyr_abs = abs((int32_t)pyr_1.bands[3][index]) * 30;
        		// val = pyr_abs - masking_threshold;
        		// masked_pyr.bands[3][index] = val;

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                int32x4_t mt2 = vld1q_s32(sum + ((i + k) * int_stride + j + k) + 4);
                uint16x8_t xt = vld1q_u16(x+index);
                uint32x4_t xt_m = vmovl_u16(vget_low_u16(xt));
                uint32x4_t xt_m_2 = vmovl_u16(vget_high_u16(xt));
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t masking_threshold_2 = vaddq_s32(vreinterpretq_s32_u32(xt_m_2), mt2);

                int16x4_t val_1_t = vld1_s16(pyr_1.bands[1] + index);
                int16x4_t val_2_t = vld1_s16(pyr_1.bands[2] + index);
                int16x4_t val_3_t = vld1_s16(pyr_1.bands[3] + index);
                int16x4_t val_1_t_2 = vld1_s16(pyr_1.bands[1] + index + 4);
                int16x4_t val_2_t_2 = vld1_s16(pyr_1.bands[2] + index + 4);
                int16x4_t val_3_t_2 = vld1_s16(pyr_1.bands[3] + index + 4);

                int16x4_t val_1abs = vabs_s16(val_1_t);
                int16x4_t val_2abs = vabs_s16(val_2_t);
                int16x4_t val_3abs = vabs_s16(val_3_t);
                int16x4_t val_1abs_2 = vabs_s16(val_1_t_2);
                int16x4_t val_2abs_2 = vabs_s16(val_2_t_2);
                int16x4_t val_3abs_2 = vabs_s16(val_3_t_2);

                int32x4_t mull1 = vmull_n_s16(val_1abs, 30); 
                int32x4_t mull2 = vmull_n_s16(val_2abs, 30);
                int32x4_t mull3 = vmull_n_s16(val_3abs, 30);
                int32x4_t mull1_2 = vmull_n_s16(val_1abs_2, 30); 
                int32x4_t mull2_2 = vmull_n_s16(val_2abs_2, 30);
                int32x4_t mull3_2 = vmull_n_s16(val_3abs_2, 30);

                int32x4_t sub_1 = vsubq_s32(mull1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(mull2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(mull3, masking_threshold);
                int32x4_t sub_1_2 = vsubq_s32(mull1_2, masking_threshold_2);
                int32x4_t sub_2_2 = vsubq_s32(mull2_2, masking_threshold_2);
                int32x4_t sub_3_2 = vsubq_s32(mull3_2, masking_threshold_2);

                vst1q_s32(masked_pyr.bands[1] + index, sub_1);
                vst1q_s32(masked_pyr.bands[2] + index, sub_2);
                vst1q_s32(masked_pyr.bands[3] + index, sub_3);
                vst1q_s32(masked_pyr.bands[1] + index + 4, sub_1_2);
                vst1q_s32(masked_pyr.bands[2] + index + 4, sub_2_2);
                vst1q_s32(masked_pyr.bands[3] + index + 4, sub_3_2);

                j+=8;
        	}                           
        }
    }
	
    if(band_index == 2)
    {
	    for (i = 0; i < height; i++)
	    {
	    	for (j = 0; j < width;)
	    	{
	    		// int32_t masking_threshold;
	    		// int32_t val;
	    		index = i * width + j;
	    		// masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
	    		// val = masked_pyr.bands[1][index] - masking_threshold;
	    		// masked_pyr.bands[1][index] = val;
	    		// val = masked_pyr.bands[2][index] - masking_threshold;
	    		// masked_pyr.bands[2][index] = val;
	    		// val = masked_pyr.bands[3][index] - masking_threshold;
	    		// masked_pyr.bands[3][index] = val;

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                uint16x4_t xt = vld1_u16(x+index);
                uint32x4_t xt_m = vmovl_u16(xt);
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t val_1 = vld1q_s32(masked_pyr.bands[1] + index);
                int32x4_t val_2 = vld1q_s32(masked_pyr.bands[2] + index);
                int32x4_t val_3 = vld1q_s32(masked_pyr.bands[3] + index);
                int32x4_t sub_1 = vsubq_s32(val_1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(val_2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(val_3, masking_threshold);
                vst1q_s32(masked_pyr.bands[1] + index, sub_1);
                vst1q_s32(masked_pyr.bands[2] + index, sub_2);
                vst1q_s32(masked_pyr.bands[3] + index, sub_3);

                j+=4;
	    	}
	    }
    }
	/*
	** For band 3 loop the final value is clipped
	** to minimum of zero.
	*/
    if(band_index == 3)
    {
        int32x4_t lower  = vdupq_n_s32 (0); // do this only once before the loops
	    for (i = 0; i < height; i++)
	    {
	    	for (j = 0; j < width;)
	    	{
	    		// int32_t masking_threshold;
	    		// int32_t val;
	    		index = i * width + j;
	    		// masking_threshold = (int32_t)x[index] + sum[(i + k) * int_stride + j + k]; // x + mx
	    		// val = masked_pyr.bands[1][index] - masking_threshold;
	    		// masked_pyr.bands[1][index] = (int32_t)clip(val, 0.0, val);
	    		// val = masked_pyr.bands[2][index] - masking_threshold;
	    		// masked_pyr.bands[2][index] = (int32_t)clip(val, 0.0, val);
	    		// val = masked_pyr.bands[3][index] - masking_threshold;
	    		// masked_pyr.bands[3][index] = (int32_t)clip(val, 0.0, val);

                int32x4_t mt = vld1q_s32(sum + ((i + k) * int_stride + j + k));
                uint16x4_t xt = vld1_u16(x+index);
                uint32x4_t xt_m = vmovl_u16(xt);
                int32x4_t masking_threshold = vaddq_s32(vreinterpretq_s32_u32(xt_m), mt);
                int32x4_t val_1 = vld1q_s32(masked_pyr.bands[1] + index);
                int32x4_t val_2 = vld1q_s32(masked_pyr.bands[2] + index);
                int32x4_t val_3 = vld1q_s32(masked_pyr.bands[3] + index);
                int32x4_t sub_1 = vsubq_s32(val_1, masking_threshold);
                int32x4_t sub_2 = vsubq_s32(val_2, masking_threshold);
                int32x4_t sub_3 = vsubq_s32(val_3, masking_threshold);
                
                int32x4_t x1 = vmaxq_s32 (sub_1, lower); 
                int32x4_t x2 = vmaxq_s32 (sub_2, lower); 
                int32x4_t x3 = vmaxq_s32 (sub_3, lower); 
                vst1q_s32(masked_pyr.bands[1] + index, x1);
                vst1q_s32(masked_pyr.bands[2] + index, x2);
                vst1q_s32(masked_pyr.bands[3] + index, x3);

                j+=4;
	    	}
	    }
    }
    
    free(temp_sum);
    free(sum);
    free(x_pad);
}


int compare(const int32_t* _x, const int32_t* _x_simd, int iwidth, int iheight, int band)
{
    int index = 0;
    int count = 0;
    for(size_t i = 0; i < iheight; i++)
    {
        for(size_t j = 0; j < iwidth; j++)
        {
            index = i*iwidth+j;
            if(index == 100)
                printf("c: %d\t neon: %d\n", _x[index], _x_simd[index]);
            if(_x[index] != _x_simd[index])
            {
                printf("mismatch element C: %d, ARM: %d, column: %ld, row: %ld, index(row*width+col): %d \n", _x[index], _x_simd[index], j, i, index);
                count++;   
            }
        }
    }

    printf("total mismatches in band %d : %d\n",band, count);
    printf("total elements in band %d : %d\n",band, iheight*iwidth);
    printf("-------------\n");

    return 1;
}


// void integer_dlm_contrast_mask_one_way(i_dwt2buffers pyr_1, u_adm_buffers pyr_2, i_adm_buffers masked_pyr, size_t width, size_t height)
int main()
{
    // int k;
    int height = 540;
    int width = 960;
    
    // int16_t *pyr_1[4];
    // uint16_t *pyr_2[4];
    // int32_t *masked_pyr[4], *masked_pyr_neon[4];

    u16_adm_buffers pyr_2;
    i16_adm_buffers pyr_1;
    i32_adm_buffers masked_pyr, masked_pyr_neon;

    pyr_1.bands[1] = (int16_t*)malloc(sizeof(int16_t) * height * width);
    pyr_1.bands[2] = (int16_t*)malloc(sizeof(int16_t) * height * width);
    pyr_1.bands[3] = (int16_t*)malloc(sizeof(int16_t) * height * width);
    pyr_2.bands[1]  = (uint16_t*)malloc(sizeof(uint16_t) * height * width);
    pyr_2.bands[2]  = (uint16_t*)malloc(sizeof(uint16_t) * height * width);
    pyr_2.bands[3]  = (uint16_t*)malloc(sizeof(uint16_t) * height * width);
    masked_pyr.bands[1] = (int32_t*)malloc(sizeof(int32_t) * height * width);
    masked_pyr.bands[2] = (int32_t*)malloc(sizeof(int32_t) * height * width);
    masked_pyr.bands[3] = (int32_t*)malloc(sizeof(int32_t) * height * width);
    masked_pyr_neon.bands[1] = (int32_t*)malloc(sizeof(int32_t) * height * width);
    masked_pyr_neon.bands[2] = (int32_t*)malloc(sizeof(int32_t) * height * width);
    masked_pyr_neon.bands[3] = (int32_t*)malloc(sizeof(int32_t) * height * width);

    //(i_dlm_rest) pyr_1 = 2387, - 2387 (int16) // 3 bands
    //(idlm_Add) pyr_2 = 47750, 0 (uint16) // 3 bands

    int max = 2387;
    int min = -2387;

    int max2 = 47750;
    int min2 = 0;
    int index = 0;

    for(size_t xh = 0; xh < height; xh++)
    {
        for(size_t yw = 0; yw < width; yw++)
        {
            for (int k = 1; k < 4; k++)
            {
                index = xh*width+yw;
                pyr_1.bands[k][index] = (int16_t)((rand() % (max + 1 - min)) + min);
                pyr_2.bands[k][index] = (uint16_t)((rand() % (max2 + 1 - min2)) + min2);
            }
        }
    }

    clock_t startTime, stopTime;
    double msecElapsed;
    startTime = clock();
    for (int k = 1; k < 4; k++)
    {
        integer_integral_image_adm_sums(pyr_1, pyr_2.bands[k], 3, 1, masked_pyr, width, height, k);        
    }
    stopTime = clock();
    msecElapsed = (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    printf("Overall adm c time: %f ms\n-------------\n", msecElapsed);

    startTime = clock();
    for (int k = 1; k < 4; k++)
    {
        integer_integral_image_adm_sums_neon(pyr_1, pyr_2.bands[k], 3, 1, masked_pyr_neon, width, height, k);
    }
    stopTime = clock();
    msecElapsed = (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    printf("Overall adm neon time: %f ms\n-------------\n", msecElapsed);

    for (int k = 1; k < 4; k++)
    {
        compare(masked_pyr.bands[k], masked_pyr_neon.bands[k], width, height, k);
    }
    return 0;
}



