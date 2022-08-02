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
#include <mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "integer_funque_filters.h"
#include "integer_funque_vif.h"
#include "common/macros.h"

#define VIF_COMPUTE_METRIC_R_SHIFT 6

void int16_frame_to_csv(int16_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%d,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void int32_frame_to_csv(int32_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%d,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void int64_frame_to_csv(int64_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%ld,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

// just change the store offset to reduce multiple calculation when getting log value
void funque_log_generate(uint32_t* log_18)
{
    uint64_t i;
    uint64_t start = (unsigned int)pow(2, 17);
    uint64_t end = (unsigned int)pow(2, 18);
	for (i = start; i < end; i++)
    {
		log_18[i] = (uint32_t)round(log2((double)i) * (1 << 26));
    }
}

// uint32_t log_18(uint32_t input)
// {
    
//     uint32_t log_out_1 = (uint32_t)round(log2((double)input) * (1 << 26));
//     return log_out_1;
// }

FORCE_INLINE inline uint32_t get_best_18bitsfixed_opt_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 46) 
    {
        k -= 46;
        temp = temp << k;
        *x = k;

    }
    else if (k < 45) 
    {
        k = 46 - k;
        temp = temp >> k;
        *x = -k;
    }
    else
    {
        *x = 0;
        if (temp >> 18)
        {
            temp = temp >> 1;
            *x = -1;
        }
    }

    return (uint32_t)temp;
}

/**
 * Works similar to get_best_16bitsfixed_opt function but for 64 bit input
 */
FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp); // for long

    if (k > 48)  // temp < 2^47
    {
        k -= 48;
        temp = temp << k;
        *x = k;

    }
    else if (k < 47)  // temp > 2^48
    {
        k = 48 - k;
        temp = temp >> k;
        *x = -k;
    }
    else
    {
        *x = 0;
        if (temp >> 16)
        {
            temp = temp >> 1;
            *x = -1;
        }
    }

    return (uint16_t)temp;
}

void integer_reflect_pad(const dwt2_dtype* src, size_t width, size_t height, int reflect, dwt2_dtype* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(dwt2_dtype) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
    }
}

void integer_integral_image_2(const dwt2_dtype* src1, const dwt2_dtype* src2, size_t width, size_t height, int64_t* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            int64_t val = (((int64_t)src1[(i - 1) * width + (j - 1)] * (int64_t)src2[(i - 1) * width + (j - 1)]));
            val += (int64_t)(sum[(i - 1) * (width + 1) + j]);
            val += (int64_t)(sum[i * (width + 1) + j - 1]) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]);
            sum[i * (width + 1) + j] = val;
        }
    }

}

void integer_integral_image(const dwt2_dtype* src, size_t width, size_t height, int64_t* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            int64_t val = (int64_t)(src[(i - 1) * width + (j - 1)]); //64 to avoid overflow  

            val += (int64_t)(sum[(i - 1) * (width + 1) + j]);
            val += (int64_t)(sum[i * (width + 1) + j - 1]) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]);
            sum[i * (width + 1) + j] = val;
        }
    }
}

/**
 * To get the var_x, var_y, cov_xy only the previous kw x kh pixels are required from the current pixel
 * Hence the integral image function is modified to store only required data
 * The compute metrics function is also brought into this function itself
 */
void integer_vif_comp_integral(const dwt2_dtype* src_x,
							  const dwt2_dtype* src_y, 
                             size_t width, size_t height, 
						     int64_t* int_1_x, int64_t* int_1_y,
							 int64_t* int_2_x, int64_t* int_2_y,
							 int64_t* int_x_y,
							 int kw, int kh, double kNorm)
{
	int width_p1  = (width + 1);
	int height_p1 = (height + 1);
	int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    int32_t mx, my; 
    int64_t vx, vy, cxy;

    int64_t *interim_2_x = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	int32_t *interim_1_x = (int32_t*)malloc(width_p1 * height_p1 * sizeof(int32_t));
	
	int64_t *interim_2_y = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	int32_t *interim_1_y = (int32_t*)malloc(width_p1 * height_p1 * sizeof(int32_t));
	
	int64_t *interim_x_y = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	
	int32_t *src_x_sq    = (int32_t*)malloc(width * sizeof(int32_t));
	int32_t *src_y_sq    = (int32_t*)malloc(width * sizeof(int32_t));
	int32_t *src_x_y     = (int32_t*)malloc(width * sizeof(int32_t));
	
	//memset 1st row to 0
	memset(int_1_x,0,width_p1 * sizeof(int64_t));
	memset(int_2_x,0,width_p1 * sizeof(int64_t));
	
	memset(int_1_y,0,width_p1 * sizeof(int64_t));
	memset(int_2_y,0,width_p1 * sizeof(int64_t));
	
	memset(int_x_y,0,width_p1 * sizeof(int64_t));
    size_t i;
    int row_offset, pre_kh_kw_offset;

    /**
     * To get the var_x, var_y, cov_xy only the previous kw x kh pixels are required from the current pixel
     * Hence, the loop is divided into 2 parts, because till we reach kh row we have to sum all previous row pixels, 
     * similar rule is applied for loop across width
     */
    //1st loop across height, sums all previous rows intermediate(kw pixel) sums i.e. previous_available_rows x kw sums 
    for (i=1; i<kh+1; i++)
    {
		row_offset = i*width_p1;
		int src_offset = (i - 1) * width;		
		int pre_row_offset = (i-1) * width_p1;

		interim_2_x[row_offset] = 0; // 1st coloumn is set to 0	
		interim_1_x[row_offset] = 0; // 1st coloumn is set to 0	
		
		interim_2_y[row_offset] = 0; // 1st coloumn is set to 0	
		interim_1_y[row_offset] = 0; // 1st coloumn is set to 0	
		
		interim_x_y[row_offset] = 0; // 1st coloumn is set to 0	
		
        //The loop across width is divided into 2 parts, 
        //1st loop across width, upto kw pixels since all pixels has to be summed up till here
        for(size_t j=1; j<kw+1; j++)
        {
			int j_minus1 = j - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];			
			
            int32_t src_sq_val_x = (int32_t) src_x_val * src_x_val;
			int32_t src_sq_val_y = (int32_t) src_y_val * src_y_val;
			int32_t src_val_xy   = (int32_t) src_x_val * src_y_val;
			
			src_x_sq [j_minus1] = src_sq_val_x; // store the square in temp row buffer
			src_y_sq [j_minus1] = src_sq_val_y; // store the square in temp row buffer			
			src_x_y[j_minus1]   = src_val_xy; // store the x*y in temp row buffer
			
            //These buffers will hold the sum of previous pixels in the row
            interim_2_x[row_offset + j] = interim_2_x[row_offset + j_minus1] + src_sq_val_x;			
			interim_1_x[row_offset + j] = interim_1_x[row_offset + j_minus1] + src_x_val;
			
			interim_2_y[row_offset + j] = interim_2_y[row_offset + j_minus1] + src_sq_val_y;			
			interim_1_y[row_offset + j] = interim_1_y[row_offset + j_minus1] + src_y_val;
			
			interim_x_y[row_offset + j] = interim_x_y[row_offset + j_minus1] + src_val_xy;
        }
        //2nd loop across width, stores only sum of previous kw pixels
        for(size_t j=kw+1; j<width_p1; j++)
        {
            int j_minus1 = j - 1;
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];			
			
            int32_t src_sq_val_x = (int32_t) src_x_val * src_x_val;
			int32_t src_sq_val_y = (int32_t) src_y_val * src_y_val;
			int32_t src_val_xy   = (int32_t) src_x_val * src_y_val;
			
			src_x_sq [j_minus1] = src_sq_val_x; // store the square in temp row buffer
			src_y_sq [j_minus1] = src_sq_val_y; // store the square in temp row buffer			
			src_x_y[j_minus1]   = src_val_xy; // store the x*y in temp row buffer
			
            //These buffers will hold the sum of previous kw pixels in the row
            interim_2_x[row_offset + j] = interim_2_x[row_offset + j_minus1] + 
			                              src_sq_val_x - src_x_sq[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
			
			interim_1_x[row_offset + j] = interim_1_x[row_offset + j_minus1] + 
			                              src_x_val - src_x[src_offset + j_minus1 - kw]; // subtarct src from -kw pos;
									   
		    interim_2_y[row_offset + j] = interim_2_y[row_offset + j_minus1] + 
										  src_sq_val_y - src_y_sq[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
			
			interim_1_y[row_offset + j] = interim_1_y[row_offset + j_minus1] + 
			                              src_y_val - src_y[src_offset + j_minus1 - kw]; // subtarct src from -kw pos;
									   
			interim_x_y[row_offset + j] = interim_x_y[row_offset + j_minus1] + 
										  src_val_xy - src_x_y[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
			
        }
        /**
         * These buffers will hold the sum of previous rows intermediate sums i.e.
         * This will hold sums of kw x (available prev rows) pixels
         */
        for (size_t j=1; j<width_p1; j++)
        {
            int_2_x[row_offset + j] = interim_2_x[row_offset + j] + int_2_x[pre_row_offset + j];
            int_1_x[row_offset + j] = interim_1_x[row_offset + j] + int_1_x[pre_row_offset + j];
            
            int_2_y[row_offset + j] = interim_2_y[row_offset + j] + int_2_y[pre_row_offset + j];
            int_1_y[row_offset + j] = interim_1_y[row_offset + j] + int_1_y[pre_row_offset + j];
            
            int_x_y[row_offset + j] = interim_x_y[row_offset + j] + int_x_y[pre_row_offset + j]; 
        }
    }
    // /**
    //  * The var_x, var_y, cov_xy is stored for 1 row
    //  * Size of var_x, var_y, cov_xy is (width-kw)x(height-kh)
    //  * 1st row of var_x, var_y, cov_xy is computed from int_*[kh * (width - kw) + kw]
    //  * and computation for only 1st row is done in previous loops, 
    //  * hence storing values for 1st rows of var_x, var_y, cov_xy
    // */
    // pre_kh_kw_offset = (i-1-kh) * (width_p1-kw);
    // for (size_t j = kw; j < width_p1; j++)
    // {
    //     mx = int_1_x[row_offset + j];
    //     my = int_1_y[row_offset + j];

    //     vx = int_2_x[row_offset + j] - (((int64_t)mx*mx*knorm_fact)>>knorm_shift);
    //     vy = int_2_y[row_offset + j] - (((int64_t)my*my*knorm_fact)>>knorm_shift);
    //     cxy = int_x_y[row_offset + j] - (((int64_t)mx*my*knorm_fact)>>knorm_shift);

    //     var_x[pre_kh_kw_offset + j - kw] = vx < 0 ? 0 : (int32_t) (vx >> VIF_COMPUTE_METRIC_R_SHIFT); 
    //     var_y[pre_kh_kw_offset + j - kw] = vy < 0 ? 0 : (int32_t) (vy >> VIF_COMPUTE_METRIC_R_SHIFT);
    //     cov_xy[pre_kh_kw_offset + j -kw] = (vx < 0 || vy < 0) ? 0 : (int32_t) (cxy >> VIF_COMPUTE_METRIC_R_SHIFT);
    // }

    //2nd loop across height, sums previous kh rows intermediate(kw pixel) sums i.e. kh x kw sums 
    for ( ; i<height_p1; i++)
    {
		row_offset = i*width_p1;
		int src_offset = (i - 1) * width;		
		int pre_row_offset = (i-1) * width_p1;
        int pre_kh_offset = (i-kh) * width_p1;
        pre_kh_kw_offset = (i-kh) * (width_p1-kw);
		interim_2_x[row_offset] = 0; // 1st coloumn is set to 0	
		interim_1_x[row_offset] = 0; // 1st coloumn is set to 0	
		
		interim_2_y[row_offset] = 0; // 1st coloumn is set to 0	
		interim_1_y[row_offset] = 0; // 1st coloumn is set to 0	
		
		interim_x_y[row_offset] = 0; // 1st coloumn is set to 0	
		
        //1st loop across width, upto kw pixels since all pixels has to be summed up till here
        for(size_t j=1; j<kw+1; j++)
        {
			int j_minus1 = j - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];			
			
            int32_t src_sq_val_x = (int32_t) src_x_val * src_x_val;
			int32_t src_sq_val_y = (int32_t) src_y_val * src_y_val;
			int32_t src_val_xy   = (int32_t) src_x_val * src_y_val;
			
			src_x_sq [j_minus1] = src_sq_val_x; // store the square in temp row buffer
			src_y_sq [j_minus1] = src_sq_val_y; // store the square in temp row buffer			
			src_x_y[j_minus1]   = src_val_xy; // store the x*y in temp row buffer
			
            interim_2_x[row_offset + j] = interim_2_x[row_offset + j_minus1] + src_sq_val_x;			
			interim_1_x[row_offset + j] = interim_1_x[row_offset + j_minus1] + src_x_val;
			
			interim_2_y[row_offset + j] = interim_2_y[row_offset + j_minus1] + src_sq_val_y;			
			interim_1_y[row_offset + j] = interim_1_y[row_offset + j_minus1] + src_y_val;
			
			interim_x_y[row_offset + j] = interim_x_y[row_offset + j_minus1] + src_val_xy;
        }
        //2nd loop across width, stores only sum of previous kw pixels
        for(size_t j=kw+1; j<width_p1; j++)
        {
            int j_minus1 = j - 1;
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];			
			
            int32_t src_sq_val_x = (int32_t) src_x_val * src_x_val;
			int32_t src_sq_val_y = (int32_t) src_y_val * src_y_val;
			int32_t src_val_xy   = (int32_t) src_x_val * src_y_val;
			
			src_x_sq [j_minus1] = src_sq_val_x; // store the square in temp row buffer
			src_y_sq [j_minus1] = src_sq_val_y; // store the square in temp row buffer			
			src_x_y[j_minus1]   = src_val_xy; // store the x*y in temp row buffer
			
            interim_2_x[row_offset + j] = interim_2_x[row_offset + j_minus1] + 
			                              src_sq_val_x - src_x_sq[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
			
			interim_1_x[row_offset + j] = interim_1_x[row_offset + j_minus1] + 
			                              src_x_val - src_x[src_offset + j_minus1 - kw]; // subtarct src from -kw pos;
									   
		    interim_2_y[row_offset + j] = interim_2_y[row_offset + j_minus1] + 
										  src_sq_val_y - src_y_sq[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
			
			interim_1_y[row_offset + j] = interim_1_y[row_offset + j_minus1] + 
			                              src_y_val - src_y[src_offset + j_minus1 - kw]; // subtarct src from -kw pos;
									   
			interim_x_y[row_offset + j] = interim_x_y[row_offset + j_minus1] + 
										  src_val_xy - src_x_y[j_minus1 - kw]; // subtarct src_x_sq from -kw pos; 
        }
        //The loop is split due to requirement of cov_xy, var_x, var_y because the 1st element is kw element of int_*
        //Operations to store int_* values is same in both loops
        /**
         * These buffers will hold the sum of previous kh rows intermediate sums i.e.
         * This will hold sums of kh x kw pixels
         */
        for (size_t j=1; j<kw; j++)
        {
            int_2_x[row_offset + j] = interim_2_x[row_offset+j] + int_2_x[pre_row_offset + j] - interim_2_x[pre_kh_offset + j];
			int_1_x[row_offset + j] = interim_1_x[row_offset + j] + int_1_x[pre_row_offset + j] - interim_1_x[pre_kh_offset + j];
			
			int_2_y[row_offset + j] = interim_2_y[row_offset+j] + int_2_y[pre_row_offset + j] - interim_2_y[pre_kh_offset + j];
			int_1_y[row_offset + j] = interim_1_y[row_offset + j] + int_1_y[pre_row_offset + j] - interim_1_y[pre_kh_offset + j];
			
			int_x_y[row_offset + j] = interim_x_y[row_offset+j] + int_x_y[pre_row_offset + j] - interim_x_y[pre_kh_offset + j];
        }
        //This loop is similar to previous loop for storing int_* i.e. int_1_x, int_1_y, int_2_x, int_2_y, int_x_y
        /**
         * 1st row of var_x, var_y, cov_xy is stored earlier, from 2nd row values are stored here
         * Size of var_x, var_y, cov_xy is (width-kw)x(height-kh)
         * 1st row of var_x, var_y, cov_xy is computed from int_*[kh * (width - kw) + kw] 
         * values of int_* buffers from [i * (width - kw) + j] to end where i starts from kh to height, j starts from kw to width 
         */
        for (size_t j=kw; j < width_p1; j++)
        {
            int_2_x[row_offset + j] = interim_2_x[row_offset+j] + int_2_x[pre_row_offset + j] - interim_2_x[pre_kh_offset + j];
			int_1_x[row_offset + j] = interim_1_x[row_offset + j] + int_1_x[pre_row_offset + j] - interim_1_x[pre_kh_offset + j];
			
			int_2_y[row_offset + j] = interim_2_y[row_offset+j] + int_2_y[pre_row_offset + j] - interim_2_y[pre_kh_offset + j];
			int_1_y[row_offset + j] = interim_1_y[row_offset + j] + int_1_y[pre_row_offset + j] - interim_1_y[pre_kh_offset + j];
			
			int_x_y[row_offset + j] = interim_x_y[row_offset+j] + int_x_y[pre_row_offset + j] - interim_x_y[pre_kh_offset + j];
        
        
            // mx = int_1_x[row_offset + j];
            // my = int_1_y[row_offset + j];

            // vx = int_2_x[row_offset + j] - (((int64_t)mx*mx*knorm_fact)>>knorm_shift);
            // vy = int_2_y[row_offset + j] - (((int64_t)my*my*knorm_fact)>>knorm_shift);
            // cxy = int_x_y[row_offset + j] - (((int64_t)mx*my*knorm_fact)>>knorm_shift);

            // var_x[pre_kh_kw_offset + j - kw] = vx < 0 ? 0 : (int32_t) (vx >> VIF_COMPUTE_METRIC_R_SHIFT); 
            // var_y[pre_kh_kw_offset + j - kw] = vy < 0 ? 0 : (int32_t) (vy >> VIF_COMPUTE_METRIC_R_SHIFT);
            // cov_xy[pre_kh_kw_offset + j -kw] = (vx < 0 || vy < 0) ? 0 : (int32_t) (cxy >> VIF_COMPUTE_METRIC_R_SHIFT);

        }
    }	
	
    free(interim_2_x);
	free(interim_1_x);
	free(src_x_sq);
	
	free(interim_2_y);
	free(interim_1_y);
	free(src_y_sq);
	
	free(interim_x_y);
	free(src_x_y);
}

void integer_vif_comp_integral_mod(const dwt2_dtype* src_x,
							  const dwt2_dtype* src_y, 
                             size_t width, size_t height, 
						     int64_t* int_1_x, int64_t* int_1_y,
							 int64_t* int_2_x, int64_t* int_2_y,
							 int64_t* int_x_y,
							 int kw, int kh, double kNorm)
{
	int width_p1  = (width + 1);
	int height_p1 = (height + 1);
	int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    int32_t mx, my; 
    int64_t vx, vy, cxy;

    int64_t *interim_2_x = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	int32_t *interim_1_x = (int32_t*)malloc(width_p1 * height_p1 * sizeof(int32_t));
	
	int64_t *interim_2_y = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	int32_t *interim_1_y = (int32_t*)malloc(width_p1 * height_p1 * sizeof(int32_t));
	
	int64_t *interim_x_y = (int64_t*)malloc(width_p1 * height_p1 * sizeof(int64_t));
	
	int32_t *src_x_sq    = (int32_t*)malloc(height * sizeof(int32_t));
	int32_t *src_y_sq    = (int32_t*)malloc(height * sizeof(int32_t));
	int32_t *src_x_y     = (int32_t*)malloc(height * sizeof(int32_t));
	
	//memset 1st row to 0
	memset(int_1_x,0,width_p1 * sizeof(int64_t));
	memset(int_2_x,0,width_p1 * sizeof(int64_t));
	
	memset(int_1_y,0,width_p1 * sizeof(int64_t));
	memset(int_2_y,0,width_p1 * sizeof(int64_t));
	
	memset(int_x_y,0,width_p1 * sizeof(int64_t));
    size_t i = 0;
    size_t j = 0;
    int row_offset, pre_kh_kw_offset;
    int col_offset;
    row_offset = i*width_p1;
    int32_t int_1_x_pre = 0;
    int32_t int_1_y_pre = 0;
    int64_t int_2_x_pre = 0;
    int64_t int_2_y_pre = 0;
    int64_t int_x_y_pre = 0;
    for (j=1; j<kw+1; j++)
    {
        col_offset = j*height_p1;
        int j_minus1 = j - 1;
// row_offset
        // interim_1_x[col_offset] = 0;
        // interim_1_y[col_offset] = 0;
        // interim_2_x[col_offset] = 0;
        // interim_x_y[col_offset] = 0;
        int_1_x_pre = 0;
        int_1_y_pre = 0;
        int_2_x_pre = 0;
        int_2_y_pre = 0;
        int_x_y_pre = 0;

        for (i=1; i<kh+1; i++)
        {
            row_offset = i * width_p1;
            int src_offset = (i-1) * width;

            int i_minus1 = i - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            src_x_sq[i_minus1] = src_xx_val; // store the square in temp row buffer
			src_y_sq[i_minus1] = src_yy_val; // store the square in temp row buffer			
			src_x_y[i_minus1]  = src_xy_val; // store the x*y in temp row buffer

            int32_t inter_1_x = int_1_x_pre + src_x_val;
            int32_t inter_1_y = int_1_y_pre + src_y_val;
            int64_t inter_2_x = int_2_x_pre + src_xx_val;
            int64_t inter_2_y = int_2_y_pre + src_yy_val;
            int64_t inter_x_y = int_x_y_pre + src_xy_val;

            interim_1_x[row_offset + j] = inter_1_x;
            interim_1_y[row_offset + j] = inter_1_y;
            interim_2_x[row_offset + j] = inter_2_x;
            interim_2_y[row_offset + j] = inter_2_y;
            interim_x_y[row_offset + j] = inter_x_y;
            // interim_1_x[col_offset+i] = inter_1_x;
            // interim_1_y[col_offset+i] = inter_1_y;
            // interim_2_x[col_offset+i] = inter_2_x;
            // interim_x_y[col_offset+i] = inter_x_y;
            int_1_x_pre = inter_1_x;
            int_1_y_pre = inter_1_y;
            int_2_x_pre = inter_2_x;
            int_2_y_pre = inter_2_y;
            int_x_y_pre = inter_x_y;
        }
        for (; i<height_p1; i++)
        {
            row_offset = i * width_p1;
            int src_offset = (i-1) * width;
            int pre_kh_offset = (i-1-kh) * width;
            // int pre_row_offset = (i-1) * width_p1;

            int i_minus1 = i - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            src_x_sq[i_minus1] = src_xx_val; // store the square in temp row buffer
			src_y_sq[i_minus1] = src_yy_val; // store the square in temp row buffer			
			src_x_y[i_minus1]  = src_xy_val; // store the x*y in temp row buffer

            int32_t inter_1_x = int_1_x_pre + src_x_val - src_x[pre_kh_offset + j_minus1];
            int32_t inter_1_y = int_1_y_pre + src_y_val - src_y[pre_kh_offset + j_minus1];
            int64_t inter_2_x = int_2_x_pre + src_xx_val - src_x_sq[i_minus1 - kh];
            int64_t inter_2_y = int_2_y_pre + src_yy_val - src_y_sq[i_minus1 - kh];
            int64_t inter_x_y = int_x_y_pre + src_xy_val - src_x_y[i_minus1 - kh];

            interim_1_x[row_offset + j] = inter_1_x;
            interim_1_y[row_offset + j] = inter_1_y;
            interim_2_x[row_offset + j] = inter_2_x;
            interim_2_y[row_offset + j] = inter_2_y;
            interim_x_y[row_offset + j] = inter_x_y;
            // interim_1_x[col_offset+i] = inter_1_x;
            // interim_1_y[col_offset+i] = inter_1_y;
            // interim_2_x[col_offset+i] = inter_2_x;
            // interim_x_y[col_offset+i] = inter_x_y;
            int_1_x_pre = inter_1_x;
            int_1_y_pre = inter_1_y;
            int_2_x_pre = inter_2_x;
            int_2_y_pre = inter_2_y;
            int_x_y_pre = inter_x_y;
        }
        for (i=1; i<height_p1; i++)
        {
            row_offset = i * width_p1;
            int pre_row_offset = (i-1) * width_p1;
            int_2_x[row_offset + j] = interim_2_x[row_offset + j] + int_2_x[row_offset + j_minus1];
            int_1_x[row_offset + j] = interim_1_x[row_offset + j] + int_1_x[row_offset + j_minus1];
            
            int_2_y[row_offset + j] = interim_2_y[row_offset + j] + int_2_y[row_offset + j_minus1];
            int_1_y[row_offset + j] = interim_1_y[row_offset + j] + int_1_y[row_offset + j_minus1];
            
            int_x_y[row_offset + j] = interim_x_y[row_offset + j] + int_x_y[row_offset + j_minus1];
            // int_2_x[row_offset + j] = interim_2_x[col_offset + i] + int_2_x[pre_row_offset + j];
            // int_1_x[row_offset + j] = interim_1_x[col_offset + i] + int_1_x[pre_row_offset + j];
            
            // int_2_y[row_offset + j] = interim_2_y[col_offset + i] + int_2_y[pre_row_offset + j];
            // int_1_y[row_offset + j] = interim_1_y[col_offset + i] + int_1_y[pre_row_offset + j];
            
            // int_x_y[row_offset + j] = interim_x_y[col_offset + i] + int_x_y[pre_row_offset + j];
        }
    }
    for ( ; j < width_p1; j++)
    {
        col_offset = j*height_p1;
        int j_minus1 = j - 1;

        // interim_1_x[col_offset] = 0;
        // interim_1_y[col_offset] = 0;
        // interim_2_x[col_offset] = 0;
        // interim_x_y[col_offset] = 0;
        int_1_x_pre = 0;
        int_1_y_pre = 0;
        int_2_x_pre = 0;
        int_2_y_pre = 0;
        int_x_y_pre = 0;

        for (i=1; i<kh+1; i++)
        {
            row_offset = i * width_p1;
            int src_offset = (i-1) * width;
            // int pre_row_offset = (i-1) * width_p1;

            int i_minus1 = i - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            src_x_sq[i_minus1] = src_xx_val; // store the square in temp row buffer
			src_y_sq[i_minus1] = src_yy_val; // store the square in temp row buffer			
			src_x_y[i_minus1]  = src_xy_val; // store the x*y in temp row buffer

            int32_t inter_1_x = int_1_x_pre + src_x_val;
            int32_t inter_1_y = int_1_y_pre + src_y_val;
            int64_t inter_2_x = int_2_x_pre + src_xx_val;
            int64_t inter_2_y = int_2_y_pre + src_yy_val;
            int64_t inter_x_y = int_x_y_pre + src_xy_val;

            interim_1_x[row_offset + j] = inter_1_x;
            interim_1_y[row_offset + j] = inter_1_y;
            interim_2_x[row_offset + j] = inter_2_x;
            interim_2_y[row_offset + j] = inter_2_y;
            interim_x_y[row_offset + j] = inter_x_y;
            // interim_1_x[col_offset+i] = inter_1_x;
            // interim_1_y[col_offset+i] = inter_1_y;
            // interim_2_x[col_offset+i] = inter_2_x;
            // interim_x_y[col_offset+i] = inter_x_y;
            int_1_x_pre = inter_1_x;
            int_1_y_pre = inter_1_y;
            int_2_x_pre = inter_2_x;
            int_2_y_pre = inter_2_y;
            int_x_y_pre = inter_x_y;
        }
        for (; i<height_p1; i++)
        {
            row_offset = i * width_p1;
            int src_offset = (i-1) * width;
            int pre_kh_offset = (i-1-kh) * width;
            // int pre_row_offset = (i-1) * width_p1;

            int i_minus1 = i - 1;
			
			dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            src_x_sq[i_minus1] = src_xx_val; // store the square in temp row buffer
			src_y_sq[i_minus1] = src_yy_val; // store the square in temp row buffer			
			src_x_y[i_minus1]  = src_xy_val; // store the x*y in temp row buffer

            int32_t inter_1_x = int_1_x_pre + src_x_val - src_x[pre_kh_offset + j_minus1];
            int32_t inter_1_y = int_1_y_pre + src_y_val - src_y[pre_kh_offset + j_minus1];
            int64_t inter_2_x = int_2_x_pre + src_xx_val - src_x_sq[i_minus1 - kh];
            int64_t inter_2_y = int_2_y_pre + src_yy_val - src_y_sq[i_minus1 - kh];
            int64_t inter_x_y = int_x_y_pre + src_xy_val - src_x_y[i_minus1 - kh];

            interim_1_x[row_offset + j] = inter_1_x;
            interim_1_y[row_offset + j] = inter_1_y;
            interim_2_x[row_offset + j] = inter_2_x;
            interim_2_y[row_offset + j] = inter_2_y;
            interim_x_y[row_offset + j] = inter_x_y;
            // interim_1_x[col_offset+i] = inter_1_x;
            // interim_1_y[col_offset+i] = inter_1_y;
            // interim_2_x[col_offset+i] = inter_2_x;
            // interim_x_y[col_offset+i] = inter_x_y;
            int_1_x_pre = inter_1_x;
            int_1_y_pre = inter_1_y;
            int_2_x_pre = inter_2_x;
            int_2_y_pre = inter_2_y;
            int_x_y_pre = inter_x_y;
        }
        for (i=1; i<height_p1; i++)
        {
            row_offset = i * width_p1;
            // int pre_kh_offset = (j-kh) * height_p1;
            int pre_kh_offset = (i-kh) * width_p1;
            int pre_row_offset = (i-1) * width_p1;
            int_2_x[row_offset + j] = interim_2_x[row_offset + j] + int_2_x[row_offset + j_minus1] - interim_2_x[row_offset + j - kw];
            int_1_x[row_offset + j] = interim_1_x[row_offset + j] + int_1_x[row_offset + j_minus1] - interim_1_x[row_offset + j - kw];
            
            int_2_y[row_offset + j] = interim_2_y[row_offset + j] + int_2_y[row_offset + j_minus1] - interim_2_y[row_offset + j - kw];
            int_1_y[row_offset + j] = interim_1_y[row_offset + j] + int_1_y[row_offset + j_minus1] - interim_1_y[row_offset + j - kw];
            
            int_x_y[row_offset + j] = interim_x_y[row_offset + j] + int_x_y[row_offset + j_minus1] - interim_x_y[row_offset + j - kw];

            // int_2_x[row_offset + j] = interim_2_x[col_offset + i] + int_2_x[pre_row_offset + j] - interim_2_x[pre_kh_offset + i];
            // int_1_x[row_offset + j] = interim_1_x[col_offset + i] + int_1_x[pre_row_offset + j] - interim_1_x[pre_kh_offset + i];
            
            // int_2_y[row_offset + j] = interim_2_y[col_offset + i] + int_2_y[pre_row_offset + j] - interim_2_y[pre_kh_offset + i];
            // int_1_y[row_offset + j] = interim_1_y[col_offset + i] + int_1_y[pre_row_offset + j] - interim_1_y[pre_kh_offset + i];
            
            // int_x_y[row_offset + j] = interim_x_y[col_offset + i] + int_x_y[pre_row_offset + j] - interim_x_y[pre_kh_offset + i];
        }
    }
    free(interim_2_x);
	free(interim_1_x);
	free(src_x_sq);
	
	free(interim_2_y);
	free(interim_1_y);
	free(src_y_sq);
	
	free(interim_x_y);
	free(src_x_y);
}

void integer_vif_comp_integral_mod_memopt(const dwt2_dtype* src_x,
							  const dwt2_dtype* src_y, 
                             size_t width, size_t height, 
						     int64_t* int_1_x, int64_t* int_1_y,
							 int64_t* int_2_x, int64_t* int_2_y,
							 int64_t* int_x_y,
							 int kw, int kh, double kNorm)
{
	int width_p1  = (width + 1);
	int height_p1 = (height + 1);
	int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    int32_t mx, my; 
    int64_t vx, vy, cxy;

    int64_t *interim_2_x = (int64_t*)malloc(width_p1 * sizeof(int64_t));
	int32_t *interim_1_x = (int32_t*)malloc(width_p1 * sizeof(int32_t));
	
	int64_t *interim_2_y = (int64_t*)malloc(width_p1 * sizeof(int64_t));
	int32_t *interim_1_y = (int32_t*)malloc(width_p1 * sizeof(int32_t));
	
	int64_t *interim_x_y = (int64_t*)malloc(width_p1 * sizeof(int64_t));

	memset(interim_2_x, 0, width_p1 * sizeof(int64_t));
	memset(interim_1_x, 0, width_p1 * sizeof(int32_t));
	memset(interim_2_y, 0, width_p1 * sizeof(int64_t));
	memset(interim_1_y, 0, width_p1 * sizeof(int32_t));
	memset(interim_x_y, 0, width_p1 * sizeof(int64_t));
	
	//memset 1st row to 0
	memset(int_1_x,0,width_p1 * sizeof(int64_t));
	memset(int_2_x,0,width_p1 * sizeof(int64_t));
	
	memset(int_1_y,0,width_p1 * sizeof(int64_t));
	memset(int_2_y,0,width_p1 * sizeof(int64_t));
	
	memset(int_x_y,0,width_p1 * sizeof(int64_t));
    size_t i = 0;
    size_t j = 0;

    for (i=1; i<kh+1; i++)
    {
        int row_offset = i * width_p1;
        int src_offset = (i-1) * width;

        for (j=1; j<width_p1; j++)
        {
            int j_minus1 = j-1;
            dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            interim_1_x[j] = interim_1_x[j] + src_x_val;
            interim_2_x[j] = interim_2_x[j] + src_xx_val;
            interim_1_y[j] = interim_1_y[j] + src_y_val;
            interim_2_y[j] = interim_2_y[j] + src_yy_val;
            interim_x_y[j] = interim_x_y[j] + src_xy_val;

        }
        for (j=1; j<kw+1; j++)
        {
            int j_minus1 = j-1;
            int_2_x[row_offset + j] = interim_2_x[j] + int_2_x[row_offset + j_minus1];
            int_1_x[row_offset + j] = interim_1_x[j] + int_1_x[row_offset + j_minus1];
            
            int_2_y[row_offset + j] = interim_2_y[j] + int_2_y[row_offset + j_minus1];
            int_1_y[row_offset + j] = interim_1_y[j] + int_1_y[row_offset + j_minus1];
            
            int_x_y[row_offset + j] = interim_x_y[j] + int_x_y[row_offset + j_minus1];
        }
        for (; j<width_p1; j++)
        {
            int j_minus1 = j-1;
            int_2_x[row_offset + j] = interim_2_x[j] + int_2_x[row_offset + j_minus1] - interim_2_x[j - kw];
            int_1_x[row_offset + j] = interim_1_x[j] + int_1_x[row_offset + j_minus1] - interim_1_x[j - kw];
            
            int_2_y[row_offset + j] = interim_2_y[j] + int_2_y[row_offset + j_minus1] - interim_2_y[j - kw];
            int_1_y[row_offset + j] = interim_1_y[j] + int_1_y[row_offset + j_minus1] - interim_1_y[j - kw];
            
            int_x_y[row_offset + j] = interim_x_y[j] + int_x_y[row_offset + j_minus1] - interim_x_y[j - kw];
        }
    }
    for(; i<height_p1; i++)
    {
        int row_offset = i * width_p1;
        int src_offset = (i-1) * width;
        int pre_kh_src_offset = (i-1-kh) * width;
        for (j=1; j<width_p1; j++)
        {
            int j_minus1 = j-1;
            dwt2_dtype src_x_val = src_x[src_offset + j_minus1];
			dwt2_dtype src_y_val = src_y[src_offset + j_minus1];

            dwt2_dtype src_x_prekh_val = src_x[pre_kh_src_offset + j_minus1];
            dwt2_dtype src_y_prekh_val = src_y[pre_kh_src_offset + j_minus1];
            int32_t src_xx_val = (int32_t) src_x_val * src_x_val;
			int32_t src_yy_val = (int32_t) src_y_val * src_y_val;
			int32_t src_xy_val = (int32_t) src_x_val * src_y_val;

            int32_t src_xx_prekh_val = (int32_t) src_x_prekh_val * src_x_prekh_val;
			int32_t src_yy_prekh_val = (int32_t) src_y_prekh_val * src_y_prekh_val;
			int32_t src_xy_prekh_val = (int32_t) src_x_prekh_val * src_y_prekh_val;

            interim_1_x[j] = interim_1_x[j] + src_x_val - src_x_prekh_val;
            interim_2_x[j] = interim_2_x[j] + src_xx_val - src_xx_prekh_val;
            interim_1_y[j] = interim_1_y[j] + src_y_val - src_y_prekh_val;
            interim_2_y[j] = interim_2_y[j] + src_yy_val - src_yy_prekh_val;
            interim_x_y[j] = interim_x_y[j] + src_xy_val - src_xy_prekh_val;

        }
        for (j=1; j<kw+1; j++)
        {
            int j_minus1 = j-1;
            int_2_x[row_offset + j] = interim_2_x[j] + int_2_x[row_offset + j_minus1];
            int_1_x[row_offset + j] = interim_1_x[j] + int_1_x[row_offset + j_minus1];
            
            int_2_y[row_offset + j] = interim_2_y[j] + int_2_y[row_offset + j_minus1];
            int_1_y[row_offset + j] = interim_1_y[j] + int_1_y[row_offset + j_minus1];
            
            int_x_y[row_offset + j] = interim_x_y[j] + int_x_y[row_offset + j_minus1];
        }
        for (; j<width_p1; j++)
        {
            int j_minus1 = j-1;
            int_2_x[row_offset + j] = interim_2_x[j] + int_2_x[row_offset + j_minus1] - interim_2_x[j - kw];
            int_1_x[row_offset + j] = interim_1_x[j] + int_1_x[row_offset + j_minus1] - interim_1_x[j - kw];
            
            int_2_y[row_offset + j] = interim_2_y[j] + int_2_y[row_offset + j_minus1] - interim_2_y[j - kw];
            int_1_y[row_offset + j] = interim_1_y[j] + int_1_y[row_offset + j_minus1] - interim_1_y[j - kw];
            
            int_x_y[row_offset + j] = interim_x_y[j] + int_x_y[row_offset + j_minus1] - interim_x_y[j - kw];
        }
    }

    free(interim_2_x);
	free(interim_1_x);
	free(interim_2_y);
	free(interim_1_y);
	free(interim_x_y);
}

void integer_compute_metrics(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int32_t* var_x, int32_t* var_y, int32_t* cov_xy)
{
    int32_t mx, my; 
    int64_t vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = int_1_x[(i + kh) * width + j + kw];
            my = int_1_y[(i + kh) * width + j + kw];

            vx = int_2_x[(i + kh) * width + j + kw] - (((int64_t)mx*mx)/kNorm);
            vy = int_2_y[(i + kh) * width + j + kw] - (((int64_t)my*my)/kNorm);
            cxy = int_xy[(i + kh) * width + j + kw] - (((int64_t)mx*my)/kNorm);
            // var_x[i * (width - kw) + j] = vx < 0 ? 0 : vx; 
            // var_y[i * (width - kw) + j] = vy < 0 ? 0 : vy;
            // cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : cxy;

            // mx = int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw];
            // my = int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw];

            // // (1/knorm) pending on all these (vx, vy ,cxy) - do this in next function
            // vx = (int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) - (((int64_t)mx*(int64_t)mx)/kNorm); 
            // vy = (int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) - (((int64_t)my * (int64_t)my)/kNorm);
            // cxy = (int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) - (((int64_t)mx *(int64_t) my)/kNorm);

            var_x[i * (width - kw) + j] = (int32_t) (vx >> VIF_COMPUTE_METRIC_R_SHIFT); 
            var_y[i * (width - kw) + j] = (int32_t) (vy >> VIF_COMPUTE_METRIC_R_SHIFT);
            cov_xy[i * (width - kw) + j] = (int32_t) (cxy >> VIF_COMPUTE_METRIC_R_SHIFT);

        }
    }
}

int integer_compute_vif_funque(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, double* score, double* score_num, double* score_den, int k, int stride, double sigma_nsq, int64_t shift_val, uint32_t* log_18)
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = k * k;

    int x_reflect = (int)((kh - stride) / 2); // amount for reflecting
    int y_reflect = (int)((kw - stride) / 2);

    size_t r_width = width + (2 * x_reflect); // after reflect pad
    size_t r_height = height + (2 * x_reflect);

    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;
    // double exp = (double)1e-10;
    int index = 0;

    dwt2_dtype* x_pad_t, *y_pad_t;
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));
    integer_reflect_pad(x_t, width, height, x_reflect, x_pad_t);
    integer_reflect_pad(y_t, width, height, y_reflect, y_pad_t);

    int64_t* int_1_x_t, * int_1_y_t, * int_2_x_t, * int_2_y_t, * int_xy_t;
    int32_t var_x_t, var_y_t, cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    
    integer_vif_comp_integral_mod_memopt(x_pad_t, y_pad_t,
                            r_width, r_height,
                            int_1_x_t , int_1_y_t,
                            int_2_x_t , int_2_y_t, int_xy_t,
                            kw, kh, (double)k_norm);

    uint32_t sv_sq_t;
    int64_t exp_t = 1;//exp*shift_val*shift_val; // using 1 because exp in Q32 format is still 0
    uint32_t sigma_nsq_t = (int64_t)((int64_t)sigma_nsq*shift_val*shift_val*k_norm) >> VIF_COMPUTE_METRIC_R_SHIFT ;

    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;

    int64_t score_num_t = 0;
    int64_t num_power = 0;
    int64_t score_den_t = 0;
    int64_t den_power = 0;

    int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 
    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;
            int32_t mx = int_1_x_t[(i+kh)*(r_width+1) + j + kw];
            int32_t my = int_1_y_t[(i+kh)*(r_width+1) + j + kw];
            var_x_t = (int_2_x_t[(i+kh)*(r_width+1) + j + kw] - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;
            var_y_t = (int_2_y_t[(i+kh)*(r_width+1) + j + kw] - (((int64_t) my * my * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;
            cov_xy_t = (int_xy_t[(i+kh)*(r_width+1) + j + kw] - (((int64_t) mx * my * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;
            //These 2 loops can be kept in prev function also
            if (var_x_t < exp_t)
            {
                var_x_t = 0;
                cov_xy_t = 0;
            }
            
            if (var_y_t < exp_t)
            {
                var_y_t = 0;
                cov_xy_t = 0;
            }
            int32_t g_t_num = cov_xy_t;
            int32_t g_den = var_x_t + exp_t*k_norm;

            sv_sq_t = (var_y_t - ((int64_t)g_t_num * cov_xy_t)/g_den);


            if((g_t_num < 0 && g_den > 0) || (g_den < 0 && g_t_num > 0))
            {
                sv_sq_t = var_x_t;
                g_t_num = 0;
            }

            if (sv_sq_t < (exp_t * k_norm))
                sv_sq_t = exp_t * k_norm;

            int64_t p1 = ((int64_t)g_t_num * (int64_t)g_t_num)/(int64_t)g_den;
            uint32_t p2 = (uint32_t)(var_x_t);
            int64_t n1 = p1 * (int64_t)p2;
            int64_t n2 = (int64_t) g_den * ((int64_t) sv_sq_t + (int64_t)sigma_nsq_t);
            int64_t num_t = n2 + n1;
            int64_t num_den_t = n2;
            int x1, x2;
  
            uint32_t log_in_num_1 = get_best_18bitsfixed_opt_64((uint64_t)num_t, &x1);
            uint32_t log_in_num_2 = get_best_18bitsfixed_opt_64((uint64_t)num_den_t, &x2);
            int32_t temp_numerator = (int64_t)log_18[log_in_num_1] - (int64_t)log_18[log_in_num_2];
            int32_t temp_power_num = -x1 + x2; 
            score_num_t += temp_numerator;
            num_power += temp_power_num;

            uint32_t d1 = ((uint32_t)sigma_nsq_t + (uint32_t)(var_x_t));
            uint32_t d2 = (sigma_nsq_t);
            int y1, y2;

            uint32_t log_in_den_1 = get_best_18bitsfixed_opt_64((uint64_t)d1, &y1);
            uint32_t log_in_den_2 = get_best_18bitsfixed_opt_64((uint64_t)d2, &y2);
            int32_t temp_denominator =  (int64_t)log_18[log_in_den_1] - (int64_t)log_18[log_in_den_2];
            int32_t temp_power_den = -y1 + y2;
            score_den_t += temp_denominator;
            den_power += temp_power_den;
        }
    }

    double add_exp = 1e-4*s_height*s_width;

    double power_double_num = (double)num_power;
    double power_double_den = (double)den_power;

    *score_num = (((double)score_num_t/(double)(1 << 26)) + power_double_num) + add_exp;
    *score_den = (((double)score_den_t/(double)(1<<26)) + power_double_den) + add_exp;
    *score += *score_num / *score_den;

    free(x_pad_t);
    free(y_pad_t);
    free(int_1_x_t);
    free(int_1_y_t);
    free(int_2_x_t);
    free(int_2_y_t);
    free(int_xy_t);

    ret = 0;

    return ret;
}