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

#include "integer_filters.h"
#include "common/macros.h"

#define VIF_COMPUTE_METRIC_R_SHIFT 6

//increase storage value to remove calculation to get log value
// uint32_t log_18[262144];

// just change the store offset to reduce multiple calculation when getting log value
void log_generate(uint32_t* log_18)
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

void integer_compute_metrics(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int32_t* var_x, int32_t* var_y, int32_t* cov_xy)
{
    int32_t mx, my;
    int64_t vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw];
            my = int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw];

            // (1/knorm) pending on all these (vx, vy ,cxy) - do this in next function
            vx = (int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) - (((int64_t)mx*mx)/kNorm); 
            vy = (int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) - (((int64_t)my * my)/kNorm);
            cxy = (int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) - (((int64_t)mx * my)/kNorm);

            var_x[i * (width - kw) + j] = vx < 0 ? 0 : (int32_t) (vx >> VIF_COMPUTE_METRIC_R_SHIFT); 
            var_y[i * (width - kw) + j] = vy < 0 ? 0 : (int32_t) (vy >> VIF_COMPUTE_METRIC_R_SHIFT);
            cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : (int32_t) (cxy >> VIF_COMPUTE_METRIC_R_SHIFT);
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
    int32_t* var_x_t, *var_y_t; 
    int32_t* cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    integer_integral_image(x_pad_t, r_width, r_height, int_1_x_t); 
    integer_integral_image(y_pad_t, r_width, r_height, int_1_y_t); 
    integer_integral_image_2(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t); 
    integer_integral_image_2(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t); 
    integer_integral_image_2(x_pad_t, y_pad_t, r_width, r_height, int_xy_t); 

    var_x_t = (int32_t*)malloc(sizeof(int32_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y_t = (int32_t*)malloc(sizeof(int32_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy_t = (int32_t*)malloc(sizeof(int32_t) * (r_width + 1 - kw) * (r_height + 1 - kh));

    integer_compute_metrics(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t);

    int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    int32_t* sv_sq_t = (int32_t*)malloc(sizeof(int32_t) * s_width * s_height);

    int64_t exp_t = 1;//exp*shift_val*shift_val; // using 1 because exp in Q32 format is still 0
    int64_t sigma_nsq_t = (int64_t)(sigma_nsq*shift_val*shift_val) >> VIF_COMPUTE_METRIC_R_SHIFT ;

    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;

    int64_t score_num_t = 0;
    int64_t num_power = 0;
    int64_t score_den_t = 0;
    int64_t den_power = 0;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;
            int32_t g_t_num = cov_xy_t[index]/k_norm;
            int32_t g_den = (var_x_t[index] + exp_t * k_norm)/k_norm;

            sv_sq_t[index] = (var_y_t[index] - ((int64_t)g_t_num * cov_xy_t[index])/g_den)/k_norm;

            if (var_x_t[index] < exp_t)
            {
                g_t_num = 0;
                sv_sq_t[index] = var_y_t[index];
                var_x_t[index] = 0;
            }
            
            if (var_y_t[index] < exp_t)
            {
                g_t_num = 0;
                sv_sq_t[index] = 0;
            }

            if((g_t_num < 0 && g_den > 0) || (g_den < 0 && g_t_num > 0))
            {
                sv_sq_t[index] = var_x_t[index];
                g_t_num = 0;
            }

            if (sv_sq_t[index] < exp_t)
                sv_sq_t[index] = exp_t;

            int64_t p1 = ((int64_t)g_t_num * g_t_num)/g_den;
            int32_t p2 = (var_x_t[index]/k_norm);
            int64_t n1 = p1 * p2;
            int64_t n2 = (((int64_t)g_den*(sv_sq_t[index])) + (int64_t)g_den*sigma_nsq_t);
            int64_t num_t = n2 + n1;
            int64_t num_den_t = n2;
            int x1, x2;
  
            uint32_t log_in_num_1 = get_best_18bitsfixed_opt_64((uint64_t)num_t, &x1);
            uint32_t log_in_num_2 = get_best_18bitsfixed_opt_64((uint64_t)num_den_t, &x2);
            int64_t temp_numerator = (int64_t)log_18[log_in_num_1] - (int64_t)log_18[log_in_num_2];
            int32_t temp_power_num = -x1 + x2; 
            score_num_t += temp_numerator;
            num_power += temp_power_num;

            uint32_t d1 = (uint32_t) sigma_nsq_t + (var_x_t[index]/k_norm);
            uint32_t d2 = (uint32_t) sigma_nsq_t;
            int y1, y2;

            uint32_t log_in_den_1 = get_best_18bitsfixed_opt_64((uint64_t)d1, &y1);
            uint32_t log_in_den_2 = get_best_18bitsfixed_opt_64((uint64_t)d2, &y2);
            int64_t temp_denominator =  (int64_t)log_18[log_in_den_1] - (int64_t)log_18[log_in_den_2];
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
    free(var_x_t);
    free(var_y_t);
    free(cov_xy_t);
    free(g_t);
    free(sv_sq_t);

    ret = 0;

    return ret;
}