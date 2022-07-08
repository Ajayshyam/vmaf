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

#define FLOATING_POINT 0
#define GET_VARIABLE_NAME(Variable) (#Variable)

//increase storage value to remove calculation to get log value
// uint32_t log_values[65537];
// uint32_t log_values[2147483648 * 2];

//just change the store offset to reduce multiple calculation when getting log value
// void log_generate()
// {
//     uint64_t i;
//     uint64_t start = (unsigned int)pow(2, 31);
//     uint64_t end = (unsigned int)pow(2, 32);
//     uint64_t diff = end - start;
// 	for (i = start; i < end; i++)
//     // for (i = 32767; i < 65536; i++)
//     {
//         // log_values[i] = (uint16_t)round(log2f((float)i) * 2048);
// 		log_values[i] = (uint32_t)round(log2((double)i) * (1 << 26));
//     }
// }

uint32_t log_32(uint32_t input)
{
    
    uint32_t log_out_1 = (uint32_t)round(log2((double)input) * (1 << 26));
    return log_out_1;
            
}

//divide get_best_16bitsfixed_opt for more improved performance as for input greater than 16 bit
// FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_greater(uint32_t temp, int *x)
// {
// 	int k = __builtin_clz(temp); // for int
// 	k = 16 - k;
// 	temp = temp >> k;
// 	*x = -k;
// 	return temp;
// }

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


FORCE_INLINE inline uint32_t get_best_32bitsfixed_opt_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp); // for long

    if (k > 32)  // temp < 2^47
    {
        k -= 32;
        temp = temp << k;
        *x = k;

    }
    else if (k < 31)  // temp > 2^48
    {
        k = 32 - k;
        temp = temp >> k;
        *x = -k;
    }
    else
    {
        *x = 0;
        if (temp >> 32)
        {
            temp = temp >> 1;
            *x = -1;
        }
    }

    return (uint32_t)temp;
}


//divide get_best_16bitsfixed_opt_64 for more improved performance as for input greater than 16 bit
// FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_greater_64(uint64_t temp, int *x)
// {
//     int k = __builtin_clzll(temp); // for long
//     k = 16 - k;
//     temp = temp >> k;
//     *x = -k;
//     return (uint16_t)temp;
// }

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

void integer_integral_image_2(const dwt2_dtype* src1, const dwt2_dtype* src2, size_t width, size_t height, int64_t* sum, int kw, int kh)
{
    int64_t *temp_sum = (int64_t*)calloc((width + 1) * (height + 1), sizeof(int64_t));

    for (size_t i=1; i<(height+1); i++)
    {
        for(size_t j=1; j<kw+1; j++)
        {
            int64_t val = (int64_t) src1[(i - 1) * width + (j - 1)] * src2[(i - 1) * width + (j - 1)];
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + val;
        }
        for(size_t j=kw+1; j<(width+1); j++)
        {
            int64_t val = (int64_t) src1[(i - 1) * width + (j - 1)] * src2[(i - 1) * width + (j - 1)];
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + val - ((int32_t)src1[(i-1)*width+j-kw-1] * src2[(i-1)*width+j-kw-1]);
        }
    }
    for (size_t j=1; j<(width+1); j++)
    {
        for (size_t i=1; i<kh+1; i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j];
        }
        for (size_t i=kh+1; i<(height+1); i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j] - temp_sum[(i-kh) * (width + 1) + j];
        }
    }
    free(temp_sum);
}

void integer_integral_image(const dwt2_dtype* src, size_t width, size_t height, int64_t* sum, int kw, int kh)
{
    int32_t *temp_sum = (int64_t*)calloc((width + 1) * (height + 1), sizeof(int32_t));

    for (size_t i=1; i<(height+1); i++)
    {
        for(size_t j=1; j<kw+1; j++)
        {
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + src[(i - 1) * width + (j - 1)];
        }
        for(size_t j=kw+1; j<(width+1); j++)
        {
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + src[(i - 1) * width + (j - 1)] - src[(i-1)*width+j-kw-1];
        }
    }
    for (size_t j=1; j<(width+1); j++)
    {
        for (size_t i=1; i<kh+1; i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j];
        }
        for (size_t i=kh+1; i<(height+1); i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j] - temp_sum[(i-kh) * (width + 1) + j];
        }
    }
    free(temp_sum);
}

void integer_compute_metrics(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int64_t* var_x, int64_t* var_y, int64_t* cov_xy)
{
    int64_t mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = int_1_x[(i + kh) * width + j + kw];
            my = int_1_y[(i + kh) * width + j + kw];

            vx = int_2_x[(i + kh) * width + j + kw] - ((mx*mx)/kNorm);
            vy = int_2_y[(i + kh) * width + j + kw] - ((my*my)/kNorm);
            cxy = int_xy[(i + kh) * width + j + kw] - ((mx * my)/kNorm);
            var_x[i * (width - kw) + j] = vx < 0 ? 0 : vx; 
            var_y[i * (width - kw) + j] = vy < 0 ? 0 : vy;
            cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : cxy;
        }
    }
}

int integer_compute_vif_funque(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, double* score, double* score_num, double* score_den, int k, int stride, double sigma_nsq, int64_t shift_val)
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
    int64_t* var_x_t, * var_y_t, * cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    //Q64
    integer_integral_image(x_pad_t, r_width, r_height, int_1_x_t, kw, kh); 
    integer_integral_image(y_pad_t, r_width, r_height, int_1_y_t, kw, kh); 
    integer_integral_image_2(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t, kw, kh); 
    integer_integral_image_2(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t, kw, kh); 
    integer_integral_image_2(x_pad_t, y_pad_t, r_width, r_height, int_xy_t, kw, kh); 

    var_x_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));

    //Q64
    integer_compute_metrics(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t);

    int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    int64_t* sv_sq_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);

    int64_t exp_t = 1;//exp*shift_val*shift_val; // using 1 because exp in Q32 format is still 0
    int64_t sigma_nsq_t = sigma_nsq*shift_val*shift_val ;

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
            int64_t g_t_num = cov_xy_t[index]/k_norm;
            int64_t g_den = (var_x_t[index] + exp_t * k_norm)/k_norm;

            sv_sq_t[index] = var_y_t[index] - (g_t_num * cov_xy_t[index])/g_den;

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

            int64_t p1 = (g_t_num * g_t_num)/g_den;
            int64_t p2 = (var_x_t[index]/k_norm);
            int64_t n1 = p1 * p2;
            int64_t n2 = ((g_den*(sv_sq_t[index]/k_norm)) + g_den*sigma_nsq_t);
            int64_t num_t = n2 + n1;
            int64_t num_den_t = n2;
            int x1, x2;

            uint32_t log_in_num_1 = get_best_32bitsfixed_opt_64((uint64_t)num_t, &x1);
            uint32_t log_in_num_2 = get_best_32bitsfixed_opt_64((uint64_t)num_den_t, &x2);

            int64_t temp_numerator = (int64_t)log_32(log_in_num_1) - (int64_t)log_32(log_in_num_2);
            int64_t temp_power_num = -x1 + x2; // 2^28
            score_num_t += temp_numerator;
            num_power += temp_power_num;

            int64_t d1 = sigma_nsq_t + (var_x_t[index]/k_norm);
            int64_t d2 = sigma_nsq_t;
            int y1, y2;

            uint32_t log_in_den_1 = get_best_32bitsfixed_opt_64((uint64_t)d1, &y1);
            uint32_t log_in_den_2 = get_best_32bitsfixed_opt_64((uint64_t)d2, &y2);
            int64_t temp_denominator =  (int64_t)log_32(log_in_den_1) - (int64_t)log_32(log_in_den_2);
            int64_t temp_power_den = -y1 + y2;
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