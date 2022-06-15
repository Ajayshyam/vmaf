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
#include <stdint.h>
#include "common/macros.h"

//increase storage value to remove calculation to get log value
uint16_t log_values[65537];
int64_t shift_k = (int64_t)pow(2,15);
int64_t shift_d = (int64_t)pow(2,16);

//just change the store offset to reduce multiple calculation when getting log value
void log_generate()
{
    int i;
	for (i = 32767; i < 65536; i++)
    {
		log_values[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
}

//divide get_best_16bitsfixed_opt for more improved performance as for input greater than 16 bit
FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_greater(uint32_t temp, int *x)
{
	int k = __builtin_clz(temp); // for int
	k = 16 - k;
	temp = temp >> k;
	*x = -k;
	return temp;
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

//divide get_best_16bitsfixed_opt_64 for more improved performance as for input greater than 16 bit
FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_greater_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp); // for long
    k = 16 - k;
    temp = temp >> k;
    *x = -k;
    return (uint16_t)temp;
}


void reflect_pad(const double* src, size_t width, size_t height, int reflect, double* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(double) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(double) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(double) * out_width);
    }
}

void integral_image_2(const double* src1, const double* src2, size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            double val = (double)src1[(i - 1) * width + (j - 1)] * (double)src2[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void integral_image(const double* src, size_t width, size_t height, double* sum)
{
    int index = 0;
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;
            index = i * (width + 1) + j;
            double val = (double)src[(i - 1) * width + (j - 1)];

            if (i >= 1)
            {
                val += sum[(i - 1) * (width + 1) + j];
                if (j >= 1)
                {
                    val += sum[i * (width + 1) + j - 1] - sum[(i - 1) * (width + 1) + j - 1];
                }
            }
            else {
                if (j >= 1)
                {
                    val += sum[i * width + j - 1];
                }
            }
            sum[i * (width + 1) + j] = val;
        }
    }
}

void integral_image_2_int(const double* src1, const double* src2, size_t width, size_t height, int64_t* sum)
{
     int64_t shift = (int64_t)pow(2,31);
    int64_t shift_0 = 1;

    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            uint64_t val = (uint64_t)((src1[(i - 1) * width + (j - 1)] * src2[(i - 1) * width + (j - 1)]) * shift);

            val += (int64_t)(sum[(i - 1) * (width + 1) + j]*shift_0);
            val += (int64_t)(sum[i * (width + 1) + j - 1]*shift_0) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]*shift_0);

            sum[i * (width + 1) + j] = val;
        }
    }
}

void integral_image_int(const double* src, size_t width, size_t height, int64_t* sum)
{
    int64_t shift = (int64_t)pow(2,31);
    int64_t shift_0 = 1;

    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            int64_t val = (int64_t)(src[(i - 1) * width + (j - 1)] *shift); //64 to avoid overflow  

            val += (int64_t)(sum[(i - 1) * (width + 1) + j]*shift_0);
            val += (int64_t)(sum[i * (width + 1) + j - 1]*shift_0) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]*shift_0);
            sum[i * (width + 1) + j] = val;

        }
    }
}

void compute_metrics_int(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int64_t* var_x, int64_t* var_y, int64_t* cov_xy)
{
    double kNorm_inv = 1/kNorm;
    int64_t norm_inv = (int64_t)(kNorm_inv* shift_k);
    int64_t mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx =  (((int_1_x[i * width + j] * norm_inv) - (int_1_x[i * width + j + kw]* norm_inv ) - (int_1_x[(i + kh) * width + j] * norm_inv) + (int_1_x[(i + kh) * width + j + kw]* norm_inv ))/shift_k) ;
            my = (((int_1_y[i * width + j] * norm_inv) - (int_1_y[i * width + j + kw]* norm_inv ) - (int_1_y[(i + kh) * width + j]* norm_inv ) + (int_1_y[(i + kh) * width + j + kw] * norm_inv))/shift_k) ;

            vx = ((((int_2_x[i * width + j] ) - (int_2_x[i * width + j + kw] ) - (int_2_x[(i + kh) * width + j] ) + (int_2_x[(i + kh) * width + j + kw] ))) * norm_inv*shift_d) - ((mx * mx));
            vy = ((((int_2_y[i * width + j] ) - (int_2_y[i * width + j + kw] ) - (int_2_y[(i + kh) * width + j] ) + (int_2_y[(i + kh) * width + j + kw] ))) * norm_inv*shift_d) - ((my * my));

            cxy = ((((int_xy[i * width + j] ) - (int_xy[i * width + j + kw] ) - (int_xy[(i + kh) * width + j] ) + (int_xy[(i + kh) * width + j + kw] ))) * norm_inv*shift_d) - ((mx * my));

            // var_x[i * (width - kw) + j] = vx < 0 ? 0 : (double)vx/ (double)(shift_k*shift_k*shift_d*shift_d);
            // var_y[i * (width - kw) + j] = vy < 0 ? 0 : (double)vy/ (double)(shift_k*shift_k*shift_d*shift_d);
            // cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : (double)cxy/ (double)(shift_k*shift_k*shift_d*shift_d);

            var_x[i * (width - kw) + j] = vx < 0 ? 0 : vx;
            var_y[i * (width - kw) + j] = vy < 0 ? 0 : vy;
            cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? 0 : cxy;
        }
    }
}

void compute_metrics(const double* int_1_x, const double* int_1_y, const double* int_2_x, const double* int_2_y, const double* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, double* var_x, double* var_y, double* cov_xy)
{
    double mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = (int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw]) / kNorm;
            my = (int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw]) / kNorm;

            vx = ((int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) / kNorm) - (mx * mx);
            vy = ((int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) / kNorm) - (my * my);

            cxy = ((int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) / kNorm) - (mx * my);

            var_x[i * (width - kw) + j] = vx < 0 ? (double)0 : vx;
            var_y[i * (width - kw) + j] = vy < 0 ? (double)0 : vy;
            cov_xy[i * (width - kw) + j] = (vx < 0 || vy < 0) ? (double)0 : cxy;
            int jk = -1;
        }
    }
}

double largest(double *arr, int n)
{
    int i;
    
    // Initialize maximum element
    double max = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
 
    return max;
}

uint64_t largest_int(uint64_t *arr, int n)
{
    int i;
    
    // Initialize maximum element
    uint64_t max = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
 
    return max;
}

int compute_vif_funque(const double* x, const double* y, size_t width, size_t height, double* score, double* score_num, double* score_den, int k, int stride, double sigma_nsq)
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = k * k;

    double* x_pad, * y_pad;

    int x_reflect = (int)((kh - stride) / 2);
    int y_reflect = (int)((kw - stride) / 2);

    x_pad = (double*)malloc(sizeof(double) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (double*)malloc(sizeof(double) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

    reflect_pad(x, width, height, x_reflect, x_pad);
    reflect_pad(y, width, height, y_reflect, y_pad);
    size_t r_width = width + (2 * x_reflect);
    size_t r_height = height + (2 * x_reflect);

    double* int_1_x, * int_1_y, * int_2_x, * int_2_y, * int_xy;
    double* var_x, * var_y, * cov_xy;

    int_1_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_1_y = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_2_x = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_2_y = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));
    int_xy = (double*)calloc((r_width + 1) * (r_height + 1), sizeof(double));

    integral_image(x_pad, r_width, r_height, int_1_x);
    integral_image(y_pad, r_width, r_height, int_1_y);
    integral_image_2(x_pad, x_pad, r_width, r_height, int_2_x);
    integral_image_2(y_pad, y_pad, r_width, r_height, int_2_y);
    integral_image_2(x_pad, y_pad, r_width, r_height, int_xy);

    var_x = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));

    compute_metrics(int_1_x, int_1_y, int_2_x, int_2_y, int_xy, r_width + 1, r_height + 1, kh, kw, k_norm, var_x, var_y, cov_xy);

    // double d1 = largest(var_x, (r_width + 1 - kw) * (r_height + 1 - kh));
    // double d2 = largest(var_y, (r_width + 1 - kw) * (r_height + 1 - kh));
    // double d3 = largest(cov_xy, (r_width + 1 - kw) * (r_height + 1 - kh));

    int64_t* int_1_x_t, * int_1_y_t, * int_2_x_t, * int_2_y_t, * int_xy_t;
    int64_t* var_x_t, * var_y_t, * cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    integral_image_int(x_pad, r_width, r_height, int_1_x_t);
    integral_image_int(y_pad, r_width, r_height, int_1_y_t);
    integral_image_2_int(x_pad, x_pad, r_width, r_height, int_2_x_t);
    integral_image_2_int(y_pad, y_pad, r_width, r_height, int_2_y_t);
    integral_image_2_int(x_pad, y_pad, r_width, r_height, int_xy_t);

    var_x_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));

    compute_metrics_int(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t);

    // double d1_t = largest(var_x_t, (r_width + 1 - kw) * (r_height + 1 - kh));
    // double d2_t = largest(var_y_t, (r_width + 1 - kw) * (r_height + 1 - kh));
    // double d3_t = largest(cov_xy_t, (r_width + 1 - kw) * (r_height + 1 - kh));
    
    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;

    double* g = (double*)malloc(sizeof(double) * s_width * s_height);
    double* sv_sq = (double*)malloc(sizeof(double) * s_width * s_height);
    int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    int64_t* sv_sq_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    double exp = (double)1e-10;
    int index;
    int64_t pending_shifts = shift_k*shift_k*shift_d*shift_d;
    int64_t exp_t = exp *shift_d*shift_d; // Q32
    int64_t sigma_nsq_t = sigma_nsq * shift_d * shift_d; //Q32

    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;
    
    double score_t = (double)0;
    double score_num_t = (double)0;
    double score_den_t = (double)0;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;
            g[index] = cov_xy[index] / (var_x[index] + exp);
            sv_sq[index] = var_y[index] - g[index] * cov_xy[index];
            
            //Q32 = Q62/Q30
            var_x_t[index] = var_x_t[index] >> 30;

            //Q30 = Q62/ Q32
            g_t[index] = cov_xy_t[index] / (var_x_t[index] + exp_t);

            //Q32 = Q62/Q30
            cov_xy_t[index] = cov_xy_t[index] >> 30;

            //Q62 = Q62 - Q30*Q32
            sv_sq_t[index] = var_y_t[index] - g_t[index] * cov_xy_t[index];

            if (var_x[index] < exp)
            {
                g[index] = (double)0;
                sv_sq[index] = var_y[index];
                var_x[index] = (double)0;
            }

            if (var_y[index] < exp)
            {
                g[index] = (double)0;
                sv_sq[index] = (double)0;
            }

             if (g[index] < 0)
            {
                sv_sq[index] = var_x[index];
                g[index] = (double)0;
            }

            if (sv_sq[index] < exp)
                sv_sq[index] = exp;

            if (var_x_t[index] < exp_t)
            {
                g_t[index] = 0;
                sv_sq_t[index] = var_y_t[index];
                var_x_t[index] = 0;
            }
            
            if (var_y_t[index] < exp_t)
            {
                g_t[index] = 0;
                sv_sq_t[index] = 0;
            }

            if (g_t[index] < 0)
            {
                sv_sq_t[index] = var_x_t[index];
                g_t[index] = 0;
            }

            if (sv_sq_t[index] < exp)
                sv_sq_t[index] = exp;

            //Q16 = Q30/Q14
            g_t[index] = g_t[index] >> 14;

            //Q32 = Q62/Q30
            sv_sq_t[index] = sv_sq_t[index] >> 30;

            //Q30 = Q32/Q2
            var_x_t[index] = var_x_t[index] >> 2;

            //Q30
            // int64_t exp_add = (1e-4 * (1 << 30));

            //Q30 = Q30+(Q16*Q16*Q30/(Q32 + Q32))
            int64_t num_t= (1 << 30) +  g_t[index] * g_t[index] * var_x_t[index] / (sv_sq_t[index] + sigma_nsq_t);
            int64_t tmp_num_t = num_t ;
            int x;
            uint16_t log_in_num = get_best_16bitsfixed_opt_64((uint64_t)tmp_num_t, &x);
            // num_t_temp = log_in * 2^x;  log2f(log_in * 2^x/2^offset) = log2f(log_in) + log2f(2^x) - log2f(2^pffset) = log2f(log_in) + x - offset
            score_num_t += log_values[log_in_num] + (-x - 30) * 2048;
            double tn = score_num_t / 2048 ;

            //Q30 = Q30+(Q30*Q16*Q16)/Q32
            int64_t den_t = (1 <<30) + (var_x_t[index] *shift_d * shift_d)/sigma_nsq_t;
            int64_t tmp_den_t = den_t;
            int y;
            uint16_t log_in_den = get_best_16bitsfixed_opt_64((uint64_t)tmp_den_t, &y);
            score_den_t += log_values[log_in_den] + (-y - 30) * 2048;
            double td = score_den_t / 2048;

            // double num_sum = (double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq);
            double num_int = (double)num_t/(double)(pow(2, 30));
            // double den_sum = (double)1 + var_x[index] / sigma_nsq;
            double den_int = (double)den_t/(double)(pow(2, 30));
            // *score_num += (log((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)));
            // *score_den += (log((double)1 + var_x[index] / sigma_nsq));

            *score_num += (log(num_int) + (double)1e-4);
            *score_den += (log(den_int) + (double)1e-4);
            // int tmpr = 0;
        }
    }
    double add_exp = 1e-4*s_height*width;
    *score += ((*score_num + add_exp) / (*score_den + add_exp));
    score_t += ((((double)score_num_t) + add_exp)/(((double)score_den_t)+add_exp));


    free(x_pad);
    free(y_pad);
    free(int_1_x);
    free(int_1_y);
    free(int_2_x);
    free(int_2_y);
    free(int_xy);
    free(var_x);
    free(var_y);
    free(cov_xy);
    free(g);
    free(sv_sq);
    ret = 0;

    return ret;
}