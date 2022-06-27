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
uint32_t log_values[2147483648 * 2];

//just change the store offset to reduce multiple calculation when getting log value
void log_generate()
{
    uint64_t i;
    uint64_t start = (unsigned int)pow(2, 31);
    uint64_t end = (unsigned int)pow(2, 32);
    uint64_t diff = end - start;
	for (i = start; i < end; i++)
    // for (i = 32767; i < 65536; i++)
    {
        // log_values[i] = (uint16_t)round(log2f((float)i) * 2048);
		log_values[i] = (uint32_t)round(log2((double)i) * (1 << 26));
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
FORCE_INLINE inline uint16_t get_best_16bitsfixed_opt_greater_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp); // for long
    k = 16 - k;
    temp = temp >> k;
    *x = -k;
    return (uint16_t)temp;
}


void reflect_pad(const funque_dtype* src, size_t width, size_t height, int reflect, funque_dtype* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(funque_dtype) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(funque_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(funque_dtype) * out_width);
    }
}

void reflect_pad_int(const dwt2_dtype* src, size_t width, size_t height, int reflect, dwt2_dtype* dest)
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


void integral_image_2(const funque_dtype* src1, const funque_dtype* src2, size_t width, size_t height, double* sum)
{
    // int index = 0;
    //   FILE * fp = fopen("x_float.txt", "w");
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            // index = i * (width + 1) + j;
            // if(index == 4410)
            //     printf("here \n");

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
            // fprintf(fp, " %f\n", sum[i * (width + 1) + j]);
            
        }
    }
    // fclose(fp);
  
    // int tmp = 0;
}

void integral_image(const funque_dtype* src, size_t width, size_t height, double* sum)
{
    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

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

void dump_data_d(char* file_name, float* data, int width, int height)
{
    FILE * fp = fopen(file_name, "w");
    int index = 0;
    for (size_t i = 0; i < (height); ++i)
    {  
        for (size_t j = 0; j < (width); ++j)
        {  
            index = i*width+j;
            if(j == (width-1))
                fprintf(fp, " %f\n", (float)data[index]);
            else
                fprintf(fp, " %f,", (float)data[index]);
        } 
    }

    fclose(fp);
}

void dump_data_i(char* file_name, int16_t* data, int width, int height)
{
    FILE * fp = fopen(file_name, "w");
    int index = 0;
    for (size_t i = 0; i < (height); ++i)
    {  
        for (size_t j = 0; j < (width); ++j)
        {  
            index = i*width+j;
            if(j == (width-1))
                fprintf(fp, " %ld\n", (int16_t)data[index]);
            else
                fprintf(fp, " %ld,", (int16_t)data[index]);
        } 
    }

    fclose(fp);
}

void dump_data_i_convert(char* file_name, int16_t* data, int width, int height, int64_t shift)
{
    FILE * fp = fopen(file_name, "w");
    int index = 0;
    for (size_t i = 0; i < (height); ++i)
    {  
        for (size_t j = 0; j < (width); ++j)
        {  
            index = i*width+j;
            double value = (double)data[index]/((double)shift);
            if(j == (width-1))
                fprintf(fp, " %f\n", value);
            else
                fprintf(fp, " %f,", value);
        } 
    }

    fclose(fp);
}

void dump_data(char* file_name, double* data, int width, int height)
{
    FILE * fp = fopen(file_name, "w");
    int index = 0;
    for (size_t i = 0; i < (height); ++i)
    {  
        for (size_t j = 0; j < (width); ++j)
        {  
            index = i*width+j;
            if(j == (width-1))
                fprintf(fp, " %f\n", (double)data[index]);
            else
                fprintf(fp, " %f,", (double)data[index]);
        } 
    }

    fclose(fp);
}

void dump_data_int(char* file_name, int64_t* data, int width, int height)
{
    FILE * fp = fopen(file_name, "w");
    int index = 0;
    for (size_t i = 0; i < (height); ++i)
    {  
        for (size_t j = 0; j < (width); ++j)
        {  
            index = i*width+j;
            if(j == (width-1))
                fprintf(fp, " %ld\n", (int64_t)data[index]);
            else
                fprintf(fp, " %ld,", (int64_t)data[index]);
        } 
    }

    fclose(fp);
}

void integral_image_2_int(const dwt2_dtype* src1, const dwt2_dtype* src2, size_t width, size_t height, int64_t* sum)
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

void integral_image_int(const dwt2_dtype* src, size_t width, size_t height, int64_t* sum)
{
    double st1, st2, st3;
    
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

void compute_metrics_int(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int64_t* var_x, int64_t* var_y, int64_t* cov_xy, int64_t shift_val)
{
    int64_t mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            mx = int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw];
            my = int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw];

            // (1/knorm) pending on all these (vx, vy ,cxy)
            vx = (int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) - ((mx*mx)/kNorm); 
            vy = (int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) - ((my * my)/kNorm);
            cxy = (int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) - ((mx * my)/kNorm);

            // double mdx =  ((double)(mx)/(double)(shift_val))/kNorm;
            // double mdy =  ((double)(my)/(double)(shift_val))/kNorm;
            // double vdx = ((double)(vx)/(double)(shift_val*shift_val))/kNorm;
            // double vdy = ((double)(vy)/(double)(shift_val*shift_val))/kNorm;
            // double cdxy = ((double)(cxy)/(double)(shift_val*shift_val))/kNorm;

            var_x[i * (width - kw) + j] = vx < 0 ? 0 : vx; // Divide by Q2n to get back float
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
        }
    }
}


int compute_vif_funque(const dwt2_dtype* x_t, const dwt2_dtype* y_t, const funque_dtype* x, const funque_dtype* y, size_t width, size_t height, double* score, double* score_num, double* score_den, int k, int stride, double sigma_nsq, int64_t shift_val)
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
    double exp = (double)1e-10;
    int index = 0;

#if FLOATING_POINT
    funque_dtype* x_pad, * y_pad;

    x_pad = (funque_dtype*)malloc(sizeof(funque_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (funque_dtype*)malloc(sizeof(funque_dtype) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

    reflect_pad(x, width, height, x_reflect, x_pad);
    reflect_pad(y, width, height, y_reflect, y_pad);

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

    double* g = (double*)malloc(sizeof(double) * s_width * s_height);
    double* sv_sq = (double*)malloc(sizeof(double) * s_width * s_height);

    double score_double = (double)0;
    double score_num_double = 0;
    double score_den_double = 0;
#endif

    dwt2_dtype* x_pad_t, *y_pad_t;
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));
    reflect_pad_int(x_t, width, height, x_reflect, x_pad_t);
    reflect_pad_int(y_t, width, height, y_reflect, y_pad_t);

    int64_t* int_1_x_t, * int_1_y_t, * int_2_x_t, * int_2_y_t, * int_xy_t;
    int64_t* var_x_t, * var_y_t, * cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    integral_image_int(x_pad_t, r_width, r_height, int_1_x_t);
    integral_image_int(y_pad_t, r_width, r_height, int_1_y_t);
    integral_image_2_int(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t);
    integral_image_2_int(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t);
    integral_image_2_int(x_pad_t, y_pad_t, r_width, r_height, int_xy_t);

    var_x_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));

    compute_metrics_int(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t, shift_val);

    int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    int64_t* sv_sq_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);

    int64_t exp_t = exp*shift_val*shift_val;
    int64_t sigma_nsq_t = sigma_nsq*shift_val*shift_val ;

    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;

    int64_t score_num_t = 0;
    int64_t score_den_t = 0;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;
            int64_t g_t_num = cov_xy_t[index]/k_norm;
            int64_t g_den = (var_x_t[index] + exp_t * k_norm)/k_norm;

            sv_sq_t[index] = var_y_t[index] - (g_t_num * cov_xy_t[index])/g_den;

#if FLOATING_POINT
            g[index] = cov_xy[index] / (var_x[index] + exp);
            sv_sq[index] = var_y[index] - g[index] * cov_xy[index];

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
#endif

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

            if (sv_sq_t[index] < exp)
                sv_sq_t[index] = exp;

            // double works till here
            // double gd = ((double)g_t_num/((double)shift_val*shift_val))/ ((double)g_den / ((double)shift_val*shift_val));
            // double vard = (double)var_x_t[index]/((double)shift_val*shift_val*k_norm);
            // double svsqd = (double)sv_sq_t[index]/((double)shift_val*shift_val*k_norm);
            // double score_num_tmp = (log((double)1 + gd*gd * vard / (svsqd + sigma_nsq)));
            // double score_den_tmp = (log((double)1 + vard / sigma_nsq));
  
            int64_t n1 = (g_t_num * g_t_num * var_x_t[index]/k_norm)/g_den;
            int64_t n2 = ((g_den*sv_sq_t[index]/k_norm) + g_den*sigma_nsq_t);
            int64_t num_t = n2 + n1;
            int64_t num_den_t = n2;
            int x1, x2;
            // uint16_t log_in_num_1 = get_best_16bitsfixed_opt_64((uint64_t)num_t, &x1);
            // uint16_t log_in_num_2 = get_best_16bitsfixed_opt_64((uint64_t)num_den_t, &x2);
            uint32_t log_in_num_1 = get_best_32bitsfixed_opt_64((uint64_t)num_t, &x1);
            uint32_t log_in_num_2 = get_best_32bitsfixed_opt_64((uint64_t)num_den_t, &x2);

            uint64_t temp_numerator = (log_values[log_in_num_1] + (-x1) *  (1 << 26))- (log_values[log_in_num_2] + (-x2) * (1 << 26));
            score_num_t += temp_numerator;

            int64_t d1 = sigma_nsq_t + (var_x_t[index]/k_norm);
            int64_t d2 = sigma_nsq_t;
            int y1, y2;
            // uint16_t log_in_den_1 = get_best_16bitsfixed_opt_64((uint64_t)d1, &y1);
            // uint16_t log_in_den_2 = get_best_16bitsfixed_opt_64((uint64_t)d2, &y2);
            uint32_t log_in_den_1 = get_best_32bitsfixed_opt_64((uint64_t)d1, &y1);
            uint32_t log_in_den_2 = get_best_32bitsfixed_opt_64((uint64_t)d2, &y2);
            // score_den_t += log_values[log_in_den_1] + (-y1 - 12) * 2048 - log_values[log_in_den_2] + (-y2 - 12) * 2048;
            uint64_t temp_denominator =  (log_values[log_in_den_1] + (-y1) * (1 << 26)) - (log_values[log_in_den_2] +(-y2) * (1 << 26));
            score_den_t += temp_denominator;

#if FLOATING_POINT
            // double num_sum = (double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq);
            // double den_sum = (double)1 + var_x[index] / sigma_nsq;

            score_num_double += (log2(((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq))));
            score_den_double += (log2(((double)1 + var_x[index] / sigma_nsq)));

            // double numf = log2f((float)((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)));
            // double denf = log2f((float)((double)1 + var_x[index] / sigma_nsq));
            // double numd = ((double)temp_numerator)/((double)(1<<26));
            // double dend = ((double)temp_denominator)/((double) (1<<26));
#endif
            
        }
    }
    double add_exp = 1e-4*s_height*s_width;

#if FLOATING_POINT
    score_double += ((score_num_double + add_exp) / (score_den_double + add_exp));
#endif

    double num = ((double)score_num_t)/((double) (1<<26));
    double den = ((double)score_den_t)/((double)(1<<26));
    *score_num = num;
    *score_den = den;
    *score += (( num + add_exp) / ( den + add_exp));

#if FLOATING_POINT
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
#endif

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