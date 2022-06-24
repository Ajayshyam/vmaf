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

#define FIXED_POINT 1
#define VIF_KNORM_SHIFT 12

#define GET_VARIABLE_NAME(Variable) (#Variable)

//increase storage value to remove calculation to get log value
uint32_t log_values[65537];
int64_t shift_k = (int64_t)pow(2,15);
int64_t shift_d = (int64_t)pow(2,16);

//just change the store offset to reduce multiple calculation when getting log value
void log_generate()
{
    uint32_t i;
	for (i = (1 << 31); i < (1 << 32); i++)
    {
		log_values[i] = (uint32_t)round(log2f((float)i) * (1 << 15) * 2048);
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

//Qn input, Qn output (with overflow, hence stored in Q64)
// void integral_image_2_int(const dwt2_dtype* src1, const dwt2_dtype* src2, size_t width, size_t height, double* tmp_sum, int64_t* sum, int64_t shift_val)
void integral_image_2_int(const dwt2_dtype* src1, const dwt2_dtype* src2, size_t width, size_t height, int64_t* sum)
{
    //  double st1 = 0, st2 = 0, st3 = 0;

    for (size_t i = 0; i < (height + 1); ++i)
    {
        for (size_t j = 0; j < (width + 1); ++j)
        {
            if (i == 0 || j == 0)
                continue;

            int64_t val = (((int64_t)src1[(i - 1) * width + (j - 1)] * (int64_t)src2[(i - 1) * width + (j - 1)]));
            val += (int64_t)(sum[(i - 1) * (width + 1) + j]);
            // st1 = (double)val/((double)shift_val*shift_val);
            val += (int64_t)(sum[i * (width + 1) + j - 1]) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]);
            // st2 = (double)val/((double)shift_val*shift_val);
            sum[i * (width + 1) + j] = val;
            // st3 = (double)val/((double)shift_val*shift_val);

            // tmp_sum[i * (width + 1) + j] = (double)val/((double)shift_val*shift_val);

            // int jtk = 0;
        }
    }

}

//Qn input, Qn output (with overflow, hence stored in Q64)
// void integral_image_int(const dwt2_dtype* src, size_t width, size_t height, double* tmp_sum, int64_t* sum, int64_t shift_val)
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
            // st1 = (double)val/((double)shift_val);
            val += (int64_t)(sum[i * (width + 1) + j - 1]) - (int64_t)(sum[(i - 1) * (width + 1) + j - 1]);
            // st2 = (double)val/((double)shift_val);
            sum[i * (width + 1) + j] = val;
            // st3 = (double)val/((double)shift_val);

            // tmp_sum[i * (width + 1) + j] = (double)val/((double)shift_val);

            // int jtk = 0;

              
        }
    }
}

//Returns Q2n as output when Qn is input
// void compute_metrics_int(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int64_t* var_x, int64_t* var_y, int64_t* cov_xy, int64_t shift_val)
void compute_metrics_int(const int64_t* int_1_x, const int64_t* int_1_y, const int64_t* int_2_x, const int64_t* int_2_y, const int64_t* int_xy, size_t width, size_t height, size_t kh, size_t kw, double kNorm, int64_t* var_x, int64_t* var_y, int64_t* cov_xy, int64_t shift_val)
{
    // assert(VIF_KNORM_SHIFT >= 12);

    double kNorm_inv = 1/kNorm;
    int64_t norm_inv = (int64_t)(kNorm_inv * (1 << VIF_KNORM_SHIFT)); //VIF_KNORM_SHIFT Q12
    int64_t mx, my, vx, vy, cxy;

    for (size_t i = 0; i < (height - kh); i++)
    {
        for (size_t j = 0; j < (width - kw); j++)
        {
            // Qn = (Qn * Qn)/Qn
            // mx =  (((int_1_x[i * width + j] * norm_inv) - (int_1_x[i * width + j + kw]* norm_inv ) - (int_1_x[(i + kh) * width + j] * norm_inv) + (int_1_x[(i + kh) * width + j + kw]* norm_inv )) >> 6);
            // my = (((int_1_y[i * width + j] * norm_inv) - (int_1_y[i * width + j + kw]* norm_inv ) - (int_1_y[(i + kh) * width + j]* norm_inv ) + (int_1_y[(i + kh) * width + j + kw] * norm_inv)) >> 6) ;

            //Qshift_val
            mx = int_1_x[i * width + j] - int_1_x[i * width + j + kw] - int_1_x[(i + kh) * width + j] + int_1_x[(i + kh) * width + j + kw];
            my = int_1_y[i * width + j] - int_1_y[i * width + j + kw] - int_1_y[(i + kh) * width + j] + int_1_y[(i + kh) * width + j + kw];

            // Q2n = (Q2n * Qn)
            // vx = ((((int_2_x[i * width + j] ) - (int_2_x[i * width + j + kw] ) - (int_2_x[(i + kh) * width + j] ) + (int_2_x[(i + kh) * width + j + kw] ))) * norm_inv) - ((mx * mx));
            // vy = ((((int_2_y[i * width + j] ) - (int_2_y[i * width + j + kw] ) - (int_2_y[(i + kh) * width + j] ) + (int_2_y[(i + kh) * width + j + kw] ))) * norm_inv) - ((my * my));
            // cxy = ((((int_xy[i * width + j] ) - (int_xy[i * width + j + kw] ) - (int_xy[(i + kh) * width + j] ) + (int_xy[(i + kh) * width + j + kw] ))) * norm_inv) - ((mx * my));

            // (1/knorm) pending on all these (vx, vy ,cxy)
            vx = (int_2_x[i * width + j] - int_2_x[i * width + j + kw] - int_2_x[(i + kh) * width + j] + int_2_x[(i + kh) * width + j + kw]) - ((mx*mx)/kNorm); 
            vy = (int_2_y[i * width + j] - int_2_y[i * width + j + kw] - int_2_y[(i + kh) * width + j] + int_2_y[(i + kh) * width + j + kw]) - ((my * my)/kNorm);
            cxy = (int_xy[i * width + j] - int_xy[i * width + j + kw] - int_xy[(i + kh) * width + j] + int_xy[(i + kh) * width + j + kw]) - ((mx * my)/kNorm);

            // double mdx =  ((double)(mx)/(double)(shift_val))/kNorm;
            // double mdy =  ((double)(my)/(double)(shift_val))/kNorm;
            // double vdx = ((double)(vx)/(double)(shift_val*shift_val))/kNorm;
            // double vdy = ((double)(vy)/(double)(shift_val*shift_val))/kNorm;
            // double cdxy = ((double)(cxy)/(double)(shift_val*shift_val))/kNorm;

            // var_x[i * (width - kw) + j] = vdx < 0 ? 0 : vdx; // Divide by Q2n to get back float
            // var_y[i * (width - kw) + j] = vdy < 0 ? 0 : vdy;
            // cov_xy[i * (width - kw) + j] = (vdx < 0 || vdy < 0) ? 0 : cdxy;

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

    funque_dtype* x_pad, * y_pad;

    int x_reflect = (int)((kh - stride) / 2);
    int y_reflect = (int)((kw - stride) / 2);

    x_pad = (funque_dtype*)malloc(sizeof(funque_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (funque_dtype*)malloc(sizeof(funque_dtype) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

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

    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;

    double* g = (double*)malloc(sizeof(double) * s_width * s_height);
    double* sv_sq = (double*)malloc(sizeof(double) * s_width * s_height);
    
    double exp = (double)1e-10;
    int index;

#if FIXED_POINT
    // dwt2_dtype* x_pad_t, *y_pad_t;
    // x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    // y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));
    // reflect_pad_int(x_t, width, height, x_reflect, x_pad_t);
    // reflect_pad_int(y_t, width, height, y_reflect, y_pad_t);

    // int64_t* int_1_x_t, * int_1_y_t, * int_2_x_t, * int_2_y_t, * int_xy_t;
    // int64_t* var_x_t, * var_y_t, * cov_xy_t;

    // int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    // int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    // int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    // int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    // int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    // integral_image_int(x_pad_t, r_width, r_height, int_1_x_t);
    // integral_image_int(y_pad_t, r_width, r_height, int_1_y_t);
    // integral_image_2_int(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t);
    // integral_image_2_int(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t);
    // integral_image_2_int(x_pad_t, y_pad_t, r_width, r_height, int_xy_t);

    // // dump_data("int_1x_float.csv", int_1_x, r_width+1, r_height+1);
    // // dump_data_int("int_1x_int.csv", int_1_x_t, r_width+1, r_height+1);
    // // dump_data("int_2x_float.csv", int_2_x, r_width+1, r_height+1);
    // // dump_data_int("int_2x_int.csv", int_2_x_t, r_width+1, r_height+1);

    // var_x_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    // var_y_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));
    // cov_xy_t = (double*)malloc(sizeof(double) * (r_width + 1 - kw) * (r_height + 1 - kh));

    // compute_metrics_int(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t, shift_val);

    // int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    // int64_t* sv_sq_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    // int64_t pending_shifts = shift_val*shift_val; // Q2n
    // int64_t exp_t = exp *shift_d*shift_d; // Q32
    // int64_t sigma_nsq_t = sigma_nsq * shift_d * shift_d; //Q32

    // double score_t = (double)0;
    // int64_t score_num_t = 0;
    // int64_t score_den_t = 0;

    //-----------------------------------------------------------------------------------------------------------

    dwt2_dtype* x_pad_t, *y_pad_t;
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));
    reflect_pad_int(x_t, width, height, x_reflect, x_pad_t);
    reflect_pad_int(y_t, width, height, y_reflect, y_pad_t);

    // dump_data_i("reflect_int.csv", x_pad_t, r_width, r_height);
    // dump_data_i_convert("reflect_int_convert.csv", x_pad_t, r_width, r_height, shift_val);
    // dump_data_d("reflect_float.csv", x_pad, r_width, r_height);

    int64_t* int_1_x_t, * int_1_y_t, * int_2_x_t, * int_2_y_t, * int_xy_t;
    int64_t* var_x_t, * var_y_t, * cov_xy_t;

    int_1_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_1_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_x_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_2_y_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int_xy_t = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    int64_t* t1 = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int64_t* t2 = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int64_t* t3 = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int64_t* t4 = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));
    int64_t* t5 = (int64_t*)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

    // integral_image_int(x_pad_t, r_width, r_height, int_1_x_t, t1, shift_val);
    // integral_image_int(y_pad_t, r_width, r_height, int_1_y_t, t2, shift_val);
    // integral_image_2_int(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t, t3, shift_val);
    // integral_image_2_int(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t, t4, shift_val);
    // integral_image_2_int(x_pad_t, y_pad_t, r_width, r_height, int_xy_t, t5, shift_val);

    integral_image_int(x_pad_t, r_width, r_height, int_1_x_t);
    integral_image_int(y_pad_t, r_width, r_height, int_1_y_t);
    integral_image_2_int(x_pad_t, x_pad_t, r_width, r_height, int_2_x_t);
    integral_image_2_int(y_pad_t, y_pad_t, r_width, r_height, int_2_y_t);
    integral_image_2_int(x_pad_t, y_pad_t, r_width, r_height, int_xy_t);

    // dump_data("int_1x_float.csv", int_1_x, r_width+1, r_height+1);
    // dump_data_int("int_1x_int.csv", int_1_x_t, r_width+1, r_height+1);
    // dump_data("int_2x_float.csv", int_2_x, r_width+1, r_height+1);
    // dump_data("int_2x_int.csv", int_2_x_t, r_width+1, r_height+1);

    var_x_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    var_y_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));
    cov_xy_t = (int64_t*)malloc(sizeof(int64_t) * (r_width + 1 - kw) * (r_height + 1 - kh));

    compute_metrics_int(int_1_x_t, int_1_y_t, int_2_x_t, int_2_y_t, int_xy_t, r_width + 1, r_height + 1, kh, kw, (double)k_norm, var_x_t, var_y_t, cov_xy_t, shift_val);

    int64_t* g_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    int64_t* sv_sq_t = (int64_t*)malloc(sizeof(int64_t) * s_width * s_height);
    // double pending_shifts = shift_val*shift_val; // Q2n
    int64_t exp_t = exp*shift_val*shift_val; // Q32
    int64_t sigma_nsq_t = sigma_nsq*shift_val*shift_val ; //Q32

    double score_t = (double)0;
    double score_num_t = 0;
    double score_den_t = 0;
#endif

    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;
            g[index] = cov_xy[index] / (var_x[index] + exp);
            sv_sq[index] = var_y[index] - g[index] * cov_xy[index];
            
#if FIXED_POINT
            // g_t[index] = cov_xy_t[index] / (var_x_t[index] + exp * k_norm);
            // sv_sq_t[index] = var_y_t[index] - g_t[index] * cov_xy_t[index];

            int64_t g_t_num = cov_xy_t[index]/k_norm;
            int64_t g_den = (var_x_t[index] + exp_t * k_norm)/k_norm;
            // -ve,  var_x_t[index] + exp * k_norm) < 0 ||  cov_xy_t[index] < 0

            
            sv_sq_t[index] = var_y_t[index] - (g_t_num * cov_xy_t[index])/g_den;

            // double tmp = (double)var_x_t[index]/(double)(pending_shifts*pending_shifts);
            // //Q32 = Q62/Q30 [~Q27.98]
            // var_x_t[index] = var_x_t[index] >> 30;

            // //Q30 = Q62/ Q32
            // g_t[index] = cov_xy_t[index] / (var_x_t[index] + exp_t);

            // //Q32 = Q62/Q30
            // cov_xy_t[index] = cov_xy_t[index] >> 30;

            // //Q62 = Q62 - Q30*Q32
            // sv_sq_t[index] = var_y_t[index] - g_t[index] * cov_xy_t[index];
#endif

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

#if FIXED_POINT
            if (var_x_t[index] < exp_t)
            {
                // g_t[index] = 0;
                g_t_num = 0;
                sv_sq_t[index] = var_y_t[index];
                var_x_t[index] = 0;
            }
            
            if (var_y_t[index] < exp_t)
            {
                // g_t[index] = 0;
                g_t_num = 0;
                sv_sq_t[index] = 0;
            }

            // if (g_t[index] < 0)
            if((g_t_num < 0 && g_den > 0) || (g_den < 0 && g_t_num > 0))
            {
                sv_sq_t[index] = var_x_t[index];
                // g_t[index] = 0;
                g_t_num = 0;
            }

            if (sv_sq_t[index] < exp)
                sv_sq_t[index] = exp;

            // //Q16 = Q30/Q14
            // g_t[index] = g_t[index] >> 14;

            // //Q32 = Q62/Q30
            // sv_sq_t[index] = sv_sq_t[index] >> 30;

            // //Q30 = Q32/Q2
            // var_x_t[index] = var_x_t[index] >> 2;

            // //Q30
            // // int64_t exp_add = (1e-4 * (1 << 30));

            // //METHOD -1
            // //--------------------------------------------------------------------------------------------------
            // //Q30 = Q30+(Q16*Q16*Q30/(Q32 + Q32))
            // int64_t num_t= (1 << 30) +  g_t[index] * g_t[index] * var_x_t[index] / (sv_sq_t[index] + sigma_nsq_t);
            // int64_t tmp_num_t = num_t;
            // int x;
            // uint16_t log_in_num = get_best_16bitsfixed_opt_64((uint64_t)tmp_num_t, &x);
            // // num_t_temp = log_in * 2^x;  log2f(log_in * 2^x/2^offset) = log2f(log_in) + log2f(2^x) - log2f(2^pffset) = log2f(log_in) + x - offset
            // score_num_t += log_values[log_in_num] + (-x - 30) * 2048;
            // double tn = score_num_t / 2048 ;

            // //Q30 = Q30+(Q30*Q16*Q16)/Q32
            // int64_t den_t = (1 <<30) + (var_x_t[index] *shift_d * shift_d)/sigma_nsq_t;
            // int64_t tmp_den_t = den_t;
            // int y;
            // uint16_t log_in_den = get_best_16bitsfixed_opt_64((uint64_t)tmp_den_t, &y);
            // score_den_t += log_values[log_in_den] + (-y - 30) * 2048;
            // double td = score_den_t / 2048;
            // //--------------------------------------------------------------------------------------------------

            // // METHOD - 2
            // //--------------------------------------------------------------------------------------------------
            // // int64_t a = 1 << 30; //Q30
            // // int64_t b = g_t[index] * g_t[index] * var_x_t[index]; // Q62 = Q16*Q16*Q30
            // // int64_t c = sv_sq_t[index] + sigma_nsq_t; // Q32 = Q32 + Q32
            // // //log(ac+b) - log(c)
            // // //ac - Q62 = Q30*Q32
            // // //b - Q62
            // // // ac + b - Q62
            // // int x1, x2;
            // // int64_t tmp_num_t = (a*c) + b; //Q62 
            // // uint16_t log_in_num_1 = get_best_16bitsfixed_opt_64((uint64_t)tmp_num_t, &x1);
            // // uint16_t log_in_num_2 = get_best_16bitsfixed_opt_64((uint64_t)c, &x2);
            // // score_num_t += (log_values[log_in_num_1] + (-x1 - 62) * 2048) - (log_values[log_in_num_2] + (-x2 - 32) * 2048);

            // // //log(a + var_x/sigma_nsq)
            // // //log((a*sigma_nsq) + var_x) - log(sigma_nsq) 
            // // //a*sigma_nsq - Q62 = Q30*Q32
            // // int64_t tmp_den_t = ((a*sigma_nsq_t) >> 32) + var_x_t[index]; //Q30   
            // // int y1, y2;
            // // uint16_t log_in_den_1 = get_best_16bitsfixed_opt_64((uint64_t)tmp_den_t, &y1); 
            // // uint16_t log_in_den_2 = get_best_16bitsfixed_opt_64((uint64_t)sigma_nsq_t, &y2); 
            // // score_den_t += (log_values[log_in_den_1] + (-y1 - 30) * 2048) - (log_values[log_in_den_2] + (-y2 - 32) * 2048);
            // //--------------------------------------------------------------------------------------------------

            double gd = ((double)g_t_num/((double)shift_val*shift_val))/ ((double)g_den / ((double)shift_val*shift_val));
            double vard = (double)var_x_t[index]/((double)shift_val*shift_val*k_norm);
            double svsqd = (double)sv_sq_t[index]/((double)shift_val*shift_val*k_norm);
            double score_num_tmp = (log((double)1 + gd*gd * vard / (svsqd + sigma_nsq)));
            double score_den_tmp = (log((double)1 + vard / sigma_nsq));

            int64_t n1 = (g_t_num * g_t_num * var_x_t[index]/k_norm)/g_den;
            int64_t n2 = ((g_den*sv_sq_t[index]/k_norm) + g_den*sigma_nsq_t);
            int64_t num_t = n2 + n1;
            int64_t num_den_t = n2;
            double sum_temp_log = log(1+(double)n1/(double)n2);
            double t1 = log((double)num_t/(double)num_den_t);
            double t2 = (double)t1/((double)shift_val*shift_val);
            int x1, x2;
            uint16_t log_in_num_1 = get_best_32bitsfixed_opt_64((uint64_t)num_t, &x1);
            uint16_t log_in_num_2 = get_best_32bitsfixed_opt_64((uint64_t)num_den_t, &x2);
            double sum_tmp_log_2 = log(log_in_num_1 >> x1) - log(log_in_num_2 >> x2);
            score_num_t += (log_values[log_in_num_1] + (-x1) * 2048 *  (1 << 15))- (log_values[log_in_num_2] + (-x2) * 2048 *  (1 << 15));

            int64_t d1 = sigma_nsq_t + (var_x_t[index]/k_norm);
            int64_t d2 = sigma_nsq_t;
            int y1, y2;
            uint16_t log_in_den_1 = get_best_16bitsfixed_opt_64((uint64_t)d1, &y1);
            uint16_t log_in_den_2 = get_best_16bitsfixed_opt_64((uint64_t)d2, &y2);
            score_den_t += log_values[log_in_den_1] + (-y1 - 12) * 2048 - log_values[log_in_den_2] + (-y2 - 12) * 2048;


           


// 1+((g*g*var_x)/(svsq+sigmansq)) ==log ((svsq+sigmansq)*g_den*g_den) + (g_num*g_num*var_x)) - log((svsq+sigmansq)*g_den*g_den) )
            // score_num_t += (log((double)1 + g_t[index] * g_t[index] * var_x_t[index] / (sv_sq_t[index] + sigma_nsq)));
            // score_den_t += (log((double)1 + var_x_t[index] / sigma_nsq));
#endif

            double num_sum = (double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq);
            // double num_int = (double)num_t/(double)(pow(2, 30));
            double den_sum = (double)1 + var_x[index] / sigma_nsq;
            // double den_int = (double)den_t/(double)(pow(2, 30));
            *score_num += (log((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)));
            *score_den += (log((double)1 + var_x[index] / sigma_nsq));

            // int tmpr = 0;
        }
    }
    double add_exp = 1e-4*s_height*s_width;
    *score += ((*score_num + add_exp) / (*score_den + add_exp));

#if FIXED_POINT
    score_t += ((score_num_t + add_exp) / (score_den_t + add_exp));

    // score_t += ((((double)score_num_t/2048) + add_exp)/(((double)score_den_t/2048)+add_exp));
    // score_t += (score_num_t + add_exp) / (score_den_t + add_exp);
#endif


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

#if FIXED_POINT
    free(x_pad_t);
    free(y_pad_t);

    free(t1);
    free(t2);
    free(t3);
    free(t4);
    free(t5);

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
#endif

    ret = 0;

    return ret;
}