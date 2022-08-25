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

#include "funque_filters.h"
#include "funque_vif_options.h"

void reflect_pad(const float* src, size_t width, size_t height, int reflect, float* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(float) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(float) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(float) * out_width);
    }
}

void integral_image_2(const float* src1, const float* src2, size_t width, size_t height, double* sum)
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

void integral_image(const float* src, size_t width, size_t height, double* sum)
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

#if USE_DYNAMIC_SIGMA_NSQ
int compute_vif_funque(const float* x, const float* y, size_t width, size_t height, 
                        double* score, double* score_num, double* score_den, int k, 
                        int stride, double sigma_nsq_arg, int vif_level)
#else
int compute_vif_funque(const float* x, const float* y, size_t width, size_t height, 
                        double* score, double* score_num, double* score_den, int k, 
                        int stride, double sigma_nsq_arg)
#endif
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = k * k;

    float* x_pad, * y_pad;

    int x_reflect = (int)((kh - stride) / 2);
    int y_reflect = (int)((kw - stride) / 2);

    x_pad = (float*)malloc(sizeof(float) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));
    y_pad = (float*)malloc(sizeof(float) * (width + (2 * y_reflect)) * (height + (2 * y_reflect)));

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
	
	double sigma_nsq = sigma_nsq_arg;
#if VIF_STABILITY
	double sigma_nsq_base = sigma_nsq_arg / (255.0*255.0);	
	double sigma_max_inv = 4.0;
	sigma_nsq = sigma_nsq_base;
#if USE_DYNAMIC_SIGMA_NSQ
	sigma_nsq = sigma_nsq_base * (2 << (vif_level + 1));
#endif
#endif


    *score = (double)0;
    *score_num = (double)0;
    *score_den = (double)0;
	
	double num_val, den_val;

    for (unsigned int i = 0; i < s_height; i++)
    {
        for (unsigned int j = 0; j < s_width; j++)
        {
            index = i * s_width + j;

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

#if VIF_STABILITY
			num_val = (log2((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)));
			den_val = (log2((double)1 + var_x[index] / sigma_nsq));

			if (cov_xy[index] < 0.0f) {
				num_val = 0.0f;
			}

			if (var_x[index] < sigma_nsq) {
				num_val = 1.0f - var_y[index] * sigma_max_inv;
				den_val = 1.0f;
			}
#else

			num_val = (log2((double)1 + g[index] * g[index] * var_x[index] / (sv_sq[index] + sigma_nsq)) + (double)1e-4);
			den_val = (log2((double)1 + var_x[index] / sigma_nsq) + (double)1e-4);

#endif

			*score_num += num_val;
			*score_den += den_val;
        }
    }

#if VIF_STABILITY
	*score += ((*score_den) == 0.0) ? 1.0 : ((*score_num) / (*score_den));
#else
    *score += ((*score_num) / (*score_den));
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
    ret = 0;

    return ret;
}