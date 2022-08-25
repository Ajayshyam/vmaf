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
#include <arm_neon.h>

#include "../funque_vif_options.h"
#include "../integer_funque_filters.h"
#include "../common/macros.h"
#include "../integer_funque_vif.h"
#include "integer_funque_vif_neon.h"

#if USE_DYNAMIC_SIGMA_NSQ
int integer_compute_vif_funque_neon(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                    double* score, double* score_num, double* score_den, int k, int stride, 
                                    double sigma_nsq_arg, int64_t shift_val, uint32_t* log_18, int vif_level)
#else
int integer_compute_vif_funque_neon(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                    double* score, double* score_num, double* score_den, int k, int stride, 
                                    double sigma_nsq_arg, int64_t shift_val, uint32_t* log_18)
#endif
{
    int ret = 1;

    int kh = k;
    int kw = k;
    int k_norm = kw * kh;

    int x_reflect = (int)((kh - stride) / 2); // amount for reflecting
    int y_reflect = (int)((kw - stride) / 2);
    size_t vif_width, vif_height;

#if VIF_REFLECT_PAD
    vif_width  = width;
    vif_height = height;
#else
    vif_width = width - (2 * y_reflect);
    vif_height = height - (2 * x_reflect);
#endif

    size_t r_width = vif_width + (2 * x_reflect); // after reflect pad
    size_t r_height = vif_height + (2 * x_reflect);

    size_t s_width = (r_width + 1) - kw;
    size_t s_height = (r_height + 1) - kh;

    dwt2_dtype* x_pad_t, *y_pad_t;
#if VIF_REFLECT_PAD
    x_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (vif_width + (2 * x_reflect)) * (vif_height + (2 * x_reflect)));
    y_pad_t = (dwt2_dtype*)malloc(sizeof(dwt2_dtype*) * (vif_width + (2 * y_reflect)) * (vif_height + (2 * y_reflect)));
    integer_reflect_pad(x_t, vif_width, vif_height, x_reflect, x_pad_t);
    integer_reflect_pad(y_t, vif_width, vif_height, y_reflect, y_pad_t);
#else
    x_pad_t = x_t;
    y_pad_t = y_t;
#endif

    // int32_t int_1_x, int_1_y;
    // int64_t int_2_x, int_2_y, int_x_y;

    int64_t exp_t = 1; // using 1 because exp in Q32 format is still 0
    int32_t sigma_nsq_t = (int64_t)((int64_t)sigma_nsq_arg*shift_val*shift_val*k_norm) >> VIF_COMPUTE_METRIC_R_SHIFT;
#if VIF_STABILITY
	double sigma_nsq_base = sigma_nsq_arg / (255.0*255.0);	
#if USE_DYNAMIC_SIGMA_NSQ
	sigma_nsq_base = sigma_nsq_base * (2 << (vif_level + 1));
#endif
	sigma_nsq_t = (int64_t)((int64_t)sigma_nsq_base*shift_val*shift_val*k_norm) >> VIF_COMPUTE_METRIC_R_SHIFT;
#endif
    int64_t score_num_t = 0;
    int64_t num_power = 0;
    int64_t score_den_t = 0;
    int64_t den_power = 0;

    int16_t knorm_fact = 25891;   // (2^21)/81 knorm factor is multiplied and shifted instead of division
    int16_t knorm_shift = 21; 

    {
        int width_p1 = r_width + 1;
        int height_p1 = r_height + 1;
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

        int16x8_t src16x8_X, src16x8_Y, prekh_src16x8_X, prekh_src16x8_Y;
        int16x4_t src16x4_X, src16x4_Y, prekh_src16x4_X, prekh_src16x4_Y;
        int32x4_t mul32x4_XXlo, mul32x4_XXhi, mul32x4_YYlo, mul32x4_YYhi;
        int32x4_t mul32x4_XYlo, mul32x4_XYhi;
        int32x4_t sub32x4_Xlo, sub32x4_Xhi, sub32x4_Ylo, sub32x4_Yhi;
        int32x4_t itrm1_32x4_Xlo, itrm1_32x4_Xhi, itrm1_32x4_Ylo, itrm1_32x4_Yhi;
        int64x2_t intr2X0, intr2X1, intr2X2, intr2X3;
        int64x2_t intr2Y0, intr2Y1, intr2Y2, intr2Y3;
        int64x2_t intr2XY0, intr2XY1, intr2XY2, intr2XY3;
        int srcOffset, prekhSrcOffset;

        int i = 0, j;

        //The height loop is broken into 2 parts, 
        //1st loop, prev kh row is not available to subtract during vertical summation
        for (i=1; i<kh+1; i++)
        {
            // int row_offset = i * width_p1;
            int src_offset = (i-1) * r_width;

            /**
             * In this loop the pixels are summated vertically and stored in interim buffer
             * The interim buffer is of size 1 row
             * inter_sum = prev_inter_sum + cur_metric_val
             * 
             * where inter_sum will have vertical pixel sums, 
             * prev_inter_sum will have prev rows inter_sum and 
             * cur_metric_val can be srcx or srcy or srcxx or srcyy or srcxy
             * The previous kh row metric val is not subtracted since it is not available here 
            */
            for (j = 1; j <= width_p1 - 8; j += 8)
            {
                srcOffset = src_offset + j - 1;
                src16x8_X = vld1q_s16(x_pad_t + srcOffset);
                src16x8_Y = vld1q_s16(y_pad_t + srcOffset);
                src16x4_X = vget_low_s16(src16x8_X);
                src16x4_Y = vget_low_s16(src16x8_Y);

                mul32x4_XXhi = vmull_high_s16(src16x8_X, src16x8_X);
                mul32x4_YYhi = vmull_high_s16(src16x8_Y, src16x8_Y);
                mul32x4_XYhi = vmull_high_s16(src16x8_X, src16x8_Y);
                mul32x4_XXlo = vmull_s16(src16x4_X, src16x4_X);
                mul32x4_YYlo = vmull_s16(src16x4_Y, src16x4_Y);
                mul32x4_XYlo = vmull_s16(src16x4_X, src16x4_Y);

                intr2X0 = vld1q_s64(interim_2_x + j);
                intr2X1 = vld1q_s64(interim_2_x + j + 2);
                intr2X2 = vld1q_s64(interim_2_x + j + 4);
                intr2X3 = vld1q_s64(interim_2_x + j + 6);
                intr2Y0 = vld1q_s64(interim_2_y + j);
                intr2Y1 = vld1q_s64(interim_2_y + j + 2);
                intr2Y2 = vld1q_s64(interim_2_y + j + 4);
                intr2Y3 = vld1q_s64(interim_2_y + j + 6);
                intr2XY0 = vld1q_s64(interim_x_y + j);
                intr2XY1 = vld1q_s64(interim_x_y + j + 2);
                intr2XY2 = vld1q_s64(interim_x_y + j + 4);
                intr2XY3 = vld1q_s64(interim_x_y + j + 6);

                intr2X0 = vaddw_s32(intr2X0, vget_low_s32(mul32x4_XXlo));
                intr2X1 = vaddw_high_s32(intr2X1, mul32x4_XXlo);
                intr2X2 = vaddw_s32(intr2X2, vget_low_s32(mul32x4_XXhi));
                intr2X3 = vaddw_high_s32(intr2X3, mul32x4_XXhi);
                intr2Y0 = vaddw_s32(intr2Y0, vget_low_s32(mul32x4_YYlo));
                intr2Y1 = vaddw_high_s32(intr2Y1, mul32x4_YYlo);
                intr2Y2 = vaddw_s32(intr2Y2, vget_low_s32(mul32x4_YYhi));
                intr2Y3 = vaddw_high_s32(intr2Y3, mul32x4_YYhi);
                intr2XY0 = vaddw_s32(intr2XY0, vget_low_s32(mul32x4_XYlo));
                intr2XY1 = vaddw_high_s32(intr2XY1, mul32x4_XYlo);
                intr2XY2 = vaddw_s32(intr2XY2, vget_low_s32(mul32x4_XYhi));
                intr2XY3 = vaddw_high_s32(intr2XY3, mul32x4_XYhi);

                vst1q_s64(interim_2_x + j, intr2X0);
                vst1q_s64(interim_2_x + j + 2, intr2X1);
                vst1q_s64(interim_2_x + j + 4, intr2X2);
                vst1q_s64(interim_2_x + j + 6, intr2X3);
                vst1q_s64(interim_2_y + j, intr2Y0);
                vst1q_s64(interim_2_y + j + 2, intr2Y1);
                vst1q_s64(interim_2_y + j + 4, intr2Y2);
                vst1q_s64(interim_2_y + j + 6, intr2Y3);
                vst1q_s64(interim_x_y + j, intr2XY0);
                vst1q_s64(interim_x_y + j + 2, intr2XY1);
                vst1q_s64(interim_x_y + j + 4, intr2XY2);
                vst1q_s64(interim_x_y + j + 6, intr2XY3);

                itrm1_32x4_Xlo = vld1q_s32(interim_1_x + j);
                itrm1_32x4_Xhi = vld1q_s32(interim_1_x + j + 4);
                itrm1_32x4_Ylo = vld1q_s32(interim_1_y + j);
                itrm1_32x4_Yhi = vld1q_s32(interim_1_y + j + 4);

                itrm1_32x4_Xlo = vaddw_s16(itrm1_32x4_Xlo, src16x4_X);
                itrm1_32x4_Xhi = vaddw_high_s16(itrm1_32x4_Xhi, src16x8_X);
                itrm1_32x4_Ylo = vaddw_s16(itrm1_32x4_Ylo, src16x4_Y);
                itrm1_32x4_Yhi = vaddw_high_s16(itrm1_32x4_Yhi, src16x8_Y);

                vst1q_s32(interim_1_x + j, itrm1_32x4_Xlo);
                vst1q_s32(interim_1_x + j + 4, itrm1_32x4_Xhi);
                vst1q_s32(interim_1_y + j, itrm1_32x4_Ylo);
                vst1q_s32(interim_1_y + j + 4, itrm1_32x4_Yhi);
            }

            for (; j < width_p1; j++)
            {
                int j_minus1 = j - 1;
                dwt2_dtype src_x_val = x_pad_t[src_offset + j_minus1];
                dwt2_dtype src_y_val = y_pad_t[src_offset + j_minus1];

                int32_t src_xx_val = (int32_t)src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t)src_y_val * src_y_val;
                int32_t src_xy_val = (int32_t)src_x_val * src_y_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val;
                interim_x_y[j] = interim_x_y[j] + src_xy_val;
            }
        }
        /**
         * The vif score calculations would start from the kh,kw index of var & covar
         * Hence horizontal sum of first kh rows are not used, hence that computation is avoided
         */
        //score computation for 1st row of variance & covariance i.e. kh row of padded img
        vif_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift,
                             exp_t, sigma_nsq_t, log_18,
                             interim_1_x, interim_1_y,
                             interim_2_x, interim_2_y, interim_x_y,
                             &score_num_t, &num_power, &score_den_t, &den_power);

        //2nd loop, core loop 
        for(; i<height_p1; i++)
        {
            // int row_offset = i * width_p1;
            int src_offset = (i-1) * r_width;
            int pre_kh_src_offset = (i-1-kh) * r_width;
            /**
             * This loop is similar to the loop across columns seen in 1st for loop
             * In this loop the pixels are summated vertically and stored in interim buffer
             * The interim buffer is of size 1 row
             * inter_sum = prev_inter_sum + cur_metric_val - prev_kh-row_metric_val
            */
            for (j = 1; j <= width_p1 - 8; j += 8)
            {
                srcOffset = src_offset + j - 1;
                prekhSrcOffset = pre_kh_src_offset + j - 1;

                src16x8_X = vld1q_s16(x_pad_t + srcOffset);
                src16x8_Y = vld1q_s16(y_pad_t + srcOffset);
                prekh_src16x8_X = vld1q_s16(x_pad_t + prekhSrcOffset);
                prekh_src16x8_Y = vld1q_s16(y_pad_t + prekhSrcOffset);

                src16x4_X = vget_low_s16(src16x8_X);
                src16x4_Y = vget_low_s16(src16x8_Y);
                prekh_src16x4_X = vget_low_s16(prekh_src16x8_X);
                prekh_src16x4_Y = vget_low_s16(prekh_src16x8_Y);

                mul32x4_XXlo = vmull_s16(src16x4_X, src16x4_X);
                mul32x4_XXhi = vmull_high_s16(src16x8_X, src16x8_X);
                mul32x4_YYlo = vmull_s16(src16x4_Y, src16x4_Y);
                mul32x4_YYhi = vmull_high_s16(src16x8_Y, src16x8_Y);
                mul32x4_XYlo = vmull_s16(src16x4_X, src16x4_Y);
                mul32x4_XYhi = vmull_high_s16(src16x8_X, src16x8_Y);

                mul32x4_XXlo = vmlsl_s16(mul32x4_XXlo, prekh_src16x4_X, prekh_src16x4_X);
                mul32x4_XXhi = vmlsl_high_s16(mul32x4_XXhi, prekh_src16x8_X, prekh_src16x8_X);
                mul32x4_YYlo = vmlsl_s16(mul32x4_YYlo, prekh_src16x4_Y, prekh_src16x4_Y);
                mul32x4_YYhi = vmlsl_high_s16(mul32x4_YYhi, prekh_src16x8_Y, prekh_src16x8_Y);
                mul32x4_XYlo = vmlsl_s16(mul32x4_XYlo, prekh_src16x4_X, prekh_src16x4_Y);
                mul32x4_XYhi = vmlsl_high_s16(mul32x4_XYhi, prekh_src16x8_X, prekh_src16x8_Y);

                intr2X0 = vld1q_s64(interim_2_x + j);
                intr2X1 = vld1q_s64(interim_2_x + j + 2);
                intr2X2 = vld1q_s64(interim_2_x + j + 4);
                intr2X3 = vld1q_s64(interim_2_x + j + 6);
                intr2Y0 = vld1q_s64(interim_2_y + j);
                intr2Y1 = vld1q_s64(interim_2_y + j + 2);
                intr2Y2 = vld1q_s64(interim_2_y + j + 4);
                intr2Y3 = vld1q_s64(interim_2_y + j + 6);
                intr2XY0 = vld1q_s64(interim_x_y + j);
                intr2XY1 = vld1q_s64(interim_x_y + j + 2);
                intr2XY2 = vld1q_s64(interim_x_y + j + 4);
                intr2XY3 = vld1q_s64(interim_x_y + j + 6);

                intr2X0 = vaddw_s32(intr2X0, vget_low_s32(mul32x4_XXlo));
                intr2X1 = vaddw_high_s32(intr2X1, mul32x4_XXlo);
                intr2X2 = vaddw_s32(intr2X2, vget_low_s32(mul32x4_XXhi));
                intr2X3 = vaddw_high_s32(intr2X3, mul32x4_XXhi);
                intr2Y0 = vaddw_s32(intr2Y0, vget_low_s32(mul32x4_YYlo));
                intr2Y1 = vaddw_high_s32(intr2Y1, mul32x4_YYlo);
                intr2Y2 = vaddw_s32(intr2Y2, vget_low_s32(mul32x4_YYhi));
                intr2Y3 = vaddw_high_s32(intr2Y3, mul32x4_YYhi);
                intr2XY0 = vaddw_s32(intr2XY0, vget_low_s32(mul32x4_XYlo));
                intr2XY1 = vaddw_high_s32(intr2XY1, mul32x4_XYlo);
                intr2XY2 = vaddw_s32(intr2XY2, vget_low_s32(mul32x4_XYhi));
                intr2XY3 = vaddw_high_s32(intr2XY3, mul32x4_XYhi);

                vst1q_s64(interim_2_x + j, intr2X0);
                vst1q_s64(interim_2_x + j + 2, intr2X1);
                vst1q_s64(interim_2_x + j + 4, intr2X2);
                vst1q_s64(interim_2_x + j + 6, intr2X3);
                vst1q_s64(interim_2_y + j, intr2Y0);
                vst1q_s64(interim_2_y + j + 2, intr2Y1);
                vst1q_s64(interim_2_y + j + 4, intr2Y2);
                vst1q_s64(interim_2_y + j + 6, intr2Y3);
                vst1q_s64(interim_x_y + j, intr2XY0);
                vst1q_s64(interim_x_y + j + 2, intr2XY1);
                vst1q_s64(interim_x_y + j + 4, intr2XY2);
                vst1q_s64(interim_x_y + j + 6, intr2XY3);

                itrm1_32x4_Xlo = vld1q_s32(interim_1_x + j);
                itrm1_32x4_Xhi = vld1q_s32(interim_1_x + j + 4);
                itrm1_32x4_Ylo = vld1q_s32(interim_1_y + j);
                itrm1_32x4_Yhi = vld1q_s32(interim_1_y + j + 4);

                sub32x4_Xlo = vsubl_s16(src16x4_X, prekh_src16x4_X);
                sub32x4_Xhi = vsubl_high_s16(src16x8_X, prekh_src16x8_X);
                sub32x4_Ylo = vsubl_s16(src16x4_Y, prekh_src16x4_Y);
                sub32x4_Yhi = vsubl_high_s16(src16x8_Y, prekh_src16x8_Y);

                itrm1_32x4_Xlo = vaddq_s32(sub32x4_Xlo, itrm1_32x4_Xlo);
                itrm1_32x4_Xhi = vaddq_s32(sub32x4_Xhi, itrm1_32x4_Xhi);
                itrm1_32x4_Ylo = vaddq_s32(sub32x4_Ylo, itrm1_32x4_Ylo);
                itrm1_32x4_Yhi = vaddq_s32(sub32x4_Yhi, itrm1_32x4_Yhi);

                vst1q_s32(interim_1_x + j, itrm1_32x4_Xlo);
                vst1q_s32(interim_1_x + j + 4, itrm1_32x4_Xhi);
                vst1q_s32(interim_1_y + j, itrm1_32x4_Ylo);
                vst1q_s32(interim_1_y + j + 4, itrm1_32x4_Yhi);
            }

            for (; j < width_p1; j++)
            {
                int j_minus1 = j - 1;
                dwt2_dtype src_x_val = x_pad_t[src_offset + j_minus1];
                dwt2_dtype src_y_val = y_pad_t[src_offset + j_minus1];

                dwt2_dtype src_x_prekh_val = x_pad_t[pre_kh_src_offset + j_minus1];
                dwt2_dtype src_y_prekh_val = y_pad_t[pre_kh_src_offset + j_minus1];
                int32_t src_xx_val = (int32_t)src_x_val * src_x_val;
                int32_t src_yy_val = (int32_t)src_y_val * src_y_val;
                int32_t src_xy_val = (int32_t)src_x_val * src_y_val;

                int32_t src_xx_prekh_val = (int32_t)src_x_prekh_val * src_x_prekh_val;
                int32_t src_yy_prekh_val = (int32_t)src_y_prekh_val * src_y_prekh_val;
                int32_t src_xy_prekh_val = (int32_t)src_x_prekh_val * src_y_prekh_val;

                interim_1_x[j] = interim_1_x[j] + src_x_val - src_x_prekh_val;
                interim_2_x[j] = interim_2_x[j] + src_xx_val - src_xx_prekh_val;
                interim_1_y[j] = interim_1_y[j] + src_y_val - src_y_prekh_val;
                interim_2_y[j] = interim_2_y[j] + src_yy_val - src_yy_prekh_val;
                interim_x_y[j] = interim_x_y[j] + src_xy_val - src_xy_prekh_val;
            }


            //horizontal summation and score compuations
            vif_horz_integralsum(kw, width_p1, knorm_fact, knorm_shift, 
                                 exp_t, sigma_nsq_t, log_18, 
                                 interim_1_x, interim_1_y,
                                 interim_2_x, interim_2_y, interim_x_y,
                                 &score_num_t, &num_power, 
                                 &score_den_t, &den_power);
        }

        free(interim_2_x);
        free(interim_1_x);
        free(interim_2_y);
        free(interim_1_y);
        free(interim_x_y);
    }

    double add_exp = 1e-4*s_height*s_width;

    double power_double_num = (double)num_power;
    double power_double_den = (double)den_power;
#if VIF_STABILITY
	*score_num = (((double)score_num_t/(double)(1<<26)) + power_double_num);
    *score_den = (((double)score_den_t/(double)(1<<26)) + power_double_den);
	*score += ((*score_den) == 0.0) ? 1.0 : ((*score_num) / (*score_den));
#else
    *score_num = (((double)score_num_t/(double)(1<<26)) + power_double_num) + add_exp;
    *score_den = (((double)score_den_t/(double)(1<<26)) + power_double_den) + add_exp;
    *score = *score_num / *score_den;
#endif

#if VIF_REFLECT_PAD
    free(x_pad_t);
    free(y_pad_t);
#endif

    ret = 0;

    return ret;
}
