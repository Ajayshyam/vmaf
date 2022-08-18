
#include "../integer_funque_filters.h"
#include <arm_neon.h>
#include <math.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline int16_t ssim_get_best_i16_from_u64(uint64_t temp, int *power)
{
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (int16_t)temp;
}

int integer_compute_ssim_funque_neon(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup)
{
    int ret = 1;
    size_t width = ref->width;
    size_t height = ref->height;

    dwt2_dtype mx, my;
    ssim_inter_dtype var_x, var_y, cov_xy, var_x_band0, var_y_band0, cov_xy_band0;
    ssim_inter_dtype map, l_num, l_den, cs_num, cs_den;
    ssim_inter_dtype C1 = ((K1 * max_val) * (K1 * max_val) * ((pending_div * pending_div) << (2 - SSIM_INTER_L_SHIFT)));
    ssim_inter_dtype C2 = ((K2 * max_val) * (K2 * max_val) * ((pending_div * pending_div) >> (SSIM_INTER_VAR_SHIFTS + SSIM_INTER_CS_SHIFT - 2)));
    ssim_accum_dtype accum_map = 0;
    ssim_accum_dtype accum_map_sq = 0;
    ssim_accum_dtype map_sq_insum = 0;
    int index = 0, i, j, k;
    int16_t i16_map_den;

    int16x8_t ref16x8_b0, dist16x8_b0, ref16x8_b1, dist16x8_b1;
    int16x8_t ref16x8_b2, dist16x8_b2, ref16x8_b3, dist16x8_b3;
    int16x4_t tmpRef16x8_low, tmpDist16x8_low;
    int32x4_t var32x4_lb0, var32x4_hb0, cov32x4_lb0, cov32x4_hb0;
    int32x4_t lDen32x4_lb0, lDen32x4_hb0, csDen32x4_lb1, csDen32x4_hb1;
    int32x4_t addlNum32x4_lb0, addlNum32x4_hb0, addlDen32x4_lb0, addlDen32x4_hb0;
    int32x4_t varSftX32x4_lb1, varSftX32x4_hb1, varSftY32x4_lb1, varSftY32x4_hb1;
    int32x4_t varSft32x4_lb1, varSft32x4_hb1, covSft32x4_lb1, covSft32x4_hb1;
    int32x4_t addcsNum32x4_lb1, addcsNum32x4_hb1, addcsDen32x4_lb1, addcsDen32x4_hb1;
    int64x2_t mulNum64x2_0lo, mulNum64x2_0hi, mulNum64x2_1lo, mulNum64x2_1hi, mulDen64x2_0lo, mulDen64x2_0hi, mulDen64x2_1lo, mulDen64x2_1hi;
    int32x4_t varX32x4_lb1, varX32x4_hb1, varY32x4_lb1, varY32x4_hb1, cov32x4_lb1, cov32x4_hb1;
    int32x4_t dupC1 = vdupq_n_s32(C1);
    int32x4_t dupC2 = vdupq_n_s32(C2);

    int64_t *numVal = (int64_t *)malloc(width * sizeof(int64_t *));
    int64_t *denVal = (int64_t *)malloc(width * sizeof(int64_t *));

    for (i = 0; i < height; i++)
    {
        for (j = 0; j <= width - 8; j += 8)
        {
            index = i * width + j;
            ref16x8_b0 = vld1q_s16(ref->bands[0] + index);
            dist16x8_b0 = vld1q_s16(dist->bands[0] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b0);
            tmpDist16x8_low = vget_low_s16(dist16x8_b0);

            var32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            var32x4_hb0 = vmull_high_s16(ref16x8_b0, ref16x8_b0);
            var32x4_lb0 = vmlal_s16(var32x4_lb0, tmpDist16x8_low, tmpDist16x8_low);
            var32x4_hb0 = vmlal_high_s16(var32x4_hb0, dist16x8_b0, dist16x8_b0);
            cov32x4_lb0 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb0 = vmull_high_s16(ref16x8_b0, dist16x8_b0);

            lDen32x4_lb0 = vshrq_n_s32(var32x4_lb0, SSIM_INTER_L_SHIFT);
            lDen32x4_hb0 = vshrq_n_s32(var32x4_hb0, SSIM_INTER_L_SHIFT);

            addlNum32x4_lb0 = vaddq_s32(cov32x4_lb0, dupC1);
            addlNum32x4_hb0 = vaddq_s32(cov32x4_hb0, dupC1);
            addlDen32x4_lb0 = vaddq_s32(lDen32x4_lb0, dupC1);
            addlDen32x4_hb0 = vaddq_s32(lDen32x4_hb0, dupC1);

            ref16x8_b1 = vld1q_s16(ref->bands[1] + index);
            dist16x8_b1 = vld1q_s16(dist->bands[1] + index);
            ref16x8_b2 = vld1q_s16(ref->bands[2] + index);
            dist16x8_b2 = vld1q_s16(dist->bands[2] + index);
            ref16x8_b3 = vld1q_s16(ref->bands[3] + index);
            dist16x8_b3 = vld1q_s16(dist->bands[3] + index);

            tmpRef16x8_low = vget_low_s16(ref16x8_b1);
            tmpDist16x8_low = vget_low_s16(dist16x8_b1);
            varX32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmull_high_s16(ref16x8_b1, ref16x8_b1);
            varY32x4_lb1 = vmull_s16(tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmull_high_s16(dist16x8_b1, dist16x8_b1);
            cov32x4_lb1 = vmull_s16(tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmull_high_s16(ref16x8_b1, dist16x8_b1);

            tmpRef16x8_low = vget_low_s16(ref16x8_b2);
            tmpDist16x8_low = vget_low_s16(dist16x8_b2);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b2, ref16x8_b2);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b2, dist16x8_b2);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b2, dist16x8_b2);

            tmpRef16x8_low = vget_low_s16(ref16x8_b3);
            tmpDist16x8_low = vget_low_s16(dist16x8_b3);
            varX32x4_lb1 = vmlal_s16(varX32x4_lb1, tmpRef16x8_low, tmpRef16x8_low);
            varX32x4_hb1 = vmlal_high_s16(varX32x4_hb1, ref16x8_b3, ref16x8_b3);
            varY32x4_lb1 = vmlal_s16(varY32x4_lb1, tmpDist16x8_low, tmpDist16x8_low);
            varY32x4_hb1 = vmlal_high_s16(varY32x4_hb1, dist16x8_b3, dist16x8_b3);
            cov32x4_lb1 = vmlal_s16(cov32x4_lb1, tmpRef16x8_low, tmpDist16x8_low);
            cov32x4_hb1 = vmlal_high_s16(cov32x4_hb1, ref16x8_b3, dist16x8_b3);

            covSft32x4_lb1 = vshrq_n_s32(cov32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            covSft32x4_hb1 = vshrq_n_s32(cov32x4_hb1, SSIM_INTER_VAR_SHIFTS);
            varSftX32x4_lb1 = vshrq_n_s32(varX32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            varSftX32x4_hb1 = vshrq_n_s32(varX32x4_hb1, SSIM_INTER_VAR_SHIFTS);
            varSftY32x4_lb1 = vshrq_n_s32(varY32x4_lb1, SSIM_INTER_VAR_SHIFTS);
            varSftY32x4_hb1 = vshrq_n_s32(varY32x4_hb1, SSIM_INTER_VAR_SHIFTS);

            varSft32x4_lb1 = vaddq_s32(varSftX32x4_lb1, varSftY32x4_lb1);
            varSft32x4_hb1 = vaddq_s32(varSftX32x4_hb1, varSftY32x4_hb1);
            csDen32x4_lb1 = vshrq_n_s32(varSft32x4_lb1, SSIM_INTER_CS_SHIFT);
            csDen32x4_hb1 = vshrq_n_s32(varSft32x4_hb1, SSIM_INTER_CS_SHIFT);

            addcsNum32x4_lb1 = vaddq_s32(covSft32x4_lb1, dupC2);
            addcsNum32x4_hb1 = vaddq_s32(covSft32x4_hb1, dupC2);
            addcsDen32x4_lb1 = vaddq_s32(csDen32x4_lb1, dupC2);
            addcsDen32x4_hb1 = vaddq_s32(csDen32x4_hb1, dupC2);

            mulNum64x2_0lo = vmull_s32(vget_low_s32(addlNum32x4_lb0), vget_low_s32(addcsNum32x4_lb1));
            mulNum64x2_0hi = vmull_high_s32(addlNum32x4_lb0, addcsNum32x4_lb1);
            mulNum64x2_1lo = vmull_s32(vget_low_s32(addlNum32x4_hb0), vget_low_s32(addcsNum32x4_hb1));
            mulNum64x2_1hi = vmull_high_s32(addlNum32x4_hb0, addcsNum32x4_hb1);

            mulDen64x2_0lo = vmull_s32(vget_low_s32(addlDen32x4_lb0), vget_low_s32(addcsDen32x4_lb1));
            mulDen64x2_0hi = vmull_high_s32(addlDen32x4_lb0, addcsDen32x4_lb1);
            mulDen64x2_1lo = vmull_s32(vget_low_s32(addlDen32x4_hb0), vget_low_s32(addcsDen32x4_hb1));
            mulDen64x2_1hi = vmull_high_s32(addlDen32x4_hb0, addcsDen32x4_hb1);

            vst1q_s64(numVal + j, mulNum64x2_0lo);
            vst1q_s64(numVal + j + 2, mulNum64x2_0hi);
            vst1q_s64(numVal + j + 4, mulNum64x2_1lo);
            vst1q_s64(numVal + j + 6, mulNum64x2_1hi);

            vst1q_s64(denVal + j, mulDen64x2_0lo);
            vst1q_s64(denVal + j + 2, mulDen64x2_0hi);
            vst1q_s64(denVal + j + 4, mulDen64x2_1lo);
            vst1q_s64(denVal + j + 6, mulDen64x2_1hi);
        }
        for (; j < width; j++)
        {
            index = i * width + j;
            mx = ref->bands[0][index];
            my = dist->bands[0][index];

            var_x = 0;
            var_y = 0;
            cov_xy = 0;

            for (int k = 1; k < 4; k++)
            {
                var_x += ((ssim_inter_dtype)ref->bands[k][index] * ref->bands[k][index]);
                var_y += ((ssim_inter_dtype)dist->bands[k][index] * dist->bands[k][index]);
                cov_xy += ((ssim_inter_dtype)ref->bands[k][index] * dist->bands[k][index]);
            }
            var_x_band0 = (ssim_inter_dtype)mx * mx;
            var_y_band0 = (ssim_inter_dtype)my * my;
            cov_xy_band0 = (ssim_inter_dtype)mx * my;

            var_x = (var_x >> SSIM_INTER_VAR_SHIFTS);
            var_y = (var_y >> SSIM_INTER_VAR_SHIFTS);
            cov_xy = (cov_xy >> SSIM_INTER_VAR_SHIFTS);

            l_num = (cov_xy_band0 + C1);
            l_den = (((var_x_band0 + var_y_band0) >> SSIM_INTER_L_SHIFT) + C1);
            cs_num = (cov_xy + C2);
            cs_den = (((var_x + var_y) >> SSIM_INTER_CS_SHIFT) + C2);

            numVal[j] = (ssim_accum_dtype)l_num * cs_num;
            denVal[j] = (ssim_accum_dtype)l_den * cs_den;
        }
        for (k = 0; k < width; k++)
        {
            int power_val;
            i16_map_den = ssim_get_best_i16_from_u64((uint64_t)denVal[k], &power_val);
            map = ((numVal[k] >> power_val) * div_lookup[i16_map_den + 32768]) >> SSIM_SHIFT_DIV;

            accum_map += map;
            map_sq_insum += (ssim_accum_dtype)(((ssim_accum_dtype)map * map));
        }
    }

    accum_map_sq = map_sq_insum / (height * width);
    double ssim_mean = (double)accum_map / (height * width);
    double ssim_std;
    ssim_std = sqrt(MAX(0, ((double)accum_map_sq - ssim_mean * ssim_mean)));
    *score = (ssim_std / ssim_mean);

    free(numVal);
    free(denVal);
    ret = 0;
    return ret;
}