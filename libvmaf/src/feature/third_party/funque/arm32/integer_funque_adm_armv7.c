#include <arm_neon.h>
#include "integer_funque_adm_armv7.h"

void integer_dlm_decouple_armv7(i_dwt2buffers ref, i_dwt2buffers dist,
                                i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add,
                                int32_t *adm_div_lookup, float border_size, double *adm_score_den)
{
    int width = ref.width;
    int height = ref.height;
    int i, j, k, l, index, addIndex, restIndex;

    int angle_flagC;
    adm_i16_dtype tmp_val;
    uint16_t angle_flag[8];

    adm_i32_dtype ot_dp, o_mag_sq, t_mag_sq;
    int border_h = (border_size * height);
    int border_w = (border_size * width);
    int64_t den_sum[3] = {0};
    int64_t den_row_sum[3] = {0};
    int64_t col0_ref_cube[3] = {0};
    int loop_h, loop_w, dlm_width, dlm_height;
    int extra_sample_h = 0, extra_sample_w = 0;
    adm_i64_dtype den_cube[3] = {0};

    /**
    DLM has the configurability of computing the metric only for the
    centre region. currently border_size defines the percentage of pixels to be avoided
    from all sides so that size of centre region is defined.
    */
#if ADM_REFLECT_PAD
    extra_sample_w = 0;
    extra_sample_h = 0;
#else
    extra_sample_w = 1;
    extra_sample_h = 1;
#endif

    border_h -= extra_sample_h;
    border_w -= extra_sample_w;

#if !ADM_REFLECT_PAD
    // If reflect pad is disabled & if border_size is 0, process 1 row,col pixels lesser
    border_h = MAX(1, border_h);
    border_w = MAX(1, border_w);
#endif

    loop_h = height - border_h;
    loop_w = width - border_w;

    dlm_height = height - (border_h << 1);
    dlm_width = width - (border_w << 1);

    // The width of i_dlm_add buffer will be extra only if padding is enabled
    int dlm_add_w = dlm_width + (ADM_REFLECT_PAD << 1);
    int16x4_t src16x4_rH_hi, src16x4_rV_hi, src16x4_rD_hi, src16x4_dH_hi, src16x4_dV_hi;
    int16x4_t src16x4_rH_lo, src16x4_rV_lo, src16x4_rD_lo, src16x4_dH_lo, src16x4_dV_lo;
    int32x4_t mul32x4_rdH0, mul32x4_rdH1, mul32x4_rrH0, mul32x4_rrH1, mul32x4_ddH0, mul32x4_ddH1;
    int32x4_t mul32x4_rrV0, mul32x4_rrV1, mul32x4_rrD0, mul32x4_rrD1;
    int64x2_t otdp_64x2_rdH00, otdp_64x2_rdH01, otdp_64x2_rdH10, otdp_64x2_rdH11;
    int64x2_t otmag_64x2_rrH00, otmag_64x2_rrH01, otmag_64x2_rrH10, otmag_64x2_rrH11;
    uint32x4_t chkZero32x4_rHlo, chkZero32x4_rHhi, chkZero32x4_rVlo, chkZero32x4_rVhi, chkZero32x4_rDlo, chkZero32x4_rDhi;
    int32x4_t src32x4_b1lo, src32x4_b1hi, src32x4_b2lo, src32x4_b2hi, src32x4_b3lo, src32x4_b3hi;
    int32x4_t src32x4_dH_lo, src32x4_dH_hi, src32x4_dV_lo, src32x4_dV_hi, src32x4_dD_lo, src32x4_dD_hi;
    int64x2_t mul64x2_H0, mul64x2_H1, mul64x2_H2, mul64x2_H3;
    int64x2_t mul64x2_V0, mul64x2_V1, mul64x2_V2, mul64x2_V3;
    int64x2_t mul64x2_D0, mul64x2_D1, mul64x2_D2, mul64x2_D3;
    int32x4_t sft32x4_Hlo, sft32x4_Hhi, sft32x4_Vlo, sft32x4_Vhi, sft32x4_Dlo, sft32x4_Dhi;
    int32x4_t tmp_Hlo, tmp_Hhi, tmp_Vlo, tmp_Vhi, tmp_Dlo, tmp_Dhi, admAdd_lo, admAdd_hi;
    int32x4_t tmpVal_Hlo, tmpVal_Hhi, tmpVal_Vlo, tmpVal_Vhi, tmpVal_Dlo, tmpVal_Dhi;
    uint16x4_t kh16x4_Hlo, kh16x4_Hhi, kh16x4_Vlo, kh16x4_Vhi, kh16x4_Dlo, kh16x4_Dhi;
    int16x8_t sft16x8_H, sft16x8_V, sft16x8_D, dlmRest_H, dlmRest_V, dlmRest_D, src16x8_dD;
    uint16x8_t angFlagBuf, angBuf0, angBuf1;
    int16x8_t sftModH, srcModH, sftModV, srcModV, sftModD, srcModD;
    int64x2_t mul64x2_rrrH, mul64x2_rrrV, mul64x2_rrrD;

    int32x4_t dupVal0 = vdupq_n_s32(32768);
    uint16x4_t dupVal1 = vdup_n_u16(32768);
    uint16x8_t dupConst1 = vdupq_n_u16(1);
    int32x4_t dupConstZero = vdupq_n_s32(0);

    int32_t buf1_adm_div[8] = {};
    int32_t buf2_adm_div[8] = {};
    int32_t buf3_adm_div[8] = {};
    int32_t buf_ot_dp[8] = {};
    int64_t buf_ot_dp_sq[8] = {};
    int64_t buf_ot_mag[8] = {};

    dwt2_dtype *refBandH = ref.bands[1];
    dwt2_dtype *refBandV = ref.bands[2];
    dwt2_dtype *refBandD = ref.bands[3];
    dwt2_dtype *distBandH = dist.bands[1];
    dwt2_dtype *distBandV = dist.bands[2];
    dwt2_dtype *distBandD = dist.bands[3];

    for (i = border_h; i < loop_h; i++)
    {
        if (extra_sample_w)
        {
            for (k = 1; k < 4; k++)
            {
                int16_t ref_abs = abs(ref.bands[k][i * width + border_w]);
                col0_ref_cube[k - 1] = (int64_t)ref_abs * ref_abs * ref_abs;
            }
        }
        for (j = border_w; j <= loop_w - 8; j += 8)
        {
            index = i * width + j;
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;
            restIndex = (i - border_h) * (dlm_width) + j - border_w;

            src16x4_rH_lo = vld1_s16(refBandH + index);
            src16x4_rV_lo = vld1_s16(refBandV + index);
            src16x4_dH_lo = vld1_s16(distBandH + index);
            src16x4_dV_lo = vld1_s16(distBandV + index);
            src16x4_rH_hi = vld1_s16(refBandH + index + 4);
            src16x4_rV_hi = vld1_s16(refBandV + index + 4);
            src16x4_dH_hi = vld1_s16(distBandH + index + 4);
            src16x4_dV_hi = vld1_s16(distBandV + index + 4);

            mul32x4_rdH0 = vmull_s16(src16x4_rH_lo, src16x4_dH_lo);
            mul32x4_rdH1 = vmull_s16(src16x4_rH_hi, src16x4_dH_hi);
            mul32x4_rrH0 = vmull_s16(src16x4_rH_lo, src16x4_rH_lo);
            mul32x4_rrH1 = vmull_s16(src16x4_rH_hi, src16x4_rH_hi);
            mul32x4_rrV0 = vmull_s16(src16x4_rV_lo, src16x4_rV_lo);
            mul32x4_rrV1 = vmull_s16(src16x4_rV_hi, src16x4_rV_hi);

            src16x8_dD = vld1q_s16(distBandD + index);
            src16x4_rD_lo = vld1_s16(refBandD + index);
            src16x4_rD_hi = vld1_s16(refBandD + index + 4);

            tmp_Hlo = vmovl_s16(src16x4_rH_lo);
            tmp_Hhi = vmovl_s16(src16x4_rH_hi);
            tmp_Vlo = vmovl_s16(src16x4_rV_lo);
            tmp_Vhi = vmovl_s16(src16x4_rV_hi);
            tmp_Dlo = vmovl_s16(src16x4_rD_lo);
            tmp_Dhi = vmovl_s16(src16x4_rD_hi);

            mul32x4_rrD0 = vmull_s16(src16x4_rD_lo, src16x4_rD_lo);
            mul32x4_rrD1 = vmull_s16(src16x4_rD_hi, src16x4_rD_hi);
            mul64x2_rrrH = vmull_s32(vget_low_s32(vabsq_s32(mul32x4_rrH0)), vget_low_s32(vabsq_s32(tmp_Hlo)));
            mul64x2_rrrV = vmull_s32(vget_low_s32(vabsq_s32(mul32x4_rrV0)), vget_low_s32(vabsq_s32(tmp_Vlo)));
            mul64x2_rrrD = vmull_s32(vget_low_s32(vabsq_s32(mul32x4_rrD0)), vget_low_s32(vabsq_s32(tmp_Dlo)));

            mul64x2_rrrH = vmlal_s32(mul64x2_rrrH, vget_high_s32(vabsq_s32(mul32x4_rrH0)), vget_high_s32(vabsq_s32(tmp_Hlo)));
            mul64x2_rrrV = vmlal_s32(mul64x2_rrrV, vget_high_s32(vabsq_s32(mul32x4_rrV0)), vget_high_s32(vabsq_s32(tmp_Vlo)));
            mul64x2_rrrD = vmlal_s32(mul64x2_rrrD, vget_high_s32(vabsq_s32(mul32x4_rrD0)), vget_high_s32(vabsq_s32(tmp_Dlo)));

            mul64x2_rrrH = vmlal_s32(mul64x2_rrrH, vget_low_s32(vabsq_s32(mul32x4_rrH1)), vget_low_s32(vabsq_s32(tmp_Hhi)));
            mul64x2_rrrV = vmlal_s32(mul64x2_rrrV, vget_low_s32(vabsq_s32(mul32x4_rrV1)), vget_low_s32(vabsq_s32(tmp_Vhi)));
            mul64x2_rrrD = vmlal_s32(mul64x2_rrrD, vget_low_s32(vabsq_s32(mul32x4_rrD1)), vget_low_s32(vabsq_s32(tmp_Dhi)));

            mul64x2_rrrH = vmlal_s32(mul64x2_rrrH, vget_high_s32(vabsq_s32(mul32x4_rrH1)), vget_high_s32(vabsq_s32(tmp_Hhi)));
            mul64x2_rrrV = vmlal_s32(mul64x2_rrrV, vget_high_s32(vabsq_s32(mul32x4_rrV1)), vget_high_s32(vabsq_s32(tmp_Vhi)));
            mul64x2_rrrD = vmlal_s32(mul64x2_rrrD, vget_high_s32(vabsq_s32(mul32x4_rrD1)), vget_high_s32(vabsq_s32(tmp_Dhi)));

            den_row_sum[0] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrH, 0);
            den_row_sum[1] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrV, 0);
            den_row_sum[2] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrD, 0);
            den_row_sum[0] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrH, 1);
            den_row_sum[1] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrV, 1);
            den_row_sum[2] += (adm_i64_dtype)vgetq_lane_s64(mul64x2_rrrD, 1);

            mul32x4_rdH0 = vmlal_s16(mul32x4_rdH0, src16x4_rV_lo, src16x4_dV_lo);
            mul32x4_rdH1 = vmlal_s16(mul32x4_rdH1, src16x4_rV_hi, src16x4_dV_hi);
            mul32x4_rrH0 = vmlal_s16(mul32x4_rrH0, src16x4_rV_lo, src16x4_rV_lo);
            mul32x4_rrH1 = vmlal_s16(mul32x4_rrH1, src16x4_rV_hi, src16x4_rV_hi);

            mul32x4_ddH0 = vmull_s16(src16x4_dH_lo, src16x4_dH_lo);
            mul32x4_ddH1 = vmull_s16(src16x4_dH_hi, src16x4_dH_hi);
            mul32x4_ddH0 = vmlal_s16(mul32x4_ddH0, src16x4_dV_lo, src16x4_dV_lo);
            mul32x4_ddH1 = vmlal_s16(mul32x4_ddH1, src16x4_dV_hi, src16x4_dV_hi);

            otdp_64x2_rdH00 = vmull_s32(vget_low_s32(mul32x4_rdH0), vget_low_s32(mul32x4_rdH0));
            otdp_64x2_rdH01 = vmull_s32(vget_high_s32(mul32x4_rdH0), vget_high_s32(mul32x4_rdH0));
            otdp_64x2_rdH10 = vmull_s32(vget_low_s32(mul32x4_rdH1), vget_low_s32(mul32x4_rdH1));
            otdp_64x2_rdH11 = vmull_s32(vget_high_s32(mul32x4_rdH1), vget_high_s32(mul32x4_rdH1));

            otmag_64x2_rrH00 = vmull_s32(vget_low_s32(mul32x4_rrH0), vget_low_s32(mul32x4_ddH0));
            otmag_64x2_rrH01 = vmull_s32(vget_high_s32(mul32x4_rrH0), vget_high_s32(mul32x4_ddH0));
            otmag_64x2_rrH10 = vmull_s32(vget_low_s32(mul32x4_rrH1), vget_low_s32(mul32x4_ddH1));
            otmag_64x2_rrH11 = vmull_s32(vget_high_s32(mul32x4_rrH1), vget_high_s32(mul32x4_ddH1));

            vst1q_s32(buf_ot_dp, mul32x4_rdH0);
            vst1q_s32(buf_ot_dp + 4, mul32x4_rdH1);

            vst1q_s64(buf_ot_dp_sq, otdp_64x2_rdH00);
            vst1q_s64(buf_ot_dp_sq + 2, otdp_64x2_rdH01);
            vst1q_s64(buf_ot_dp_sq + 4, otdp_64x2_rdH10);
            vst1q_s64(buf_ot_dp_sq + 6, otdp_64x2_rdH11);

            vst1q_s64(buf_ot_mag, otmag_64x2_rrH00);
            vst1q_s64(buf_ot_mag + 2, otmag_64x2_rrH01);
            vst1q_s64(buf_ot_mag + 4, otmag_64x2_rrH10);
            vst1q_s64(buf_ot_mag + 6, otmag_64x2_rrH11);

            for (l = 0; l < 8; l++)
            {
                angle_flag[l] = (uint16_t)((buf_ot_dp[l] >= 0) && (buf_ot_dp_sq[l] >= COS_1DEG_SQ * buf_ot_mag[l]));
                buf1_adm_div[l] = adm_div_lookup[ref.bands[1][index + l] + 32768];
                buf2_adm_div[l] = adm_div_lookup[ref.bands[2][index + l] + 32768];
                buf3_adm_div[l] = adm_div_lookup[ref.bands[3][index + l] + 32768];
            }

            src32x4_b1lo = vld1q_s32(buf1_adm_div);
            src32x4_b1hi = vld1q_s32(buf1_adm_div + 4);
            src32x4_dH_lo = vmovl_s16(src16x4_dH_lo);
            src32x4_dH_hi = vmovl_s16(src16x4_dH_hi);

            src32x4_b2lo = vld1q_s32(buf2_adm_div);
            src32x4_b2hi = vld1q_s32(buf2_adm_div + 4);
            src32x4_dV_lo = vmovl_s16(src16x4_dV_lo);
            src32x4_dV_hi = vmovl_s16(src16x4_dV_hi);

            src32x4_b3lo = vld1q_s32(buf3_adm_div);
            src32x4_b3hi = vld1q_s32(buf3_adm_div + 4);
            src32x4_dD_lo = vmovl_s16(vget_low_s16(src16x8_dD));
            src32x4_dD_hi = vmovl_s16(vget_high_s16(src16x8_dD));

            mul64x2_H0 = vmull_s32(vget_low_s32(src32x4_b1lo), vget_low_s32(src32x4_dH_lo));
            mul64x2_H1 = vmull_s32(vget_high_s32(src32x4_b1lo), vget_high_s32(src32x4_dH_lo));
            mul64x2_H2 = vmull_s32(vget_low_s32(src32x4_b1hi), vget_low_s32(src32x4_dH_hi));
            mul64x2_H3 = vmull_s32(vget_high_s32(src32x4_b1hi), vget_high_s32(src32x4_dH_hi));

            mul64x2_V0 = vmull_s32(vget_low_s32(src32x4_b2lo), vget_low_s32(src32x4_dV_lo));
            mul64x2_V1 = vmull_s32(vget_high_s32(src32x4_b2lo), vget_high_s32(src32x4_dV_lo));
            mul64x2_V2 = vmull_s32(vget_low_s32(src32x4_b2hi), vget_low_s32(src32x4_dV_hi));
            mul64x2_V3 = vmull_s32(vget_high_s32(src32x4_b2hi), vget_high_s32(src32x4_dV_hi));

            mul64x2_D0 = vmull_s32(vget_low_s32(src32x4_b3lo), vget_low_s32(src32x4_dD_lo));
            mul64x2_D1 = vmull_s32(vget_high_s32(src32x4_b3lo), vget_high_s32(src32x4_dD_lo));
            mul64x2_D2 = vmull_s32(vget_low_s32(src32x4_b3hi), vget_low_s32(src32x4_dD_hi));
            mul64x2_D3 = vmull_s32(vget_high_s32(src32x4_b3hi), vget_high_s32(src32x4_dD_hi));

            sft32x4_Hlo = vcombine_s32(vrshrn_n_s64(mul64x2_H0, 15), vrshrn_n_s64(mul64x2_H1, 15));
            sft32x4_Hhi = vcombine_s32(vrshrn_n_s64(mul64x2_H2, 15), vrshrn_n_s64(mul64x2_H3, 15));
            sft32x4_Vlo = vcombine_s32(vrshrn_n_s64(mul64x2_V0, 15), vrshrn_n_s64(mul64x2_V1, 15));
            sft32x4_Vhi = vcombine_s32(vrshrn_n_s64(mul64x2_V2, 15), vrshrn_n_s64(mul64x2_V3, 15));
            sft32x4_Dlo = vcombine_s32(vrshrn_n_s64(mul64x2_D0, 15), vrshrn_n_s64(mul64x2_D1, 15));
            sft32x4_Dhi = vcombine_s32(vrshrn_n_s64(mul64x2_D2, 15), vrshrn_n_s64(mul64x2_D3, 15));

            chkZero32x4_rHlo = vceqq_s32(tmp_Hlo, dupConstZero);
            chkZero32x4_rHhi = vceqq_s32(tmp_Hhi, dupConstZero);
            chkZero32x4_rVlo = vceqq_s32(tmp_Vlo, dupConstZero);
            chkZero32x4_rVhi = vceqq_s32(tmp_Vhi, dupConstZero);
            chkZero32x4_rDlo = vceqq_s32(tmp_Dlo, dupConstZero);
            chkZero32x4_rDhi = vceqq_s32(tmp_Dhi, dupConstZero);

            tmp_Hlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rHlo, vreinterpretq_u32_s32(dupVal0)));
            tmp_Hhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rHhi, vreinterpretq_u32_s32(dupVal0)));
            tmp_Vlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rVlo, vreinterpretq_u32_s32(dupVal0)));
            tmp_Vhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rVhi, vreinterpretq_u32_s32(dupVal0)));
            tmp_Dlo = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rDlo, vreinterpretq_u32_s32(dupVal0)));
            tmp_Dhi = vreinterpretq_s32_u32(vandq_u32(chkZero32x4_rDhi, vreinterpretq_u32_s32(dupVal0)));

            tmpVal_Hlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rHlo), vreinterpretq_u32_s32(sft32x4_Hlo)));
            tmpVal_Hhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rHhi), vreinterpretq_u32_s32(sft32x4_Hhi)));
            tmpVal_Vlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rVlo), vreinterpretq_u32_s32(sft32x4_Vlo)));
            tmpVal_Vhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rVhi), vreinterpretq_u32_s32(sft32x4_Vhi)));
            tmpVal_Dlo = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rDlo), vreinterpretq_u32_s32(sft32x4_Dlo)));
            tmpVal_Dhi = vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(chkZero32x4_rDhi), vreinterpretq_u32_s32(sft32x4_Dhi)));

            tmp_Hlo = vaddq_s32(tmp_Hlo, tmpVal_Hlo);
            tmp_Hhi = vaddq_s32(tmp_Hhi, tmpVal_Hhi);
            tmp_Vlo = vaddq_s32(tmp_Vlo, tmpVal_Vlo);
            tmp_Vhi = vaddq_s32(tmp_Vhi, tmpVal_Vhi);
            tmp_Dlo = vaddq_s32(tmp_Dlo, tmpVal_Dlo);
            tmp_Dhi = vaddq_s32(tmp_Dhi, tmpVal_Dhi);

            kh16x4_Hlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Hlo));
            kh16x4_Hhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Hhi));
            kh16x4_Vlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Vlo));
            kh16x4_Vhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Vhi));
            kh16x4_Dlo = vmin_u16(dupVal1, vqmovun_s32(tmp_Dlo));
            kh16x4_Dhi = vmin_u16(dupVal1, vqmovun_s32(tmp_Dhi));

            tmpVal_Hlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Hlo)), vmovl_s16(src16x4_rH_lo));
            tmpVal_Hhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Hhi)), vmovl_s16(src16x4_rH_hi));
            tmpVal_Vlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Vlo)), vmovl_s16(src16x4_rV_lo));
            tmpVal_Vhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Vhi)), vmovl_s16(src16x4_rV_hi));
            tmpVal_Dlo = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Dlo)), vmovl_s16(src16x4_rD_lo));
            tmpVal_Dhi = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(kh16x4_Dhi)), vmovl_s16(src16x4_rD_hi));

            sft16x8_H = vcombine_s16(vrshrn_n_s32((tmpVal_Hlo), 15), vrshrn_n_s32((tmpVal_Hhi), 15));
            sft16x8_V = vcombine_s16(vrshrn_n_s32((tmpVal_Vlo), 15), vrshrn_n_s32((tmpVal_Vhi), 15));
            sft16x8_D = vcombine_s16(vrshrn_n_s32((tmpVal_Dlo), 15), vrshrn_n_s32((tmpVal_Dhi), 15));

            angFlagBuf = vld1q_u16(angle_flag);
            angBuf1 = vcltq_u16(angFlagBuf, dupConst1);
            angBuf0 = vcgtq_s16(vreinterpretq_s16_u16(angFlagBuf), vreinterpretq_s16_s32(dupConstZero));

            sftModH = vandq_s16(sft16x8_H, vreinterpretq_s16_u16(angBuf1));
            sftModV = vandq_s16(sft16x8_V, vreinterpretq_s16_u16(angBuf1));
            sftModD = vandq_s16(sft16x8_D, vreinterpretq_s16_u16(angBuf1));

            srcModH = vandq_s16(vcombine_s16(src16x4_dH_lo, src16x4_dH_hi), vreinterpretq_s16_u16(angBuf0));
            srcModV = vandq_s16(vcombine_s16(src16x4_dV_lo, src16x4_dV_hi), vreinterpretq_s16_u16(angBuf0));
            srcModD = vandq_s16(src16x8_dD, vreinterpretq_s16_u16(angBuf0));

            dlmRest_H = vaddq_s16(sftModH, srcModH);
            dlmRest_V = vaddq_s16(sftModV, srcModV);
            dlmRest_D = vaddq_s16(sftModD, srcModD);

            admAdd_lo = vabdl_s16(src16x4_dH_lo, vget_low_s16(dlmRest_H));
            admAdd_hi = vabdl_s16(src16x4_dH_hi, vget_high_s16(dlmRest_H));
            admAdd_lo = vabal_s16(admAdd_lo, src16x4_dV_lo, vget_low_s16(dlmRest_V));
            admAdd_hi = vabal_s16(admAdd_hi, src16x4_dV_hi, vget_high_s16(dlmRest_V));
            admAdd_lo = vabal_s16(admAdd_lo, vget_low_s16(src16x8_dD), vget_low_s16(dlmRest_D));
            admAdd_hi = vabal_s16(admAdd_hi, vget_high_s16(src16x8_dD), vget_high_s16(dlmRest_D));

            vst1q_s16((i_dlm_rest.bands[1] + restIndex), dlmRest_H);
            vst1q_s16((i_dlm_rest.bands[2] + restIndex), dlmRest_V);
            vst1q_s16((i_dlm_rest.bands[3] + restIndex), dlmRest_D);
            vst1q_s32((i_dlm_add + addIndex), admAdd_lo);
            vst1q_s32((i_dlm_add + addIndex + 4), admAdd_hi);
        }
        for (; j < loop_w; j++)
        {
            index = i * width + j;
            // If padding is enabled the computation of i_dlm_add will be from 1,1 & later padded
            addIndex = (i + ADM_REFLECT_PAD - border_h) * (dlm_add_w) + j + ADM_REFLECT_PAD - border_w;
            restIndex = (i - border_h) * (dlm_width) + j - border_w;
            ot_dp = ((adm_i32_dtype)ref.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * dist.bands[2][index]);
            o_mag_sq = ((adm_i32_dtype)ref.bands[1][index] * ref.bands[1][index]) + ((adm_i32_dtype)ref.bands[2][index] * ref.bands[2][index]);
            t_mag_sq = ((adm_i32_dtype)dist.bands[1][index] * dist.bands[1][index]) + ((adm_i32_dtype)dist.bands[2][index] * dist.bands[2][index]);
            angle_flagC = ((ot_dp >= 0) && (((adm_i64_dtype)ot_dp * ot_dp) >= COS_1DEG_SQ * ((adm_i64_dtype)o_mag_sq * t_mag_sq)));
            i_dlm_add[addIndex] = 0;
            for (k = 1; k < 4; k++)
            {
                /**
                 * Division dist/ref is carried using lookup table adm_div_lookup and converted to multiplication
                 */
                adm_i32_dtype tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((adm_i64_dtype)adm_div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
                adm_u16_dtype kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
                /**
                 * kh is in Q15 type and ref.bands[k][index] is in Q16 type hence shifted by
                 * 15 to make result Q16
                 */
                tmp_val = (((adm_i32_dtype)kh * ref.bands[k][index]) + 16384) >> 15;

                i_dlm_rest.bands[k][restIndex] = angle_flagC ? dist.bands[k][index] : tmp_val;
                /**
                 * Absolute is taken here for the difference value instead of
                 * taking absolute of pyr_2 in integer_dlm_contrast_mask_one_way function
                 */
                i_dlm_add[addIndex] += (int32_t)abs(dist.bands[k][index] - i_dlm_rest.bands[k][restIndex]);

                // Accumulating denominator score to avoid load in next stage
                int16_t ref_abs = abs(ref.bands[k][index]);
                den_cube[k - 1] = (adm_i64_dtype)ref_abs * ref_abs * ref_abs;

                den_row_sum[k - 1] += den_cube[k - 1];
            }
        }
        if (extra_sample_w)
        {
            for (k = 0; k < 3; k++)
            {
                den_row_sum[k] -= den_cube[k];
                den_row_sum[k] -= col0_ref_cube[k];
            }
        }
        if ((i != border_h && i != (loop_h - 1)) || !extra_sample_h)
        {
            for (k = 0; k < 3; k++)
            {
                den_sum[k] += den_row_sum[k];
            }
        }
        den_row_sum[0] = 0;
        den_row_sum[1] = 0;
        den_row_sum[2] = 0;

        if (!extra_sample_w)
        {
            addIndex = (i + 1 - border_h) * (dlm_add_w);
            i_dlm_add[addIndex + 0] = i_dlm_add[addIndex + 2];
            i_dlm_add[addIndex + dlm_width + 1] = i_dlm_add[addIndex + dlm_width - 1];
        }
    }

    if (!extra_sample_h)
    {
        int row2Idx = 2 * (dlm_add_w);
        int rowLast2Idx = (dlm_height - 1) * (dlm_add_w);
        int rowLastPadIdx = (dlm_height + 1) * (dlm_add_w);

        memcpy(&i_dlm_add[0], &i_dlm_add[row2Idx], sizeof(int32_t) * (dlm_add_w));
        memcpy(&i_dlm_add[rowLastPadIdx], &i_dlm_add[rowLast2Idx], sizeof(int32_t) * (dlm_width + 2));
    }

    // Calculating denominator score
    double den_band = 0;
    for (k = 0; k < 3; k++)
    {
        double accum_den = (double)den_sum[k] / ADM_CUBE_DIV;
        den_band += powf((double)(accum_den), 1.0 / 3.0);
    }
    // compensation for the division by thirty in the numerator
    *adm_score_den = (den_band * 30) + 1e-4;
}
