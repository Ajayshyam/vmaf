#include "common.h"
#include <arm_neon.h>

#define USE_SHIFT_BY_8 1
void integer_funque_dwt2_neon(
								spat_fil_output_dtype *src, 
								i_dwt2buffers *dwt2_dst, 
								ptrdiff_t dst_stride, 
								int width, 
								int height
							 )
{
// The below praogram is same as C
#if 0

    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT

#if !USE_SHIFT_BY_8
    dwt2_dtype filter_coeff_lo[2] = {23170,  23170};
    dwt2_dtype filter_coeff_hi[2] = {23170, -23170};
#else
    dwt2_dtype filter_coeff_lo[2] = {181, 181};
    dwt2_dtype filter_coeff_hi[2] = {181, -181};
#endif

    dwt2_dtype *tmplo = malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)));
    dwt2_dtype *tmphi = malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)));

    dwt2_dtype *band_a = dwt2_dst->bands[0];
    dwt2_dtype *band_h = dwt2_dst->bands[1];
    dwt2_dtype *band_v = dwt2_dst->bands[2];
    dwt2_dtype *band_d = dwt2_dst->bands[3];
    dwt2_accum_dtype accum;
    int16_t row_idx0, row_idx1, col_idx0, col_idx1;

    for (unsigned i=0; i < (height+1)/2; ++i)
    {
        row_idx0 = 2*i;
        // row_idx0 = row_idx0 < height ? row_idx0 : height;
        row_idx1 = 2*i+1;
        row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

        /* Vertical pass. */
        for(unsigned j=0; j<width; ++j){
            accum = 0;
            accum += filter_coeff_lo[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_lo[1] * src[(row_idx1)*width+j];
            tmplo[j] = (dwt2_dtype) (accum >> DWT2_INTER_SHIFT);

            accum = 0;
            accum += filter_coeff_hi[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_hi[1] * src[(row_idx1)*width+j];
            tmphi[j] = (dwt2_dtype) (accum >> DWT2_INTER_SHIFT);
        }

        /* Horizontal pass (lo and hi). */
        for(unsigned j=0; j<(width+1)/2; ++j)
        {
            col_idx0 = 2*j;
            col_idx1 = 2*j+1;
            col_idx1 = col_idx1 < width ? col_idx1 : 2*j;

            accum = 0;
            accum += filter_coeff_lo[0] * tmplo[col_idx0];
            accum += filter_coeff_lo[1] * tmplo[col_idx1];
            band_a[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = 0;
            accum += filter_coeff_lo[0] * tmphi[col_idx0];
            accum += filter_coeff_lo[1] * tmphi[col_idx1];
            band_h[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = 0;
            accum += filter_coeff_hi[0] * tmplo[col_idx0];
            accum += filter_coeff_hi[1] * tmplo[col_idx1];
            band_v[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);

            accum = 0;
            accum += filter_coeff_hi[0] * tmphi[col_idx0];
            accum += filter_coeff_hi[1] * tmphi[col_idx1];
            band_d[i*dst_px_stride+j] = (dwt2_dtype) (accum >> DWT2_OUT_SHIFT);
        }
    }

    free(tmplo);
    free(tmphi);

// SIMD code is below
#else

		int16x8_t src16x8_0, src16x8_1;
		int32x4_t add32x4_lo, add32x4_hi, sub32x4_lo, sub32x4_hi;
		int32x4_t mul32x4_0_lo, mul32x4_0_hi, mul32x4_1_lo, mul32x4_1_hi;
		int32x4x2_t arr32x4_lo, arr32x4_hi;
		int16x4_t sft16x4_0, sft16x4_1, sft16x4_2, sft16x4_3;
		int32_t filter_coeff = 181;
		int16_t row_idx0, row_idx1;
		int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
		dwt2_dtype *band_a = dwt2_dst->bands[0];
		dwt2_dtype *band_h = dwt2_dst->bands[1];
		dwt2_dtype *band_v = dwt2_dst->bands[2];
		dwt2_dtype *band_d = dwt2_dst->bands[3];

		for (unsigned i=0; i < (height+1)/2; ++i)
		{
			row_idx0 = 2*i;
			// row_idx0 = row_idx0 < height ? row_idx0 : height;
			row_idx1 = 2*i+1;
			row_idx1 = row_idx1 < height ? row_idx1 : 2*i;

			/* Vertical pass. */
			for(unsigned j = 0; j < width; j += 8) {

				src16x8_0 = vld1q_s16(src + row_idx0 * width + j);
				src16x8_1 = vld1q_s16(src + row_idx1 * width + j);

				add32x4_lo = vaddl_s16(vget_low_s16(src16x8_0), vget_low_s16(src16x8_1));
				add32x4_hi = vaddl_high_s16(src16x8_0, src16x8_1);
				mul32x4_0_lo = vmulq_n_s32(add32x4_lo, filter_coeff);    // tmplo results
				mul32x4_0_hi = vmulq_n_s32(add32x4_hi, filter_coeff);    // tmplo results

				sub32x4_lo = vsubl_s16(vget_low_s16(src16x8_0), vget_low_s16(src16x8_1));
				sub32x4_hi = vsubl_high_s16(src16x8_0, src16x8_1);
				mul32x4_1_lo = vmulq_n_s32(sub32x4_lo, filter_coeff);    // tmphi results
				mul32x4_1_hi = vmulq_n_s32(sub32x4_hi, filter_coeff);    // tmphi results

				add32x4_lo = vpaddq_s32(mul32x4_0_lo, mul32x4_0_hi);    // use only first 4
				add32x4_hi = vpaddq_s32(mul32x4_1_lo, mul32x4_1_hi);    // use only first 4

				arr32x4_lo = vuzpq_s32(mul32x4_0_lo, mul32x4_0_hi);    // use only first 4
				arr32x4_hi = vuzpq_s32(mul32x4_1_lo, mul32x4_1_hi);    // use only first 4
				sub32x4_lo = vsubq_s32(arr32x4_lo.val[1], arr32x4_lo.val[0]);
				sub32x4_hi = vsubq_s32(arr32x4_hi.val[1], arr32x4_hi.val[0]);

				mul32x4_0_lo = vmulq_n_s32(add32x4_lo, filter_coeff);    // tmplo results
				mul32x4_0_hi = vmulq_n_s32(add32x4_hi, filter_coeff);    // tmplo results
				mul32x4_1_lo = vmulq_n_s32(sub32x4_lo, filter_coeff);    // tmphi results
				mul32x4_1_hi = vmulq_n_s32(sub32x4_hi, filter_coeff);    // tmphi results

//				half shift and store the results
//				sft16x4_0 = vshrn_n_s32(mul32x4_0_lo, DWT2_OUT_SHIFT);
//				sft16x4_1 = vshrn_n_s32(mul32x4_0_hi, DWT2_OUT_SHIFT);
//				sft16x4_2 = vshrn_n_s32(mul32x4_1_lo, DWT2_OUT_SHIFT);
//				sft16x4_3 = vshrn_n_s32(mul32x4_1_hi, DWT2_OUT_SHIFT);

				vst1_s16(band_a + i * dst_px_stride + (j >> 1), vmovn_s32(mul32x4_0_lo));
				vst1_s16(band_h + i * dst_px_stride + (j >> 1), vmovn_s32(mul32x4_0_hi));
				vst1_s16(band_v + i * dst_px_stride + (j >> 1), vmovn_s32(mul32x4_1_lo));
				vst1_s16(band_d + i * dst_px_stride + (j >> 1), vmovn_s32(mul32x4_1_hi));
			}
		}
#endif
}


