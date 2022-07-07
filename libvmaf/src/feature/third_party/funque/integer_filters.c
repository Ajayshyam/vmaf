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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mem.h"
#include "offset.h"
#include "integer_filters.h"

void funque_dwt2(funque_dtype *src, dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(funque_dtype);
    funque_dtype filter_coeff_lo[2] = {0.707106781,  0.707106781};
    funque_dtype filter_coeff_hi[2] = {0.707106781, -0.707106781};

    funque_dtype *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(funque_dtype)), MAX_ALIGN);
    funque_dtype *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(funque_dtype)), MAX_ALIGN);

    funque_dtype *band_a = dwt2_dst->bands[0];
    funque_dtype *band_h = dwt2_dst->bands[1];
    funque_dtype *band_v = dwt2_dst->bands[2];
    funque_dtype *band_d = dwt2_dst->bands[3];
    funque_dtype accum;
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
            tmplo[j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * src[(row_idx0)*width+j];
            accum += filter_coeff_hi[1] * src[(row_idx1)*width+j];
            tmphi[j] = accum;
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
            band_a[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_lo[0] * tmphi[col_idx0];
            accum += filter_coeff_lo[1] * tmphi[col_idx1];
            band_h[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmplo[col_idx0];
            accum += filter_coeff_hi[1] * tmplo[col_idx1];
            band_v[i*dst_px_stride+j] = accum;

            accum = 0;
            accum += filter_coeff_hi[0] * tmphi[col_idx0];
            accum += filter_coeff_hi[1] * tmphi[col_idx1];
            band_d[i*dst_px_stride+j] = accum;
        }
    }
    aligned_free(tmplo);
    aligned_free(tmphi);
}

void float_frame_to_csv(float *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%f,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void int16_frame_to_csv(int16_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%d,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void int32_frame_to_csv(int32_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%d,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void int64_frame_to_csv(int64_t *ptr_frm, int width, int height, char *filename)
{
    FILE *fptr = fopen(filename, "w");
    fprintf(fptr, ",");
    for(int idx_w=0; idx_w<width; idx_w++)
    {
        fprintf(fptr, "%d,", idx_w);
    }
    fprintf(fptr, "\n");

    for(int idx_h=0; idx_h<height; idx_h++)
    {
        fprintf(fptr, "%d,", idx_h);
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            fprintf(fptr, "%ld,", ptr_frm[idx_h*width+idx_w]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}


void integer_funque_dwt2(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height)
{
    int dst_px_stride = dst_stride / sizeof(dwt2_dtype);
    // Filter coefficients are upshifted by DWT2_COEFF_UPSHIFT
    dwt2_dtype filter_coeff_lo[2] = {23170,  23170};
    dwt2_dtype filter_coeff_hi[2] = {23170, -23170};

    dwt2_dtype *tmplo = aligned_malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)), MAX_ALIGN);
    dwt2_dtype *tmphi = aligned_malloc(ALIGN_CEIL(width * sizeof(dwt2_dtype)), MAX_ALIGN);

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
    aligned_free(tmplo);
    aligned_free(tmphi);
}

//Convolution using coefficients from python workspace
void spatial_filter(funque_dtype *src, funque_dtype *dst, ptrdiff_t dst_stride, int width, int height)
{
    //Copied the coefficients from python coefficients
    funque_dtype filter_coeffs[21] = {-0.01373464, -0.01608515, -0.01890698, -0.02215702, -0.02546262, 
                             -0.02742965, -0.02361034, -0.00100996,  0.07137023,  0.22121922,
                             0.3279824 ,  0.22121922,  0.07137023, -0.00100996, -0.02361034,
                             -0.02742965, -0.02546262, -0.02215702, -0.01890698, -0.01608515,
                             -0.01373464};
    
    int src_px_stride = width;
    int dst_px_stride = width;

    funque_dtype *tmp = aligned_malloc(ALIGN_CEIL(src_px_stride * sizeof(funque_dtype)), MAX_ALIGN);
    funque_dtype fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;
    int fwidth = 21;
    for (i = 0; i < height; ++i) {

        /* Vertical pass. */
        for (j = 0; j < width; ++j) {
            double accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = filter_coeffs[fi];
                
                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -(ii+1)  : (ii >= height ? 2 * height - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += (double) fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < width; ++j) {
            funque_dtype accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = filter_coeffs[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -(jj+1) : (jj >= width ? 2 * width - jj - 1 : jj);

                imgcoeff = tmp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);

    return;
}

void convert_int16_2_float(int16_t *fixed_src, funque_dtype *float_dst, int width, int height, int downshift_factor) {
    double divide_val = pow(2, downshift_factor);
    for(int idx_h=0; idx_h<height; idx_h++)
    {
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            float_dst[idx_h*width+idx_w] = (funque_dtype) (fixed_src[idx_h*width+idx_w] / divide_val);
        }
    }
}

void convert_int32_2_float(int32_t *fixed_src, funque_dtype *float_dst, int width, int height, int downshift_factor) {
   
    for(int idx_h=0; idx_h<height; idx_h++)
    {
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            float_dst[idx_h*width+idx_w] = (funque_dtype) (fixed_src[idx_h*width+idx_w] >> downshift_factor);
        }
    }
}

void convert_int64_2_float(int32_t *fixed_src, funque_dtype *float_dst, int width, int height, int downshift_factor) {
   
    for(int idx_h=0; idx_h<height; idx_h++)
    {
        for(int idx_w=0; idx_w<width; idx_w++)
        {
            float_dst[idx_h*width+idx_w] = (funque_dtype) (fixed_src[idx_h*width+idx_w] >> downshift_factor);
        }
    }
}

void fix2float(void *fixed_src, funque_dtype *float_dst, int width, int height, int downshift_factor, int fixed_sz)
{
    if(fixed_sz == sizeof(int16_t))
        convert_int16_2_float((int16_t*)fixed_src, float_dst, width, height, downshift_factor);
    else if(fixed_sz == sizeof(int32_t))
        convert_int32_2_float((int32_t*)fixed_src, float_dst, width, height, downshift_factor);
    else if(fixed_sz == sizeof(int64_t))
        convert_int32_2_float((int64_t*)fixed_src, float_dst, width, height, downshift_factor);
    else
        printf("Unsupported input datatype for fixed to float concersion");
}

void integer_spatial_filter(uint8_t *src, spat_fil_output_dtype *dst, int width, int height)
{
    funque_dtype filter_coeffs[21] = {-0.01373464, -0.01608515, -0.01890698, -0.02215702, -0.02546262, 
                             -0.02742965, -0.02361034, -0.00100996,  0.07137023,  0.22121922,
                             0.3279824 ,  0.22121922,  0.07137023, -0.00100996, -0.02361034,
                             -0.02742965, -0.02546262, -0.02215702, -0.01890698, -0.01608515,
                             -0.01373464};
    // //Copied the coefficients from python coefficients
    // spat_fil_coeff_dtype filter_coeffs[21] = {-450, -527, -620, -726, -834, -899, -774, -33,
    //                                            2339, 7249, 10747, 7249, 2339, -33, -774, -899,
    //                                           -834, -726, -620, -527, -450};

    //This for loop has to be removed and use constant values for coefficients
    //This loop is just for experimental purpose
    // The floating-point coefficients are upshifted for fixed point implementation by SPAT_FILTER_COEFF_SHIFT
    static spat_fil_coeff_dtype i_filter_coeffs[21], cnt=0;
    if(cnt == 0)
    {
        for(int i=0; i<21; i++)
        {
            i_filter_coeffs[i] = round(filter_coeffs[i]*pow(2, SPAT_FILTER_COEFF_SHIFT));
        }
        cnt++;
    }

    int src_px_stride = width;
    int dst_px_stride = width;

    spat_fil_inter_dtype *tmp = aligned_malloc(ALIGN_CEIL(src_px_stride * sizeof(spat_fil_inter_dtype)), MAX_ALIGN);
    spat_fil_inter_dtype imgcoeff;

    int i, j, fi, fj, ii, jj;
    int fwidth = 21;
    for (i = 0; i < height; ++i) {

        /* Vertical pass. */
        for (j = 0; j < width; ++j) {
            spat_fil_accum_dtype accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                // fcoeff = filter_coeffs[fi];
                
                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -(ii+1)  : (ii >= height ? 2 * height - ii - 1 : ii);

                // imgcoeff = src[ii * src_px_stride + j];

                accum += i_filter_coeffs[fi] * src[ii * src_px_stride + j];
            }

            tmp[j] = (spat_fil_inter_dtype) (accum >> SPAT_FILTER_INTER_SHIFT);
        }

        /* Horizontal pass. */
        for (j = 0; j < width; ++j) {
            spat_fil_accum_dtype accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                // fcoeff = filter_coeffs[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -(jj+1) : (jj >= width ? 2 * width - jj - 1 : jj);

                // imgcoeff = tmp[jj];

                accum += i_filter_coeffs[fj] * tmp[jj];
            }

            dst[i * dst_px_stride + j] = (spat_fil_output_dtype) (accum >> SPAT_FILTER_OUT_SHIFT);
        }
    }

    aligned_free(tmp);

    return;
}

// void convert_fix2float(funque_dtype *src, )
void normalize_bitdepth(funque_dtype *src, funque_dtype *dst, int scaler, ptrdiff_t dst_stride, int width, int height)
{
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            dst[j] = src[j] / scaler;
        }
        dst += dst_stride / sizeof(funque_dtype);
        src += dst_stride / sizeof(funque_dtype);
    }
    return;
}
