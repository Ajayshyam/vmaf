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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <time.h>

const int INTER_RESIZE_COEF_BITS = 11;
// const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
const int INTER_RESIZE_COEF_SCALE = 2048;
static const int MAX_ESIZE = 16;

#define CLIP3(X,MIN,MAX) ((X < MIN) ? MIN : (X > MAX) ? MAX : X)
#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
#define MIN(LEFT, RIGHT) (LEFT < RIGHT ? LEFT : RIGHT)

#define MEASURE_SUB_COMPONENTS 0
#define MEASURE_TOTAL 1
#define USE_C_VRESIZE 1

static void interpolateCubic(float x, float* coeffs)
{
    const float A = -0.75f;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

void hresize_neon(const unsigned char** src, int** dst, int count,
    const int* xofs, const short* alpha,
    int swidth, int dwidth, int cn, int xmin, int xmax) {
    int first_col_count = 0;
    uint8x8_t src1_8x8, src2_8x8,src3_8x8;
    int simd_loop = (xmax/8)*8;
    int num_pix = 8;
    
    for (int k = 0; k < count; k++)
    {
        const unsigned char* S = src[k];
        int* D = dst[k];
        int dx = 0, limit = xmin;
        for (;;)
        {
            for (; dx < limit; dx++)
            {
                int j, sx = xofs[dx] - cn;
                int v = 0;
                for (j = 0; j < 4; j++)
                {
                    int sxj = sx + j * cn;
                    if ((unsigned)sxj >= (unsigned)swidth)
                    {
                        while (sxj < 0)
                            sxj += cn;
                        while (sxj >= swidth)
                            sxj -= cn;
                    }
                    v += S[sxj] * alpha[j];
                }
                D[dx] = v;
            }
            if (limit == dwidth)
                break;
            int sx = xofs[1];
            int start = sx-cn;
            src1_8x8 = vld1_u8(S+start);
           
            for (; dx < simd_loop; )
            {
                start+=num_pix;
                src2_8x8 = vld1_u8(S+start);
                start+=num_pix;
                src3_8x8 = vld1_u8(S+start);
 
                uint16x8_t movl1_16x8 = vmovl_u8(src1_8x8);
                uint16x8_t movl2_16x8 = vmovl_u8(src2_8x8);
                uint16x8_t movl3_16x8 = vmovl_u8(src3_8x8);
                int16x8_t s_movl1_16x8 = vreinterpretq_s16_u16(movl1_16x8);
                int16x8_t s_movl2_16x8 = vreinterpretq_s16_u16(movl2_16x8);
                int16x8_t s_movl3_16x8 = vreinterpretq_s16_u16(movl3_16x8);
                int16x8x2_t t1 = vuzpq_s16(s_movl1_16x8, s_movl2_16x8);// 0 odd, 1 even
                int16x8x2_t t2 = vuzpq_s16(s_movl3_16x8, s_movl3_16x8);
                int16x8_t vx1 = vextq_s16(t1.val[0], t2.val[0],1);//s_movl3_16x8,1);
                int16x8_t vx2 = vextq_s16(t1.val[1], t2.val[1],1);
                int32x4_t m1_l = vmull_n_s16(vget_low_s16(t1.val[0]), alpha[0]);
                int32x4_t m1_h = vmull_n_s16(vget_high_s16(t1.val[0]), alpha[0]);
                int32x4_t m2_l = vmlal_n_s16(m1_l, vget_low_s16(vx1), alpha[1]);
                int32x4_t m2_h = vmlal_n_s16(m1_h, vget_high_s16(vx1), alpha[1]);
                int32x4_t m3_l = vmlal_n_s16(m2_l, vget_low_s16(t1.val[1]), alpha[2]);
                int32x4_t m3_h = vmlal_n_s16(m2_h, vget_high_s16(t1.val[1]), alpha[2]);
                int32x4_t out_l = vmlal_n_s16(m3_l, vget_low_s16(vx2), alpha[3]); // final out
                int32x4_t out_h = vmlal_n_s16(m3_h, vget_high_s16(vx2), alpha[3]); // final out

                vst1q_s32(D+dx, out_l);
                dx+=4;
                vst1q_s32(D+dx, out_h);
                dx+=4;
                src1_8x8 = src3_8x8;
            }

            for (; dx < xmax; dx++ )
            {
                int sx2 = xofs[dx]; //sx - 2, 4, 6, 8....
                D[dx] = S[sx2 - cn] * alpha[0] + S[sx2] * alpha[1] + S[sx2 + cn] * alpha[2] + S[sx2 + cn * 2] * alpha[3];
            }
            
            limit = dwidth;
        }
    }
}

//   clock_t startTime_g, stopTime_g;
//     double msecElapsed_hresize_g = 0;

void hresize(const unsigned char** src, int** dst, int count,
    const int* xofs, const short* alpha,
    int swidth, int dwidth, int cn, int xmin, int xmax) {
    
    for (int k = 0; k < count; k++)
    {
        const unsigned char* S = src[k];
        int* D = dst[k];
        int dx = 0, limit = xmin;
        for (;;)
        {
            for (; dx < limit; dx++)
            {
                int j, sx = xofs[dx] - cn;
                int v = 0;
                for (j = 0; j < 4; j++)
                {
                    int sxj = sx + j * cn;
                    if ((unsigned)sxj >= (unsigned)swidth)
                    {
                        while (sxj < 0)
                            sxj += cn;
                        while (sxj >= swidth)
                            sxj -= cn;
                    }
                    v += S[sxj] * alpha[j];
                }
                D[dx] = v;
            }
            if (limit == dwidth)
                break;
            // startTime_g = clock();

            for (; dx < xmax; dx++)
            {
                int sx = xofs[dx]; //sx - 2, 4, 6, 8....
                D[dx] = S[sx - cn] * alpha[0] + S[sx] * alpha[1] + S[sx + cn] * alpha[2] + S[sx + cn * 2] * alpha[3];
            }
                //  stopTime_g = clock();
            // msecElapsed_hresize_g += (stopTime_g - startTime_g) * 1000.0 / CLOCKS_PER_SEC;
       
            limit = dwidth;
        }
    }
}

unsigned char castOp(int val)
{
    int bits = 22;
    int SHIFT = bits;
    int DELTA = (1 << (bits - 1));
    return CLIP3((val + DELTA) >> SHIFT, 0, 255);
}

void vresize(const int** src, unsigned char* dst, const short* beta, int width)
{
    int b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
    const int* S0 = src[0], * S1 = src[1], * S2 = src[2], * S3 = src[3];

    for (int x = 0; x < width; x++)
        dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
}

void vresize_neon(const int** src, unsigned char* dst, const short* beta, int width)
{
    int32x4_t src_1, src_2, src_3, src_4, src_1_mul;
    int32x4_t d4_q;
    int32x4_t add_1;
    int32x4_t add_delta;
    int32x4_t shift_right_32x4;
    uint16x4_t shift_right_16x4;
    uint16x8_t shift_right_16x8;
    int32x4_t dt;
    uint8x8_t dt2;

    int bits = 22;
    // int32x4_t SHIFT = vdupq_n_s32(bits);
    int DELTA = (1 << (bits - 1));
    // b1_vq = vdupq_n_s32(beta[0]);
    // b2_vq = vdupq_n_s32(beta[1]);
    // b3_vq = vdupq_n_s32(beta[2]);
    // b4_vq = vdupq_n_s32(beta[3]);
    d4_q = vdupq_n_s32(DELTA);
    src_1_mul = vdupq_n_s32(0);

    int32x4_t lower  = vdupq_n_s32(0);
    int32x4_t higher = vdupq_n_s32(255);

    for (int x = 0; x < width; x+=4)
    {
        src_1 = vld1q_s32(src[0] + x);
        src_2 = vld1q_s32(src[1] + x);
        src_3 = vld1q_s32(src[2] + x);
        src_4 = vld1q_s32(src[3] + x);

        add_1 = vmlaq_n_s32(src_1_mul, src_1, beta[0]);
        add_1 = vmlaq_n_s32(add_1, src_2, beta[1]);
        add_1 = vmlaq_n_s32(add_1, src_3, beta[2]);
        add_1 = vmlaq_n_s32(add_1, src_4, beta[3]);

        add_delta = vaddq_s32(add_1, d4_q);

        shift_right_32x4 = vshrq_n_s32(add_delta, bits); // 32x4

        dt = vminq_s32(shift_right_32x4, higher);
        dt = vmaxq_s32(dt, lower);

        // shift_right_32x4 = vshrq_n_s32(add_delta, bits); // 32x4
        shift_right_16x4 = vqmovun_s32(dt); //16x4
        shift_right_16x8 = vcombine_u16(shift_right_16x4, shift_right_16x4); //16x8
        dt2 = vqmovn_u16(shift_right_16x8); // 8x8
        
        vst1_lane_u32((unsigned int*)(dst+x), vreinterpret_u32_u8(dt2), 0);
    }
}

static int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

void step(const unsigned char* _src, unsigned char* _dst, const int* xofs, const int* yofs, const short* _alpha, const short* _beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
{
    int dy, cn = channels;

    int bufstep = (int)((dwidth + 16 - 1) & -16);
    int* _buffer = (int*)malloc(bufstep * ksize * sizeof(int));
    if (_buffer == NULL)
    {
        printf("malloc fails\n");
    }
    const unsigned char* srows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int* rows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int prev_sy[MAX_ESIZE];

    for (int k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = _buffer + bufstep * k;
    }

#if MEASURE_SUB_COMPONENTS
    clock_t startTime, stopTime;
    double msecElapsed_hresize = 0, msecElapsed_vresize = 0;
#endif
       
    for (dy = start; dy < end; dy++)
    {
        int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (int k = 0; k < ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, iheight);
            for (k1 = MAX(k1, k); k1 < ksize; k1++)
            {
                if (k1 < MAX_ESIZE && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = MIN(k0, k); // remember the first row that needs to be computed
            srows[k] = _src + (sy * iwidth);
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
        {
#if MEASURE_SUB_COMPONENTS
            startTime = clock();
#endif

            hresize((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                iwidth, dwidth, cn, xmin, xmax);

#if MEASURE_SUB_COMPONENTS
            stopTime = clock();
            msecElapsed_hresize += (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
#endif
           
        }
#if MEASURE_SUB_COMPONENTS
        startTime = clock();
#endif
        vresize((const int**)rows, (_dst + dwidth * dy), _beta, dwidth);
#if MEASURE_SUB_COMPONENTS
        stopTime = clock();
        msecElapsed_vresize += (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
#endif
       
    }

#if MEASURE_SUB_COMPONENTS
    printf("H resize time: %f ms\n", msecElapsed_hresize);
    printf("V resize time: %f ms\n", msecElapsed_vresize);
#endif
    free(_buffer);
}

void step_neon(const unsigned char* _src, unsigned char* _dst, const int* xofs, const int* yofs, const short* _alpha, const short* _beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax)
{
    int dy, cn = channels;

    int bufstep = (int)((dwidth + 16 - 1) & -16);
    int* _buffer = (int*)malloc(bufstep * ksize * sizeof(int));
    if (_buffer == NULL)
    {
        printf("malloc fails\n");
    }
    const unsigned char* srows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int* rows[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int prev_sy[MAX_ESIZE];

    for (int k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = _buffer + bufstep * k;
    }

#if MEASURE_SUB_COMPONENTS
    clock_t startTime, stopTime;
    double msecElapsed_hresize = 0, msecElapsed_vresize = 0;
#endif

    for (dy = start; dy < end; dy++)
    {
        int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (int k = 0; k < ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, iheight);
            for (k1 = MAX(k1, k); k1 < ksize; k1++)
            {
                if (k1 < MAX_ESIZE && sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = MIN(k0, k); // remember the first row that needs to be computed
            srows[k] = _src + (sy * iwidth);
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
        {
#if MEASURE_SUB_COMPONENTS
            startTime = clock();
#endif

            hresize_neon((srows + k0), (rows + k0), ksize - k0, xofs, _alpha,
                iwidth, dwidth, cn, xmin, xmax);

#if MEASURE_SUB_COMPONENTS
            stopTime = clock();
            msecElapsed_hresize += (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
#endif
        }

#if MEASURE_SUB_COMPONENTS
        startTime = clock();
#endif

#if USE_C_VRESIZE
        vresize((const int**)rows, (_dst + dwidth * dy), _beta, dwidth);
#elif !USE_C_VRESIZE
        vresize_neon((const int**)rows, (_dst + dwidth * dy), _beta, dwidth);
#endif

#if MEASURE_SUB_COMPONENTS
        stopTime = clock();
        msecElapsed_vresize += (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
#endif
       
    }

#if MEASURE_SUB_COMPONENTS
    printf("H resize time neon: %f ms\n", msecElapsed_hresize);
    printf("V resize time neon: %f ms\n", msecElapsed_vresize);
#endif
    
    free(_buffer);
}


void resize(const unsigned char* _src, unsigned char* _dst, int iwidth, int iheight, int dwidth, int dheight, int neon)
{
    double  inv_scale_x = (double)dwidth / iwidth;
    double  inv_scale_y = (double)dheight / iheight;

    int  depth = 0, cn = 1;

    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;

    int iscale_x = (int)scale_x;
    int iscale_y = (int)scale_y;

    int k, sx, sy, dx, dy;

    int xmin = 0, xmax = dwidth, width = dwidth * cn;

    float fx, fy;
    int ksize = 4, ksize2;
    ksize2 = ksize / 2;

    unsigned char* _buffer = (unsigned char*)malloc((width + dheight) * (sizeof(int) + sizeof(float) * ksize));

    int* xofs = (int*)_buffer;
    int* yofs = xofs + width;
    float cbuf[4] = { 0 };

    for (dx = 0; dx < dwidth; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = (int)floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1)
        {
            xmin = dx + 1;
        }

        if (sx + ksize2 >= iwidth)
        {
            xmax = MIN(xmax, dx);
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;
    }

    for (dy = 0; dy < dheight; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = (int)floor(fy);
        fy -= sy;

        yofs[dy] = sy;
    }

    const short ibeta[] = {-192, 1216, 1216, -192};
    const short ialpha[] = {-192, 1216, 1216, -192};

    if(neon == 1)
        step_neon(_src, _dst, xofs, yofs, ialpha, ibeta, iwidth, iheight, dwidth, dheight, cn, ksize, 0, dheight, xmin, xmax);
    else
        step(_src, _dst, xofs, yofs, ialpha, ibeta, iwidth, iheight, dwidth, dheight, cn, ksize, 0, dheight, xmin, xmax);
}

int compare(const unsigned char* _x, const unsigned char* _x_simd, int iwidth, int iheight)
{
    int index = 0;
    int count = 0;
    for(size_t i = 0; i < iheight; i++)
    {
        for(size_t j = 0; j < iwidth; j++)
        {
            index = i*iwidth+j;
            if(_x[index] != _x_simd[index])
            {
                printf("mismatch element C: %u, ARM: %u, column: %ld, row: %ld, index(row*width+col): %d \n", _x[index], _x_simd[index], j, i, index);
                count++;
                goto label;
               
            }
        }
    }

    label:
    printf("total mismatches: %d\n", count);
    printf("total elements: %d\n",iheight*iwidth);

    return 1;
}

int main()
{
    int height = 7680;
    int width = 4320;

    // int height = 1920;
    // int width = 1080;

    int max = 255;
    int min = 0;

    int index = 0;

    unsigned char* x = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 2);
    unsigned char* y = (unsigned char*)malloc(sizeof(unsigned char) * (height / 2) * (width / 2));
    unsigned char* y_neon = (unsigned char*)malloc(sizeof(unsigned char) * (height / 2) * (width / 2));

    for(size_t xh = 0; xh < height; xh++)
    {
        for(size_t yw = 0; yw < width; yw++)
        {
            index = xh*width+yw;
            x[index] = (unsigned char)((rand() % (max + 1 - min)) + min);
        }
    }

#if MEASURE_TOTAL
    clock_t startTime, stopTime;
    double msecElapsed;
    startTime = clock();
#endif

    resize(x, y, width, height, width/2, height/2, 0);

#if MEASURE_TOTAL
    stopTime = clock();
    msecElapsed = (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    // printf("hresize: %f ms\n  ",  msecElapsed_hresize_g);
    printf("Overall resize time: %f ms\n-------------\n", msecElapsed);

    startTime = clock();
#endif

    resize(x, y_neon, width, height, width/2, height/2, 1);

#if MEASURE_TOTAL
    stopTime = clock();
    msecElapsed = (stopTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    printf("Overall resize time with neon: %f ms\n", msecElapsed);
#endif
    compare(y, y_neon, width/2, height/2);

    free(x);
    free(y);
    free(y_neon);
    return 0;
}