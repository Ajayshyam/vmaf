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

// #include "integer_funque_filters.h"

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE 2048
#define MAX_ESIZE 16

// enabled by default for funque since resize factor is always 0.5, disabled otherwise
#define OPTIMISED_COEFF 1
#define USE_C_VRESIZE 1

#define CLIP3(X, MIN, MAX) ((X < MIN) ? MIN : (X > MAX) ? MAX \
                                                        : X)
#define MAX(LEFT, RIGHT) (LEFT > RIGHT ? LEFT : RIGHT)
#define MIN(LEFT, RIGHT) (LEFT < RIGHT ? LEFT : RIGHT)

typedef struct ResizerState
{
#if OPTIMISED_COEFF
    void (*resizer_step)(const unsigned char *_src, unsigned char *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax);
#else
    void (*resizer_step)(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax);
#endif
}ResizerState;

unsigned char castOp(int val);
void vresize(const int **src, unsigned char *dst, const short *beta, int width);
#if OPTIMISED_COEFF
void step(const unsigned char *_src, unsigned char *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax);
#else
void step(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax);
#endif
void resize(ResizerState m, const unsigned char* _src, unsigned char* _dst, int iwidth, int iheight, int dwidth, int dheight);
void hbd_resize(const unsigned short *_src, unsigned short *_dst, int iwidth, int iheight, int dwidth, int dheight, int bitdepth);
