#include "utils.h"
#include "common.h"

#define PROFILE         0
#define PRINT_VAL       0

#define WIDTH_MIN       4
#define WIDTH_MAX       32
#define HEIGHT_MIN      4
#define HEIGHT_MAX      32
#define VARIANT        "MOTION SCORE"

#define FRAMES         1
#define FRAME_WIDTH    1920
#define FRAME_HEIGHT   1080


#if 0 // not used
int compare(i_dwt2buffers *dst, i_dwt2buffers *dst_simd, int dst_stride, int width, int height)
{
    int x, y;
    for(y = 0; y < height; y++)
    {
        for(x = 0; x < width; x++)
        {
            if(*(dst->bands[0] + x) != *(dst_simd->bands[0] + x) ||
               *(dst->bands[1] + x) != *(dst_simd->bands[1] + x) ||
               *(dst->bands[2] + x) != *(dst_simd->bands[2] + x) ||
               *(dst->bands[3] + x) != *(dst_simd->bands[3] + x)
            )
            {
                printf("\nbands[0]\tMismatch elements: c = %d ! ARM = %d, column %d, row %d", *(dst->bands[0] + x), *(dst_simd->bands[0] + x), x, y);
                printf("\nbands[1]\tMismatch elements: c = %d ! ARM = %d, column %d, row %d", *(dst->bands[1] + x), *(dst_simd->bands[1] + x), x, y);
                printf("\nbands[2]\tMismatch elements: c = %d ! ARM = %d, column %d, row %d", *(dst->bands[2] + x), *(dst_simd->bands[2] + x), x, y);
                printf("\nbands[3]\tMismatch elements: c = %d ! ARM = %d, column %d, row %d", *(dst->bands[3] + x), *(dst_simd->bands[3] + x), x, y);
            }
            else
            {
#if PRINT_VAL
                printf("\t%d = %d", *(dst->bands[0] + x), *(dst_simd->bands[0] + x));
                printf("\t%d = %d", *(dst->bands[1] + x), *(dst_simd->bands[1] + x));
                printf("\t%d = %d", *(dst->bands[2] + x), *(dst_simd->bands[2] + x));
                printf("\t%d = %d", *(dst->bands[3] + x), *(dst_simd->bands[3] + x));
#endif
            }
#if PRINT_VAL
        printf("\n");
#endif
        }
        dst->bands[0] += dst_stride;
        dst_simd->bands[0] += dst_stride;
        dst->bands[1] += dst_stride;
        dst_simd->bands[1] += dst_stride;
        dst->bands[2] += dst_stride;
        dst_simd->bands[2] += dst_stride;
        dst->bands[3] += dst_stride;
        dst_simd->bands[3] += dst_stride;
#if PRINT_VAL
        printf("\n\n");
#endif
    }
    return 1;
}
#endif

int main()
{
    int i, j, k, l;
    struct timeval s_tv;
    const int multiplier = 1;
    double     cTime, simdTime;

    int32_t stride0 = 128;
    int32_t stride1 = 128;
    float   div_factor = 20.5;
    int blockWidth, blockHeight;
    double C_RES = 0;
    double SIMD_RES = 0;

    int16_t *src0 = (int16_t*) calloc(FRAME_WIDTH*FRAME_HEIGHT*FRAMES, sizeof(int16_t));
    int16_t *src1 = (int16_t*) calloc(FRAME_WIDTH*FRAME_HEIGHT*FRAMES, sizeof(int16_t));

    for(int l = 0; l < FRAMES; l++)
    {
        for (i = 0; i < FRAME_HEIGHT; i++)        //Case 4 : random() % 32767 // Random case
        {
            for (j = 0; j < FRAME_WIDTH; j++)
            {
                *(src0 + i * FRAME_WIDTH + j) = rand() % 32767 - rand() % 32765;
                *(src1 + i * FRAME_WIDTH + j) = rand() % 32767 - rand() % 32764;
            }
        }
    }

    for (blockHeight = HEIGHT_MIN; blockHeight <= HEIGHT_MAX; blockHeight *= 2)
    {
        for (blockWidth = WIDTH_MIN; blockWidth <= WIDTH_MAX; blockWidth *= 2)
        {
            cTime = 0;
            simdTime = 0;

            /// C func call
#if PROFILE
            PROFILE_START
            for(int l = 0; l < FRAMES; l++){
                for(int i = 0; i < (FRAME_HEIGHT); i += blockHeight){
                    for(int j = 0; j < (FRAME_WIDTH); j += blockWidth){
#endif
                        C_RES = integer_funque_image_mad_c(src0 + i * blockWidth + j, src1 + i * blockWidth + j, blockWidth, blockHeight, stride0, stride1, div_factor);
#if PROFILE
                    }
                }
            }
            cTime = PROFILE_END
#endif

            /// SIMD func call
#if PROFILE
            PROFILE_START
            for(int l = 0; l < FRAMES; l++){
                for(int i = 0; i < (FRAME_HEIGHT); i += blockHeight){
                    for(int j = 0; j < (FRAME_WIDTH); j += blockWidth){
#endif
                        SIMD_RES = integer_funque_image_mad_neon(src0 + i * blockWidth + j, src1 + i * blockWidth + j, blockWidth, blockHeight, stride0, stride1, div_factor);
#if PROFILE
                    }
                }
            }
            simdTime = PROFILE_END
#endif

            if (C_RES == SIMD_RES)
                printf("\nC and SIMD results match\n");
            else
                printf("\nC != SIMD : results failed\n");

#if PROFILE
            printf("%dx%d\t%f\t%f\t%fx", blockWidth, blockHeight, cTime, simdTime, (cTime/simdTime));
#endif
            printf("\n");
        }
    }
return 0;
}
