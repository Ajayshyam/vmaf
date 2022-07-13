#include "utils.h"
#include "common.h"

#define PROFILE     0
#define PRINT_VAL   1

#define WIDTH_MIN   4
#define WIDTH_MAX   32
#define HEIGHT_MIN  4
#define HEIGHT_MAX  32
#define VARIANT "DWT2"

#define MAX_HEIGTH 256
#define MAX_WIDTH 256

#if PROFILE
#define ITERATE 100000 // 10^5
#else
#define ITERATE 1
#endif


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
            printf("\n");
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


int main()
{
    int i, j, k = 0;
    struct timeval s_tv;
    const int multiplier = 1;
    i_dwt2buffers *C;
    i_dwt2buffers *SIMD;
    int32_t dst_stride = 128;
    int width, height;


    C    = (i_dwt2buffers*) malloc(sizeof(i_dwt2buffers));
    SIMD = (i_dwt2buffers*) malloc(sizeof(i_dwt2buffers));
    int16_t *src     = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    C->bands[0]      = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    C->bands[1]      = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    C->bands[2]      = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    C->bands[3]      = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    SIMD->bands[0]   = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    SIMD->bands[1]   = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    SIMD->bands[2]   = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));
    SIMD->bands[3]   = (int16_t*) calloc(MAX_WIDTH*MAX_HEIGTH, sizeof(int16_t));

    for (width = WIDTH_MIN; width <= WIDTH_MAX; width *= 2)
    {
        for (height = HEIGHT_MIN; height <= HEIGHT_MAX; height *= 2)
        {
                for (i = 0; i < MAX_HEIGTH; i++)        //Case 4 : random() % 32767 // Random case
                {
                    for (j = 0; j < MAX_WIDTH; j++)
                    {
                        *(src+i*MAX_WIDTH+j) = rand() % 32767 - rand() % 32767;
                    }
                }

                double    time1 = 0;
                double    time2 = 0;

                integer_funque_dwt2(src + 128, C, dst_stride, width, height);
                PROFILE_START
                for (int cnt = 0; cnt < ITERATE; cnt++)
                {
                    integer_funque_dwt2(src + 128, C, dst_stride, width, height);
                }
                time1=PROFILE_END

                integer_funque_dwt2_neon(src + 128, SIMD, dst_stride, width, height);
                PROFILE_START
                for (int cnt = 0; cnt < ITERATE; cnt++)
                {
                    integer_funque_dwt2_neon(src + 128, SIMD, dst_stride, width, height);
                }
                time2=PROFILE_END

                compare(C, SIMD, dst_stride, width, height);

#if PROFILE
                printf("%dx%d\t%f\t%f\t%fx", width, height, time1, time2, (time1/time2));
#endif
                printf("\n");
        }
    }
return 0;
}
