#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

void integer_adm_decouple_neon(i_dwt2buffers ref, i_dwt2buffers dist,
                               i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add,
                               int32_t *adm_div_lookup, float border_size, double *adm_score_den);

void integer_adm_integralimg_numscore_neon(i_dwt2buffers pyr_1, int32_t *x_pad, int k, 
                            int stride, int width, int height, adm_i32_dtype *interim_x, 
                            float border_size, double *adm_score_num);