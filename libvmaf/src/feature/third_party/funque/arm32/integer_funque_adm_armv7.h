#include "../integer_funque_filters.h"
#include "../integer_funque_adm.h"

void integer_dlm_decouple_armv7(i_dwt2buffers ref, i_dwt2buffers dist,
                               i_dwt2buffers i_dlm_rest, adm_i32_dtype *i_dlm_add,
                               int32_t *adm_div_lookup, float border_size, double *adm_score_den);