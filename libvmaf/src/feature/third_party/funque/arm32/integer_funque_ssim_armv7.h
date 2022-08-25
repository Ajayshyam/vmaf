#include "../integer_funque_filters.h"

int integer_compute_ssim_funque_armv7(i_dwt2buffers *ref, i_dwt2buffers *dist, double *score, int max_val, float K1, float K2, int pending_div, int32_t *div_lookup);