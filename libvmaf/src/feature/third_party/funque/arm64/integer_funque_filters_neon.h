#include "../integer_funque_filters.h"

void integer_funque_dwt2_neon(spat_fil_output_dtype *src, i_dwt2buffers *dwt2_dst, ptrdiff_t dst_stride, int width, int height);
void integer_spatial_filter_neon(void *src, spat_fil_output_dtype *dst, int width, int height, int bitdepth);