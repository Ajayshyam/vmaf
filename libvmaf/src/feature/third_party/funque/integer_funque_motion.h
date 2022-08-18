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

#include "integer_funque_filters.h"

double integer_funque_image_mad_c(const dwt2_dtype *img1, const dwt2_dtype *img2, int width, int height, int img1_stride, int img2_stride, float pending_div_factor);
int integer_compute_motion_funque(ModuleFunqueState m,const dwt2_dtype *ref, const dwt2_dtype *dis, int w, int h, int ref_stride, int dis_stride, float pending_div_factor, double *score);
