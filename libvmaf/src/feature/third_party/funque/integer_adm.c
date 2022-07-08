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

#include <math.h>
#include <stddef.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mem.h"
#include "integer_adm_options.h"
#include "adm_tools.h"
#include "integer_filters.h"
#include "integer_adm.h"
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef __SSE2__
#ifdef ADM_OPT_RECIP_DIVISION

#include <emmintrin.h>

static funque_dtype rcp_s(funque_dtype x)
{
    funque_dtype xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

static inline funque_dtype clip(funque_dtype value, funque_dtype low, funque_dtype high)
{
  return value < low ? low : (value > high ? high : value);
}

#define DIVS(n, d) ((n) * rcp_s(d))
#endif //ADM_OPT_RECIP_DIVISION
#else
#define DIVS(n, d) ((n) / (d))
#endif // __SSE2__


void integer_reflect_pad_adm(const dwt2_dtype* src, size_t width, size_t height, int reflect, dwt2_dtype* dest)
{
    size_t out_width = width + 2 * reflect;
    size_t out_height = height + 2 * reflect;

    for (size_t i = reflect; i != (out_height - reflect); i++) {

        for (int j = 0; j != reflect; j++)
        {
            dest[i * out_width + (reflect - 1 - j)] = src[(i - reflect) * width + j + 1];
        }

        memcpy(&dest[i * out_width + reflect], &src[(i - reflect) * width], sizeof(dwt2_dtype) * width);

        for (int j = 0; j != reflect; j++)
            dest[i * out_width + out_width - reflect + j] = dest[i * out_width + out_width - reflect - 2 - j];
    }

    for (int i = 0; i != reflect; i++) {
        memcpy(&dest[(reflect - 1) * out_width - i * out_width], &dest[reflect * out_width + (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
        memcpy(&dest[(out_height - reflect) * out_width + i * out_width], &dest[(out_height - reflect - 1) * out_width - (i + 1) * out_width], sizeof(dwt2_dtype) * out_width);
    }
}

//Qn input, Qn output (with overflow, hence stored in Q64)
void integer_integral_image_adm(const dwt2_dtype* src, size_t width, size_t height, int64_t* sum, int kw, int kh)
{
    int32_t *temp_sum = (int64_t*)calloc((width + 1) * (height + 1), sizeof(int32_t));

    for (size_t i=1; i<(height+1); i++)
    {
        for(size_t j=1; j<kw+1; j++)
        {
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + src[(i - 1) * width + (j - 1)];
        }
        for(size_t j=kw+1; j<(width+1); j++)
        {
            temp_sum[i*(width+1)+j] = temp_sum[i*(width+1)+j-1] + src[(i - 1) * width + (j - 1)] - src[(i-1)*width+j-kw-1];
        }
    }
    for (size_t j=1; j<(width+1); j++)
    {
        for (size_t i=1; i<kh+1; i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j];
        }
        for (size_t i=kh+1; i<(height+1); i++)
        {
            sum[i * (width + 1) + j] = temp_sum[i*(width+1)+j] + sum[(i-1) * (width + 1) + j] - temp_sum[(i-kh) * (width + 1) + j];
        }
    }
    free(temp_sum);
}

void integer_integral_image_adm_sums(dwt2_dtype *x, int k, int stride, int64_t *mx, int64_t *masking_threshold_int, int width, int height)
{
  dwt2_dtype *x_pad;
  int64_t *int_x;
  int i, j, index;

  int x_reflect = (int)((k - stride) / 2);

  x_pad = (dwt2_dtype *)malloc(sizeof(dwt2_dtype) * (width + (2 * x_reflect)) * (height + (2 * x_reflect)));

  integer_reflect_pad_adm(x, width, height, x_reflect, x_pad);

  size_t r_width = width + (2 * x_reflect);
  size_t r_height = height + (2 * x_reflect);

  int_x = (int64_t *)calloc((r_width + 1) * (r_height + 1), sizeof(int64_t));

  integer_integral_image_adm(x_pad, r_width, r_height, int_x, k, k);

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
	  index = i * width + j;
      // mx[index] = (int_x[i * (width + 3) + j] - int_x[i * (width + 3) + j + k] - int_x[(i + k) * (width + 3) + j] + int_x[(i + k) * (width + 3) + j + k]);
      mx[index] = int_x[(i + k) * (width + 3) + j + k];
      masking_threshold_int[index] =  (int64_t)x[index] + mx[index];
    }
  }
  
  free(x_pad);
  free(int_x);
}

void integer_dlm_contrast_mask_one_way(i_dwt2buffers pyr_1, i_dwt2buffers pyr_2, dwt2buffers masked_pyr, size_t width, size_t height)
{
  int i, k, j, index;
  int64_t val = 0, pyr_abs;
  dwt2_dtype *masking_signal;
  int64_t *masking_threshold, *masking_threshold_int;
  int64_t *integral_sum;

  masking_signal = (dwt2_dtype *)calloc(width * height, sizeof(dwt2_dtype));
  masking_threshold_int = (int64_t *)calloc(width * height, sizeof(int64_t));
  masking_threshold = (int64_t *)calloc(width * height, sizeof(int64_t));
  integral_sum = (int64_t *)calloc(width * height, sizeof(int64_t));

  for (k = 1; k < 4; k++)
  {
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
        masking_signal[index] = abs(pyr_2.bands[k][index]);
      }
    }
    integer_integral_image_adm_sums(masking_signal, 3, 1, integral_sum, masking_threshold_int, width, height);
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
		masking_threshold[index] += masking_threshold_int[index];
      }
    }
  }

  for (k = 1; k < 4; k++)
  {
    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
        index = i * width + j;
		//compensation for the division by 30 of masking_threshold
		pyr_abs = abs((int64_t)pyr_1.bands[k][index]) * 30;
        val = pyr_abs - masking_threshold[index];
        masked_pyr.bands[k][index] = (int64_t)clip(val, 0.0, val);
      }
    }
  }
  free(masking_signal);
  free(masking_threshold);
  free(masking_threshold_int);
  free(integral_sum);
}

void integer_dlm_decouple(i_dwt2buffers ref, i_dwt2buffers dist, i_dwt2buffers i_dlm_rest, i_dwt2buffers i_dlm_add)
{
#ifdef ADM_OPT_AVOID_ATAN
  const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
  float eps = 1e-30;
  size_t width = ref.width;
  size_t height = ref.height;
  int i, j, k, index;

  float val;
  int64_t tmp_val;
  int angle_flag;

#ifdef ADM_OPT_AVOID_ATAN
  int64_t *ot_dp = (int64_t *)calloc(width * height, sizeof(int64_t));
  int64_t *o_mag_sq = (int64_t *)calloc(width * height, sizeof(int64_t));
  int64_t *t_mag_sq = (int64_t *)calloc(width * height, sizeof(int64_t));
#else
  float *psi_ref = (float *)calloc(width * height, sizeof(float));
  float *psi_dist = (float *)calloc(width * height, sizeof(float));
  float *psi_diff = (float *)calloc(width * height, sizeof(float));
#endif

  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      index = i * width + j;
#ifdef ADM_OPT_AVOID_ATAN
      ot_dp[index] = ((int64_t)ref.bands[1][index] * dist.bands[1][index]) + ((int64_t)ref.bands[2][index] * dist.bands[2][index]);
      o_mag_sq[index] = ((int64_t)ref.bands[1][index] * ref.bands[1][index]) + ((int64_t)ref.bands[2][index] * ref.bands[2][index]);
      t_mag_sq[index] = ((int64_t)dist.bands[1][index] * dist.bands[1][index]) + ((int64_t)dist.bands[2][index] * dist.bands[2][index]);

      /** angle_flag is calculated in floating-point by converting fixed-point variables back to floating-point  */
      angle_flag = (((float)ot_dp[index] / 4096.0) >= 0.0f) &&
                   (((float)ot_dp[index] / 4096.0) * ((float)ot_dp[index] / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq[index] / 4096.0) * ((float)t_mag_sq[index] / 4096.0));
#else
      psi_ref[index] = atanf(ref.bands[2][index] / (ref.bands[1][index] + eps)) + M_PI * ((ref.bands[1][index] < 0));
      psi_dist[index] = atanf(dist.bands[2][index] / (dist.bands[1][index] + eps)) + M_PI * ((dist.bands[1][index] < 0));
      psi_diff[index] = 180 * fabsf(psi_ref[index] - psi_dist[index]) / M_PI;
      angle_flag = psi_diff[index] < 1;

#endif
      for (k = 1; k < 4; k++)
      {
        /**
         * Division dist/ref is carried using lookup table and converted to multiplication
         */
        int64_t tmp_k = (ref.bands[k][index] == 0) ? 32768 : (((int64_t)div_lookup[ref.bands[k][index] + 32768] * dist.bands[k][index]) + 16384) >> 15;
        int64_t kh = tmp_k < 0 ? 0 : (tmp_k > 32768 ? 32768 : tmp_k);
        /**
         * kh is in Q15 type and ref.bands[k][index] is in Q16 type hence shifted by
         * 15 to make result Q16
         */
        tmp_val = ((kh * ref.bands[k][index]) + 16384) >> 15;

        i_dlm_rest.bands[k][index] = angle_flag ? dist.bands[k][index] : tmp_val;
        i_dlm_add.bands[k][index] = dist.bands[k][index] - i_dlm_rest.bands[k][index];
      }
    }
  }

#ifdef ADM_OPT_AVOID_ATAN
  free(ot_dp);
  free(o_mag_sq);
  free(t_mag_sq);
#else
  free(psi_ref);
  free(psi_dist);
  free(psi_diff);
#endif
}

int compute_integer_adm_funque(i_dwt2buffers i_ref, i_dwt2buffers i_dist, double *adm_score, double *adm_score_num, double *adm_score_den, size_t width, size_t height, funque_dtype border_size)
{
  // TODO: assert len(pyr_ref) == len(pyr_dist),'Pyramids must be of equal height.'
  div_lookup_generator();
  int n_levels = 1;
  int i, j, k, index;
  int64_t num_sum = 0, den_sum = 0, ref_abs; 
  double num_band = 0, den_band = 0;
  dwt2buffers dlm_rest, dlm_add, pyr_rest, ref;
  i_dwt2buffers i_dlm_rest, i_dlm_add;
  i_dlm_rest.bands[0] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_rest.bands[1] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_rest.bands[2] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_rest.bands[3] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_add.bands[0] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_add.bands[1] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_add.bands[2] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  i_dlm_add.bands[3] = (int32_t *)malloc(sizeof(int32_t) * height * width);
  dlm_rest.bands[0] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[1] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[2] = (float *)malloc(sizeof(float) * height * width);
  dlm_rest.bands[3] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[0] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[1] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[2] = (float *)malloc(sizeof(float) * height * width);
  dlm_add.bands[3] = (float *)malloc(sizeof(float) * height * width);
  pyr_rest.bands[0] = (int64_t *)malloc(sizeof(int64_t) * height * width);
  pyr_rest.bands[1] = (int64_t *)malloc(sizeof(int64_t) * height * width);
  pyr_rest.bands[2] = (int64_t *)malloc(sizeof(int64_t) * height * width);
  pyr_rest.bands[3] = (int64_t *)malloc(sizeof(int64_t) * height * width);
  ref.bands[0] = (float *)malloc(sizeof(float) * height * width);
  ref.bands[1] = (float *)malloc(sizeof(float) * height * width);
  ref.bands[2] = (float *)malloc(sizeof(float) * height * width);
  ref.bands[3] = (float *)malloc(sizeof(float) * height * width);

  integer_dlm_decouple(i_ref, i_dist, i_dlm_rest, i_dlm_add);

  integer_dlm_contrast_mask_one_way(i_dlm_rest, i_dlm_add, pyr_rest, width, height);

  int border_h = (border_size * height);
  int border_w = (border_size * width);
  int loop_h = height - border_h;
  int loop_w = width - border_w;
  double row_num[loop_h], row_den[loop_h], accum_num = 0, accum_den = 0;

  for (k = 1; k < 4; k++)
  {
    for (i = border_h; i < loop_h; i++)
    {
      for (j = border_w; j < loop_w; j++)
      {
        index = i * width + j;
        num_sum += pyr_rest.bands[k][index] * pyr_rest.bands[k][index] * pyr_rest.bands[k][index];
		//compensation for the division by thirty in the numerator
		ref_abs = abs((int64_t)i_ref.bands[k][index]) * 30;
        den_sum += ref_abs * ref_abs * ref_abs;
      }
      row_num[i] = num_sum / 16320;
      row_den[i] = den_sum / 16320;
      accum_num += row_num[i];
      accum_den += row_den[i];
      num_sum = 0;
      den_sum = 0;
    }

    den_band += powf((double)(accum_den), 1.0 / 3.0);
    num_band += powf((double)(accum_num), 1.0 / 3.0);
    accum_num = 0;
    accum_den = 0;
  }

  *adm_score_num = num_band + 1e-4;
  *adm_score_den = den_band + 1e-4;
  *adm_score = (*adm_score_num) / (*adm_score_den);

  for (int i = 0; i < 4; i++)
  {
    free(dlm_rest.bands[i]);
    free(dlm_add.bands[i]);
    free(i_dlm_rest.bands[i]);
    free(i_dlm_add.bands[i]);
    free(pyr_rest.bands[i]);
    free(ref.bands[i]);
  }

  int ret = 0;
  return ret;
}
