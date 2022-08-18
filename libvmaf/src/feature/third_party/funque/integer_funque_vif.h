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
#ifndef FEATURE_INTFUNQUE_VIF_H_
#define FEATURE_INTFUNQUE_VIF_H_

#define VIF_COMPUTE_METRIC_R_SHIFT 6

void funque_log_generate(uint32_t* log_18);

void integer_reflect_pad(const dwt2_dtype* src, size_t width, size_t height, int reflect, dwt2_dtype* dest);

int integer_compute_vif_funque_c(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, double *score, double *score_num, double *score_den, int k, int stride, double sigma_nsq, int64_t shift_val, uint32_t* log_18);

FORCE_INLINE inline uint32_t get_best_18bitsfixed_opt_64(uint64_t temp, int *x)
{
    int k = __builtin_clzll(temp);

    if (k > 46) 
    {
        k -= 46;
        temp = temp << k;
        *x = k;

    }
    else if (k < 45) 
    {
        k = 46 - k;
        temp = temp >> k;
        *x = -k;
    }
    else
    {
        *x = 0;
        if (temp >> 18)
        {
            temp = temp >> 1;
            *x = -1;
        }
    }

    return (uint32_t)temp;
}

/**
 * This function accumulates the numerator and denominator values & their powers 
 */
static inline vif_stats_calc(int32_t int_1_x, int32_t int_1_y, 
                             int64_t int_2_x, int64_t int_2_y, int64_t int_x_y, 
                             int16_t knorm_fact, int16_t knorm_shift, int k_norm, 
                             int16_t exp_t, int32_t sigma_nsq_t, uint32_t *log_18,
                             int64_t *score_num_t, int64_t *num_power,
                             int64_t *score_den_t, int64_t *den_power)
{
    int32_t mx = int_1_x;
    int32_t my = int_1_y;
    int32_t var_x = (int_2_x - (((int64_t) mx * mx * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;
    int32_t var_y_t = (int_2_y - (((int64_t) my * my * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;
    int32_t cov_xy_t = (int_x_y - (((int64_t) mx * my * knorm_fact) >> knorm_shift)) >> VIF_COMPUTE_METRIC_R_SHIFT;

    if (var_x < exp_t)
    {
        var_x = 0;
        cov_xy_t = 0;
    }
    
    if (var_y_t < exp_t)
    {
        var_y_t = 0;
        cov_xy_t = 0;
    }
    int32_t g_t_num = cov_xy_t;
    int32_t g_den = var_x + exp_t*k_norm;

    int32_t sv_sq_t = (var_y_t - ((int64_t)g_t_num * cov_xy_t)/g_den);


    if((g_t_num < 0 && g_den > 0) || (g_den < 0 && g_t_num > 0))
    {
        sv_sq_t = var_x;
        g_t_num = 0;
    }

    if (sv_sq_t < (exp_t * k_norm))
        sv_sq_t = exp_t * k_norm;

    int64_t p1 = ((int64_t)g_t_num * g_t_num)/g_den;
    int32_t p2 = var_x;
    int64_t n1 = p1 * p2;
    int64_t n2 = g_den * ((int64_t) sv_sq_t + sigma_nsq_t);
    int64_t num_t = n2 + n1;
    int64_t num_den_t = n2;
    int x1, x2;

    uint32_t log_in_num_1 = get_best_18bitsfixed_opt_64((uint64_t)num_t, &x1);
    uint32_t log_in_num_2 = get_best_18bitsfixed_opt_64((uint64_t)num_den_t, &x2);
    int32_t temp_numerator = (int64_t)log_18[log_in_num_1] - (int64_t)log_18[log_in_num_2];
    int32_t temp_power_num = -x1 + x2; 
    *score_num_t += temp_numerator;
    *num_power += temp_power_num;

    uint32_t d1 = ((uint32_t)sigma_nsq_t + (uint32_t)(var_x));
    uint32_t d2 = (sigma_nsq_t);
    int y1, y2;

    uint32_t log_in_den_1 = get_best_18bitsfixed_opt_64((uint64_t)d1, &y1);
    uint32_t log_in_den_2 = get_best_18bitsfixed_opt_64((uint64_t)d2, &y2);
    int32_t temp_denominator =  (int64_t)log_18[log_in_den_1] - (int64_t)log_18[log_in_den_2];
    int32_t temp_power_den = -y1 + y2;
    *score_den_t += temp_denominator;
    *den_power += temp_power_den;
}

//This function does summation of horizontal intermediate_vertical_sums & then 
//numerator denominator score calculations are done
static inline vif_horz_integralsum(int kw, int width_p1, 
                                   int16_t knorm_fact, int16_t knorm_shift, int k_norm, 
                                   int16_t exp_t, int32_t sigma_nsq_t, uint32_t *log_18,
                                   int32_t *interim_1_x, int32_t *interim_1_y,
                                   int64_t *interim_2_x, int64_t *interim_2_y, int64_t *interim_x_y,
                                   int64_t *score_num_t, int64_t *num_power,
                                   int64_t *score_den_t, int64_t *den_power)
{
    int32_t int_1_x, int_1_y;
    int64_t int_2_x, int_2_y, int_x_y;

    //1st column vals are 0, hence intialising to 0
    int_1_x = 0;
    int_1_y = 0;
    int_2_x = 0;
    int_2_y = 0;
    int_x_y = 0;
    /**
     * The horizontal accumulation similar to vertical accumulation
     * metric_sum = prev_col_metric_sum + interim_metric_vertical_sum
     * The previous kw col interim metric sum is not subtracted since it is not available here
     */
    for (size_t j=1; j<kw+1; j++)
    {
        int j_minus1 = j-1;
        int_2_x = interim_2_x[j] + int_2_x;
        int_1_x = interim_1_x[j] + int_1_x;
        
        int_2_y = interim_2_y[j] + int_2_y;
        int_1_y = interim_1_y[j] + int_1_y;
        
        int_x_y = interim_x_y[j] + int_x_y;
    }
    /**
     * The score needs to be calculated for kw column as well, 
     * whose interim result calc is different from rest of the columns, 
     * hence calling vif_stats_calc for kw column separately 
     */
    vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                    knorm_fact, knorm_shift, k_norm, 
                    exp_t, sigma_nsq_t, log_18,
                    score_num_t, num_power, score_den_t, den_power);

    //Similar to prev loop, but previous kw col interim metric sum is subtracted
    for (size_t j=kw+1; j<width_p1; j++)
    {
        int j_minus1 = j-1;
        int_2_x = interim_2_x[j] + int_2_x - interim_2_x[j - kw];
        int_1_x = interim_1_x[j] + int_1_x - interim_1_x[j - kw];
        
        int_2_y = interim_2_y[j] + int_2_y - interim_2_y[j - kw];
        int_1_y = interim_1_y[j] + int_1_y - interim_1_y[j - kw];
        
        int_x_y = interim_x_y[j] + int_x_y - interim_x_y[j - kw];

        vif_stats_calc(int_1_x, int_1_y, int_2_x, int_2_y, int_x_y,
                        knorm_fact, knorm_shift, k_norm, 
                        exp_t, sigma_nsq_t, log_18,
                        score_num_t, num_power, score_den_t, den_power);
    }

}

#endif /* _FEATURE_INTFUNQUE_VIF_H_ */