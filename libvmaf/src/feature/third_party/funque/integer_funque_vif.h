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

void funque_log_generate(uint32_t* log_18);

int integer_compute_vif_funque(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, double *score, double *score_num, double *score_den, int k, int stride, double sigma_nsq, int64_t shift_val, uint32_t* log_18
#if PROFILE_IND_MODULES
                                    , double *pad_time
#endif
                                    );