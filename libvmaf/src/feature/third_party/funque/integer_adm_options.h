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

#pragma once

#ifndef ADM_OPTIONS_H_
#define ADM_OPTIONS_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

/* Contrast sensitivity function */
enum ADM_CSF_MODE {
    ADM_CSF_MODE_WATSON97 = 0,
};

/* Default contrast sensitivity function */
#define DEFAULT_ADM_CSF_MODE (ADM_CSF_MODE_WATSON97)

#define DEFAULT_VIF_LEVELS 2

static int32_t div_lookup[65537];
static const int32_t div_Q_factor = 1073741824; // 2^30

static inline void div_lookup_generator()
{
    for (int i = 1; i <= 32768; ++i)
    {
        int32_t recip = (int32_t)(div_Q_factor / i);
        div_lookup[32768 + i] = recip;
        div_lookup[32768 - i] = 0 - recip;
    }
}

#define SHIFT_ADM_DECOUPLE_FINAL 16

#endif /* ADM_OPTIONS_H_ */
