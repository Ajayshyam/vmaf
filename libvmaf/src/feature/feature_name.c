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

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static const char *template_list[] = {
    "VMAF_integer_feature_adm2_score",
    "VMAF_integer_feature_vif_scale0_score",
    "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score",
    "VMAF_integer_feature_vif_scale3_score",
};

static int in_template_list(char *name)
{
    const unsigned n = sizeof(template_list) / sizeof(template_list[0]);
    for(unsigned i = 0; i < n; i++) {
        if (!strcmp(name, template_list[i]))
            return 1;
    }
    return 0;
}

char *vmaf_feature_name(char *name, char *key, double val,
                        char *buf, size_t buf_sz)
{
    if (!key) return name;
    if (!in_template_list(name)) return name;

    memset(buf, 0, buf_sz);
    snprintf(buf, buf_sz - 1, "%s_%s_%.2f", name, key, val);
    return buf;
}
