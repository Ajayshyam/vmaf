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

#include <errno.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

// #include "config.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "integer_funque_filters.h"
#include "integer_funque_vif.h"
#include "funque_vif_options.h"
#include "integer_funque_adm.h"
#include "funque_adm_options.h"
#include "integer_funque_motion.h"
#include "integer_funque_ssim.h"
#include "resizer.h"

#include "funque_profiler.h"
typedef struct IntFunqueState
{
    size_t float_stride;
    dwt2_dtype *i_prev_ref_dwt2;
    bool debug;

    VmafPicture res_ref_pic;
    VmafPicture res_dist_pic;

    size_t i_dwt2_stride;
    spat_fil_output_dtype *spat_filter;
    i_dwt2buffers i_ref_dwt2out;
    i_dwt2buffers i_dist_dwt2out;

    // funque configurable parameters
    bool enable_resize;
    int vif_levels;

    // VIF extra variables
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    uint32_t log_18[262144];

    // ADM extra variables
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode;
	int32_t adm_div_lookup[65537];

    // motion score extra variables
    unsigned index;
    double score;
    bool motion_force_zero;

    // SSIM extra variables
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    double max_db;

    VmafDictionary *feature_name_dict;
} IntFunqueState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(IntFunqueState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_resize",
        .alias = "rsz",
        .help = "Enable resize for funque",
        .offset = offsetof(IntFunqueState, enable_resize),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "vif_levels",
        .alias = "vifl",
        .help = "Number of levels in VIF",
        .offset = offsetof(IntFunqueState, vif_levels),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_VIF_LEVELS,
        // Update this when the support is added
        .min = MIN_VIF_LEVELS,
        .max = MAX_VIF_LEVELS,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(IntFunqueState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(IntFunqueState, vif_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(IntFunqueState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(IntFunqueState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdf",
        .help = "reference display height in pixels",
        .offset = offsetof(IntFunqueState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(IntFunqueState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(IntFunqueState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },

    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(IntFunqueState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(IntFunqueState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(IntFunqueState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },

    {0}};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    IntFunqueState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        goto fail;

    if (s->enable_resize)
    {
        w = (w + 1) >> 1;
        h = (h + 1) >> 1;
    }

    s->float_stride = ALIGN_CEIL(w * sizeof(float));

    if (s->enable_resize)
    {
        s->res_ref_pic.data[0] = aligned_malloc(s->float_stride * h, 32);
        if (!s->res_ref_pic.data[0])
            goto fail;
        s->res_dist_pic.data[0] = aligned_malloc(s->float_stride * h, 32);
        if (!s->res_dist_pic.data[0])
            goto fail;
    }

    s->spat_filter = aligned_malloc(ALIGN_CEIL(w * sizeof(spat_fil_output_dtype)) * h, 32);
    if (!s->spat_filter)
        goto fail;

    // dwt output dimensions
    s->i_ref_dwt2out.width = (int)(w + 1) / 2;
    s->i_ref_dwt2out.height = (int)(h + 1) / 2;
    s->i_dist_dwt2out.width = (int)(w + 1) / 2;
    s->i_dist_dwt2out.height = (int)(h + 1) / 2;

    s->i_dwt2_stride = (s->float_stride + 3) / 4;
    s->i_prev_ref_dwt2 = aligned_malloc(s->i_dwt2_stride * s->i_ref_dwt2out.height, 32);
    if (!s->i_prev_ref_dwt2)
        goto fail;
    // Memory allocation for dwt output bands
    for (unsigned i = 0; i < 4; i++)
    {
        s->i_ref_dwt2out.bands[i] = aligned_malloc(s->i_dwt2_stride * s->i_ref_dwt2out.height, 32);
        if (!s->i_ref_dwt2out.bands[i])
            goto fail;

        s->i_dist_dwt2out.bands[i] = aligned_malloc(s->i_dwt2_stride * s->i_dist_dwt2out.height, 32);
        if (!s->i_dist_dwt2out.bands[i])
            goto fail;
    }

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db)
    {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    }
    else
    {
        s->max_db = INFINITY;
    }
    funque_log_generate(s->log_18);
	div_lookup_generator(s->adm_div_lookup);

    return 0;

fail:
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
    if (s->spat_filter)
        aligned_free(s->spat_filter);
    if (s->i_prev_ref_dwt2)
        aligned_free(s->i_prev_ref_dwt2);

    for (unsigned i = 0; i < 4; i++)
    {
        if (s->i_ref_dwt2out.bands[i])
            aligned_free(s->i_ref_dwt2out.bands[i]);
        if (s->i_dist_dwt2out.bands[i])
            aligned_free(s->i_dist_dwt2out.bands[i]);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
#if PROFILE_FUNQUE
    struct timeval start_time, end_time;
    struct timeval resize_start_time, resize_end_time;
    struct timeval spat_start_time, spat_end_time; 
    struct timeval prefil_start_time, prefil_end_time;
    struct timeval motion_start_time, motion_end_time;
    struct timeval vif_start_time, vif_end_time;
    struct timeval adm_start_time, adm_end_time;
    struct timeval ssim_start_time, ssim_end_time;
    double spat_time = 0;
    double pad_time1, pad_time2;
    // start timer.
    gettimeofday(&start_time, NULL);
#endif
    IntFunqueState *s = fex->priv;
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    VmafPicture *res_ref_pic = &s->res_ref_pic;
    VmafPicture *res_dist_pic = &s->res_dist_pic;
#if PROFILE_IND_MODULES
    gettimeofday(&resize_start_time, NULL);
#endif
    if (s->enable_resize)
    {
        res_ref_pic->bpc = ref_pic->bpc;
        res_ref_pic->h[0] = ref_pic->h[0] / 2;
        res_ref_pic->w[0] = ref_pic->w[0] / 2;
        res_ref_pic->stride[0] = ref_pic->stride[0] / 2;
        res_ref_pic->pix_fmt = ref_pic->pix_fmt;
        res_ref_pic->ref = ref_pic->ref;

        res_dist_pic->bpc = dist_pic->bpc;
        res_dist_pic->h[0] = dist_pic->h[0] / 2;
        res_dist_pic->w[0] = dist_pic->w[0] / 2;
        res_dist_pic->stride[0] = dist_pic->stride[0] / 2;
        res_dist_pic->pix_fmt = dist_pic->pix_fmt;
        res_dist_pic->ref = dist_pic->ref;

        resize(ref_pic->data[0], res_ref_pic->data[0], ref_pic->w[0], ref_pic->h[0], res_ref_pic->w[0], res_ref_pic->h[0]);
        resize(dist_pic->data[0], res_dist_pic->data[0], dist_pic->w[0], dist_pic->h[0], res_dist_pic->w[0], res_dist_pic->h[0]);
    }
    else
    {
        res_ref_pic = ref_pic;
        res_dist_pic = dist_pic;
    }
#if PROFILE_IND_MODULES
    gettimeofday(&resize_end_time, NULL);
    prefil_start_time = resize_end_time;
#endif
    // TODO: Move to lookup table for optimization
    int bitdepth_pow2 = (int)pow(2, res_ref_pic->bpc) - 1;
#if PROFILE_IND_MODULES
    gettimeofday(&spat_start_time, NULL);
#endif
    integer_spatial_filter(res_ref_pic->data[0], s->spat_filter, res_ref_pic->w[0], res_ref_pic->h[0]);
#if PROFILE_IND_MODULES
    gettimeofday(&spat_end_time, NULL);
    spat_time += ((spat_end_time.tv_sec - spat_start_time.tv_sec) * 1e6 +
                  (spat_end_time.tv_usec - spat_start_time.tv_usec)) * 1e-6;
#endif
    integer_funque_dwt2(s->spat_filter, &s->i_ref_dwt2out, s->i_dwt2_stride, res_ref_pic->w[0], res_ref_pic->h[0]);
#if PROFILE_IND_MODULES
    gettimeofday(&spat_start_time, NULL);
#endif
    integer_spatial_filter(res_dist_pic->data[0], s->spat_filter, res_dist_pic->w[0], res_dist_pic->h[0]);
#if PROFILE_IND_MODULES
    gettimeofday(&spat_end_time, NULL);
    spat_time += ((spat_end_time.tv_sec - spat_start_time.tv_sec) * 1e6 +
                  (spat_end_time.tv_usec - spat_start_time.tv_usec)) * 1e-6;
#endif
    integer_funque_dwt2(s->spat_filter, &s->i_dist_dwt2out, s->i_dwt2_stride, res_dist_pic->w[0], res_dist_pic->h[0]);

    int16_t spatfilter_shifts = 2 * SPAT_FILTER_COEFF_SHIFT - SPAT_FILTER_INTER_SHIFT - SPAT_FILTER_OUT_SHIFT;
    int16_t dwt_shifts = 2 * DWT2_COEFF_UPSHIFT - DWT2_INTER_SHIFT - DWT2_OUT_SHIFT;
    float pending_div_factor = (1 << ( spatfilter_shifts + dwt_shifts)) * bitdepth_pow2;

#if PROFILE_IND_MODULES
    gettimeofday(&prefil_end_time, NULL);
    motion_start_time = prefil_end_time;
#endif

    if (index == 0)
    {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_integer_feature_motion_score", 0., index);
        memcpy(s->i_prev_ref_dwt2, s->i_ref_dwt2out.bands[0],
               s->i_ref_dwt2out.width * s->i_ref_dwt2out.height * sizeof(dwt2_dtype));

        if (err)
            return err;
    }
    else
    {
        double motion_score;

        err |= integer_compute_motion_funque(s->i_prev_ref_dwt2, s->i_ref_dwt2out.bands[0],
                                             s->i_ref_dwt2out.width, s->i_ref_dwt2out.height,
                                             s->i_dwt2_stride, s->i_dwt2_stride,
                                             pending_div_factor,
                                             &motion_score);
        memcpy(s->i_prev_ref_dwt2, s->i_ref_dwt2out.bands[0],
               s->i_ref_dwt2out.width * s->i_ref_dwt2out.height * sizeof(dwt2_dtype));

        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                                                       s->feature_name_dict, "FUNQUE_integer_feature_motion_score", motion_score, index);

        if (err)
            return err;
    }

#if PROFILE_IND_MODULES
    gettimeofday(&motion_end_time, NULL);
    adm_start_time = motion_end_time;
#endif

    double adm_score, adm_score_num, adm_score_den;
    double ssim_score;

    err = integer_compute_adm_funque(s->i_ref_dwt2out, s->i_dist_dwt2out, &adm_score, &adm_score_num, &adm_score_den, s->i_ref_dwt2out.width, s->i_ref_dwt2out.height, 0.2, (int16_t) pending_div_factor, s->adm_div_lookup);
    if (err)
        return err;
    err |= vmaf_feature_collector_append(feature_collector, "FUNQUE_integer_feature_adm2_score",
                                         adm_score, index);

#if PROFILE_IND_MODULES
    gettimeofday(&adm_end_time, NULL);
    ssim_start_time = adm_end_time;
#endif

    err = integer_compute_ssim_funque(&s->i_ref_dwt2out, &s->i_dist_dwt2out, &ssim_score, 1, 0.01, 0.03,
                                      pow(2, 2 * SPAT_FILTER_COEFF_SHIFT - SPAT_FILTER_INTER_SHIFT - SPAT_FILTER_OUT_SHIFT + 2 * DWT2_COEFF_UPSHIFT - DWT2_INTER_SHIFT - DWT2_OUT_SHIFT) * bitdepth_pow2, s->adm_div_lookup);

    err |= vmaf_feature_collector_append(feature_collector, "FUNQUE_integer_feature_ssim",
                                         ssim_score, index);

#if PROFILE_IND_MODULES
    gettimeofday(&ssim_end_time, NULL);
    vif_start_time = ssim_end_time;
#endif

    double vif_score[MAX_VIF_LEVELS], vif_score_num[MAX_VIF_LEVELS], vif_score_den[MAX_VIF_LEVELS];

    err = integer_compute_vif_funque(s->i_ref_dwt2out.bands[0], s->i_dist_dwt2out.bands[0], s->i_ref_dwt2out.width, s->i_ref_dwt2out.height, 
                    &vif_score[0], &vif_score_num[0], &vif_score_den[0], 9, 1, (double)5.0, (int16_t) pending_div_factor, s->log_18 
#if PROFILE_IND_MODULES
                    , &pad_time1
#endif
                    );
    if (err) return err;

    int vifdwt_stride = (s->i_dwt2_stride + 1)/2;
    int vifdwt_width  = s->i_ref_dwt2out.width;
    int vifdwt_height = s->i_ref_dwt2out.height;
    //The VIF function reuses the band1, band2, band3 of s->ref_dwt2out & s->dist_dwt2out
    //Hence VIF is called in the end
    //If the individual modules(VIF,ADM,motion,ssim) are moved to different files,
    // separate memory allocation for higher level VIF buffers might be needed  
    for(int vif_level=1; vif_level<s->vif_levels; vif_level++)
    {
        int16_t vif_pending_div = (1 << ( spatfilter_shifts + (dwt_shifts << vif_level))) * bitdepth_pow2;;
        integer_funque_vifdwt2_band0(s->i_ref_dwt2out.bands[vif_level-1], s->i_ref_dwt2out.bands[vif_level], vifdwt_stride, vifdwt_width, vifdwt_height);
        integer_funque_vifdwt2_band0(s->i_dist_dwt2out.bands[vif_level-1], s->i_dist_dwt2out.bands[vif_level], vifdwt_stride, vifdwt_width, vifdwt_height);
        vifdwt_stride = (vifdwt_stride + 1)/2;
        vifdwt_width = (vifdwt_width + 1)/2;
        vifdwt_height = (vifdwt_height + 1)/2;

        err = integer_compute_vif_funque(s->i_ref_dwt2out.bands[vif_level], s->i_dist_dwt2out.bands[vif_level], vifdwt_width, vifdwt_height, 
                                    &vif_score[vif_level], &vif_score_num[vif_level], &vif_score_den[vif_level], 9, 1, (double)5.0, (int16_t) vif_pending_div, s->log_18
#if PROFILE_IND_MODULES
                    , &pad_time2
#endif
                    );        
        if (err) return err;
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_integer_feature_vif_scale0_score",
            vif_score[0], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_integer_feature_vif_scale1_score",
            vif_score[1], index);
	
    if (s->vif_levels > 2)
    {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_integer_feature_vif_scale2_score",
            vif_score[2], index);
        if (s->vif_levels > 3)
        {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "FUNQUE_integer_feature_vif_scale3_score",
            vif_score[3], index);
        }
    }
#if PROFILE_IND_MODULES
    gettimeofday(&vif_end_time, NULL);
#endif

#if PROFILE_FUNQUE
    gettimeofday(&end_time, NULL);
    double time_taken;
    time_taken = (end_time.tv_sec - start_time.tv_sec) * 1e6;
    time_taken = (time_taken + (end_time.tv_usec - 
                              start_time.tv_usec)) * 1e-6;
    if(index==0)
    {
        printf("frame_num,time_taken");
#if PROFILE_IND_MODULES
        printf(",resize,pre_filters,spatial_filter,motion,vif,adm,ssim,vifpad1,vifpad2");
#endif
        printf("\n");
    }
    printf("%d,%f", (index+1), time_taken);
#if PROFILE_IND_MODULES
    double resize_time, pre_filt_time, motion_time, vif_time, adm_time, ssim_time;
    resize_time   = ((resize_end_time.tv_sec - resize_start_time.tv_sec) * 1e6 +
                    (resize_end_time.tv_usec - resize_start_time.tv_usec)) * 1e-6;
    pre_filt_time = ((prefil_end_time.tv_sec - prefil_start_time.tv_sec) * 1e6 +
                    (prefil_end_time.tv_usec - prefil_start_time.tv_usec)) * 1e-6;
    motion_time   = ((motion_end_time.tv_sec - motion_start_time.tv_sec) * 1e6 +
                    (motion_end_time.tv_usec - motion_start_time.tv_usec)) * 1e-6;
    vif_time      = ((vif_end_time.tv_sec - vif_start_time.tv_sec) * 1e6 +
                    (vif_end_time.tv_usec - vif_start_time.tv_usec)) * 1e-6;
    adm_time      = ((adm_end_time.tv_sec - adm_start_time.tv_sec) * 1e6 +
                    (adm_end_time.tv_usec - adm_start_time.tv_usec)) * 1e-6;
    ssim_time     = ((ssim_end_time.tv_sec - ssim_start_time.tv_sec) * 1e6 +
                    (ssim_end_time.tv_usec - ssim_start_time.tv_usec)) * 1e-6;

    printf(",%f,%f,%f,%f,%f,%f,%f", resize_time, pre_filt_time, spat_time, motion_time, vif_time, adm_time, ssim_time, pad_time1, pad_time2);
#endif
    printf("\n");
#endif

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    IntFunqueState *s = fex->priv;
    if (s->res_ref_pic.data[0])
        aligned_free(s->res_ref_pic.data[0]);
    if (s->res_dist_pic.data[0])
        aligned_free(s->res_dist_pic.data[0]);
    if (s->spat_filter)
        aligned_free(s->spat_filter);
    if (s->i_prev_ref_dwt2)
        aligned_free(s->i_prev_ref_dwt2);

    for (unsigned i = 0; i < 4; i++)
    {
        if (s->i_ref_dwt2out.bands[i])
            aligned_free(s->i_ref_dwt2out.bands[i]);
        if (s->i_dist_dwt2out.bands[i])
            aligned_free(s->i_dist_dwt2out.bands[i]);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "FUNQUE_integer_feature_vif_scale0_score", "FUNQUE_integer_feature_vif_scale1_score",
    "FUNQUE_integer_feature_vif_scale2_score", "FUNQUE_integer_feature_vif_scale3_score",

    "FUNQUE_integer_feature_adm2_score", "FUNQUE_integer_feature_adm_scale0_score",
    "FUNQUE_integer_feature_adm_scale1_score", "FUNQUE_integer_feature_adm_scale2_score",
    "FUNQUE_integer_feature_adm_scale3_score",

    "FUNQUE_integer_feature_motion_score", "FUNQUE_integer_feature_motion2_score",
    "FUNQUE_integer_feature_motion2_score",

    "FUNQUE_integer_feature_ssim",

    NULL};

VmafFeatureExtractor vmaf_fex_integer_funque = {
    .name = "integer_funque",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(IntFunqueState),
    .provided_features = provided_features,
};