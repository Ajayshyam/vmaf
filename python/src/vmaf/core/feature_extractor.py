from abc import ABCMeta, abstractmethod
import os
from xml.etree import ElementTree

from vmaf.tools.misc import make_absolute_path, run_process
from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import re
import numpy as np
import ast

from vmaf import ExternalProgramCaller
from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.executor import Executor
from vmaf.core.result import Result
from vmaf.tools.reader import YuvReader


class FeatureExtractor(Executor):
    """
    FeatureExtractor takes in a list of assets, and run feature extraction on
    them, and return a list of corresponding results. A FeatureExtractor must
    specify a unique type and version combination (by the TYPE and VERSION
    attribute), so that the Result generated by it can be identified.

    A derived class of FeatureExtractor must:
        1) Override TYPE and VERSION
        2) Override _generate_result(self, asset), which call a
        command-line executable and generate feature scores in a log file.
        3) Override _get_feature_scores(self, asset), which read the feature
        scores from the log file, and return the scores in a dictionary format.
    For an example, follow VmafFeatureExtractor.
    """

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def ATOM_FEATURES(self):
        raise NotImplementedError

    def _read_result(self, asset):
        result = {}
        result.update(self._get_feature_scores(asset))
        executor_id = self.executor_id
        return Result(asset, executor_id, result)

    @classmethod
    def get_scores_key(cls, atom_feature):
        return "{type}_{atom_feature}_scores".format(
            type=cls.TYPE, atom_feature=atom_feature)

    @classmethod
    def get_score_key(cls, atom_feature):
        return "{type}_{atom_feature}_score".format(
            type=cls.TYPE, atom_feature=atom_feature)

    def _get_feature_scores(self, asset):
        # routine to read the feature scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)

        atom_feature_scores_dict = {}
        atom_feature_idx_dict = {}
        for atom_feature in self.ATOM_FEATURES:
            atom_feature_scores_dict[atom_feature] = []
            atom_feature_idx_dict[atom_feature] = 0

        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                for atom_feature in self.ATOM_FEATURES:
                    re_template = "{af}: ([0-9]+) ([a-zA-Z0-9.-]+)".format(af=atom_feature)
                    mo = re.match(re_template, line)
                    if mo:

                        cur_idx = int(mo.group(1))
                        assert cur_idx == atom_feature_idx_dict[atom_feature]

                        # parse value, allowing NaN and inf
                        val = float(mo.group(2))
                        if np.isnan(val) or np.isinf(val):
                            val = None

                        atom_feature_scores_dict[atom_feature].append(val)
                        atom_feature_idx_dict[atom_feature] += 1
                        continue

        len_score = len(atom_feature_scores_dict[self.ATOM_FEATURES[0]])
        assert len_score != 0
        for atom_feature in self.ATOM_FEATURES[1:]:
            assert len_score == len(atom_feature_scores_dict[atom_feature]), \
                "Feature data possibly corrupt. Run cleanup script and try again."

        feature_result = {}

        for atom_feature in self.ATOM_FEATURES:
            scores_key = self.get_scores_key(atom_feature)
            feature_result[scores_key] = atom_feature_scores_dict[atom_feature]

        return feature_result


class VmafFeatureExtractor(FeatureExtractor):

    TYPE = "VMAF_feature"

    # VERSION = '0.1' # vmaf_study; Anush's VIF fix
    # VERSION = '0.2' # expose vif_num, vif_den, adm_num, adm_den, anpsnr
    # VERSION = '0.2.1' # expose vif num/den of each scale
    # VERSION = '0.2.2'  # adm abs-->fabs, corrected border handling, uniform reading with option of offset for input YUV, updated VIF corner case
    # VERSION = '0.2.2b'  # expose adm_den/num_scalex
    # VERSION = '0.2.3'  # AVX for VMAF convolution; update adm features by folding noise floor into per coef
    # VERSION = '0.2.4'  # Fix a bug in adm feature passing scale into dwt_quant_step
    # VERSION = '0.2.4b'  # Modify by adding ADM noise floor outside cube root; add derived feature motion2
    VERSION = '0.2.4c'  # Modify by moving motion2 to c code

    ATOM_FEATURES = ['vif', 'adm', 'ansnr', 'motion', 'motion2',
                     'vif_num', 'vif_den', 'adm_num', 'adm_den', 'anpsnr',
                     'vif_num_scale0', 'vif_den_scale0',
                     'vif_num_scale1', 'vif_den_scale1',
                     'vif_num_scale2', 'vif_den_scale2',
                     'vif_num_scale3', 'vif_den_scale3',
                     'adm_num_scale0', 'adm_den_scale0',
                     'adm_num_scale1', 'adm_den_scale1',
                     'adm_num_scale2', 'adm_den_scale2',
                     'adm_num_scale3', 'adm_den_scale3',
                     ]

    DERIVED_ATOM_FEATURES = ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                             'vif2', 'adm2', 'adm3',
                             'adm_scale0', 'adm_scale1', 'adm_scale2', 'adm_scale3',
                             ]

    ADM2_CONSTANT = 0
    ADM_SCALE_CONSTANT = 0

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        ExternalProgramCaller.call_vmaf_feature(yuv_type, ref_path, dis_path, w, h, log_file_path, logger)

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        result = super(VmafFeatureExtractor, cls)._post_process_result(result)

        # adm2 =
        # (adm_num + ADM2_CONSTANT) / (adm_den + ADM2_CONSTANT)
        adm2_scores_key = cls.get_scores_key('adm2')
        adm_num_scores_key = cls.get_scores_key('adm_num')
        adm_den_scores_key = cls.get_scores_key('adm_den')
        result.result_dict[adm2_scores_key] = list(
            (np.array(result.result_dict[adm_num_scores_key]) + cls.ADM2_CONSTANT) /
            (np.array(result.result_dict[adm_den_scores_key]) + cls.ADM2_CONSTANT)
        )

        # vif_scalei = vif_num_scalei / vif_den_scalei, i = 0, 1, 2, 3
        vif_num_scale0_scores_key = cls.get_scores_key('vif_num_scale0')
        vif_den_scale0_scores_key = cls.get_scores_key('vif_den_scale0')
        vif_num_scale1_scores_key = cls.get_scores_key('vif_num_scale1')
        vif_den_scale1_scores_key = cls.get_scores_key('vif_den_scale1')
        vif_num_scale2_scores_key = cls.get_scores_key('vif_num_scale2')
        vif_den_scale2_scores_key = cls.get_scores_key('vif_den_scale2')
        vif_num_scale3_scores_key = cls.get_scores_key('vif_num_scale3')
        vif_den_scale3_scores_key = cls.get_scores_key('vif_den_scale3')
        vif_scale0_scores_key = cls.get_scores_key('vif_scale0')
        vif_scale1_scores_key = cls.get_scores_key('vif_scale1')
        vif_scale2_scores_key = cls.get_scores_key('vif_scale2')
        vif_scale3_scores_key = cls.get_scores_key('vif_scale3')
        result.result_dict[vif_scale0_scores_key] = list(
            (np.array(result.result_dict[vif_num_scale0_scores_key])
             / np.array(result.result_dict[vif_den_scale0_scores_key]))
        )
        result.result_dict[vif_scale1_scores_key] = list(
            (np.array(result.result_dict[vif_num_scale1_scores_key])
             / np.array(result.result_dict[vif_den_scale1_scores_key]))
        )
        result.result_dict[vif_scale2_scores_key] = list(
            (np.array(result.result_dict[vif_num_scale2_scores_key])
             / np.array(result.result_dict[vif_den_scale2_scores_key]))
        )
        result.result_dict[vif_scale3_scores_key] = list(
            (np.array(result.result_dict[vif_num_scale3_scores_key])
             / np.array(result.result_dict[vif_den_scale3_scores_key]))
        )

        # vif2 =
        # ((vif_num_scale0 / vif_den_scale0) + (vif_num_scale1 / vif_den_scale1) +
        # (vif_num_scale2 / vif_den_scale2) + (vif_num_scale3 / vif_den_scale3)) / 4.0
        vif_scores_key = cls.get_scores_key('vif2')
        result.result_dict[vif_scores_key] = list(
            (
                (np.array(result.result_dict[vif_num_scale0_scores_key])
                 / np.array(result.result_dict[vif_den_scale0_scores_key])) +
                (np.array(result.result_dict[vif_num_scale1_scores_key])
                 / np.array(result.result_dict[vif_den_scale1_scores_key])) +
                (np.array(result.result_dict[vif_num_scale2_scores_key])
                 / np.array(result.result_dict[vif_den_scale2_scores_key])) +
                (np.array(result.result_dict[vif_num_scale3_scores_key])
                 / np.array(result.result_dict[vif_den_scale3_scores_key]))
            ) / 4.0
        )

        # adm_scalei = adm_num_scalei / adm_den_scalei, i = 0, 1, 2, 3
        adm_num_scale0_scores_key = cls.get_scores_key('adm_num_scale0')
        adm_den_scale0_scores_key = cls.get_scores_key('adm_den_scale0')
        adm_num_scale1_scores_key = cls.get_scores_key('adm_num_scale1')
        adm_den_scale1_scores_key = cls.get_scores_key('adm_den_scale1')
        adm_num_scale2_scores_key = cls.get_scores_key('adm_num_scale2')
        adm_den_scale2_scores_key = cls.get_scores_key('adm_den_scale2')
        adm_num_scale3_scores_key = cls.get_scores_key('adm_num_scale3')
        adm_den_scale3_scores_key = cls.get_scores_key('adm_den_scale3')
        adm_scale0_scores_key = cls.get_scores_key('adm_scale0')
        adm_scale1_scores_key = cls.get_scores_key('adm_scale1')
        adm_scale2_scores_key = cls.get_scores_key('adm_scale2')
        adm_scale3_scores_key = cls.get_scores_key('adm_scale3')
        result.result_dict[adm_scale0_scores_key] = list(
            (np.array(result.result_dict[adm_num_scale0_scores_key]) + cls.ADM_SCALE_CONSTANT)
            / (np.array(result.result_dict[adm_den_scale0_scores_key]) + cls.ADM_SCALE_CONSTANT)
        )
        result.result_dict[adm_scale1_scores_key] = list(
            (np.array(result.result_dict[adm_num_scale1_scores_key]) + cls.ADM_SCALE_CONSTANT)
            / (np.array(result.result_dict[adm_den_scale1_scores_key]) + cls.ADM_SCALE_CONSTANT)
        )
        result.result_dict[adm_scale2_scores_key] = list(
            (np.array(result.result_dict[adm_num_scale2_scores_key]) + cls.ADM_SCALE_CONSTANT)
            / (np.array(result.result_dict[adm_den_scale2_scores_key]) + cls.ADM_SCALE_CONSTANT)
        )
        result.result_dict[adm_scale3_scores_key] = list(
            (np.array(result.result_dict[adm_num_scale3_scores_key]) + cls.ADM_SCALE_CONSTANT)
            / (np.array(result.result_dict[adm_den_scale3_scores_key]) + cls.ADM_SCALE_CONSTANT)
        )

        # adm3 = \
        # (((adm_num_scale0 + ADM_SCALE_CONSTANT) / (adm_den_scale0 + ADM_SCALE_CONSTANT))
        #  + ((adm_num_scale1 + ADM_SCALE_CONSTANT) / (adm_den_scale1 + ADM_SCALE_CONSTANT))
        #  + ((adm_num_scale2 + ADM_SCALE_CONSTANT) / (adm_den_scale2 + ADM_SCALE_CONSTANT))
        #  + ((adm_num_scale3 + ADM_SCALE_CONSTANT) / (adm_den_scale3 + ADM_SCALE_CONSTANT))) / 4.0
        adm3_scores_key = cls.get_scores_key('adm3')
        result.result_dict[adm3_scores_key] = list(
            (
                ((np.array(result.result_dict[adm_num_scale0_scores_key]) + cls.ADM_SCALE_CONSTANT)
                 / (np.array(result.result_dict[adm_den_scale0_scores_key]) + cls.ADM_SCALE_CONSTANT)) +
                ((np.array(result.result_dict[adm_num_scale1_scores_key]) + cls.ADM_SCALE_CONSTANT)
                 / (np.array(result.result_dict[adm_den_scale1_scores_key]) + cls.ADM_SCALE_CONSTANT)) +
                ((np.array(result.result_dict[adm_num_scale2_scores_key]) + cls.ADM_SCALE_CONSTANT)
                 / (np.array(result.result_dict[adm_den_scale2_scores_key]) + cls.ADM_SCALE_CONSTANT)) +
                ((np.array(result.result_dict[adm_num_scale3_scores_key]) + cls.ADM_SCALE_CONSTANT)
                 / (np.array(result.result_dict[adm_den_scale3_scores_key]) + cls.ADM_SCALE_CONSTANT))
            ) / 4.0
        )

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result


class VifFrameDifferenceFeatureExtractor(FeatureExtractor):

    TYPE = "VifDiff_feature"

    VERSION = '0.1'

    ATOM_FEATURES = ['vifdiff',
                     'vifdiff_num', 'vifdiff_den',
                     'vifdiff_num_scale0', 'vifdiff_den_scale0',
                     'vifdiff_num_scale1', 'vifdiff_den_scale1',
                     'vifdiff_num_scale2', 'vifdiff_den_scale2',
                     'vifdiff_num_scale3', 'vifdiff_den_scale3',
                     ]

    DERIVED_ATOM_FEATURES = ['vifdiff_scale0', 'vifdiff_scale1', 'vifdiff_scale2', 'vifdiff_scale3',
                             ]

    ADM2_CONSTANT = 0
    ADM_SCALE_CONSTANT = 0

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        ExternalProgramCaller.call_vifdiff_feature(yuv_type, ref_path, dis_path, w, h, log_file_path, logger)

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        result = super(VifFrameDifferenceFeatureExtractor, cls)._post_process_result(result)

        # vifdiff_scalei = vifdiff_num_scalei / vifdiff_den_scalei, i = 0, 1, 2, 3
        vifdiff_num_scale0_scores_key = cls.get_scores_key('vifdiff_num_scale0')
        vifdiff_den_scale0_scores_key = cls.get_scores_key('vifdiff_den_scale0')
        vifdiff_num_scale1_scores_key = cls.get_scores_key('vifdiff_num_scale1')
        vifdiff_den_scale1_scores_key = cls.get_scores_key('vifdiff_den_scale1')
        vifdiff_num_scale2_scores_key = cls.get_scores_key('vifdiff_num_scale2')
        vifdiff_den_scale2_scores_key = cls.get_scores_key('vifdiff_den_scale2')
        vifdiff_num_scale3_scores_key = cls.get_scores_key('vifdiff_num_scale3')
        vifdiff_den_scale3_scores_key = cls.get_scores_key('vifdiff_den_scale3')
        vifdiff_scale0_scores_key = cls.get_scores_key('vifdiff_scale0')
        vifdiff_scale1_scores_key = cls.get_scores_key('vifdiff_scale1')
        vifdiff_scale2_scores_key = cls.get_scores_key('vifdiff_scale2')
        vifdiff_scale3_scores_key = cls.get_scores_key('vifdiff_scale3')
        result.result_dict[vifdiff_scale0_scores_key] = list(
            (np.array(result.result_dict[vifdiff_num_scale0_scores_key])
             / np.array(result.result_dict[vifdiff_den_scale0_scores_key]))
        )
        result.result_dict[vifdiff_scale1_scores_key] = list(
            (np.array(result.result_dict[vifdiff_num_scale1_scores_key])
             / np.array(result.result_dict[vifdiff_den_scale1_scores_key]))
        )
        result.result_dict[vifdiff_scale2_scores_key] = list(
            (np.array(result.result_dict[vifdiff_num_scale2_scores_key])
             / np.array(result.result_dict[vifdiff_den_scale2_scores_key]))
        )
        result.result_dict[vifdiff_scale3_scores_key] = list(
            (np.array(result.result_dict[vifdiff_num_scale3_scores_key])
             / np.array(result.result_dict[vifdiff_den_scale3_scores_key]))
        )

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result


class VmafrcFeatureExtractorMixin(object):

    def _get_feature_scores(self, asset):
        # override FeatureExtractor._get_feature_scores

        assert hasattr(self, '_get_log_file_path')
        assert hasattr(self, 'ATOM_FEATURES')
        assert hasattr(self, 'ATOM_FEATURES_TO_VMAFRC_KEY_DICT')
        assert hasattr(self, 'get_scores_key')

        log_file_path = self._get_log_file_path(asset)
        tree = ElementTree.parse(log_file_path)
        root = tree.getroot()

        feature_scores = [[] for _ in self.ATOM_FEATURES]

        for frame in root.findall('frames/frame'):
            for i_feature, feature in enumerate(self.ATOM_FEATURES):
                try:
                    feature_scores[i_feature].append(float(frame.attrib[self.ATOM_FEATURES_TO_VMAFRC_KEY_DICT[feature]]))
                except KeyError:
                    pass  # some features may be missing

        for i_feature, feature in enumerate(self.ATOM_FEATURES):
            assert len(feature_scores[i_feature]) != 0
            assert len(feature_scores[i_feature]) == len(feature_scores[0])

        feature_result = {}
        for i_feature, feature in enumerate(self.ATOM_FEATURES):
            feature_result[self.get_scores_key(feature)] = feature_scores[i_feature]

        return feature_result


class PsnrFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "PSNR_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['psnr']

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'psnr': 'float_psnr',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        ExternalProgramCaller.call_psnr(yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


class MomentFeatureExtractor(FeatureExtractor):

    TYPE = "Moment_feature"

    # VERSION = "1.0" # call executable
    VERSION = "1.1" # python only

    ATOM_FEATURES = ['ref1st', 'ref2nd', 'dis1st', 'dis2nd', ]

    DERIVED_ATOM_FEATURES = ['refvar', 'disvar', ]

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_w, quality_h = asset.quality_width_height

        ref_scores_mtx = None
        with YuvReader(filepath=asset.ref_procfile_path, width=quality_w, height=quality_h,
                       yuv_type=self._get_workfile_yuv_type(asset)) as ref_yuv_reader:
            scores_mtx_list = []
            i = 0
            for ref_yuv in ref_yuv_reader:
                ref_y = ref_yuv[0]
                ref_y = ref_y.astype(np.double)
                firstm = ref_y.mean()
                secondm = ref_y.var() + firstm**2
                scores_mtx_list.append(np.hstack(([firstm], [secondm])))
                i += 1
            ref_scores_mtx = np.vstack(scores_mtx_list)

        dis_scores_mtx = None
        with YuvReader(filepath=asset.dis_procfile_path, width=quality_w, height=quality_h,
                       yuv_type=self._get_workfile_yuv_type(asset)) as dis_yuv_reader:
            scores_mtx_list = []
            i = 0
            for dis_yuv in dis_yuv_reader:
                dis_y = dis_yuv[0]
                dis_y = dis_y.astype(np.double)
                firstm = dis_y.mean()
                secondm = dis_y.var() + firstm**2
                scores_mtx_list.append(np.hstack(([firstm], [secondm])))
                i += 1
            dis_scores_mtx = np.vstack(scores_mtx_list)

        assert ref_scores_mtx is not None and dis_scores_mtx is not None

        log_dict = {'ref_scores_mtx': ref_scores_mtx.tolist(),
                    'dis_scores_mtx': dis_scores_mtx.tolist()}

        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, 'wt') as log_file:
            log_file.write(str(log_dict))

    def _get_feature_scores(self, asset):
        # routine to read the feature scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)

        with open(log_file_path, 'rt') as log_file:
            log_str = log_file.read()
            log_dict = ast.literal_eval(log_str)
        ref_scores_mtx = np.array(log_dict['ref_scores_mtx'])
        dis_scores_mtx = np.array(log_dict['dis_scores_mtx'])

        _, num_ref_features = ref_scores_mtx.shape
        assert num_ref_features == 2 # ref1st, ref2nd
        _, num_dis_features = dis_scores_mtx.shape
        assert num_dis_features == 2 # dis1st, dis2nd

        feature_result = {}
        feature_result[self.get_scores_key('ref1st')] = list(ref_scores_mtx[:, 0])
        feature_result[self.get_scores_key('ref2nd')] = list(ref_scores_mtx[:, 1])
        feature_result[self.get_scores_key('dis1st')] = list(dis_scores_mtx[:, 0])
        feature_result[self.get_scores_key('dis2nd')] = list(dis_scores_mtx[:, 1])

        return feature_result

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        result = super(MomentFeatureExtractor, cls)._post_process_result(result)

        # calculate refvar and disvar from ref1st, ref2nd, dis1st, dis2nd
        refvar_scores_key = cls.get_scores_key('refvar')
        ref1st_scores_key = cls.get_scores_key('ref1st')
        ref2nd_scores_key = cls.get_scores_key('ref2nd')
        disvar_scores_key = cls.get_scores_key('disvar')
        dis1st_scores_key = cls.get_scores_key('dis1st')
        dis2nd_scores_key = cls.get_scores_key('dis2nd')
        get_var = lambda m: m[1] - m[0] * m[0]
        result.result_dict[refvar_scores_key] = \
            list(map(get_var, zip(result.result_dict[ref1st_scores_key],
                             result.result_dict[ref2nd_scores_key])))
        result.result_dict[disvar_scores_key] = \
            list(map(get_var, zip(result.result_dict[dis1st_scores_key],
                             result.result_dict[dis2nd_scores_key])))

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result


class SsimFeatureExtractor(FeatureExtractor):

    TYPE = "SSIM_feature"
    # VERSION = "1.0"
    VERSION = "1.1" # fix OPT_RANGE_PIXEL_OFFSET = 0

    ATOM_FEATURES = ['ssim', 'ssim_l', 'ssim_c', 'ssim_s']

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        ExternalProgramCaller.call_ssim(yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


class MsSsimFeatureExtractor(FeatureExtractor):

    TYPE = "MS_SSIM_feature"
    # VERSION = "1.0"
    VERSION = "1.1" # fix OPT_RANGE_PIXEL_OFFSET = 0

    ATOM_FEATURES = ['ms_ssim',
                     'ms_ssim_l_scale0', 'ms_ssim_c_scale0', 'ms_ssim_s_scale0',
                     'ms_ssim_l_scale1', 'ms_ssim_c_scale1', 'ms_ssim_s_scale1',
                     'ms_ssim_l_scale2', 'ms_ssim_c_scale2', 'ms_ssim_s_scale2',
                     'ms_ssim_l_scale3', 'ms_ssim_c_scale3', 'ms_ssim_s_scale3',
                     'ms_ssim_l_scale4', 'ms_ssim_c_scale4', 'ms_ssim_s_scale4',
                     ]

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        ExternalProgramCaller.call_ms_ssim(yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


