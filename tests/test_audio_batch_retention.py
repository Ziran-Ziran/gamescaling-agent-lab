import copy
import unittest

from tests import test_audio_batch_eval as fixtures
from scripts.training.audio_batch_eval import POLICIES
from scripts.training.retention_compare import MODEL_FINGERPRINT,MODEL_WEIGHTS
from scripts.training.audio_batch_retention import SUITES,compare_matrix,validate_rgb_protocol


class AudioBatchRGBRetentionProtocolTests(unittest.TestCase):
    def test_all_four_fixed_cohort_policies_and_new_rgb_suites(self):
        cohort=fixtures.AudioBatchEvaluationProtocolTests().fixture()
        for suite in SUITES:
            for policy in POLICIES:
                config={'arguments':dict(SUITES[suite]),'adapter':copy.deepcopy(cohort['policies'][policy]),
                        'model':{'fingerprint':MODEL_FINGERPRINT,'files':{'model.safetensors':MODEL_WEIGHTS}},
                        'environment':{'packages':dict(cohort['training']['paired-context']['identity']['packages'])}}
                validate_rgb_protocol(config,cohort,policy,suite)
                changed=copy.deepcopy(config)
                changed['arguments']['seed_start']=30000
                with self.assertRaisesRegex(ValueError,'protocol'):
                    validate_rgb_protocol(changed,cohort,policy,suite)
                if policy!='original':
                    changed=copy.deepcopy(config)
                    changed['adapter']=None
                    with self.assertRaisesRegex(ValueError,'adapter'):
                        validate_rgb_protocol(changed,cohort,policy,suite)
                changed=copy.deepcopy(config)
                changed['environment']['packages']['fixture']='version2'
                with self.assertRaisesRegex(ValueError,'dependencies'):
                    validate_rgb_protocol(changed,cohort,policy,suite)

    def test_partial_policy_matrix_rejected_before_reading_files(self):
        with self.assertRaisesRegex(ValueError,'four'):
            compare_matrix({'original':'not-read'},'not-read',suite='primary')


if __name__=='__main__':
    unittest.main()
