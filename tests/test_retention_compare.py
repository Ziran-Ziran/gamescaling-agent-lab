import copy
import unittest
from scripts.training.retention_compare import ADAPTERS,ADAPTER_IDENTITIES,MODEL_FINGERPRINT,MODEL_WEIGHTS,SUITES,validate_protocol


class RetentionProtocolTests(unittest.TestCase):
    def fixture(self,arm='original',suite='primary'):
        return {'arguments':dict(SUITES[suite]),'model':{'fingerprint':MODEL_FINGERPRINT,
                'files':{'model.safetensors':MODEL_WEIGHTS}},
                'adapter':copy.deepcopy(ADAPTER_IDENTITIES[arm])}

    def test_fixed_model_adapter_and_suite_not_just_self_consistent_config(self):
        for suite in SUITES:
            for arm in ADAPTERS:
                config=self.fixture(arm,suite)
                validate_protocol(config,arm,suite)
                changed=copy.deepcopy(config)
                changed['arguments']['seed_start']=12000
                with self.assertRaisesRegex(ValueError,'protocol'):
                    validate_protocol(changed,arm,suite)
        changed=self.fixture('audio-lora128')
        changed['adapter']['files']['adapter_config.json']='0'*64
        with self.assertRaisesRegex(ValueError,'adapter'):
            validate_protocol(changed,'audio-lora128','primary')
        changed=self.fixture('audio-lora128')
        changed['adapter']['files']['adapter_model.safetensors']=ADAPTERS['class-sft128']
        with self.assertRaisesRegex(ValueError,'adapter'):
            validate_protocol(changed,'audio-lora128','primary')
        changed=self.fixture()
        changed['model']['files']['model.safetensors']='0'*64
        with self.assertRaisesRegex(ValueError,'base model'):
            validate_protocol(changed,'original','primary')

    def test_wrong_horizon_history_or_games_rejected(self):
        for key,value in (('max_steps',8),('history_len',1),('games',['hazard'])):
            changed=self.fixture()
            changed['arguments'][key]=value
            with self.assertRaisesRegex(ValueError,'protocol'):
                validate_protocol(changed,'original','primary')


if __name__=='__main__':
    unittest.main()
