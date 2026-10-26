import copy
import importlib.util
import unittest

from scripts.training.vision_readout_train import validate_run_inputs
from scripts.training.vision_features import POOL_VERSION


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class ReadoutTrainingTests(unittest.TestCase):
    def test_training_requires_recorded_original_features_and_complete_balanced_missing_pairs(self):
        import torch
        metadata={'source_kinds':['recorded'],'train_rows':4,
                  'encoder_identity':{'model':{'fingerprint':'f'*64},'pool_version':POOL_VERSION}}
        rgb=torch.arange(8.).reshape(4,2)
        tensors={'rgb':torch.cat((rgb,rgb)), 'audio':torch.cat((rgb,torch.zeros_like(rgb))),
                 'present':torch.tensor([True]*4+[False]*4),'labels':torch.tensor([1,3,3,1]*2)}
        validate_run_inputs(metadata,tensors)
        for changed in ({**metadata,'source_kinds':['synthetic']},
                        {**metadata,'encoder_identity':{'fake':'fixture'}}):
            with self.assertRaises(ValueError):
                validate_run_inputs(changed,tensors)
        changed=copy.deepcopy(tensors)
        changed['labels'][4]=3
        with self.assertRaisesRegex(ValueError,'missing pairs'):
            validate_run_inputs(metadata,changed)


if __name__=='__main__':
    unittest.main()
