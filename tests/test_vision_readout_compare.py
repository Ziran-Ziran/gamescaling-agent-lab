import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from scripts.training.vision_readout_compare import audit_feature_file,comparable_identity


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class ReadoutEvidenceTests(unittest.TestCase):
    def test_numeric_replay_binds_file_features_logits_and_head(self):
        import torch
        from scripts.training.data import sha256_file
        from scripts.training.vision_features import tensor_digest
        from scripts.training.vision_readout import FeatureNormalizer,ReadoutPolicy,make_head
        norm=FeatureNormalizer(torch.zeros(4),torch.ones(4))
        policy=ReadoutPolicy(make_head(9,arm='joint-mlp'),norm,norm,arm='joint-mlp')
        rgb,audio=torch.arange(4).float(),torch.ones(4)
        logits,record=policy.predict(rgb,audio,True)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'features.pt'
            torch.save({'rgb':rgb,'audio':audio,'present':True,'logits':logits,'head_record':record},path)
            evidence={'feature_file':{'path':path.name,'sha256':sha256_file(path)},
                'inputs':{'present':True,'input_rgb_pixel_sha256':'rgb','input_spectral_pixel_sha256':'audio',
                          'model_text_used':False,'input_tokens':0,'prompt_sha256':None,'actual_visual_calls':2},
                'encoder_records':{key:{'feature_sha256':tensor_digest(value),'input_pixel_sha256':key}
                                   for key,value in (('rgb',rgb),('audio',audio))},
                'head_record':record,'class_record':{'raw_class_logits':dict(zip(('W','A','S','D','noop'),logits.tolist()))}}
            self.assertEqual(audit_feature_file(evidence,root,policy,torch.zeros(4),4),0.)
            bad=copy.deepcopy(evidence)
            bad['inputs']['actual_visual_calls']=1
            with self.assertRaisesRegex(ValueError,'call count'):
                audit_feature_file(bad,root,policy,torch.zeros(4),4)
            bad=copy.deepcopy(evidence)
            bad['encoder_records']['audio']['feature_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'feature'):
                audit_feature_file(bad,root,policy,torch.zeros(4),4)
            bad=copy.deepcopy(evidence)
            bad['feature_file']['path']='../outside.pt'
            with self.assertRaisesRegex(ValueError,'path'):
                audit_feature_file(bad,root,policy,torch.zeros(4),4)
            bad=copy.deepcopy(evidence)
            bad['class_record']['raw_class_logits']['A']+=.1
            with self.assertRaisesRegex(ValueError,'logits'):
                audit_feature_file(bad,root,policy,torch.zeros(4),4)

    def test_only_declared_arm_architecture_initialization_may_differ(self):
        common={'source':42,'arm':'rgb-mlp','architecture':'MLP','initialization_fingerprint':'same'}
        other={**common,'arm':'joint-linear','architecture':'linear','initialization_fingerprint':'different'}
        self.assertEqual(comparable_identity(common),comparable_identity(other))
        other['source']=43
        self.assertNotEqual(comparable_identity(common),comparable_identity(other))


if __name__=='__main__':
    unittest.main()
