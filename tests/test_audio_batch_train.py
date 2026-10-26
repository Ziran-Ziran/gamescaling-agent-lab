import copy
import importlib.util
from pathlib import Path
import unittest

from controller.control_core import to_action_text
from scripts.training.action_head import template_action

from scripts.training import audio_batch as module


class SensorAllowlistTests(unittest.TestCase):
    def test_image_processor_identity_binds_torchvision_and_canonical_defaults(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.training.audio_batch_train import image_processor_identity
        class ImageProcessor:
            def to_dict(self):
                return {'image_mean':(.1,.2,.3)}
        processor=SimpleNamespace(image_processor=ImageProcessor())
        with patch('scripts.training.audio_batch_train.importlib.metadata.version',return_value='version1'):
            identity=image_processor_identity(processor)
        self.assertEqual(identity,json.loads(json.dumps(identity)))
        with patch('scripts.training.audio_batch_train.importlib.metadata.version',return_value='version2'):
            self.assertNotEqual(identity,image_processor_identity(processor))

    def test_targets_resets_and_privileged_fields_never_enter_training_payload(self):
        from scripts.training.audio_batch_train import sensor_payload
        row={'sample_id':'one','system_prompt':'visible rules','rgb':{'path':'frame.png'},
             'action':'teacher','reward':99,'engine_state':'private','cue_reset':{'tone':1}}
        sensors={'one':{'audio':{'present':False},'captured_ns':7,'clock_id':'sensor',
                        'reward':88,'teacher_class':1}}
        payload=sensor_payload(row,sensors)
        self.assertEqual(set(payload),{'system_prompt','rgb','audio','captured_ns','clock_id'})
        self.assertNotIn('action',payload)
        self.assertIsNot(payload['rgb'],row['rgb'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class AudioMicrobatchTests(unittest.TestCase):
    def fixture(self):
        import torch
        torch.manual_seed(99)
        model=torch.nn.Linear(3,5,bias=False)
        rows=[{'sample_id':str(index),'action':to_action_text(template_action(label))}
              for index,label in enumerate(('A','D','D','A'))]
        inputs=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
        return model,rows,inputs

    def test_four_micro_gradients_match_mean_ce_and_optimizer_steps_once(self):
        import torch
        model,rows,inputs=self.fixture()
        reference=copy.deepcopy(model)
        expected_optimizer=torch.optim.SGD(reference.parameters(),lr=.01)
        targets=torch.tensor([1,3,3,1])
        expected=torch.nn.functional.cross_entropy(reference(inputs)/2,targets)
        expected.backward()
        expected_optimizer.step()
        class CountingSGD(torch.optim.SGD):
            calls=0
            def step(self,*args,**kwargs):
                self.calls+=1
                return super().step(*args,**kwargs)
        optimizer=CountingSGD(model.parameters(),lr=.01)
        def forward(row):
            return model(inputs[int(row['sample_id'])]),{'only':'sensors'}
        record=module.optimize_microbatch(model,optimizer,rows,forward,temperature=2.,clip=1e6)
        self.assertEqual(optimizer.calls,1)
        self.assertEqual(len(record['examples']),4)
        self.assertAlmostEqual(record['loss'],float(expected.detach()),places=6)
        torch.testing.assert_close(model.weight,reference.weight,rtol=1e-6,atol=1e-7)
        self.assertTrue(all(type(row['loss']) is float and len(row['raw_class_logits'])==5
                            for row in record['examples']))

    def test_partial_microbatch_failure_does_not_step_or_leave_partial_gradients(self):
        import torch
        model,rows,inputs=self.fixture()
        original=model.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        def forward(row):
            if row['sample_id']=='2':
                raise RuntimeError('injected third sensor failure')
            return model(inputs[int(row['sample_id'])]),{}
        with self.assertRaisesRegex(RuntimeError,'third sensor'):
            module.optimize_microbatch(model,optimizer,rows,forward)
        self.assertTrue(torch.equal(original,model.weight))
        self.assertEqual(len(optimizer.state),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
