from collections import Counter
import copy
import importlib.util
import unittest
from controller.control_core import to_action_text
from scripts.training.action_head import template_action
from tests import test_audio_batch as fixture
from scripts.training import audio_replay as module

class ReplayScheduleTests(unittest.TestCase):
    def fixture(self):
        from scripts.training.audio_batch import group_training_rows
        audio=group_training_rows(*fixture.BalancedAudioBatchTests().fixture())
        rgb=[{'sample_id':f'rgb-{label}-{i}', 'game_id':'headless-reach-v1','frame_id':0,'history':[],
              'action':to_action_text(template_action(label))} for label in ('W','A','S','D') for i in range(16)]
        return audio,module.rgb_groups(rgb)
    def test_exact_audio_exposure_missing_and_rgb_budget(self):
        audio,rgb=self.fixture();exposures={}
        for arm in module.ARMS:
            seen=Counter();labels=Counter();masked=Counter()
            for step in range(64):
                examples=module.examples_for_step(audio,rgb,arm=arm,step=step,seed=20261101)
                a=[e for e in examples if e['kind']=='audio'];r=[e for e in examples if e['kind']=='rgb']
                self.assertEqual(sorted(e['target_class'] for e in a),[1,3]);self.assertEqual(len(r),2 if arm=='rgb-replay' else 0)
                self.assertEqual(a[0]['row']['rgb']['pixel_sha256'],a[1]['row']['rgb']['pixel_sha256'])
                seen.update((e['row']['sample_id'],e['missing']) for e in a)
                labels.update(e['target_class'] for e in r)
                masked.update(e['target_class'] for e in a if e['missing'])
            self.assertEqual(sum(seen.values()),128);self.assertEqual(masked,{1:8,3:8})
            self.assertEqual(labels,{0:32,1:32,2:32,3:32} if arm=='rgb-replay' else {})
            exposures[arm]=seen
        self.assertEqual(*exposures.values())
    def test_global_step_reconstruction_and_complete_epochs(self):
        audio,rgb=self.fixture()
        for arm in module.ARMS:
            rows=[e['row']['sample_id'] for step in range(32) for e in module.examples_for_step(audio,rgb,arm=arm,step=step,seed=20261101) if e['kind']=='audio']
            self.assertEqual(len(set(rows)),64)
            for step in (0,2,7,8,31,32,99):
                self.assertEqual(module.examples_for_step(audio,rgb,arm=arm,step=step,seed=20261101),module.examples_for_step(audio,rgb,arm=arm,step=step,seed=20261101))
        with self.assertRaises(ValueError):module.examples_for_step(audio,rgb,arm='wrong',step=0,seed=1)

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
class ReplayOptimizationTests(unittest.TestCase):
    def test_control_preserves_quarter_loss_coefficient(self):
        import torch
        torch.manual_seed(1);model=torch.nn.Linear(2,5,bias=False);ref=copy.deepcopy(model)
        rows=[{'kind':'audio','target_class':i,'row':{'sample_id':str(i)}} for i in (1,3)]
        inputs=torch.eye(2);opt=torch.optim.SGD(model.parameters(),lr=.1);ropt=torch.optim.SGD(ref.parameters(),lr=.1)
        loss=torch.nn.functional.cross_entropy(ref(inputs)/2,torch.tensor([1,3]),reduction='sum')/4
        loss.backward();ropt.step()
        record=module.optimize_examples(model,opt,rows,lambda e:(model(inputs[int(e['row']['sample_id'])//2]),{}),clip=1e6)
        torch.testing.assert_close(model.weight,ref.weight,rtol=1e-6,atol=1e-7)
        self.assertAlmostEqual(record['objective'],float(loss.detach()),places=6)
        self.assertIsNone(record['rgb_mean_ce']);self.assertEqual(record['gradient_clip_factor'],1.)
    def test_failure_aborts_without_optimizer_step(self):
        import torch
        model=torch.nn.Linear(2,5);before=copy.deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters())
        examples=[{'kind':'audio','target_class':i,'row':{'sample_id':str(i)}} for i in (1,3)]
        def forward(e):
            if e['target_class']==3:raise RuntimeError('sensor failure')
            return model(torch.ones(2)),{}
        with self.assertRaisesRegex(RuntimeError,'sensor failure'):module.optimize_examples(model,opt,examples,forward)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()));self.assertEqual(len(opt.state),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_parent_copy_prevalidates_all24_and_does_not_touch_rng(self):
        import torch
        model=torch.nn.ParameterDict({f'p{i}':torch.nn.Parameter(torch.zeros(2)) for i in range(24)})
        parent={name:torch.full_like(p,3.) for name,p in model.named_parameters()};rng=torch.get_rng_state().clone()
        bad=copy.deepcopy(parent);bad['p9']=torch.ones(3)
        with self.assertRaises(ValueError):module.copy_parent_trainables(model,bad)
        self.assertTrue(all(bool((p==0).all()) for p in model.parameters()))
        module.copy_parent_trainables(model,parent)
        self.assertTrue(all(bool((p==3).all()) for p in model.parameters()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
