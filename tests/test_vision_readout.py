import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

from scripts.training.vision_readout import FeatureNormalizer, make_head, readout_inputs, ReadoutPolicy,head_fingerprint


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch CPU runtime required')
class ReadoutTests(unittest.TestCase):
    def test_inference_head_uses_only_numeric_observation_features_and_saved_train_normalization(self):
        import torch
        rgb=FeatureNormalizer.fit(torch.tensor([[1.,2.],[3.,4.]]))
        audio=FeatureNormalizer.fit(torch.tensor([[5.,6.],[7.,8.]]))
        head=make_head(5,arm='joint-mlp',seed=7)
        policy=ReadoutPolicy(head,rgb,audio,arm='joint-mlp')
        before=head_fingerprint(head)
        logits,evidence=policy.predict(torch.tensor([1.,2.]),torch.tensor([5.,6.]),True)
        self.assertEqual(logits.shape,(5,))
        self.assertFalse(logits.requires_grad)
        self.assertEqual(head_fingerprint(head),before)
        self.assertTrue(evidence['present'])
        self.assertEqual(evidence['arm'],'joint-mlp')
        with self.assertRaises(ValueError):
            policy.predict(torch.tensor([1.,2.]),torch.tensor([5.,6.]),1)

    def test_normalizer_uses_only_explicit_training_features_and_constant_channels_stay_finite(self):
        import torch
        train=torch.tensor([[1.,5.],[3.,5.]])
        normalizer=FeatureNormalizer.fit(train)
        self.assertEqual(normalizer.mean.tolist(),[2.,5.])
        self.assertTrue(torch.isfinite(normalizer.transform(torch.tensor([[1000.,5.]]))).all())
        self.assertEqual(normalizer.transform(train)[:,1].tolist(),[0.,0.])
        with self.assertRaises(ValueError):
            normalizer.transform(torch.ones(1,3))

    def test_arm_gates_remove_only_declared_modalities_and_keep_mask_explicit(self):
        import torch
        rgb=torch.tensor([[1.,2.],[3.,4.]])
        audio=torch.tensor([[5.,6.],[7.,8.]])
        present=torch.tensor([True,False])
        joint=readout_inputs(rgb,audio,present,arm='joint-mlp')
        self.assertEqual(joint.tolist(),[[1.,2.,5.,6.,1.],[3.,4.,7.,8.,0.]])
        self.assertEqual(readout_inputs(rgb,audio,present,arm='rgb-mlp').tolist(),
                         [[1.,2.,0.,0.,0.],[3.,4.,0.,0.,0.]])
        self.assertEqual(readout_inputs(rgb,audio,present,arm='audio-mlp').tolist(),
                         [[0.,0.,5.,6.,1.],[0.,0.,7.,8.,0.]])

    def test_mlp_arms_share_exact_initialization_and_all_head_parameters_have_gradient(self):
        import torch
        models=[make_head(9,arm=arm,seed=20261030) for arm in ('rgb-mlp','audio-mlp','joint-mlp')]
        for other in models[1:]:
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(models[0].parameters(),other.parameters())))
        model=models[0]
        loss=torch.nn.functional.cross_entropy(model(torch.randn(8,9)),torch.tensor([1,3]*4))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        linear=make_head(9,arm='joint-linear',seed=20261030)
        self.assertLess(sum(p.numel() for p in linear.parameters()),sum(p.numel() for p in model.parameters()))

    def test_cpu_readout_resume_matches_continuous_parameters_optimizer_and_rng(self):
        import torch
        from scripts.training.checkpoint import save_checkpoint,restore_checkpoint
        data=torch.arange(72.).reshape(8,9)/72
        labels=torch.tensor([1,3]*4)
        def train(model,opt,count):
            for _ in range(count):
                opt.zero_grad()
                loss=torch.nn.functional.cross_entropy(model(data),labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
        with tempfile.TemporaryDirectory() as folder:
            a=make_head(9,arm='joint-mlp',seed=7)
            ao=torch.optim.AdamW(a.parameters(),lr=.001)
            train(a,ao,2)
            save_checkpoint(Path(folder)/'cp',a,ao,step=2,identity={'task':'readout'})
            train(a,ao,2)
            b=make_head(9,arm='joint-mlp',seed=7)
            bo=torch.optim.AdamW(b.parameters(),lr=.001)
            restore_checkpoint(Path(folder)/'cp',b,bo,identity={'task':'readout'})
            train(b,bo,2)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
            for av,bv in zip(ao.state.values(),bo.state.values()):
                self.assertTrue(all(torch.equal(av[k],bv[k]) if torch.is_tensor(av[k]) else av[k]==bv[k] for k in av))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
