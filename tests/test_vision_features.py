import importlib.util
from pathlib import Path
import unittest

from scripts.training.vision_features import FrozenVisionEncoder, pool_tokens, tensor_digest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch CPU runtime required')
class VisionFeatureTests(unittest.TestCase):
    def test_pool_keeps_spatial_quadrants_and_detaches_encoder_graph(self):
        import torch
        tokens = torch.arange(16., requires_grad=True).reshape(16,1)
        original = tokens.detach().clone()
        result = pool_tokens(tokens, [1,8,8], merge_size=2)
        self.assertEqual(result.tolist(), [2.5,4.5,10.5,12.5])
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.device.type, 'cpu')
        self.assertTrue(torch.equal(tokens.detach(),original))
        self.assertNotEqual(tensor_digest(result),tensor_digest(result+1))

    def test_pool_rejects_unverified_video_grid_shape_and_nonfinite_features(self):
        import torch
        for tokens,grid in ((torch.ones(4,2),[2,4,4]),(torch.ones(3,2),[1,4,4]),
                            (torch.full((4,2),float('nan')),[1,4,4])):
            with self.subTest(grid=grid),self.assertRaises(ValueError):
                pool_tokens(tokens,grid,merge_size=2)

    def test_encoder_calls_only_frozen_visual_features_and_receives_no_label_or_state(self):
        import torch
        from types import SimpleNamespace
        from PIL import Image
        class Batch(dict):
            def to(self,device):
                return self
        class ImageProcessor:
            def to_dict(self):
                return {'size':256,'image_mean':(.5,.5,.5),'image_std':(.5,.5,.5)}
            def __call__(self,*,images,return_tensors):
                self.images=images
                return Batch(pixel_values=torch.ones(1,3),image_grid_thw=torch.tensor([[1,4,4]]))
        class Processor:
            def __init__(self):
                self.image_processor=ImageProcessor()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(1))
                self.config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
                self.calls=[]
            def get_image_features(self,**kwargs):
                self.calls.append(set(kwargs))
                return SimpleNamespace(pooler_output=(torch.arange(8.).reshape(4,2),))
            def forward(self,*args,**kwargs):
                raise AssertionError('language decoder must not run')
        backend=SimpleNamespace(model=Model(),processor=Processor(),device='cpu',
                                identity={'fingerprint':'f'*64},adapter_identity=None,synchronize=lambda:None)
        encoder=FrozenVisionEncoder(backend)
        feature,evidence=encoder.encode(Image.new('RGB',(256,256)))
        self.assertEqual(feature.shape,(8,))
        self.assertFalse(backend.model.weight.requires_grad)
        self.assertFalse(backend.model.training)
        self.assertEqual(backend.model.calls,[{'pixel_values','image_grid_thw','return_dict'}])
        self.assertEqual(evidence['image_count'],1)
        self.assertEqual(evidence['feature_sha256'],tensor_digest(feature))
        self.assertEqual(len(backend.processor.image_processor.images),1)
        import json
        self.assertEqual(encoder.identity,json.loads(json.dumps(encoder.identity)))
        self.assertEqual(encoder.identity['image_processor_config']['image_mean'],[.5,.5,.5])
        self.assertIn('torchvision',encoder.identity['packages'])
        backend.adapter_identity={'unexpected':'adapter'}
        with self.assertRaises(ValueError):
            FrozenVisionEncoder(backend)


if __name__ == '__main__':
    unittest.main()
