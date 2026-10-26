import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training.vision_readout_data import build_training_cache,load_training_cache,training_tensors


class Clock:
    def __init__(self):
        self.now=1_000_000_000
    def ns(self):
        return self.now
    def sleep(self,seconds):
        self.now+=round(seconds*1e9)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class FeatureDataTests(unittest.TestCase):
    def test_cache_contains_only_train_and_missing_pairs_remain_balanced(self):
        import torch
        from scripts.training.audio_cue_data import collect
        from scripts.training.vision_features import image_digest,tensor_digest
        class Encoder:
            identity={'fake':'CPUtestfixture-not-model'}
            calls=[]
            def encode(self,image):
                self.calls.append(image_digest(image))
                feature=torch.tensor([image.getpixel((0,0))[0]/255.,image.getpixel((128,128))[0]/255.])
                return feature,{'input_pixel_sha256':image_digest(image),'feature_sha256':tensor_digest(feature)}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            collect(root/'source',contexts={'train':1,'validation':1,'test':1},clock_factory=Clock)
            encoder=Encoder()
            build_training_cache(root/'source/converted',encoder,root/'cache')
            cache,metadata=load_training_cache(root/'cache',root/'source/converted')
            self.assertEqual(len(cache['samples']),4)
            self.assertTrue(all('audio-cue-reach-v1' in r['sample_id'] for r in cache['samples']))
            self.assertEqual(len(encoder.calls),len(set(encoder.calls)))
            tensors=training_tensors(cache)
            self.assertEqual(tensors['rgb'].shape,(8,2))
            self.assertEqual(tensors['present'].tolist(),[True]*4+[False]*4)
            self.assertEqual(torch.bincount(tensors['labels'][4:],minlength=5).tolist(),[0,2,0,2,0])
            self.assertTrue(torch.equal(tensors['rgb'][:4],tensors['rgb'][4:]))
            self.assertTrue(all(torch.equal(tensors['audio'][4],r) for r in tensors['audio'][4:]))
            # Even after re-signing artifact bytes, target changes cannot override
            # the independently re-audited actual source actions.
            cache['samples'][0]['target_class']=0
            torch.save(cache,root/'cache/features.pt')
            from scripts.training.data import sha256_file
            manifest_path=root/'cache/manifest.json'
            metadata['features_sha256']=sha256_file(root/'cache/features.pt')
            manifest_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'actual training source'):
                load_training_cache(root/'cache',root/'source/converted')


if __name__=='__main__':
    unittest.main()
