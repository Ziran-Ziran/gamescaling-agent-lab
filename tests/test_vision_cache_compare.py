import copy
import importlib.util
import unittest

from scripts.training.vision_cache_compare import audit_cache_trace


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class CacheTraceTests(unittest.TestCase):
    def trace(self):
        import torch
        from PIL import Image
        from scripts.training.spectral_cache import SpectralFeatureCache
        from scripts.training.vision_features import image_digest,tensor_digest
        class Encoder:
            identity={'model':'fixture'}
            def encode(self,image):
                feature=torch.tensor([float(image.getpixel((0,0))[0])])
                return feature,{'input_pixel_sha256':image_digest(image),'feature_sha256':tensor_digest(feature),
                                'image_count':1,'total_ms':1.}
        encoder=Encoder()
        cache=SpectralFeatureCache(encoder,capacity=2)
        rows=[]
        for value in (10,20,10,30):
            _,record=cache.encode(Image.new('RGB',(2,2),(value,0,0)))
            rows.append({'arm':'cached','evidence':{'inputs':{'present':True,
                'actual_visual_calls':1+record['actual_visual_calls']},'encoder_records':{'audio':record}}})
        return rows,encoder.identity,cache.stats()

    def test_reconstructs_lru_counters_and_derives_encoder_calls(self):
        rows,identity,stats=self.trace()
        report=audit_cache_trace(rows,identity,capacity=2)
        self.assertEqual(report['cache'],stats)
        self.assertEqual(report['expected_visual_calls'],[2,2,1,2])

    def test_forged_hit_feature_counter_or_causal_mask_rejected(self):
        rows,identity,_=self.trace()
        for kind in ('hit','feature','counter','causal'):
            changed=copy.deepcopy(rows)
            audio=changed[0]['evidence']['encoder_records']['audio']
            if kind=='hit': audio['cache_hit']=True
            if kind=='feature': changed[2]['evidence']['encoder_records']['audio']['feature_sha256']='0'*64
            if kind=='counter': audio['cache_after']['misses']+=1
            if kind=='causal': changed[0]['evidence']['inputs']['present']=False
            with self.assertRaises(ValueError):
                audit_cache_trace(changed,identity,capacity=2)


if __name__=='__main__':
    unittest.main()
