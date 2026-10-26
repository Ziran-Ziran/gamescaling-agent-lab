import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

from scripts.training.spectral_cache import SpectralFeatureCache,feature_from_audio


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch CPU runtime required')
class SpectralCacheTests(unittest.TestCase):
    def encoder(self):
        import torch
        from scripts.training.vision_features import image_digest,tensor_digest
        class Encoder:
            identity={'model':'fixture','processor':{'size':256},'pool':'fixture'}
            calls=0
            def encode(self,image):
                self.calls+=1
                feature=torch.arange(4,dtype=torch.float32)+int(image_digest(image)[:4],16)
                return feature,{'input_pixel_sha256':image_digest(image),'feature_sha256':tensor_digest(feature),
                                'total_ms':1000.,'image_count':1}
        return Encoder()

    def test_pixel_key_hit_clone_and_current_timing_not_old_encode_time(self):
        import torch
        from PIL import Image
        encoder=self.encoder()
        cache=SpectralFeatureCache(encoder,capacity=2)
        image=Image.new('RGB',(2,2),(30,60,90))
        a,first=cache.encode(image)
        raw=io.BytesIO()
        image.save(raw,format='PNG',compress_level=0)
        with Image.open(io.BytesIO(raw.getvalue())) as same:
            b,second=cache.encode(same)
        self.assertEqual(encoder.calls,1)
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])
        self.assertLess(second['total_ms'],second['origin_record']['total_ms'])
        b.add_(100)
        c,_=cache.encode(image)
        self.assertTrue(torch.equal(a,c))
        self.assertEqual(cache.stats()['hits'],2)

    def test_bounded_lru_eviction_and_changed_model_identity_rejected(self):
        from PIL import Image
        encoder=self.encoder()
        cache=SpectralFeatureCache(encoder,capacity=2)
        images=[Image.new('RGB',(2,2),(value,0,0)) for value in (10,20,30)]
        cache.encode(images[0]); cache.encode(images[1]); cache.encode(images[0]); cache.encode(images[2])
        self.assertEqual(cache.stats()['evictions'],1)
        self.assertEqual(cache.stats()['entries'],2)
        _,record=cache.encode(images[1])
        self.assertFalse(record['cache_hit'])
        encoder.identity=copy.deepcopy(encoder.identity)
        encoder.identity['model']='different-model'
        with self.assertRaisesRegex(ValueError,'identity'):
            cache.encode(images[0])

    def test_future_cached_waveform_masks_before_lookup_and_valid_silence_stays_present(self):
        import torch
        from scripts.training.audio_cue import waveform_bytes
        from scripts.training.data import sha256_file
        encoder=self.encoder()
        cache=SpectralFeatureCache(encoder)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'tone.wav'
            path.write_bytes(waveform_bytes(26000,0))
            record={'present':True,'path':'tone.wav','sha256':sha256_file(path),'sample_rate':16000,
                'channels':1,'encoding':'pcm_s16le','clock_id':'clock','window_start_ns':1,
                'window_end_ns':200000001,'available_ns':200000001}
            missing=torch.zeros(4)
            _,observed,_=feature_from_audio(record,root,captured_ns=200000002,clock_id='clock',
                encoder=encoder,cache=cache,missing_feature=missing)
            self.assertTrue(observed['present'])
            stats=cache.stats()
            future={**record,'path':'does-not-exist.wav','window_start_ns':300000001,
                    'window_end_ns':500000001,'available_ns':500000001}
            value,observed,evidence=feature_from_audio(future,root,captured_ns=200000002,clock_id='clock',
                encoder=encoder,cache=cache,missing_feature=missing)
            self.assertFalse(observed['present'])
            self.assertEqual(observed['missing_reason'],'future_window')
            self.assertTrue(torch.equal(value,missing))
            self.assertEqual(cache.stats(),stats)
            self.assertEqual(evidence['actual_visual_calls'],0)
            path.write_bytes(waveform_bytes(26000,0,kind='silence'))
            record['sha256']=sha256_file(path)
            _,observed,evidence=feature_from_audio(record,root,captured_ns=200000002,clock_id='clock',
                encoder=encoder,cache=cache,missing_feature=missing)
            self.assertTrue(observed['present'])
            self.assertEqual(evidence['actual_visual_calls'],1)
            other=self.encoder()
            other.identity=copy.deepcopy(other.identity)
            other.identity['model']='other-encoder'
            with self.assertRaisesRegex(ValueError,'encoder'):
                feature_from_audio(record,root,captured_ns=200000002,clock_id='clock',
                    encoder=other,cache=cache,missing_feature=missing)


if __name__=='__main__':
    unittest.main()
