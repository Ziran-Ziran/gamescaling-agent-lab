"""Bounded CPU visual-feature reuse after causal audio validation.

Keys contain exact observable image pixels and the full frozen encoder identity.
No waveform frequency, task label, reset or privileged target enters this cache.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import time

from scripts.training.audio import read_audio
from scripts.training.audio_cue import spectral_image
from scripts.training.data import require
from scripts.training.vision_features import image_digest,tensor_digest


def identity_text(identity):
    return json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)


class SpectralFeatureCache:
    def __init__(self,encoder,*,capacity=64):
        require(type(capacity) is int and 1<=capacity<=1024,'bounded positive cache capacity required')
        self.encoder,self.capacity=encoder,capacity
        self.identity=identity_text(encoder.identity)
        self.namespace=hashlib.sha256(self.identity.encode()).hexdigest()
        self.entries=OrderedDict()
        self.lookups=self.hits=self.misses=self.evictions=0

    def stats(self):
        return {'capacity':self.capacity,'entries':len(self.entries),'lookups':self.lookups,
                'hits':self.hits,'misses':self.misses,'evictions':self.evictions}

    def encode(self,image):
        import torch
        began=time.perf_counter()
        require(identity_text(self.encoder.identity)==self.identity,'frozen cache encoder identity changed')
        image=image.convert('RGB')
        pixels=image_digest(image)
        key=hashlib.sha256((self.namespace+':'+pixels).encode()).hexdigest()
        before=self.stats()
        self.lookups+=1
        hit=key in self.entries
        lookup_finished=time.perf_counter()
        if hit:
            self.hits+=1
            feature,origin=self.entries.pop(key)
            self.entries[key]=(feature,origin)
        else:
            self.misses+=1
            feature,origin=self.encoder.encode(image)
            require(feature.device.type=='cpu' and feature.dtype==torch.float32 and feature.ndim==1
                    and feature.numel()>0 and bool(torch.isfinite(feature).all()),'finite CPU float32 cache feature required')
            require(origin['input_pixel_sha256']==pixels and origin['feature_sha256']==tensor_digest(feature),
                    'cache miss encoder feature/pixel provenance mismatch')
            feature,origin=feature.detach().clone(),copy.deepcopy(origin)
            self.entries[key]=(feature,origin)
            if len(self.entries)>self.capacity:
                self.entries.popitem(last=False)
                self.evictions+=1
        clone_began=time.perf_counter()
        result=feature.clone()
        cloned=time.perf_counter()
        digest=tensor_digest(result)
        origin_copy=copy.deepcopy(origin)
        finished=time.perf_counter()
        return result,{'input_pixel_sha256':pixels,'feature_sha256':digest,
            'cache_key':key,'cache_namespace':self.namespace,'cache_lookup':True,'cache_hit':hit,
            'actual_visual_calls':0 if hit else 1,'cache_before':before,'cache_after':self.stats(),
            'lookup_ms':(lookup_finished-began)*1000,'clone_ms':(cloned-clone_began)*1000,
            'total_ms':(finished-began)*1000,'origin_record':origin_copy}


def feature_from_audio(record,root,*,captured_ns,clock_id,encoder,cache,missing_feature):
    """Validate causal availability before reading a file or touching a cache."""
    began=time.perf_counter()
    observation=read_audio(record,root,captured_ns=captured_ns,clock_id=clock_id)
    read_finished=time.perf_counter()
    image=spectral_image(observation)
    spectral_finished=time.perf_counter()
    if not observation['present']:
        feature=missing_feature.detach().float().cpu().clone()
        evidence={'input_pixel_sha256':image_digest(image),'feature_sha256':tensor_digest(feature),
            'cache_lookup':False,'cache_hit':False,'actual_visual_calls':0,
            'cache_before':cache.stats() if cache else None,'cache_after':cache.stats() if cache else None,
            'reused_fixed_missing_placeholder':True,'total_ms':0.}
    elif cache is not None:
        require(cache.encoder is encoder,'cache encoder instance differs from current request encoder')
        feature,evidence=cache.encode(image)
    else:
        feature,origin=encoder.encode(image)
        evidence={'input_pixel_sha256':origin['input_pixel_sha256'],'feature_sha256':origin['feature_sha256'],
            'cache_lookup':False,'cache_hit':False,'actual_visual_calls':1,
            'cache_before':None,'cache_after':None,'origin_record':origin,'total_ms':origin['total_ms']}
    feature_finished=time.perf_counter()
    evidence={**evidence,'feature_call_ms':(feature_finished-spectral_finished)*1000,
              'causal_read_ms':(read_finished-began)*1000,
              'spectral_ms':(spectral_finished-read_finished)*1000,
              'audio_feature_total_ms':(time.perf_counter()-began)*1000}
    return feature,observation,evidence
