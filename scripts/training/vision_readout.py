"""Small task-specific CPU decision heads on frozen original vision features.

These heads replace the decision readout in this diagnostic, not the official
language/action decoder. No head receives engine state or a teacher target.
"""
from dataclasses import dataclass
import hashlib
import json

from scripts.training.data import require

ARMS=('rgb-mlp','audio-mlp','joint-linear','joint-mlp')


@dataclass
class FeatureNormalizer:
    mean: object
    std: object

    @classmethod
    def fit(cls,training):
        import torch
        require(training.ndim==2 and training.shape[0]>0 and training.shape[1]>0
                and bool(torch.isfinite(training).all()), 'finite nonempty training features required')
        values=training.detach().float().cpu()
        return cls(values.mean(dim=0),values.std(dim=0,unbiased=False).clamp_min(1e-4))

    def transform(self,values):
        import torch
        require(values.ndim==2 and values.shape[1]==self.mean.numel() and bool(torch.isfinite(values).all()),
                'feature dimension or finiteness mismatch')
        return (values.detach().float().cpu()-self.mean)/self.std

    def state(self):
        return {'mean':self.mean.clone(),'std':self.std.clone()}

    @classmethod
    def from_state(cls,state):
        import torch
        mean,std=state['mean'].detach().float().cpu(),state['std'].detach().float().cpu()
        require(mean.ndim==1 and mean.shape==std.shape and bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all()) and bool((std>=1e-4).all()), 'invalid saved feature normalizer')
        return cls(mean,std)


def readout_inputs(rgb,audio,present,*,arm):
    import torch
    require(arm in ARMS and rgb.ndim==2 and rgb.shape==audio.shape
            and present.shape==(rgb.shape[0],) and present.dtype==torch.bool
            and bool(torch.isfinite(rgb).all()) and bool(torch.isfinite(audio).all()),
            'finite matching RGB/audio features and explicit boolean mask required')
    require(rgb.device.type==audio.device.type==present.device.type=='cpu', 'diagnostic readout is CPU-only')
    if arm=='rgb-mlp':
        audio=torch.zeros_like(audio)
        present=torch.zeros_like(present)
    if arm=='audio-mlp':
        rgb=torch.zeros_like(rgb)
    return torch.cat((rgb,audio,present.float().unsqueeze(1)),dim=1)


def make_head(input_dim,*,arm,seed=20261030):
    import torch
    require(arm in ARMS and type(input_dim) is int and input_dim>0 and type(seed) is int,
            'valid readout dimensions/arm/seed required')
    # Fork only CPU RNG. Equal-shaped MLP arms start with identical parameters,
    # and construction does not silently alter a restored global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=(torch.nn.Linear(input_dim,5) if arm=='joint-linear' else
               torch.nn.Sequential(torch.nn.Linear(input_dim,32),torch.nn.GELU(),torch.nn.Linear(32,5)))
    return model.float().cpu()


def head_fingerprint(model):
    from scripts.training.vision_features import tensor_digest
    tensors={name:tensor_digest(value) for name,value in model.named_parameters()}
    return hashlib.sha256(json.dumps(tensors,sort_keys=True).encode()).hexdigest()


class ReadoutPolicy:
    def __init__(self,model,rgb_normalizer,audio_normalizer,*,arm):
        require(arm in ARMS,'invalid readout policy arm')
        self.model=model.eval().requires_grad_(False)
        self.rgb_normalizer,self.audio_normalizer=rgb_normalizer,audio_normalizer
        self.arm=arm

    def predict(self,rgb_feature,audio_feature,present):
        import torch
        from scripts.training.vision_features import tensor_digest
        require(type(present) is bool and rgb_feature.ndim==audio_feature.ndim==1,
                'numeric observable feature vectors and boolean presence required')
        with torch.inference_mode():
            rgb=self.rgb_normalizer.transform(rgb_feature.unsqueeze(0))
            audio=self.audio_normalizer.transform(audio_feature.unsqueeze(0))
            inputs=readout_inputs(rgb,audio,torch.tensor([present],dtype=torch.bool),arm=self.arm)
            logits=self.model(inputs)[0]
        return logits,{'arm':self.arm,'present':present,
                       'rgb_feature_sha256':tensor_digest(rgb_feature),
                       'audio_feature_sha256':tensor_digest(audio_feature),
                       'head_input_sha256':tensor_digest(inputs)}
