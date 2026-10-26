"""Frozen original visual features for an explicitly separate task readout.

This does not use the GameScaling language/action decoder and does not add a
native audio encoder. Each call receives one observable image, never a label.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.training.data import require, sha256_file

POOL_VERSION = 'merged-spatial-2x2-channel-major-float32-v1'


def package_versions():
    versions={}
    for name in ('torch','torchvision','transformers','Pillow'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return versions


def image_digest(image):
    image = image.convert('RGB')
    return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes()).hexdigest()


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    header = f'{value.dtype}:{tuple(value.shape)}:'.encode()
    return hashlib.sha256(header+value.view(torch.uint8).numpy().tobytes()).hexdigest()


def pool_tokens(tokens, grid, *, merge_size):
    import torch
    require(isinstance(grid,(list,tuple)) and len(grid)==3 and all(type(v) is int and v>0 for v in grid)
            and grid[0]==1 and type(merge_size) is int and merge_size>0,
            'verified single-image merged grid required')
    require(grid[1]%merge_size==0 and grid[2]%merge_size==0, 'image grid does not divide merge size')
    height,width=grid[1]//merge_size,grid[2]//merge_size
    require(tokens.ndim==2 and tokens.shape[0]==height*width and tokens.shape[1]>0
            and bool(torch.isfinite(tokens).all()), 'finite visual tokens must match merged image grid')
    # The vision merger returns row-major merged patch groups; retain coarse
    # spatial location instead of a global average that discards left/right.
    values=tokens.detach().float().cpu().reshape(height,width,-1).permute(2,0,1)
    return torch.nn.functional.adaptive_avg_pool2d(values,(2,2)).contiguous().flatten()


class FrozenVisionEncoder:
    def __init__(self,backend):
        require(backend.adapter_identity is None, 'feature diagnostic requires original backbone without LoRA')
        self.backend=backend
        self.backend.model.eval()
        self.backend.model.requires_grad_(False)
        self.merge_size=backend.model.config.vision_config.spatial_merge_size
        processor=backend.processor.image_processor
        self.identity={'model':backend.identity,'pool_version':POOL_VERSION,
                       'dtype':str(next(backend.model.parameters()).dtype),
                       'packages':package_versions(),
                       'image_processor_type':f'{type(processor).__module__}.{type(processor).__qualname__}',
                       'image_processor_config':processor.to_dict(),
                       'hf_backend_sha256':sha256_file(Path(__file__).with_name('hf_backend.py')),
                       'source_sha256':sha256_file(Path(__file__))}
        # ImageProcessor defaults can contain tuples. Use one canonical JSON
        # representation in both tensor bundles and their JSON manifests.
        self.identity=json.loads(json.dumps(self.identity,sort_keys=True,allow_nan=False))

    def encode(self,image):
        import torch
        image=image.convert('RGB')
        self.backend.synchronize()
        began=time.perf_counter()
        batch=self.backend.processor.image_processor(images=[image],return_tensors='pt').to(self.backend.device)
        self.backend.synchronize()
        prepared=time.perf_counter()
        require(batch['image_grid_thw'].shape==(1,3), 'one image per visual encoder call required')
        grid=batch['image_grid_thw'][0].tolist()
        with torch.inference_mode():
            output=self.backend.model.get_image_features(pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],return_dict=True)
        self.backend.synchronize()
        inferred=time.perf_counter()
        require(isinstance(output.pooler_output,(tuple,list)) and len(output.pooler_output)==1,
                'model must return one merged visual token sequence')
        feature=pool_tokens(output.pooler_output[0],grid,merge_size=self.merge_size)
        finished=time.perf_counter()
        return feature,{'input_pixel_sha256':image_digest(image),'feature_sha256':tensor_digest(feature),
                        'image_count':1,'image_grid_thw':grid,'feature_shape':list(feature.shape),
                        'prepare_ms':(prepared-began)*1000,'visual_ms':(inferred-prepared)*1000,
                        'pool_and_copy_ms':(finished-inferred)*1000,'total_ms':(finished-began)*1000}


if __name__ == '__main__':
    from PIL import Image
    from scripts.training.hf_backend import RGBBackend,environment
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    backend=RGBBackend(args.model)
    encoder=FrozenVisionEncoder(backend)
    with Image.open(args.image) as image:
        features,evidence=encoder.encode(image)
    backend.torch.save({'identity':encoder.identity,'feature':features,'evidence':evidence},args.output/'feature.pt')
    report={'identity':encoder.identity,'environment':environment(),'evidence':evidence,
            'image_sha256':sha256_file(args.image),'feature_file_sha256':sha256_file(args.output/'feature.pt'),
            'peak_cuda_allocated_mib':backend.torch.cuda.max_memory_allocated()/1024**2,
            'claim':'frozen visual-only feature extraction pilot; no language decoder or learning'}
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
