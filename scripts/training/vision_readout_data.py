"""Train-only frozen visual features with source-bound labels and missing pairs."""
import argparse
import json
from pathlib import Path

from scripts.training.audio import read_audio
from scripts.training.audio_cue import spectral_image
from scripts.training.audio_cue_data import load_audio_dataset
from scripts.training.data import require,sha256_file
from scripts.training.feedback_learning import class_from_action
from scripts.training.vision_features import FrozenVisionEncoder,image_digest,tensor_digest


def preprocessing_code():
    names=('vision_readout_data.py','vision_features.py','audio.py','audio_cue.py',
           'audio_cue_data.py','data.py','feedback.py','headless.py','hf_backend.py',
           'feedback_learning.py','metrics.py','action_head.py','formatting.py','rollout.py')
    paths=[Path(__file__).with_name(n) for n in names]+[Path(__file__).resolve().parents[2]/'controller/control_core.py']
    return {p.name:sha256_file(p) for p in paths}


def source_images(row,sensor,root):
    from PIL import Image
    with Image.open(Path(root)/row['rgb']['path']) as source:
        rgb=source.convert('RGB')
    audio=read_audio(sensor['audio'],root,captured_ns=sensor['captured_ns'],clock_id=sensor['clock_id'])
    require(audio['present'],'training feature source audio must be present')
    return rgb,spectral_image(audio)


def build_training_cache(dataset_path,encoder,output):
    import torch
    dataset,manifest,digests,index=load_audio_dataset(dataset_path)
    rows=dataset['train']
    require(all(r['game_id']=='audio-cue-reach-v1' and r['frame_id']==0 and not r['history'] for r in rows),
            'only initial own audio-reach TRAIN rows support this diagnostic')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    bank,evidence,samples={},{},[]
    def encoded(image):
        key=image_digest(image)
        if key not in bank:
            feature,record=encoder.encode(image)
            feature=feature.detach().float().cpu().contiguous()
            require(feature.ndim==1 and feature.numel()>0 and bool(torch.isfinite(feature).all()),'finite vector feature required')
            require(record['input_pixel_sha256']==key and record['feature_sha256']==tensor_digest(feature),
                    'encoder feature/input digest mismatch')
            bank[key],evidence[key]=feature,record
            with (output/'encoded-images.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
        return key
    missing_key=encoded(spectral_image({'present':False}))
    for row in rows:
        rgb,audio=source_images(row,index[row['sample_id']],manifest['data_root'])
        samples.append({'sample_id':row['sample_id'],'rgb_key':encoded(rgb),'audio_key':encoded(audio),
                        'target_class':class_from_action(row['action'])})
    require(len({tuple(t.shape) for t in bank.values()})==1,'all modalities require identical frozen visual feature shape')
    cache={'schema_version':1,'encoder_identity':encoder.identity,'source_manifests':digests,
           'features':bank,'feature_evidence':evidence,'samples':samples,'missing_key':missing_key,
           'preprocessing_code':preprocessing_code()}
    torch.save(cache,output/'features.pt')
    metadata={'schema_version':1,'features_sha256':sha256_file(output/'features.pt'),
              'source_manifests':digests,'source_sha256':manifest['source_sha256'],
              'source_kinds':manifest['source_kinds'],'train_rows':len(samples),'unique_encoded_images':len(bank),
              'encoder_identity':encoder.identity,'preprocessing_code':preprocessing_code(),
              'claim':'frozen original visual features only; training labels outside encoder; no learned readout yet'}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def load_training_cache(directory,dataset_path):
    import torch
    directory=Path(directory)
    metadata=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(metadata['schema_version']==1 and metadata['features_sha256']==sha256_file(directory/'features.pt'),
            'feature cache content hash mismatch')
    require(metadata['preprocessing_code']==preprocessing_code(),'feature preprocessing code changed')
    cache=torch.load(directory/'features.pt',map_location='cpu',weights_only=True)
    dataset,manifest,digests,index=load_audio_dataset(dataset_path)
    require(cache['schema_version']==1 and cache['source_manifests']==metadata['source_manifests']==digests
            and metadata['source_sha256']==manifest['source_sha256']
            and metadata['source_kinds']==manifest['source_kinds']
            and cache['encoder_identity']==metadata['encoder_identity']
            and cache['preprocessing_code']==metadata['preprocessing_code'], 'feature cache/source identity mismatch')
    expected=[]
    for row in dataset['train']:
        rgb,audio=source_images(row,index[row['sample_id']],manifest['data_root'])
        expected.append({'sample_id':row['sample_id'],'rgb_key':image_digest(rgb),'audio_key':image_digest(audio),
                         'target_class':class_from_action(row['action'])})
    require(cache['samples']==expected,'cached labels/input keys differ from actual training source')
    expected_missing=image_digest(spectral_image({'present':False}))
    require(cache['missing_key']==expected_missing,'missing sensor feature must represent blank STFT')
    keys={expected_missing}|{s[k] for s in expected for k in ('rgb_key','audio_key')}
    require(set(cache['features'])==set(cache['feature_evidence'])==keys
            and metadata['train_rows']==len(expected) and metadata['unique_encoded_images']==len(keys),
            'feature cache includes unknown/missing input images')
    for key,feature in cache['features'].items():
        require(feature.ndim==1 and feature.device.type=='cpu' and feature.dtype==torch.float32
                and bool(torch.isfinite(feature).all()) and cache['feature_evidence'][key]['input_pixel_sha256']==key
                and tensor_digest(feature)==cache['feature_evidence'][key]['feature_sha256'],
                'feature entry integrity mismatch')
    require(len({tuple(t.shape) for t in cache['features'].values()})==1,'feature shapes differ')
    return cache,metadata


def training_tensors(cache):
    import torch
    samples,bank=cache['samples'],cache['features']
    rgb=torch.stack([bank[s['rgb_key']] for s in samples])
    audio=torch.stack([bank[s['audio_key']] for s in samples])
    missing=bank[cache['missing_key']].unsqueeze(0).expand(len(samples),-1)
    return {'rgb':torch.cat((rgb,rgb)), 'audio':torch.cat((audio,missing)),
            'present':torch.tensor([True]*len(samples)+[False]*len(samples),dtype=torch.bool),
            'labels':torch.tensor([s['target_class'] for s in samples]*2,dtype=torch.long),
            'sample_ids':[s['sample_id'] for s in samples]*2}


if __name__=='__main__':
    from scripts.training.hf_backend import RGBBackend,environment
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    backend=RGBBackend(args.model)
    report=build_training_cache(args.dataset,FrozenVisionEncoder(backend),args.output)
    report={**report,'environment':environment(),'peak_cuda_allocated_mib':backend.torch.cuda.max_memory_allocated()/1024**2}
    (args.output/'extraction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
