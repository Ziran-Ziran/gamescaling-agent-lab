"""CPU-only fixed-budget readouts on source-audited frozen visual features."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import re
import time

from scripts.training.checkpoint import restore_checkpoint,save_checkpoint
from scripts.training.action_head import LABELS
from scripts.training.data import require,sha256_file
from scripts.training.vision_features import POOL_VERSION,tensor_digest
from scripts.training.vision_readout import ARMS,FeatureNormalizer,head_fingerprint,make_head,readout_inputs
from scripts.training.vision_readout_data import load_training_cache,training_tensors


def validate_run_inputs(metadata,tensors):
    import torch
    encoder=metadata['encoder_identity']
    fingerprint=encoder.get('model',{}).get('fingerprint','')
    require(metadata['source_kinds']==['recorded'] and re.fullmatch('[0-9a-f]{64}',fingerprint)
            and encoder.get('pool_version')==POOL_VERSION,'recorded original visual features required')
    count=metadata['train_rows']
    require(type(count) is int and count>=4 and count%4==0
            and tensors['rgb'].ndim==2 and tensors['rgb'].shape==tensors['audio'].shape
            and tensors['rgb'].shape[0]==count*2 and tensors['labels'].shape==(count*2,)
            and tensors['present'].dtype==torch.bool and tensors['present'].shape==(count*2,),
            'complete augmented training feature dimensions required')
    labels=tensors['labels']
    require(torch.equal(tensors['rgb'][:count],tensors['rgb'][count:])
            and torch.equal(labels[:count],labels[count:])
            and bool(tensors['present'][:count].all()) and not bool(tensors['present'][count:].any())
            and bool((tensors['audio'][count:]==tensors['audio'][count]).all())
            and torch.bincount(labels[:count],minlength=5).tolist()==[0,count//2,0,count//2,0],
            'balanced aligned/missing pairs must preserve every training target')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--steps',type=int,choices=range(1,201),default=2)
    parser.add_argument('--save-every',type=int,choices=range(1,201),default=2)
    parser.add_argument('--seed',type=int,default=20261030)
    args=parser.parse_args()
    # Process-local CPU execution, before importing torch or loading any tensors.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from scripts.training.hf_backend import environment
    require(not torch.cuda.is_initialized() and not torch.cuda.is_available(),'readout training must remain CPU-only')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache,metadata=load_training_cache(args.features,args.dataset)
    tensors=training_tensors(cache)
    validate_run_inputs(metadata,tensors)
    rgb_norm=FeatureNormalizer.fit(tensors['rgb'])
    audio_norm=FeatureNormalizer.fit(tensors['audio'])
    inputs=readout_inputs(rgb_norm.transform(tensors['rgb']),audio_norm.transform(tensors['audio']),
                           tensors['present'],arm=args.arm)
    model=make_head(inputs.shape[1],arm=args.arm,seed=args.seed)
    initial=head_fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    paths=[Path(__file__),Path(__file__).with_name('vision_readout.py'),
           Path(__file__).with_name('vision_features.py'),Path(__file__).with_name('vision_readout_data.py'),
           Path(__file__).with_name('checkpoint.py'),Path(__file__).with_name('hf_backend.py'),
           Path(__file__).with_name('action_head.py')]
    identity={'arm':args.arm,'seed':args.seed,'feature_dim':tensors['rgb'].shape[1],'input_dim':inputs.shape[1],
              'feature_cache_sha256':metadata['features_sha256'],
              'feature_manifest_sha256':sha256_file(args.features/'manifest.json'),
              'encoder_identity':metadata['encoder_identity'],'source_manifests':metadata['source_manifests'],
              'preprocessing_code':metadata['preprocessing_code'],'classes':list(LABELS),
              'missing_feature_sha256':tensor_digest(cache['features'][cache['missing_key']]),
              'architecture':'Linear(input,5)' if args.arm=='joint-linear' else 'Linear(input,32)-GELU-Linear(32,5)',
              'initialization_fingerprint':initial,'optimizer':'AdamW','lr':.001,'weight_decay':.01,'clip':1.,
              'batch_size':len(inputs),'fixed_main_budget':200,'dtype':'float32','device':'cpu','threads':4,
              'normalizers':{k:{n:tensor_digest(v) for n,v in norm.state().items()}
                             for k,norm in (('rgb',rgb_norm),('audio',audio_norm))},
              'packages':environment()['packages'],'code':{p.name:sha256_file(p) for p in paths},
              'augmentation':'all TRAIN rows duplicated once with black STFT and present0'}
    step=restore_checkpoint(args.resume,model,optimizer,identity=identity) if args.resume else 0
    require(step<args.steps,'restored step must be below requested total')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    start_step=step
    config={'schema_version':1,'identity':identity,'environment':environment(),'source_sha256':metadata['source_sha256'],
            'start_step':step,'total_steps':args.steps,'started_utc':datetime.now(timezone.utc).isoformat(),
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'backbone_trained':False,'language_decoder_used':False,'source_train_rows':len(cache['samples']),
            'claim':'new task-specific readout of frozen vision; not original GameScaling action decoder/LoRA repair'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    model.train()
    while step<args.steps:
        began=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(inputs)
        loss=torch.nn.functional.cross_entropy(logits,tensors['labels'])
        require(bool(torch.isfinite(loss)),'nonfinite readout loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        require(bool(torch.isfinite(norm)),'nonfinite readout gradients')
        optimizer.step()
        step+=1
        correct=logits.detach().argmax(-1)==tensors['labels']
        present=tensors['present']
        record={'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm),
                'correct_present':int(correct[present].sum()),'correct_missing':int(correct[~present].sum()),
                'examples_present':int(present.sum()),'examples_missing':int((~present).sum()),
                'logits_before_update':logits.detach().tolist(),'step_ms':(time.perf_counter()-began)*1000}
        with (args.output/'steps.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='logits_before_update'}),flush=True)
        if step%args.save_every==0 or step==args.steps:
            save_checkpoint(args.output/f'checkpoint-{step:06}',model,optimizer,step=step,identity=identity)
    require(not torch.cuda.is_initialized(),'CPU head training unexpectedly initialized CUDA')
    bundle={'schema_version':1,'step':step,'identity':identity,'head_state':model.state_dict(),
            'rgb_normalizer':rgb_norm.state(),'audio_normalizer':audio_norm.state(),
            'missing_feature':cache['features'][cache['missing_key']].clone()}
    torch.save(bundle,args.output/'readout.pt')
    result={'schema_version':1,'identity':identity,'step':step,'start_step':start_step,'new_updates':step-start_step,
            'readout_sha256':sha256_file(args.output/'readout.pt'),'final_head_fingerprint':head_fingerprint(model),
            'elapsed_update_loop_seconds':time.perf_counter()-started,'cuda_initialized':torch.cuda.is_initialized(),
            'new_environment_interactions':0,'claim':config['claim']}
    (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
