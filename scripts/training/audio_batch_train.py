"""Actual local0.8B LoRA with matched balanced four-example accumulation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import time

from scripts.training.action_head import forward_action_logits
from scripts.training.audio_batch import ARMS,batch_for_step,group_training_rows,optimize_microbatch
from scripts.training.audio_cue_data import load_audio_dataset
from scripts.training.audio_vision import prepare_audio
from scripts.training.audio_vision_train import training_source_paths
from scripts.training.bandit_train import trainable_fingerprint
from scripts.training.checkpoint import restore_checkpoint,save_checkpoint
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import RGBBackend,environment


def source_paths():
    return [Path(__file__),Path(__file__).with_name('audio_batch.py'),*training_source_paths()]


def image_processor_identity(processor):
    image=processor.image_processor
    value={'type':f'{type(image).__module__}.{type(image).__qualname__}',
           'config':image.to_dict(),'torchvision':importlib.metadata.version('torchvision')}
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def sensor_payload(row,audio_index):
    sensor=audio_index[row['sample_id']]
    return {'system_prompt':row['system_prompt'],'rgb':dict(row['rgb']),
            **{key:sensor[key] for key in ('audio','captured_ns','clock_id')}}


def write_status(root,status):
    temp=root/'run-status.tmp'
    temp.write_text(json.dumps(status,indent=2))
    temp.replace(root/'run-status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--steps',type=int,choices=range(1,201),default=2)
    parser.add_argument('--save-every',type=int,choices=range(1,101),default=25)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    status={'state':'loading','planned_total_updates':args.steps,'completed_total_updates':0,
            'active_sample':None,'started_utc':datetime.now(timezone.utc).isoformat()}
    write_status(args.output,status)
    try:
        dataset,manifest,digests,audio_index=load_audio_dataset(args.dataset)
        require(manifest['source_kinds']==['recorded'],'recorded authorized TRAIN sensors required')
        groups=group_training_rows(dataset['train'],list(read_jsonl(manifest['source_path'])))
        started=time.perf_counter()
        backend=RGBBackend(args.model,rank=8,learning_seed=20261031,deterministic=True)
        torch=backend.torch
        optimizer=torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad],
                                    lr=1e-4,weight_decay=.01)
        identity={'schema_version':1,'model':backend.identity['fingerprint'],'dataset':digests,
                  'arm':args.arm,'seed':20261031,'initialization_fingerprint':trainable_fingerprint(backend.model),
                  'rank':8,'alpha':16,'dropout':0.,'target_modules':['q_proj','v_proj'],
                  'dtype':'bfloat16','optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'clip':1.,
                  'temperature':2.,'effective_batch':4,'microbatch':1,'frozen_final_updates':200,
                  'sampling':'shared8context32row-block;lastcontextmissing;per-batch2A2D;v1',
                  'mask_dropout':'globalzero-based-update%8==7;all4samples;sharedsamplemaskmultiset',
                  'inputs':'currentRGB+causalSTFTimage+mask;nohistory/target/reset/engine/reward',
                  'image_processor':image_processor_identity(backend.processor),
                  'spectrogram_packages':{name:importlib.metadata.version(name) for name in ('numpy','Pillow')},
                  'determinism':backend.determinism,'packages':environment()['packages'],
                  'code':{path.name:sha256_file(path) for path in source_paths()}}
        step=restore_checkpoint(args.resume,backend.model,optimizer,identity=identity) if args.resume else 0
        require(step<args.steps,'resume step must precede requested total')
        start_step=step
        config={'schema_version':1,'identity':identity,'model':backend.identity,'environment':environment(),
                'dataset':str(args.dataset.resolve()),'resume':str(args.resume.resolve()) if args.resume else None,
                'source_sha256':manifest['source_sha256'],'start_step':step,'total_steps':args.steps,
                'trainable_parameters':sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
                'claim':'actual original0.8B first-key LoRA, matched grouping study on artificial WAV/RGB; not native audio or commercial ability'}
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        backend.model.train()
        status.update(state='running',completed_total_updates=step)
        write_status(args.output,status)
        update_seconds=0.
        while step<args.steps:
            rows,missing=batch_for_step(groups,arm=args.arm,step=step,seed=20261031)
            backend.synchronize()
            began=time.perf_counter()
            def forward(row):
                status['active_sample']=row['sample_id']
                batch,evidence=prepare_audio(backend,sensor_payload(row,audio_index),manifest['data_root'],
                                             args.output,force_missing=missing)
                require(evidence['input_rgb_pixel_sha256']==row['rgb']['pixel_sha256'],
                        'actual TRAIN RGB differs from audited source')
                raw,candidates=forward_action_logits(backend,batch)
                return raw[candidates].float(),evidence
            record=optimize_microbatch(backend.model,optimizer,rows,forward)
            backend.synchronize()
            elapsed=time.perf_counter()-began
            update_seconds+=elapsed
            step+=1
            record.update(step=step,mask_dropout=missing,
                          target_counts=dict(Counter(str(row['target_class']) for row in record['examples'])),
                          step_ms=elapsed*1000,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            with (args.output/'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            status.update(completed_total_updates=step,active_sample=None)
            write_status(args.output,status)
            print(json.dumps({'step':step,'loss':record['loss'],'gradient_norm':record['gradient_norm'],
                              'step_ms':record['step_ms'],'peak_mib':record['peak_cuda_allocated_mib']}),flush=True)
            if step%args.save_every==0 or step==args.steps:
                checkpoint=args.output/f'checkpoint-{step:06}'
                save_checkpoint(checkpoint,backend.model,optimizer,step=step,identity=identity)
                backend.model.save_pretrained(checkpoint/'adapter',safe_serialization=True)
        result={'completed_optimizer_steps':step,'start_step':start_step,'new_updates':step-start_step,
                'new_label_uses':4*(step-start_step),'elapsed_seconds':time.perf_counter()-started,
                'optimizer_updates_seconds':update_seconds,'checkpoint':f'checkpoint-{step:06}',
                'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'claim':config['claim']}
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        status['state']='completed'
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__)
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_status(args.output,status)


if __name__=='__main__':
    main()
