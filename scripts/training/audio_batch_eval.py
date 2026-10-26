"""Fresh frozen control suites for the matched original-model audio LoRA study."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from controller.control_core import to_action_text
from scripts.training.action_head import choose_class,forward_action_logits,template_action
from scripts.training.audio_batch_train import image_processor_identity
from scripts.training.audio_cue import AudioCueGame
from scripts.training.audio_cue_data import audit_audio_episodes,record_episode
from scripts.training.audio_vision import prepare_audio
from scripts.training.audio_vision_eval import CONDITIONS,condition_input,_status
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import RGBBackend,adapter_identity,environment
from scripts.training.retention_compare import ADAPTER_IDENTITIES,MODEL_FINGERPRINT

POLICIES=('original','legacy128','paired-context','balanced-unpaired')


def frozen_schedule(suite):
    require(suite in ('transfer','iid'),'new frozen transfer/iid suite required')
    variant,start=('hazard',35000) if suite=='transfer' else ('reach',36000)
    return [{'variant':variant,'context':context,'colors':colors,'tone':tone}
            for context in range(start,start+4) for colors in (0,1) for tone in (0,1)]


def validate_cohort(value):
    require(value['schema_version']==1 and value['model_fingerprint']==MODEL_FINGERPRINT
            and set(value['policies'])==set(POLICIES),'all four frozen policies/base required')
    policies=value['policies']
    require(policies['original'] is None and policies['legacy128']==ADAPTER_IDENTITIES['audio-lora128'],
            'original/legacy adapter changed')
    require(set(value['training'])=={'paired-context','balanced-unpaired'},'both complete trained arms required')
    identities=[]
    exposures=[]
    for arm in ('paired-context','balanced-unpaired'):
        adapter=policies[arm]
        files=adapter['files']
        require(set(files)=={'adapter_config.json','adapter_model.safetensors'} and all(
            isinstance(digest,str) and len(digest)==64 and all(char in '0123456789abcdef' for char in digest)
            for digest in files.values()) and adapter['fingerprint']==hashlib.sha256(
                json.dumps(files,sort_keys=True).encode()).hexdigest(),'new adapter full identity differs')
        training=value['training'][arm]
        require(training['audited_adapter_identity']==adapter and training['export_vs_checkpoint_all24_exact'] is True,
                'cohort policy export differs from audited actual training checkpoint')
        identity=training['identity']
        require(identity['arm']==arm and identity['model']==MODEL_FINGERPRINT,'training arm/model changed')
        identities.append({key:item for key,item in identity.items() if key!='arm'})
        actual=training['actual_training']
        require(actual['updates']==200 and actual['label_uses']==800 and actual['missing_label_uses']==100
                and actual['label_counts']=={'present':{'1':350,'3':350},'missing':{'1':50,'3':50}},
                'complete frozen200x4 balanced exposures required')
        exposure=actual['per_sample_exposure']
        require(all(type(row['missing']) is bool and type(row['count']) is int and row['count']>0 for row in exposure)
                and len({row['sample_id'] for row in exposure})==64
                and len({(row['sample_id'],row['missing']) for row in exposure})==len(exposure)
                and sum(row['count'] for row in exposure)==800
                and sum(row['count'] for row in exposure if row['missing'])==100,'complete per-sample exposures required')
        exposures.append(exposure)
    require(identities[0]==identities[1],'matched initialization/training configuration differs')
    require(exposures[0]==exposures[1],'per-sample present/missing exposure differs between trained arms')


def source_paths():
    names=('audio_batch_eval.py','audio_batch_train.py','audio_batch.py','audio_vision_eval.py',
           'audio_vision.py','audio_cue.py','audio_cue_data.py','audio.py','action_head.py',
           'hf_backend.py','headless.py','feedback.py','data.py','formatting.py','metrics.py','rollout.py',
           'retention_compare.py','rollout_compare.py')
    return [*(Path(__file__).with_name(name) for name in names),Path(__file__).resolve().parents[2]/'controller/control_core.py']


def summarize(episodes,audit,root):
    frames=[frame for episode in episodes for frame in episode['frames']]
    return {'episodes':len(episodes),'actions':len(frames),'successes':audit['successes'],
        'mean_return':statistics.mean(row['return'] for row in audit['outcomes']),
        'mean_steps':len(frames)/len(episodes),
        'inference_ms_p50':statistics.median(frame['policy']['inference_ms'] for frame in frames),
        'observation_to_action_start_ms_p50':statistics.median(
            (frame['actual_action']['start_ns']-frame['captured_ns'])/1e6 for frame in frames),
        'observation_to_action_end_ms_p50':statistics.median(
            (frame['actual_action']['end_ns']-frame['captured_ns'])/1e6 for frame in frames),
        'sensor_start_to_action_end_ms_p50':statistics.median(
            (frame['actual_action']['end_ns']-frame['audio']['window_start_ns'])/1e6 for frame in frames),
        'present_inputs':sum(frame['policy']['inputs']['present'] for frame in frames),
        'episodes_sha256':sha256_file(root/'episodes.jsonl')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    parser.add_argument('--suite',choices=('transfer','iid'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cohort=json.loads(args.cohort.read_text())
    validate_cohort(cohort)
    require(adapter_identity(args.adapter)==cohort['policies'][args.policy],'requested adapter differs from frozen cohort')
    args.output.mkdir(parents=True,exist_ok=False)
    schedule=frozen_schedule(args.suite)
    status={'state':'loading','planned_episodes':128,'completed_episodes':0,'failed_episodes':0,
            'active_episode':None,'started_utc':datetime.now(timezone.utc).isoformat()}
    _status(args.output,status)
    try:
        backend=RGBBackend(args.model,adapter=args.adapter)
        require(backend.identity['fingerprint']==cohort['model_fingerprint'],'actual base model differs')
        require(environment()['packages']==cohort['training']['paired-context']['identity']['packages'],
                'evaluation model/adapter dependency versions changed from training')
        processor=image_processor_identity(backend.processor)
        require(processor==cohort['training']['paired-context']['identity']['image_processor'],
                'evaluation image processor/torchvision changed from training')
        config={'schema_version':1,'model':backend.identity,'adapter':backend.adapter_identity,
                'environment':environment(),'cohort_sha256':sha256_file(args.cohort),'policy':args.policy,
                'suite':args.suite,'conditions':list(CONDITIONS),'schedule':schedule,
                'image_processor':processor,
                'spectrogram_packages':{name:importlib.metadata.version(name) for name in ('numpy','Pillow')},
                'decoder':'argmax5class first-key safe template,0generated/1predicted','history_len':0,
                'learning_during_evaluation':False,'source_code_sha256':{p.name:sha256_file(p) for p in source_paths()}}
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        reports={}
        status['state']='running'
        for condition in CONDITIONS:
            root=args.output/condition
            root.mkdir()
            episodes=[]
            for spec in schedule:
                game=AudioCueGame(**spec)
                status['active_episode']=f'{condition}/{game.base.episode_id}'
                _status(args.output,status)
                def policy(row):
                    changed,flags,intervention=condition_input(row,root,condition,spec)
                    batch,inputs=prepare_audio(backend,changed,root,root,**flags)
                    backend.synchronize()
                    began=time.perf_counter()
                    with backend.torch.inference_mode():
                        raw,candidates=forward_action_logits(backend,batch)
                        values=raw[candidates].float().tolist()
                    backend.synchronize()
                    inference_ms=(time.perf_counter()-began)*1000
                    label,details=choose_class(values)
                    return {'completion':to_action_text(template_action(label)),
                            'evidence':{'condition':condition,'sensor_intervention':intervention,'inputs':inputs,
                                        'class_record':details,'selected_class':label,'generated_tokens':0,
                                        'predicted_tokens':1,'inference_ms':inference_ms}}
                episode=record_episode(game,root,policy=policy)
                episodes.append(episode)
                with (root/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(episode)+'\n')
                status.update(completed_episodes=status['completed_episodes']+1,active_episode=None)
                _status(args.output,status)
            audit=audit_audio_episodes(episodes,root)
            reports[condition]=summarize(episodes,audit,root)
            print(json.dumps({'condition':condition,**reports[condition]}),flush=True)
        summary={'results':reports,'peak_cuda_allocated_mib':backend.torch.cuda.max_memory_allocated()/1024**2,
                 'finished_utc':datetime.now(timezone.utc).isoformat(),
                 'claim':'new fixed own virtual-sensor controls; not native audio/commercial/general capability'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        status['state']='completed'
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__,
                      failed_episodes=int(status['active_episode'] is not None))
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        _status(args.output,status)


if __name__=='__main__':
    main()
