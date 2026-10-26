"""Fresh actual-action controls for separate heads on a frozen visual backbone."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time

from controller.control_core import to_action_text
from scripts.training.action_head import LABELS,choose_class,template_action
from scripts.training.audio import read_audio
from scripts.training.audio_cue import AudioCueGame,spectral_image
from scripts.training.audio_cue_data import audit_audio_episodes,record_episode
from scripts.training.audio_vision_compare import audit_policy_inputs
from scripts.training.audio_vision_eval import CONDITIONS,condition_input
from scripts.training.data import require,sha256_file
from scripts.training.hf_backend import RGBBackend,environment
from scripts.training.vision_features import FrozenVisionEncoder,image_digest,tensor_digest
from scripts.training.vision_readout import FeatureNormalizer,ReadoutPolicy,head_fingerprint,make_head


def readout_schedule(*,suite='transfer',pilot=False):
    require(suite in ('transfer','iid'),'unsupported readout confirmation suite')
    if pilot:
        return [{'variant':'reach','context':20000,'colors':0,'tone':0}]
    variant,start=('hazard',25000) if suite=='transfer' else ('reach',26000)
    return [{'variant':variant,'context':context,'colors':colors,'tone':tone}
            for context in range(start,start+8) for colors in (0,1) for tone in (0,1)]


def load_readout(directory):
    import torch
    directory=Path(directory)
    metadata=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    require(metadata['readout_sha256']==sha256_file(directory/'readout.pt'),'readout content hash mismatch')
    require(metadata['step']==200,'formal inference requires fixed 200-step readout')
    bundle=torch.load(directory/'readout.pt',map_location='cpu',weights_only=True)
    require(bundle['schema_version']==1 and bundle['step']==200 and bundle['identity']==metadata['identity'],
            'readout artifact identity mismatch')
    identity=bundle['identity']
    for code in (identity['code'],identity['preprocessing_code']):
        for name,digest in code.items():
            path=(Path(__file__).resolve().parents[2]/'controller/control_core.py' if name=='control_core.py'
                  else Path(__file__).with_name(name))
            require(sha256_file(path)==digest,f'readout source changed: {name}')
    require(identity['classes']==list(LABELS) and identity['input_dim']==identity['feature_dim']*2+1,
            'readout class/feature dimensions changed')
    normalizers={}
    for key in ('rgb','audio'):
        state=bundle[key+'_normalizer']
        require({name:tensor_digest(value) for name,value in state.items()}==identity['normalizers'][key],
                'saved training normalization hash mismatch')
        normalizers[key]=FeatureNormalizer.from_state(state)
        require(normalizers[key].mean.numel()==identity['feature_dim'],'saved normalizer feature dimensions changed')
    missing=bundle['missing_feature']
    require(missing.shape==(identity['feature_dim'],) and tensor_digest(missing)==identity['missing_feature_sha256'],
            'missing visual placeholder differs from frozen training encoder')
    model=make_head(identity['input_dim'],arm=identity['arm'],seed=identity['seed'])
    model.load_state_dict(bundle['head_state'],strict=True)
    require(head_fingerprint(model)==metadata['final_head_fingerprint'],'readout head parameter fingerprint mismatch')
    return ReadoutPolicy(model,normalizers['rgb'],normalizers['audio'],arm=identity['arm']),metadata,missing


def prepare_features(encoder,row,root,missing,*,force_missing=False,rgb_only=False,rgb_blank=False):
    from PIL import Image
    began=time.perf_counter()
    observation=read_audio({'present':False,'missing_reason':'controlled_mask'} if force_missing or rgb_only else row['audio'],
                           root,captured_ns=row['captured_ns'],clock_id=row['clock_id'])
    with Image.open(Path(root)/row['rgb']['path']) as source:
        rgb=source.convert('RGB')
    if rgb_blank:
        rgb=Image.new('RGB',rgb.size,(0,0,0))
    spectral=spectral_image(observation)
    rgb_feature,rgb_record=encoder.encode(rgb)
    if observation['present']:
        audio_feature,audio_record=encoder.encode(spectral)
    else:
        audio_feature=missing
        audio_record={'feature_sha256':tensor_digest(missing),'input_pixel_sha256':image_digest(spectral),
                      'reused_fixed_missing_placeholder':True}
    inputs={'present':observation['present'],'missing_reason':observation['missing_reason'],
            'image_count':1 if rgb_only else 2,'audio_sha256':observation.get('sha256'),
            'input_rgb_pixel_sha256':image_digest(rgb),
            'input_spectral_pixel_sha256':None if rgb_only else image_digest(spectral),
            'prompt_sha256':None,'input_tokens':0,'model_text_used':False,
            'actual_visual_calls':2 if observation['present'] else 1,
            'feature_prepare_ms':(time.perf_counter()-began)*1000}
    return rgb_feature,audio_feature,inputs,{'rgb':rgb_record,'audio':audio_record}


def _status(root,value):
    temporary=root/'run-status.tmp'
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(root/'run-status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--readout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true',help='one training reset, excluded from confirmation')
    parser.add_argument('--suite',choices=('transfer','iid'),default='transfer')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    schedule=readout_schedule(suite=args.suite,pilot=args.pilot)
    conditions=['aligned'] if args.pilot else list(CONDITIONS)
    status={'state':'started','planned_episodes':len(schedule)*len(conditions),'completed_episodes':0,
            'failed_episodes':0,'active_episode':None,'started_utc':datetime.now(timezone.utc).isoformat()}
    _status(args.output,status)
    try:
        import torch
        torch.set_num_threads(4)
        policy_head,metadata,missing=load_readout(args.readout)
        encoder=FrozenVisionEncoder(RGBBackend(args.model))
        require(encoder.identity==metadata['identity']['encoder_identity'],'readout frozen backbone/preprocessor changed')
        config={'schema_version':1,'model':encoder.identity,'readout':metadata,'environment':environment(),
                'schedule':schedule,'conditions':conditions,'suite':args.suite,'pilot':args.pilot,'model_text_used':False,
                'missing_placeholder':'frozen encoder black STFT feature stored in trained artifact; presentmask separate',
                'source_code_sha256':{p.name:sha256_file(p) for p in (
                    Path(__file__),*(Path(__file__).with_name(n) for n in (
                    'vision_features.py','vision_readout.py','audio.py','audio_cue.py','audio_cue_data.py',
                    'audio_vision_compare.py','audio_vision_eval.py','headless.py','feedback.py','hf_backend.py',
                    'data.py','formatting.py','metrics.py','rollout.py','action_head.py')),Path('controller/control_core.py'))},
                'claim':'new task-specific CPU head on frozen original visual features; no language/action decoder or native audio'}
        (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        reports={}
        status['state']='running'
        for condition in conditions:
            root=args.output/condition
            root.mkdir()
            episodes=[]
            for spec in schedule:
                game=AudioCueGame(**spec)
                status['active_episode']=f'{condition}/{game.base.episode_id}'
                _status(args.output,status)
                def policy(row):
                    changed,flags,intervention=condition_input(row,root,condition,spec)
                    rgb,audio,inputs,encoded=prepare_features(encoder,changed,root,missing,**flags)
                    began=time.perf_counter()
                    logits,head_record=policy_head.predict(rgb,audio,inputs['present'])
                    head_ms=(time.perf_counter()-began)*1000
                    label,details=choose_class(logits.float().tolist())
                    persisted=time.perf_counter()
                    feature_path=root/(Path(row['rgb']['path']).stem+'.features.pt')
                    require(not feature_path.exists(),'readout feature evidence already exists')
                    encoder.backend.torch.save({'rgb':rgb,'audio':audio,'present':inputs['present'],
                        'logits':logits.detach().cpu(),'head_record':head_record},feature_path)
                    feature_record={'path':feature_path.name,'sha256':sha256_file(feature_path)}
                    persist_ms=(time.perf_counter()-persisted)*1000
                    return {'completion':to_action_text(template_action(label)),
                            'evidence':{'condition':condition,'sensor_intervention':intervention,'inputs':inputs,
                                        'class_record':details,'selected_class':label,'generated_tokens':0,
                                        'predicted_tokens':0,'predicted_classes':1,
                                        'inference_ms':inputs['feature_prepare_ms']+head_ms,'head_ms':head_ms,
                                        'encoder_records':encoded,'head_record':head_record,
                                        'feature_file':feature_record,'feature_persist_ms':persist_ms}}
                episode=record_episode(game,root,policy=policy)
                audit_policy_inputs(episode,root,condition,prediction_kind='feature-readout')
                episodes.append(episode)
                with (root/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(episode)+'\n')
                status.update(completed_episodes=status['completed_episodes']+1,active_episode=None)
                _status(args.output,status)
                print(json.dumps({'condition':condition,'episode':episode['episode_id'],
                                  'return':sum(f['reward'] for f in episode['frames']),
                                  'reason':episode['frames'][-1]['terminal_reason']}),flush=True)
            audit=audit_audio_episodes(episodes,root,require_pairs=not args.pilot)
            frames=[f for e in episodes for f in e['frames']]
            reports[condition]={'episodes':len(episodes),'actions':len(frames),'successes':audit['successes'],
                'mean_return':statistics.mean(v['return'] for v in audit['outcomes']),'mean_steps':len(frames)/len(episodes),
                'inference_ms_p50':statistics.median(f['policy']['inference_ms'] for f in frames),
                'observation_to_action_start_ms_p50':statistics.median((f['actual_action']['start_ns']-f['captured_ns'])/1e6 for f in frames),
                'observation_to_action_end_ms_p50':statistics.median((f['actual_action']['end_ns']-f['captured_ns'])/1e6 for f in frames),
                'sensor_start_to_action_end_ms_p50':statistics.median((f['actual_action']['end_ns']-f['audio']['window_start_ns'])/1e6 for f in frames),
                'present_inputs':sum(f['policy']['inputs']['present'] for f in frames),
                'episodes_sha256':sha256_file(root/'episodes.jsonl')}
        summary={'results':reports,'peak_cuda_allocated_mib':encoder.backend.torch.cuda.max_memory_allocated()/1024**2,
                 'finished_utc':datetime.now(timezone.utc).isoformat(),'claim':config['claim']}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        status['state']='completed'
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__,failed_episodes=int(status['active_episode'] is not None))
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        _status(args.output,status)


if __name__=='__main__':
    main()
