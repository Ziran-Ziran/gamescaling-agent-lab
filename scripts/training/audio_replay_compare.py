"""Independent CPU controls and paired rewards for all four frozen audio policies."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.training.audio_replay_eval import source_paths
from scripts.training.audio_replay_protocol import AUDIO_POLICIES as POLICIES,CONDITIONS,frozen_schedule,validate_cohort
from scripts.training.audio_batch_train import image_processor_identity
from scripts.training.audio_replay_train import spectrogram_packages
from scripts.training.audio_cue import AudioCueGame
from scripts.training.audio_cue_data import audit_audio_episodes
from scripts.training.audio_vision_compare import audit_policy_inputs,require_complete,verify_summary
from scripts.training.audio_vision import prepare_audio
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.hf_backend import model_identity
from scripts.training.rollout_compare import audit_first_key,paired_outcomes


def paired_audio_results(before,after,*,check_inputs=True):
    require(len(before)==len(after)>0,'complete paired audio episodes required')
    for first,second in zip(before,after):
        require(first['episode_id']==second['episode_id'] and first['system_prompt']==second['system_prompt']
                and first['action_contract']==second['action_contract'],'paired reset/prompt/action contract differs')
        a,b=first['frames'][0],second['frames'][0]
        require(a['rgb']==b['rgb'] and a['audio']['sha256']==b['audio']['sha256']
                and a['feedback']['state_before']==b['feedback']['state_before'],'paired initial source RGB/audio/state differs')
        if check_inputs:
            require(all(a['policy']['inputs'][key]==b['policy']['inputs'][key] for key in (
                'present','missing_reason','image_count','audio_sha256','input_rgb_pixel_sha256',
                'input_spectral_pixel_sha256','prompt_sha256','input_tokens')),'paired actual sensor/prompt input differs')
    return paired_outcomes(before,after)


def cpu_backend(model,cohort):
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    require(spectrogram_packages()==cohort['training']['audio-only-control']['identity']['spectrogram_packages'],'CPU audit spectrogram dependencies changed')
    identity=model_identity(model)
    require(identity['fingerprint']==cohort['model_fingerprint'],'audit base model files changed')
    processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    require(image_processor_identity(processor)==cohort['training']['audio-only-control']['identity']['image_processor'],
            'CPU audit processor/torchvision differs')
    return SimpleNamespace(processor=processor,device='cpu',synchronize=lambda:None,identity=identity)


def read_run(directory,cohort_path,*,model,backend=None,input_cache=None):
    root=Path(directory).resolve()
    cohort=json.loads(Path(cohort_path).read_text())
    validate_cohort(cohort)
    backend=backend or cpu_backend(model,cohort)
    input_cache={} if input_cache is None else input_cache
    config=json.loads((root/'config.json').read_text())
    require(config['policy'] in POLICIES and config['cohort_sha256']==sha256_file(cohort_path),
            'evaluation policy/cohort file changed')
    require(config['model']==backend.identity
            and config['adapter']==cohort['policies'][config['policy']], 'evaluation actual model/adapter changed')
    require(config['environment']['packages']==cohort['training']['audio-only-control']['identity']['packages'] and config['spectrogram_packages']==cohort['training']['audio-only-control']['identity']['spectrogram_packages'],'actual evaluation inference/spectrogram dependencies differ')
    require(config['image_processor']==cohort['training']['audio-only-control']['identity']['image_processor'],
            'evaluation actual processor differs from training')
    schedule=frozen_schedule(config['suite'])
    require(config['schedule']==schedule and config['conditions']==list(CONDITIONS)
            and config['history_len']==0 and config['learning_during_evaluation'] is False
            and config['decoder']=='argmax5class first-key safe template,0generated/1predicted',
            'new frozen suite/conditions/decoder/history changed')
    require(config['source_code_sha256']=={path.name:sha256_file(path) for path in source_paths()},
            'actual evaluation code changed')
    require_complete(json.loads((root/'run-status.json').read_text()),40)
    summary=json.loads((root/'summary.json').read_text())
    require(set(summary['results'])==set(CONDITIONS),'all eight result groups required')
    result={}
    for condition in CONDITIONS:
        folder=root/condition
        episodes=list(read_jsonl(folder/'episodes.jsonl'))
        require([episode['cue_reset'] for episode in episodes]==schedule,'fresh evaluation episode schedule changed')
        audit=audit_audio_episodes(episodes,folder)
        for episode in episodes:
            default=AudioCueGame(**episode['cue_reset'])
            require(episode['source']['kind']=='recorded' and episode['source']['teacher_inputs']==[]
                    and episode['system_prompt']==default.system_prompt() and len(episode['frames'])==1
                    and episode['frames'][0]['feedback']['state_before']==default.base.snapshot(),
                    'actual default one-step policy reset/rules required')
            audit_policy_inputs(episode,folder,condition)
            for frame in episode['frames']:
                policy=frame['policy']
                audit_first_key({**frame,'policy':{**policy,'decoder_record':{
                    **policy['class_record'],'selected_class':policy['selected_class'],
                    'predicted_tokens':policy['predicted_tokens']}}})
                require(type(policy['inference_ms']) in (int,float) and math.isfinite(policy['inference_ms'])
                        and policy['inference_ms']>=0,'finite actual inference time required')
                record=policy['sensor_intervention']['input_audio_record']
                key=(condition,episode['system_prompt'],frame['rgb']['sha256'],record.get('sha256'))
                if key not in input_cache:
                    row={'system_prompt':episode['system_prompt'],'rgb':frame['rgb'],'audio':record,
                         'captured_ns':frame['captured_ns'],'clock_id':episode['clock']['id']}
                    _,inputs=prepare_audio(backend,row,folder,folder,
                        force_missing=condition=='missing',rgb_only=condition=='rgb-only',rgb_blank=condition=='rgb-blank')
                    input_cache[key]={k:v for k,v in inputs.items() if k!='prepare_ms'}
                require({k:v for k,v in policy['inputs'].items() if k!='prepare_ms'}==input_cache[key],
                        'CPU-reconstructed actual prompt/tokens/sensors differ')
        measured=verify_summary(summary['results'][condition],episodes,audit,sha256_file(folder/'episodes.jsonl'))
        result[condition]=(episodes,measured)
    return config,result


def compare_suite(runs,cohort_path,*,model):
    require(set(runs)==set(POLICIES),'all three fixed rehearsal policies required')
    cohort=json.loads(Path(cohort_path).read_text())
    validate_cohort(cohort)
    backend=cpu_backend(model,cohort)
    input_cache={}
    loaded={policy:read_run(path,cohort_path,model=model,backend=backend,input_cache=input_cache)
            for policy,path in runs.items()}
    original=loaded['parent200'][0]
    for policy,(config,_) in loaded.items():
        require(config['policy']==policy,'run path policy differs')
        require(all(config[key]==original[key] for key in (
            'model','environment','cohort_sha256','suite','conditions','schedule','image_processor',
            'spectrogram_packages','decoder','history_len','learning_during_evaluation','source_code_sha256')),
            'three-policy model/environment/observations/evaluation budget differs')
    groups={}
    for condition in CONDITIONS:
        policies={policy:loaded[policy][1][condition] for policy in POLICIES}
        paired={policy:paired_audio_results(policies['parent200'][0],rows[0])
                for policy,rows in policies.items() if policy!='parent200'}
        groups[condition]={'policies':{policy:rows[1] for policy,rows in policies.items()},
                           'versus_parent':paired,
                           'additional_rgb_objective':paired_audio_results(
                               policies['audio-only-control'][0],policies['rgb-replay'][0])}
    interventions={policy:{condition:paired_audio_results(loaded[policy][1]['aligned'][0],
        loaded[policy][1][condition][0],check_inputs=False) for condition in CONDITIONS if condition!='aligned'}
        for policy in POLICIES}
    return {'schema_version':1,'suite':original['suite'],'actual_episodes':120,
            'cohort_sha256':original['cohort_sha256'],'all_actual_input_action_reward_summary_audits':True,
            'groups':groups,'aligned_to_controls':interventions,
            'claim':'single-seed extra-RGB-objective study at common audio terms; unequal labels/compute, artificial paused task, not general ability'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    for policy in POLICIES:
        parser.add_argument('--'+policy,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare_suite({policy:getattr(args,policy.replace('-','_')) for policy in POLICIES},args.cohort,model=args.model)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({condition:{policy:row['successes'] for policy,row in group['policies'].items()}
                      for condition,group in report['groups'].items()},indent=2))
