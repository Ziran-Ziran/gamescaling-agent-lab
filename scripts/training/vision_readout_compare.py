"""CPU replay of saved visual vectors, trained heads and actual-action evidence."""
import argparse
import json
from pathlib import Path
import statistics

from scripts.training.action_head import LABELS
from scripts.training.audio_cue_data import audit_audio_episodes
from scripts.training.audio_vision_compare import audit_policy_inputs,require_complete,verify_summary
from scripts.training.audio_vision_eval import CONDITIONS
from scripts.training.data import read_jsonl,require,sha256_file
from scripts.training.vision_features import tensor_digest
from scripts.training.vision_readout import ARMS
from scripts.training.vision_readout_eval import load_readout,readout_schedule


def comparable_identity(identity):
    return {key:value for key,value in identity.items()
            if key not in ('arm','architecture','initialization_fingerprint')}


def audit_feature_file(evidence,root,policy,missing,feature_dim):
    import torch
    root=Path(root).resolve()
    reference=evidence['feature_file']
    path=(root/reference['path']).resolve()
    require(path.is_relative_to(root) and path.parent==root,'feature evidence path escapes condition root')
    require(sha256_file(path)==reference['sha256'],'feature file content hash mismatch')
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    require(set(bundle)=={'rgb','audio','present','logits','head_record'},'feature bundle fields changed')
    inputs=evidence['inputs']
    require(inputs['model_text_used'] is False and inputs['input_tokens']==0 and inputs['prompt_sha256'] is None,
            'feature readout must not receive model text')
    require(type(bundle['present']) is bool and bundle['present']==inputs['present'],'feature presence mismatch')
    require(inputs['actual_visual_calls']==(2 if bundle['present'] else 1),'actual visual call count differs from sensor mask')
    for key,pixels in (('rgb',inputs['input_rgb_pixel_sha256']),('audio',inputs['input_spectral_pixel_sha256'])):
        value=bundle[key]
        require(value.dtype==torch.float32 and value.shape==(feature_dim,) and bool(torch.isfinite(value).all()),
                'finite float32 feature vector dimensions changed')
        record=evidence['encoder_records'][key]
        require(tensor_digest(value)==record['feature_sha256'],'encoder feature digest mismatch')
        if pixels is not None:
            require(record['input_pixel_sha256']==pixels,'encoded feature source pixels differ')
    if not bundle['present']:
        require(torch.equal(bundle['audio'],missing),'masked audio must use frozen training placeholder feature')
    logits,record=policy.predict(bundle['rgb'],bundle['audio'],bundle['present'])
    require(record==bundle['head_record']==evidence['head_record'],'head input feature/hash record mismatch')
    require(torch.equal(logits,bundle['logits']),'saved logits differ from exact CPU trained-head replay')
    require(logits.tolist()==[evidence['class_record']['raw_class_logits'][label] for label in LABELS],
            'actual-action logits differ from saved head replay')
    return float((logits-bundle['logits']).abs().max())


def read_run(directory,readout):
    import torch
    torch.set_num_threads(4)
    directory=Path(directory).resolve()
    config=json.loads((directory/'config.json').read_text())
    policy,metadata,missing=load_readout(readout)
    require(config['readout']==metadata,'evaluation readout differs from supplied artifact')
    require(config['model']==metadata['identity']['encoder_identity'] and config['model_text_used'] is False,
            'evaluation frozen encoder/text contract mismatch')
    require(config['pilot'] is False and config['schedule']==readout_schedule(suite=config['suite'])
            and config['conditions']==list(CONDITIONS),'complete frozen confirmation schedule required')
    for name,digest in config['source_code_sha256'].items():
        path=(Path(__file__).resolve().parents[2]/'controller/control_core.py' if name=='control_core.py'
              else Path(__file__).with_name(name))
        require(sha256_file(path)==digest,f'evaluation source changed: {name}')
    require_complete(json.loads((directory/'run-status.json').read_text()),256)
    summary=json.loads((directory/'summary.json').read_text())
    require(set(summary['results'])==set(CONDITIONS),'unreported condition in readout summary')
    result={}
    for condition in CONDITIONS:
        root=directory/condition
        episodes=list(read_jsonl(root/'episodes.jsonl'))
        require([e['cue_reset'] for e in episodes]==config['schedule'],'actual reset order differs from frozen schedule')
        audit=audit_audio_episodes(episodes,root)
        for episode in episodes:
            require(episode['source']['teacher_inputs']==[],'expert rollout cannot replace policy evaluation')
            audit_policy_inputs(episode,root,condition,prediction_kind='feature-readout')
            for frame in episode['frames']:
                audit_feature_file(frame['policy'],root,policy,missing,metadata['identity']['feature_dim'])
        measured=verify_summary(summary['results'][condition],episodes,audit,sha256_file(root/'episodes.jsonl'))
        result[condition]=(episodes,measured)
    return config,result


def compare(runs):
    require(set(runs)==set(ARMS),'all four preregistered readout arms required')
    loaded={arm:read_run(*runs[arm]) for arm in ARMS}
    reference=loaded['joint-mlp'][0]
    identities={arm:config['readout']['identity'] for arm,(config,_) in loaded.items()}
    for arm,(config,_) in loaded.items():
        require(identities[arm]['arm']==arm,'run supplied under wrong readout arm')
        for key in ('model','environment','schedule','conditions','suite','pilot','model_text_used',
                    'missing_placeholder','source_code_sha256'):
            require(config[key]==reference[key],f'paired readout fairness mismatch: {key}')
        require(comparable_identity(identities[arm])==comparable_identity(identities['joint-mlp']),
                'readout training data/budget/normalization differs')
        if arm!='joint-linear':
            require(identities[arm]['initialization_fingerprint']==identities['joint-mlp']['initialization_fingerprint'],
                    'MLP arms must share exactly the same initialization')
    groups={}
    for condition in CONDITIONS:
        reference_episodes=loaded['joint-mlp'][1][condition][0]
        paired={}
        for arm,(_,data) in loaded.items():
            deltas=[]
            for expected,actual in zip(reference_episodes,data[condition][0]):
                require(expected['system_prompt']==actual['system_prompt'],'paired rules differ')
                a,b=expected['frames'][0],actual['frames'][0]
                require(a['rgb']==b['rgb'] and a['feedback']['state_before']==b['feedback']['state_before'],
                        'paired initial RGB/state differs')
                for key in ('present','missing_reason','image_count','audio_sha256','input_rgb_pixel_sha256',
                            'input_spectral_pixel_sha256','model_text_used','input_tokens','prompt_sha256','actual_visual_calls'):
                    require(a['policy']['inputs'][key]==b['policy']['inputs'][key],f'paired input differs: {key}')
                for key in ('rgb','audio'):
                    require(a['policy']['encoder_records'][key]['feature_sha256']==
                            b['policy']['encoder_records'][key]['feature_sha256'],'paired initial visual features differ')
                deltas.append(sum(f['reward'] for f in expected['frames'])-sum(f['reward'] for f in actual['frames']))
            paired[arm]={'joint_mlp_mean_return_delta':statistics.mean(deltas),
                         'joint_mlp_wins_ties_losses':[sum(d>1e-9 for d in deltas),
                             sum(abs(d)<=1e-9 for d in deltas),sum(d< -1e-9 for d in deltas)]}
        groups[condition]={'arms':{arm:data[condition][1] for arm,(_,data) in loaded.items()},'paired':paired}
    return {'schema_version':1,'suite':reference['suite'],'audited_actual_episodes':1024,
            'exact_cpu_head_replay':True,'maximum_logit_replay_difference':0.,
            'paired_training_sensor_feature_feedback_audits':True,'identities':identities,'groups':groups,
            'claim':'task-specific frozen-vision readout diagnostic; not original language decoder or LoRA repair'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--'+arm,nargs=2,metavar=('EVALUATION','READOUT'),required=True,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare({arm:getattr(args,arm.replace('-','_')) for arm in ARMS})
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({key:{arm:row['successes'] for arm,row in value['arms'].items()}
                      for key,value in report['groups'].items()},indent=2))
