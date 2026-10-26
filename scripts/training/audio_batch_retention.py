"""Secondary fresh RGB task retention for the same four final audio-study policies."""
import argparse
import json
from pathlib import Path

from scripts.training.audio_batch_eval import POLICIES,validate_cohort
from scripts.training.data import require,sha256_file
from scripts.training.retention_compare import MODEL_FINGERPRINT,MODEL_WEIGHTS
from scripts.training.rollout_compare import compare_loaded,read_run

SUITES={
    'primary':{'games':['reach','hazard'],'seed_start':40000,'episodes_per_game':16,'max_steps':1,
               'history_len':0,'decoder':'first-key','max_new_tokens':64},
    'secondary':{'games':['keydoor'],'seed_start':40000,'episodes_per_game':8,'max_steps':8,
                 'history_len':0,'decoder':'first-key','max_new_tokens':64}}


def validate_rgb_protocol(config,cohort,policy,suite):
    require(policy in POLICIES and suite in SUITES,'supported four-policy RGB protocol required')
    validate_cohort(cohort)
    require(all(config['arguments'][key]==value for key,value in SUITES[suite].items()),
            'fresh RGB retention protocol changed')
    require(config['model']['fingerprint']==MODEL_FINGERPRINT
            and config['model']['files']['model.safetensors']==MODEL_WEIGHTS,'actual RGB base model changed')
    require(config['adapter']==cohort['policies'][policy],'RGB policy adapter differs from final audio cohort')
    require(config['environment']['packages']==cohort['training']['paired-context']['identity']['packages'],
            'RGB inference dependencies differ from frozen training cohort')


def compare_matrix(runs,cohort_path,*,suite):
    require(set(runs)==set(POLICIES),'all four fixed RGB policies required')
    cohort=json.loads(Path(cohort_path).read_text())
    validate_cohort(cohort)
    loaded={policy:read_run(path) for policy,path in runs.items()}
    for policy,(config,_,_) in loaded.items():
        validate_rgb_protocol(config,cohort,policy,suite)
    original=loaded['original']
    comparisons={policy:compare_loaded(original,run) for policy,run in loaded.items() if policy!='original'}
    return {'schema_version':1,'suite':suite,'frozen_protocol':SUITES[suite],
            'cohort_sha256':sha256_file(cohort_path),'fixed_policy_identities':cohort['policies'],
            'actual_episodes':sum(len(run[1]) for run in loaded.values()),
            'actual_actions':sum(len(episode['frames']) for run in loaded.values() for episode in run[1]),
            'original_results':original[2],'compared_to_original':comparisons,
            'claim':'secondary RGB observation/task intervention on own toy rules; no checkpoint selection, method ranking or commercial/general retention'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--suite',choices=tuple(SUITES),required=True)
    for policy in POLICIES:
        parser.add_argument('--'+policy,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare_matrix({policy:getattr(args,policy.replace('-','_')) for policy in POLICIES},args.cohort,suite=args.suite)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({policy:{game:{key:row[key] for key in (
        'paired_mean_return_delta','retention_fraction_of_original_successes','original_success_to_failure')}
        for game,row in comparison['groups'].items()}
        for policy,comparison in report['compared_to_original'].items()},indent=2))
