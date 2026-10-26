"""Fixed five-policy retention matrix, independently bound to a frozen protocol."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.training.data import require
from .rollout_compare import compare_loaded,read_run

MODEL_FINGERPRINT='ed679f4172207776e6aa8e63915c3645721f62a38aef7c1ce3fd82a96241c87c'
MODEL_WEIGHTS='d553c0796833691e32277a937db649eaac39bae9a0028a63c964c12a261c2dfa'
ADAPTERS={
    'original':None,
    'text-sft128':'cc8a220cb138a05971e7d593547c2a1553ecfe0f3fc42a740d773e22b57aa174',
    'class-sft128':'7d727dffcc16b1ec9d6c46b651fe12ac7f14259ff457bb4181dd64bc6f00311d',
    'bandit128':'749b147d9eb0c95913b22a569afb1c1c920495e58d89abb40cb2771a1e10d31d',
    'audio-lora128':'bfa8ec85542d284f3ef123b64aa3dfd18a09918e2581884d2992c76d6cf2be56'}
ADAPTER_CONFIGS={arm:('0682c3286ced6c1a0200c5b5d4df8c0468abcaef4f7b48d14ef01d43902e5911'
                     if arm=='class-sft128' else
                     '2dc33d71032002e5ed0d6000e5ec8555d1de33062a4570a9b4900168d1a8f316')
                 for arm in ADAPTERS if arm!='original'}
ADAPTER_IDENTITIES={'original':None}
for _arm,_config in ADAPTER_CONFIGS.items():
    _files={'adapter_config.json':_config,'adapter_model.safetensors':ADAPTERS[_arm]}
    ADAPTER_IDENTITIES[_arm]={'files':_files,'fingerprint':hashlib.sha256(json.dumps(_files,sort_keys=True).encode()).hexdigest()}
SUITES={
    'primary':{'games':['reach','hazard'],'seed_start':30000,'episodes_per_game':16,'max_steps':1,
               'history_len':0,'decoder':'first-key','max_new_tokens':64},
    'secondary':{'games':['keydoor'],'seed_start':30000,'episodes_per_game':8,'max_steps':8,
                 'history_len':0,'decoder':'first-key','max_new_tokens':64}}


def validate_protocol(config,arm,suite):
    require(arm in ADAPTERS and suite in SUITES,'unsupported frozen retention arm/suite')
    require(all(config['arguments'][key]==value for key,value in SUITES[suite].items()),
            'retention protocol seed/games/horizon/history/decoder changed')
    require(config['model']['fingerprint']==MODEL_FINGERPRINT
            and config['model']['files']['model.safetensors']==MODEL_WEIGHTS,'retention base model changed')
    adapter=config['adapter']
    require(adapter==ADAPTER_IDENTITIES[arm],'retention fixed adapter weights/configuration/fingerprint differ')


def compare_matrix(runs,*,suite):
    require(set(runs)==set(ADAPTERS),'all five fixed retention policies required')
    loaded={arm:read_run(path) for arm,path in runs.items()}
    for arm,(config,_,_) in loaded.items():
        validate_protocol(config,arm,suite)
    original=loaded['original']
    reports={arm:compare_loaded(original,run) for arm,run in loaded.items() if arm!='original'}
    return {'schema_version':1,'suite':suite,'frozen_protocol':SUITES[suite],
            'model':original[0]['model'],'fixed_adapter_identities':ADAPTER_IDENTITIES,
            'actual_episodes':sum(len(run[1]) for run in loaded.values()),
            'original_results':original[2],'compared_to_original':reports,
            'claim':'descriptive task retention on fresh own RGB toy resets; unequal training objectives/costs, not method superiority or commercial capability'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=tuple(SUITES),required=True)
    for arm in ADAPTERS:
        parser.add_argument('--'+arm,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare_matrix({arm:getattr(args,arm.replace('-','_')) for arm in ADAPTERS},suite=args.suite)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({arm:{game:{key:row[key] for key in (
        'paired_mean_return_delta','original_success_to_failure','retention_fraction_of_original_successes')}
        for game,row in comparison['groups'].items()}
        for arm,comparison in report['compared_to_original'].items()},indent=2))
