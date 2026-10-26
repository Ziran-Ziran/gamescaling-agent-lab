"""Fixed small-N rehearsal protocol, all final artifacts before new metrics."""
import hashlib
import json
from collections import Counter
from scripts.training.data import require
from scripts.training.retention_compare import MODEL_FINGERPRINT
from scripts.training.audio_replay_train import PARENT_COHORT
AUDIO_POLICIES=('parent200','audio-only-control','rgb-replay')
RGB_POLICIES=('original',*AUDIO_POLICIES)
CONDITIONS=('aligned','missing','swapped','rgb-only','rgb-blank')
SUITES={
 'primary':{'games':['reach','hazard'],'seed_start':50000,'episodes_per_game':8,'max_steps':1,'history_len':0,'decoder':'first-key','max_new_tokens':64},
 'secondary':{'games':['keydoor'],'seed_start':50000,'episodes_per_game':4,'max_steps':8,'history_len':0,'decoder':'first-key','max_new_tokens':64}}

def frozen_schedule(suite):
    require(suite in ('transfer','iid'),'fresh rehearsal suite required')
    game,start=('hazard',47000) if suite=='transfer' else ('reach',48000)
    return [{'variant':game,'context':context,'colors':colors,'tone':tone} for context in range(start,start+2) for colors in (0,1) for tone in (0,1)]

def validate_cohort(value):
    require(value['schema_version']==1 and value['model_fingerprint']==MODEL_FINGERPRINT and value['parent_cohort_sha256']==PARENT_COHORT,'fixed rehearsal parent/base changed')
    require(set(value['policies'])==set(RGB_POLICIES) and value['policies']['original'] is None,'allfour frozen rehearsal identities required')
    parent=value['policies']['parent200']
    require(parent['fingerprint']=='a0f855a986117d92babdec515f0ba20b6c13a254f4d61d8a3c4225a28d6aa6ea' and parent['files']=={'adapter_config.json':'2dc33d71032002e5ed0d6000e5ec8555d1de33062a4570a9b4900168d1a8f316','adapter_model.safetensors':'e017302b535a028872b5899493259ee7429020ffb5da53b0933f1736e6f7fa44'},'namedpaired200 parent changed')
    require(set(value['training'])=={'audio-only-control','rgb-replay'},'both64update rehearsal arms required')
    common=[];exposures=[]
    for arm in ('audio-only-control','rgb-replay'):
        row=value['training'][arm];identity=row['identity'];actual=row['actual_training']
        require(row['checkpoint_step']==64 and row['trainable_tensors']==24 and row['export_vs_checkpoint_all24_exact'] is True and row['parent24_copy_initial_fingerprint_exact'] is True,'complete64update/24export/parentcopy audit required')
        adapter=value['policies'][arm];files=adapter['files']
        require(set(files)=={'adapter_config.json','adapter_model.safetensors'} and all(isinstance(d,str) and len(d)==64 and all(c in '0123456789abcdef' for c in d) for d in files.values()) and adapter['fingerprint']==hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'complete adapter file hashes/fingerprint required')
        require(row['audited_adapter_identity']==value['policies'][arm] and identity['arm']==arm and identity['parent_adapter']==parent,'rehearsal exported identity differs')
        require(identity['initialization_fingerprint']==value['parent_initialization_fingerprint'] and identity['parent_cohort']==PARENT_COHORT,'warmparent initialization differs')
        require(actual['updates']==64 and actual['audio_label_uses']==128 and actual['missing_audio_label_uses']==16 and actual['rgb_label_uses']==(128 if arm=='rgb-replay' else 0) and actual['all64_gradients_finite_nonzero'] is True,'fixed actual rehearsal budget differs')
        require(actual['audio_label_counts']=={'present':{'1':56,'3':56},'missing':{'1':8,'3':8}} and actual['rgb_label_counts']==({'0':32,'1':32,'2':32,'3':32} if arm=='rgb-replay' else {}),'balanced rehearsal label counts differ')
        exposure=actual['per_sample_audio_exposure']
        require(all(type(e['missing']) is bool and type(e['count']) is int and 0<e['count']<=2 for e in exposure) and len({(e['sample_id'],e['missing']) for e in exposure})==len(exposure),'complete per-sample/mask exposure required')
        total=Counter()
        for e in exposure:total[e['sample_id']]+=e['count']
        require(len(total)==64 and set(total.values())=={2} and sum(e['count'] for e in exposure if e['missing'])==16,'two complete64row epochs/maskbudget required')
        common.append({k:v for k,v in identity.items() if k not in ('arm','rgb_terms')});exposures.append(actual['per_sample_audio_exposure'])
    require(common[0]==common[1] and exposures[0]==exposures[1],'common audio objective/init/source/exposures differ')
