"""Independent CPU records and artifacts audit for additional RGB rehearsal."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from scripts.training.data import require

def fingerprint_tensors(tensors):
    import torch
    digest=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        tensor=value.detach().cpu().contiguous();digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}:'.encode());digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def audit_records(records,groups,rgb,*,arm,expected_inputs):
    import torch
    from scripts.training.audio_replay import examples_for_step
    require(len(records)==64,'complete64 rehearsal updates required')
    exposure=Counter();rgb_labels=Counter();audio_labels={'present':Counter(),'missing':Counter()};errors=[]
    for step,record in enumerate(records):
        expected=examples_for_step(groups,rgb,arm=arm,step=step,seed=20261101)
        require(record['step']==step+1 and record['mask_dropout'] is (step%8==7) and len(record['examples'])==len(expected),'rehearsal step/mask/term count differs')
        for example,saved in zip(expected,record['examples']):
            require(all(saved[key]==value for key,value in {'kind':example['kind'],'sample_id':example['row']['sample_id'],'target_class':example['target_class']}.items()),'actual rehearsal TRAIN sample/order/label differs')
            require({k:v for k,v in saved['sensor_evidence'].items() if k!='prepare_ms'}==expected_inputs(example),'actual rehearsal sensor inputs differ')
            values=saved['raw_class_logits'];require(isinstance(values,list) and len(values)==5 and all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite5logits required')
            ce=float(torch.nn.functional.cross_entropy(torch.tensor([values],dtype=torch.float32)/2,torch.tensor([example['target_class']])))
            error=abs(ce-saved['loss']);require(math.isfinite(saved['loss']) and error<=2e-6,'raw CE differs');errors.append(error)
            if example['kind']=='audio':
                exposure[(saved['sample_id'],example['missing'])]+=1;audio_labels['missing' if example['missing'] else 'present'][str(example['target_class'])]+=1
            else:rgb_labels[str(example['target_class'])]+=1
        actual=record['examples'];audio=[e['loss'] for e in actual if e['kind']=='audio'];rgb_loss=[e['loss'] for e in actual if e['kind']=='rgb']
        require(abs(record['objective']-sum(e['loss'] for e in actual)/4)<1e-12,'CE quarter objective differs')
        require(abs(record['audio_mean_ce']-sum(audio)/2)<1e-12 and (record['rgb_mean_ce'] is None if not rgb_loss else abs(record['rgb_mean_ce']-sum(rgb_loss)/2)<1e-12),'reported per-modality CE differs')
        require(all(type(record[k]) in (int,float) and math.isfinite(record[k]) and record[k]>=0 for k in ('gradient_norm','step_ms','peak_cuda_allocated_mib')),'nonfinite gradient/cost')
        require(record['gradient_norm']>0 and abs(record['gradient_clip_factor']-min(1.,1/(record['gradient_norm']+1e-6)))<1e-12,'gradient clip evidence differs')
    require(dict(audio_labels['present'])=={'1':56,'3':56} and dict(audio_labels['missing'])=={'1':8,'3':8},'frozen128audio/mask label budget differs')
    require(dict(rgb_labels)==({'0':32,'1':32,'2':32,'3':32} if arm=='rgb-replay' else {}),'frozen additionalRGB label budget differs')
    return {'updates':64,'audio_label_uses':sum(exposure.values()),'missing_audio_label_uses':sum(v for (s,m),v in exposure.items() if m),
            'rgb_label_uses':sum(rgb_labels.values()),'audio_label_counts':{k:dict(v) for k,v in audio_labels.items()},'rgb_label_counts':dict(rgb_labels),
            'per_sample_audio_exposure':[{'sample_id':s,'missing':m,'count':v} for (s,m),v in sorted(exposure.items())],
            'maximum_ce_difference':max(errors),'ce_absolute_tolerance':2e-6,'all64_gradients_finite_nonzero':True}


def read_segments(segments):
    from scripts.training.audio_batch_audit import load_checkpoint
    from scripts.training.audio_replay_train import source_paths,PARENT_COHORT,RGB_MANIFEST
    from scripts.training.data import read_jsonl,sha256_file
    require(segments,'rehearsal segments required');configs=[];records=[];previous=None
    for directory in segments:
        root=Path(directory).resolve();config=json.loads((root/'config.json').read_text());status=json.loads((root/'run-status.json').read_text());result=json.loads((root/'result.json').read_text());identity=config['identity']
        fixed={'seed':20261101,'parent_cohort':PARENT_COHORT,'rgb_dataset':RGB_MANIFEST,'parent_copy':'parameters_only24;freshoptimizer+newseed','rank':8,'alpha':16,'dropout':0.,'target_modules':['q_proj','v_proj'],'dtype':'bfloat16','optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'clip':1.,'temperature':2.,'audio_terms':2,'each_ce_coefficient':.25,'frozen_final_updates':64,'mask_dropout':'globalzero-based-update%8==7;bothaudio','sampling':'shuffle32sameRGBoppositetonepairs;RGBalternatingWS_AD;v1','inputs':'sensorallowlistonly;nohistory/target/reset/engine/reward'}
        require(identity['arm'] in ('audio-only-control','rgb-replay') and all(identity.get(k)==v for k,v in fixed.items()) and identity['rgb_terms']==(2 if identity['arm']=='rgb-replay' else 0),'frozen rehearsal configuration changed')
        require(identity['code']=={p.name:sha256_file(p) for p in source_paths()},'rehearsal training source changed')
        require(config['start_step']==len(records) and config['resume']==(str(previous) if previous else None),'rehearsal segment chain changed')
        require(not configs or identity==configs[0]['identity'],'rehearsal restore identity changed')
        require(status['state']=='completed' and status['active_sample'] is None and status['completed_total_updates']==status['planned_total_updates']==config['total_steps'],'incomplete rehearsal training process')
        rows=list(read_jsonl(root/'steps.jsonl'));count=config['total_steps']-config['start_step']
        require(len(rows)==count and result['completed_optimizer_steps']==config['total_steps'] and result['new_updates']==count and result['new_audio_label_uses']==2*count and result['new_rgb_label_uses']==(2*count if identity['arm']=='rgb-replay' else 0),'actual rehearsal budget differs')
        previous=root/result['checkpoint'];state=load_checkpoint(previous)
        require(state['identity']==identity and state['step']==config['total_steps'],'rehearsal checkpoint differs')
        configs.append(config);records.extend(rows)
    return configs,records,state,previous

def compare_resume(resumed,continuous):
    from scripts.training.audio_batch_audit import exact
    _,a,sa,_=read_segments(resumed);_,b,sb,_=read_segments([continuous])
    require(len(a)==len(b)==8 and exact(sa,sb),'GPU4+4vs8 rehearsal parameters/optimizer/RNG differ')
    def effective(records):
        return [{**{k:v for k,v in r.items() if k not in ('step_ms','peak_cuda_allocated_mib','examples')},'examples':[{**e,'sensor_evidence':{k:v for k,v in e['sensor_evidence'].items() if k!='prepare_ms'}} for e in r['examples']]} for r in records]
    require(effective(a)==effective(b),'rehearsal actual input/logit/loss/gradient sequence differs')
    return {'steps':8,'all24_trainables_optimizer_PythonTorchCUDA_RNG_bitwise_equal':True,'actual_inputs_logits_losses_gradients_equal':True,'maximum_parameter_difference':0.,'claim':'actualGPU4+4vs8 includes globalstep8 missing; continuous8extra labels outside mainbudget'}

def audit_training(segments,*,model,audio_dataset,rgb_dataset,cohort_path):
    import torch
    from types import SimpleNamespace,MethodType
    from transformers import AutoProcessor
    from scripts.training.audio_batch import group_training_rows
    from scripts.training.audio_batch_audit import load_checkpoint,audit_export
    from scripts.training.audio_batch_eval import validate_cohort
    from scripts.training.audio_batch_train import image_processor_identity,sensor_payload
    from scripts.training.audio_cue_data import load_audio_dataset
    from scripts.training.audio_replay import rgb_groups
    from scripts.training.audio_replay_train import PARENT_COHORT,rgb_evidence,spectrogram_packages,require_parent_dependencies
    from scripts.training.audio_vision import prepare_audio
    from scripts.training.bandit_train import checked_resets
    from scripts.training.data import read_jsonl,sha256_file
    from scripts.training.feedback import audit_feedback_episodes
    from scripts.training.hf_backend import RGBBackend,load_dataset,model_identity
    configs,records,state,checkpoint=read_segments(segments);identity=configs[0]['identity']
    require(sha256_file(cohort_path)==PARENT_COHORT,'parentcohort changed');cohort=json.loads(Path(cohort_path).read_text());validate_cohort(cohort)
    parent_path=Path(configs[0]['parent']);parent=load_checkpoint(parent_path)
    require(parent['step']==200 and parent['identity']==cohort['training']['paired-context']['identity'] and sha256_file(parent_path/'state.pt')==identity['parent_state_sha256'],'parent state provenance differs')
    require(audit_export(parent_path,parent,base_snapshot=configs[0]['model']['snapshot'])==identity['parent_adapter']==cohort['policies']['paired-context'],'parent export differs')
    require(fingerprint_tensors(parent['trainables'])==identity['initialization_fingerprint'],'actual initialization differs from copiedparent24')
    require(model_identity(model)==configs[0]['model'] and identity['model']==cohort['model_fingerprint'],'rehearsal base changed')
    audio,am,ad,ai=load_audio_dataset(audio_dataset);rgb,rm,rd=load_dataset(rgb_dataset)
    require(ad==identity['audio_dataset']==parent['identity']['dataset'] and rd==identity['rgb_dataset'],'rehearsal corpus digests differ')
    groups=group_training_rows(audio['train'],list(read_jsonl(am['source_path'])));rg=rgb_groups(rgb['train']);episodes=list(read_jsonl(rm['source_path']));audit_feedback_episodes(episodes,rm['data_root']);checked_resets(rgb['train'],episodes)
    processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    require_parent_dependencies(identity,parent['identity'])
    require(image_processor_identity(processor)==identity['image_processor'] and spectrogram_packages()==identity['spectrogram_packages'],'rehearsal actual processor/spectrogram dependencies differ')
    backend=SimpleNamespace(processor=processor,device='cpu',synchronize=lambda:None,torch=torch);backend.prepare=MethodType(RGBBackend.prepare,backend);cache={}
    def expected(e):
        row=e['row'];key=(e['kind'],row['sample_id'],e['missing'])
        if key not in cache:
            if e['kind']=='audio':_,evidence=prepare_audio(backend,sensor_payload(row,ai),am['data_root'],Path('.'),force_missing=e['missing'])
            else:_,evidence=rgb_evidence(backend,row,rm['data_root'])
            cache[key]={k:v for k,v in evidence.items() if k!='prepare_ms'}
        return cache[key]
    report=audit_records(records,groups,rg,arm=identity['arm'],expected_inputs=expected)
    adapter=audit_export(checkpoint,state,base_snapshot=configs[0]['model']['snapshot'])
    return {'schema_version':1,'identity':identity,'actual_training':report,'checkpoint_step':state['step'],'trainable_tensors':24,'audited_adapter_identity':adapter,'export_vs_checkpoint_all24_exact':True,'parent24_copy_initial_fingerprint_exact':True,'segments':len(segments)}


def freeze_cohort(control,replay,*,model,audio_dataset,rgb_dataset,cohort_path):
    from scripts.training.audio_replay_protocol import validate_cohort
    from scripts.training.data import sha256_file
    parent=json.loads(Path(cohort_path).read_text());training={}
    policies={'original':None,'parent200':parent['policies']['paired-context']}
    for arm,segments in (('audio-only-control',control),('rgb-replay',replay)):
        report=audit_training(segments,model=model,audio_dataset=audio_dataset,rgb_dataset=rgb_dataset,cohort_path=cohort_path)
        require(report['identity']['arm']==arm,'rehearsal cohort arm differs');training[arm]=report;policies[arm]=report['audited_adapter_identity']
    value={'schema_version':1,'model_fingerprint':parent['model_fingerprint'],'parent_cohort_sha256':sha256_file(cohort_path),
           'parent_initialization_fingerprint':training['audio-only-control']['identity']['initialization_fingerprint'],
           'policies':policies,'training':training,'claim':'bothcomplete64update source/input/CE/export/parentcopy audits frozen before any new metrics; unequal RGBlabels/compute'}
    validate_cohort(value);return value

if __name__=='__main__':
    import argparse
    import torch
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segments',type=Path,nargs='+');p.add_argument('--continuous',type=Path)
    p.add_argument('--control-segments',type=Path,nargs='+');p.add_argument('--replay-segments',type=Path,nargs='+')
    for name in ('model','audio-dataset','rgb-dataset','cohort'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.continuous:report=compare_resume(a.segments,a.continuous)
    else:
        kwargs={'model':a.model,'audio_dataset':a.audio_dataset,'rgb_dataset':a.rgb_dataset,'cohort_path':a.cohort}
        require(all(kwargs.values()),'model/bothTRAINdatasets/parentcohort required')
        report=freeze_cohort(a.control_segments,a.replay_segments,**kwargs) if a.control_segments else audit_training(a.segments,**kwargs)
    require(not torch.cuda.is_initialized(),'CPU audit must not initialize CUDA')
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
